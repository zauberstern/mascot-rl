"""Polaris-safe Pure-PyTorch Mamba-2 with chunked State-Space Duality (SSD)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def hippo_legs_diagonal(d_state: int) -> torch.Tensor:
    """
    HiPPO-LegS diagonal eigenvalues for selective SSM init.

    Full HiPPO-LegS is lower-triangular; diagonal entries are ``-(n+1)``.
    Selective Mamba keeps a diagonal ``A``; this is the LegS-consistent init
    (uniform historical weight / scaled Legendre measure), not a random draw.
    """
    return -torch.arange(1, d_state + 1, dtype=torch.float32)


class PureTorchMamba2(nn.Module):
    """
    Selective SSM with ZOH discretization + HiPPO-LegS ``A`` init.

    Chunked SSD (Polaris-safe): pad into blocks of `chunk_size`, run a short
    sequential scan only over the chunk length (parallel across chunks), then
    a second short scan over n_chunks ≪ L to propagate boundary states.
    Avoids `for t in range(L)` over the full sequence.

    No ``mamba_ssm`` / CUDA kernels. AWS production is CPU-only
    (``m7i-flex.large``, ``torch+cpu``); local Polaris/gfx803 is dev-only.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        self.chunk_size = max(int(chunk_size), 1)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        # A = -exp(A_log) recovers HiPPO-LegS diagonal −(n+1).
        legs = hippo_legs_diagonal(d_state)
        self.A_log = nn.Parameter(torch.log(-legs))  # log(n+1)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        b, l, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_c = self.conv1d(x_branch.transpose(1, 2))[:, :, :l].transpose(1, 2)
        x_c = F.silu(x_c)

        x_dbl = self.x_proj(x_c)
        delta_raw, B, C = torch.split(
            x_dbl, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))
        A = -torch.exp(self.A_log.float())

        y = self._ssd_chunked_scan(x_c, delta, A, B, C)
        y = y + x_c * self.D
        y = y * F.silu(z)
        return self.out_proj(y)

    def _ssd_chunked_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, h = x.shape
        n = A.shape[0]
        # Clamp chunk size to the real sequence length so L=1 never pads to 32
        # (autograd retention explosion on short feature_seq_len defaults).
        cs = min(self.chunk_size, max(int(seqlen), 1))

        pad_len = (cs - seqlen % cs) % cs
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
            delta = F.pad(delta, (0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, pad_len))
            seqlen = seqlen + pad_len

        n_chunks = seqlen // cs
        x_c = x.view(bsz, n_chunks, cs, h)
        dt_c = delta.view(bsz, n_chunks, cs, h)
        B_c = B.view(bsz, n_chunks, cs, n)
        C_c = C.view(bsz, n_chunks, cs, n)

        # Precompute ZOH coeffs: (B, C, T, H, N)
        dt = dt_c.unsqueeze(-1)
        dA = torch.exp(dt * A.view(1, 1, 1, 1, n))
        dB = (dA - 1.0) / (A.view(1, 1, 1, 1, n) + 1e-8) * B_c.unsqueeze(3)
        Bu = dB * x_c.unsqueeze(-1)

        # --- Pass 1: intra-chunk from zero (parallel across chunks) ---
        state = torch.zeros(bsz, n_chunks, h, n, device=x.device, dtype=x.dtype)
        chunk_decay = torch.ones(bsz, n_chunks, h, n, device=x.device, dtype=x.dtype)
        for t in range(cs):
            state = dA[:, :, t] * state + Bu[:, :, t]
            chunk_decay = chunk_decay * dA[:, :, t]
        chunk_final = state  # (B, C, H, N)

        # --- Pass 2: inter-chunk carry (length n_chunks ≪ L) ---
        carries = []
        carry = torch.zeros(bsz, h, n, device=x.device, dtype=x.dtype)
        for c in range(n_chunks):
            carries.append(carry)
            carry = carry * chunk_decay[:, c] + chunk_final[:, c]
        carry_in = torch.stack(carries, dim=1)  # (B, C, H, N)

        # --- Pass 3: re-scan with carry_in → outputs (parallel across chunks) ---
        state = carry_in
        ys = []
        for t in range(cs):
            state = dA[:, :, t] * state + Bu[:, :, t]
            yt = torch.einsum("bchn,bcn->bch", state, C_c[:, :, t])
            ys.append(yt)
        y = torch.stack(ys, dim=2).reshape(bsz, seqlen, h)
        if pad_len:
            y = y[:, :-pad_len, :]
        return y


class AssetTemporalMamba(nn.Module):
    """Independent Mamba-2-style SSM block for a single asset."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, chunk_size: int = 32):
        super().__init__()
        self.mamba = PureTorchMamba2(
            d_model, d_state=d_state, d_conv=d_conv, expand=2, chunk_size=chunk_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)
