"""Layer 3 master feature extractor: temporal backends + optional DHGNN."""
from __future__ import annotations

import torch
import torch.nn as nn

from mascotrl.features.mamba2 import AssetTemporalMamba
from mascotrl.features.dhgnn import SpatialDHGNN


class _AssetTemporalGRU(nn.Module):
    """GRU temporal block (spectrum ablation)."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=max(int(d_state) * 2, d_model),
            batch_first=True,
        )
        self.out = nn.Linear(max(int(d_state) * 2, d_model), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.out(y)


class _AssetTemporalLSTM(nn.Module):
    """LSTM temporal block (Jiang/Sirignano lineage)."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        # FEATNET LSTM+wide extras hits oneDNN "could not create a primitive"
        # on all CPCV folds; GRU path is fine. Disable mkldnn for LSTM only.
        if torch.backends.mkldnn.is_available():
            torch.backends.mkldnn.enabled = False
        hidden = max(int(d_state) * 2, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.out(y)


class _AssetTemporalMLP(nn.Module):
    """MLP over last timestep (Du/Huang reference architecture)."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        hidden = max(int(d_state) * 2, d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> use last step
        return self.net(x[:, -1, :])


class _AssetTemporalTransformer(nn.Module):
    """Tiny Transformer encoder over the sequence (plausible spectrum cell)."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        nhead = 2 if d_model % 2 == 0 else 1
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(int(d_state) * 4, d_model),
            batch_first=True,
            dropout=0.0,
        )
        self.enc = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.enc(x)
        return self.out(y[:, -1, :])


_BACKEND_CTORS = {
    "mamba": AssetTemporalMamba,
    "gru": _AssetTemporalGRU,
    "lstm": _AssetTemporalLSTM,
    "mlp": _AssetTemporalMLP,
    "transformer": _AssetTemporalTransformer,
}


# @lat: [[core#Feature extractor]]
class AlphaFeatureExtractor(nn.Module):
    """
    Combines independent per-asset temporal blocks with optional DHGNN.
    Produces Enriched States (E_i) for the HAPPO Policy Engine.

    Spectrum knobs:
      - temporal_backend: mlp | gru | lstm | transformer | mamba
      - use_dhgnn: if False, skip spatial hypergraph (identity pass on Z)
      - share_temporal_encoder: one temporal block applied to all assets
        (eq_alloc default; ModuleList remains the status-quo ablation)
    """

    def __init__(
        self,
        num_assets: int,
        d_model: int,
        d_state: int = 16,
        *,
        temporal_backend: str = "mamba",
        use_dhgnn: bool = True,
        spatial_mode: str = "dhgnn_copula",
        share_temporal_encoder: bool = False,
        update_incidence_at_eval: bool = False,
    ):
        super().__init__()
        self.use_dhgnn = bool(use_dhgnn)
        self.temporal_backend = str(temporal_backend).lower()
        self.spatial_mode = str(spatial_mode).lower()
        self.share_temporal_encoder = bool(share_temporal_encoder)
        self.num_assets = int(num_assets)
        _allowed_spatial = {"dhgnn_copula", "dhgnn_pearson", "none", "off"}
        if self.spatial_mode not in _allowed_spatial:
            raise ValueError(
                f"unknown spatial_mode={spatial_mode!r}; "
                f"allowed={sorted(_allowed_spatial - {'off'})}"
            )
        if self.spatial_mode in ("none", "off"):
            self.use_dhgnn = False
        ctor = _BACKEND_CTORS.get(self.temporal_backend)
        if ctor is None:
            raise ValueError(
                f"unknown temporal_backend={temporal_backend!r}; "
                f"allowed={sorted(_BACKEND_CTORS)}"
            )
        if self.share_temporal_encoder:
            # Single shared temporal block; forward loops over assets.
            self.temporal_blocks = nn.ModuleList(
                [ctor(d_model=d_model, d_state=d_state)]
            )
        else:
            self.temporal_blocks = nn.ModuleList(
                [ctor(d_model=d_model, d_state=d_state) for _ in range(num_assets)]
            )
        if self.use_dhgnn:
            dhgnn_mode = (
                "dhgnn_copula"
                if self.spatial_mode in ("none", "off")
                else self.spatial_mode
            )
            self.spatial_dhgnn = SpatialDHGNN(
                d_model=d_model,
                num_assets=num_assets,
                spatial_mode=dhgnn_mode,
                update_incidence_at_eval=bool(update_incidence_at_eval),
                # Explicit spatial_mode=dhgnn_pearson is the allow signal at this layer.
                allow_pearson_incidence=(dhgnn_mode == "dhgnn_pearson"),
            )
        else:
            self.spatial_dhgnn = None

    def forward(self, raw_states: torch.Tensor, iv_features: torch.Tensor) -> torch.Tensor:
        # raw_states: (Batch, num_assets, Seq_Len, d_model)
        n_assets = raw_states.shape[1]
        if self.share_temporal_encoder:
            # Vectorized shared path: (B, K, L, D) -> (B*K, L, D) -> one call.
            bsz, k, seq_len, d_model = raw_states.shape
            flat = raw_states.reshape(bsz * k, seq_len, d_model)
            t_out = self.temporal_blocks[0](flat)
            if t_out.dim() == 3:
                t_out = t_out[:, -1, :]
            Z = t_out.reshape(bsz, k, -1)
        else:
            temporal_outputs = []
            for k, block in enumerate(self.temporal_blocks):
                t_out = block(raw_states[:, k, :, :])
                # MLP/Transformer return (B, D); Mamba/GRU/LSTM return (B, T, D) or (B, D)
                if t_out.dim() == 3:
                    t_out = t_out[:, -1, :]
                temporal_outputs.append(t_out)
            Z = torch.stack(temporal_outputs, dim=1)
        if not self.use_dhgnn:
            return Z
        return self.spatial_dhgnn(Z, iv_features)
