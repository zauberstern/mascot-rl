"""Causal panel observation builder: Phase L cube + portfolio state → flat obs."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.features.blocks.assemble import assemble_equity_feature_cube
from src.features.blocks.portfolio_state import build_portfolio_state_features

# Static C + portfolio channels: w_prev, days_held, cum_cost, w_base.
_PORTFOLIO_CHANNEL_NAMES = ("w_prev", "days_held", "cum_cost", "w_base")


def equal_weight_on_mask(mask: np.ndarray) -> np.ndarray:
    """EW reference portfolio on the active slot mask (RASP ``w_base``)."""
    m = np.asarray(mask, dtype=np.float64).reshape(-1)
    active = float(m.sum())
    if active <= 0.0:
        return np.full(m.size, 1.0 / max(m.size, 1), dtype=np.float64)
    return m / active


class PanelObservationBuilder:
    """Precompute ``(T, K, C)`` features; at step ``t`` emit flat ``(K * C_obs,)``.

    Static channels come from :func:`assemble_equity_feature_cube`. Dynamic
    portfolio state (``w_prev``, days held, cum cost, ``w_base``) is appended
    per asset so the policy can learn cost-aware trading and active tilts
    around the EW-on-mask reference (Phase L9 / Jiang PVM / RASP).
    """

    def __init__(
        self,
        returns: np.ndarray,
        *,
        factors: np.ndarray | None = None,
        extras: dict[str, Any] | None = None,
        seq_len: int = 1,
        normalize: bool = True,
    ) -> None:
        r = np.asarray(returns, dtype=np.float64)
        if r.ndim != 2:
            raise ValueError(f"returns must be (T, K), got {r.shape}")
        self.returns = r
        self.T, self.K = int(r.shape[0]), int(r.shape[1])
        self.seq_len = max(1, int(seq_len))
        ex = dict(extras or {})
        if factors is not None:
            ex = {**ex, "factors": factors}
        slot_valid = ex.get("slot_valid_mask")
        if slot_valid is None:
            slot_valid = ex.pop("slot_mask", None)
        if slot_valid is not None:
            ex["slot_valid_mask"] = slot_valid
        self.cube, self.names = assemble_equity_feature_cube(
            r, extras=ex, normalize=normalize
        )
        # float32 halves the resident (T, K, C) cube for sequence models at K=100.
        self.cube = np.asarray(self.cube, dtype=np.float32)
        self.n_channels = int(self.cube.shape[-1])
        self.portfolio_channel_names = list(_PORTFOLIO_CHANNEL_NAMES)
        # Static C + 4 portfolio-state channels (includes w_base).
        self.obs_channels_per_asset = self.n_channels + len(self.portfolio_channel_names)
        self._days_held = np.zeros(self.K, dtype=np.float64)
        self._cum_cost = np.zeros(self.K, dtype=np.float64)
        self._w_last = np.zeros(self.K, dtype=np.float64)
        self._slot_mask = np.ones(self.K, dtype=np.float64)

    def set_slot_mask(self, mask: np.ndarray | None) -> None:
        """Update the active-slot mask used for the ``w_base`` channel."""
        if mask is None:
            self._slot_mask = np.ones(self.K, dtype=np.float64)
            return
        m = np.asarray(mask, dtype=np.float64).reshape(-1)
        if m.size != self.K:
            raise ValueError(f"slot mask size {m.size} != K={self.K}")
        self._slot_mask = (m > 0.5).astype(np.float64)

    def reset_portfolio_state(self) -> None:
        self._days_held[:] = 0.0
        self._cum_cost[:] = 0.0
        self._w_last[:] = 0.0

    def update_portfolio_state(
        self,
        w: np.ndarray,
        *,
        step_cost: float = 0.0,
    ) -> None:
        """Call after each env step with post-fill weights and total step cost."""
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.size != self.K:
            raise ValueError(f"w size {w.size} != K={self.K}")
        same_sign = (np.sign(w) == np.sign(self._w_last)) & (np.abs(w) > 1e-12)
        self._days_held = np.where(same_sign, self._days_held + 1.0, 1.0)
        self._days_held = np.where(np.abs(w) <= 1e-12, 0.0, self._days_held)
        # Allocate step cost by |Δw| share (implementation-state proxy).
        dw = np.abs(w - self._w_last)
        denom = float(dw.sum())
        if denom > 1e-12 and float(step_cost) != 0.0:
            self._cum_cost = self._cum_cost + float(step_cost) * (dw / denom)
        self._w_last = w.copy()

    def _portfolio_block(self, w_prev: np.ndarray) -> np.ndarray:
        """Return ``(K, 4)`` portfolio channels including ``w_base``."""
        w = np.asarray(w_prev, dtype=np.float64).reshape(-1)
        if w.size != self.K:
            w = np.zeros(self.K, dtype=np.float64)
        port3 = build_portfolio_state_features(w, self._days_held, self._cum_cost)
        w_base = equal_weight_on_mask(self._slot_mask).reshape(self.K, 1)
        return np.concatenate([port3, w_base], axis=-1)

    def __call__(self, t: int, w_prev: np.ndarray) -> np.ndarray:
        idx = int(max(0, min(int(t), self.T - 1)))
        static = self.cube[idx]  # (K, C)
        static = np.nan_to_num(static, nan=0.0, posinf=0.0, neginf=0.0)
        # EarnMore-style: replace invalid slots before the body sees features.
        if float(np.min(self._slot_mask)) < 0.5:
            from src.features.mask_tokens import apply_mask_tokens_to_cube

            static = apply_mask_tokens_to_cube(static, self._slot_mask)
            self._representation_masked = True
        else:
            self._representation_masked = False
        port = self._portfolio_block(w_prev)
        n_port = int(port.shape[-1])
        # (K, C+4)
        feats = np.concatenate([static, port], axis=-1)
        if self.seq_len <= 1:
            return feats.reshape(-1)
        # Causal lag stack: [t-seq+1 ... t] flattened as (K, seq, C_obs) → flat.
        start = max(0, idx + 1 - self.seq_len)
        hist = self.cube[start : idx + 1]
        if hist.shape[0] < self.seq_len:
            pad = np.zeros(
                (self.seq_len - hist.shape[0], self.K, self.n_channels),
                dtype=np.float64,
            )
            hist = np.concatenate([pad, hist], axis=0)
        hist = np.nan_to_num(hist, nan=0.0)
        if float(np.min(self._slot_mask)) < 0.5:
            from src.features.mask_tokens import apply_mask_tokens_to_cube

            # hist is (seq, K, C); broadcast slot mask across seq.
            hist = apply_mask_tokens_to_cube(
                hist, np.broadcast_to(self._slot_mask[None, :], hist.shape[:2])
            )
            self._representation_masked = True
        # Broadcast portfolio state across the seq axis (current implementation state).
        port_b = np.broadcast_to(port[None, :, :], (self.seq_len, self.K, n_port))
        stacked = np.concatenate([hist, port_b], axis=-1)  # (seq, K, C_obs)
        # Asset-major flatten for single-agent MLP: (K, seq, C) → (K*seq*C,)
        return np.transpose(stacked, (1, 0, 2)).reshape(-1)
