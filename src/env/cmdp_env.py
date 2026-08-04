"""CMDP environment: mark-to-market PnL only; frictions live in cvxpylayers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.features.extractor import AlphaFeatureExtractor
from src.policy.happo import HAPPOEngine
from src.pricing.interface import get_portfolio_greeks

# Spectrum train worlds (declared, not exclusive). Aliases normalized in __init__.
ALLOWED_TRANSITION_SOURCES = frozenset(
    {
        "historical",
        "rbergomi",
        "gbm",
        "heston",
        "garch",
        "sabr",
        "hybrid_pretrain_finetune",
    }
)


@dataclass
class StepResult:
    enriched: torch.Tensor
    macro: torch.Tensor
    deltas: torch.Tensor
    reward: torch.Tensor
    done: bool
    info: dict
    truncated: bool = False
    terminated: bool = False


# @lat: [[core#CMDPEnv]]
class CMDPEnv:
    """
    Reward law: R_t = w_exec · (P_t - P_{t-1}) − optional execution drag.

    Default drag is zero when both coeffs are 0. Overnight configs set a small
    linear spread + √turnover impact in R_t *alongside* hard CMDP τ/δ (not as a
    soft substitute for the projection — Buehler collapse risk remains gated by QP).

    execution_drag_mode:
      - fixed (status quo): drag = (bps/1e4) * ||Δw||₁
      - vol_scaled: drag *= (σ_ATM / σ_ref) clipped to [vol_floor, vol_cap]
    """

    def __init__(
        self,
        surfaces: torch.Tensor,
        feature_extractor: AlphaFeatureExtractor,
        policy: HAPPOEngine,
        d_model: int,
        macro_dim: int,
        use_gpu: bool = True,
        rate: float = 0.02,
        macro_series: torch.Tensor | None = None,
        seq_len: int = 32,
        execution_spread_bps: float = 0.0,
        execution_impact_coef: float = 0.0,
        risk_guard: object | None = None,
        execution_drag_mode: str = "fixed",
        execution_vol_ref: float = 0.20,
        execution_vol_floor: float = 0.05,
        execution_vol_cap: float = 1.0,
        funding: Any | None = None,
        *,
        acceptance_mode: bool = False,
        transition_source: str = "rbergomi",
        spot_paths: torch.Tensor | None = None,
        atm_iv_paths: torch.Tensor | None = None,
    ):
        # surfaces: [P, K, T, S, M]
        src = str(transition_source).lower().strip()
        # Alias legacy labels onto spectrum train_world ids.
        aliases = {
            "synthetic": "rbergomi",
            "rbergomi_dupire": "rbergomi",
            "sim": "rbergomi",
            "simulator": "rbergomi",
            "optionmetrics": "historical",
        }
        src = aliases.get(src, src)
        if src not in ALLOWED_TRANSITION_SOURCES:
            raise ValueError(
                f"unknown transition_source={transition_source!r}; "
                f"allowed={sorted(ALLOWED_TRANSITION_SOURCES)}"
            )
        # Declared world is recorded; acceptance no longer forbids synth when listed.
        # spot_paths may be omitted at construction (allowlist / stamp tests) but
        # non-rbergomi worlds fail closed on reset/step/_evolve_spot without them.
        self.acceptance_mode = bool(acceptance_mode)
        self.transition_source = src
        self.surfaces = surfaces
        self.spot_paths = spot_paths  # optional [P, K, T]
        self.atm_iv_paths = atm_iv_paths  # optional [P, K, T]
        if self.spot_paths is not None:
            if self.spot_paths.ndim != 3:
                raise ValueError("spot_paths must be [P, K, T]")
            if tuple(self.spot_paths.shape[:3]) != (
                surfaces.shape[0],
                surfaces.shape[1],
                surfaces.shape[2],
            ):
                raise ValueError(
                    f"spot_paths shape {tuple(self.spot_paths.shape)} incompatible "
                    f"with surfaces {tuple(surfaces.shape)}"
                )
        self.fe = feature_extractor
        self.policy = policy
        self.d_model = d_model
        self.macro_dim = macro_dim
        self.use_gpu = use_gpu
        self.rate = rate
        self.seq_len = max(4, int(seq_len))
        self.P, self.K, self.T, self.S, self.M = surfaces.shape
        self.macro_series = macro_series  # [T_macro, macro_dim] PIT/ffill series
        self.execution_spread_bps = float(execution_spread_bps)
        self.execution_impact_coef = float(execution_impact_coef)
        self.execution_drag_mode = str(execution_drag_mode)
        self.execution_vol_ref = float(execution_vol_ref)
        self.execution_vol_floor = float(execution_vol_floor)
        self.execution_vol_cap = float(execution_vol_cap)
        self.funding = funding
        self.risk_guard = risk_guard
        self.t = 0
        self.path = 0
        self.macro_start_idx = 0
        self._ep_gen: torch.Generator | None = None
        self.w = torch.zeros(1, self.K)
        self.prev_price: torch.Tensor | None = None
        self.spot = torch.full((self.K,), 100.0)

    def reset(
        self,
        path: int = 0,
        start_t: int = 1,
        *,
        episode_seed: int | None = None,
    ) -> StepResult:
        self.path = int(path) % self.P
        self.t = int(max(1, min(start_t, self.T - 2)))
        self.w = torch.zeros(1, self.K)
        self.prev_price = None  # CRITICAL: do not leak prices across episodes
        if self.transition_source != "rbergomi":
            self._require_spot_paths()
            self.spot = self.spot_paths[self.path, :, self.t].detach().float().clone()
        elif self.spot_paths is not None:
            self.spot = self.spot_paths[self.path, :, self.t].detach().float().clone()
        else:
            self.spot = torch.full((self.K,), 100.0)
        # Dedicated episode RNG: macro window + physical Brownian share one stream
        # so ablation arms with the same (seed, ep) match env noise even when
        # architecture init/act burned unequal global draws. Weight init still
        # differs by architecture (expected — not a confound to eliminate).
        self._ep_gen = None
        if episode_seed is not None:
            self._ep_gen = torch.Generator(device="cpu")
            self._ep_gen.manual_seed(int(episode_seed) % (2**63 - 1))
        if self.macro_series is not None and self.macro_series.shape[0] > self.T:
            max_start = int(self.macro_series.shape[0]) - int(self.T) - 1
            hi = max(1, max_start + 1)
            if self._ep_gen is not None:
                self.macro_start_idx = int(
                    torch.randint(0, hi, (1,), generator=self._ep_gen).item()
                )
            else:
                self.macro_start_idx = int(torch.randint(0, hi, (1,)).item())
        else:
            self.macro_start_idx = 0
        return self._observe(reward=torch.zeros(1), done=False)

    def _macro_at(self) -> torch.Tensor:
        """Chronological macro row — no white-noise fill, no path-scrambled lookahead.

        Training indexes ``macro_series`` by **integer row**
        (``macro_start_idx + t``), never by calendar date / Arctic ``as_of``.
        Duplicate event-time rows in the source lake therefore appear only as a
        repeated adjacent observation in the tensor, not as a PIT date join bug.
        """
        m = torch.zeros(1, self.macro_dim)
        if self.macro_series is None or self.macro_series.numel() == 0:
            # Deterministic surface summary only (still not Gaussian noise).
            iv = self.surfaces[self.path, :, self.t, self.S // 2, self.M // 2]
            m[0, 0] = iv.mean()
            m[0, 1] = self.t / max(self.T, 1)
            if self.macro_dim > 2:
                m[0, 2] = iv.std()
            return m
        # Episode clock walks from randomized macro_start_idx through history.
        curr = int(self.macro_start_idx) + int(self.t)
        idx = min(curr, int(self.macro_series.shape[0]) - 1)
        row = self.macro_series[idx]
        n = min(self.macro_dim, row.numel())
        m[0, :n] = row[:n]
        return m

    def _build_raw_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-channel raw states: ATM + IV log-returns (no sinusoid expand).

        Synth / surface path stacks at least two channels so the observation
        channel matrix is not rank-1. Pad/truncate to ``d_model`` via the shared
        numpy builder.
        """
        from src.features.raw_state import build_raw_states_from_feature_tensor

        seq = min(self.t, self.seq_len)
        start = self.t - seq
        s_mid = self.S // 2
        m_mid = self.M // 2
        iv = self.surfaces[self.path, :, start : self.t, s_mid, m_mid]  # (K, seq)
        iv_np = iv.detach().cpu().numpy().astype(np.float64, copy=False)
        # Simple causal IV returns as a second channel (interim synth features).
        iv_ret = np.zeros_like(iv_np)
        if iv_np.shape[1] > 1:
            prev = np.clip(iv_np[:, :-1], 1e-6, None)
            iv_ret[:, 1:] = np.log(np.clip(iv_np[:, 1:], 1e-6, None) / prev)
        feat = np.stack([iv_np, iv_ret], axis=-1)  # (K, seq, 2)
        raw_np = build_raw_states_from_feature_tensor(feat, d_model=self.d_model)
        raw = torch.from_numpy(raw_np).unsqueeze(0)
        iv_feat = iv[:, -1].unsqueeze(0)
        return raw, iv_feat

    def _require_spot_paths(self) -> None:
        if self.spot_paths is None:
            raise ValueError(
                f"transition_source={self.transition_source!r} requires spot_paths "
                "(fail closed: no silent Euler fallback)"
            )

    def _evolve_spot(self) -> None:
        """Advance spot under the declared transition_source.

        - rbergomi (default): Euler step using ATM local-vol as instantaneous sigma.
        - gbm|heston|garch|sabr|historical|hybrid_*: replay generator/OM spot_paths
          (no re-simulation), so leverage / vol-of-vol from the world survive.
        """
        if self.transition_source != "rbergomi":
            self._require_spot_paths()
            nxt = min(self.t + 1, self.T - 1)
            self.spot = self.spot_paths[self.path, :, nxt].detach().float().clone()
            return
        atm = self.surfaces[self.path, :, self.t, self.S // 2, self.M // 2].clamp_min(1e-3)
        dt = 1.0 / 252.0
        # Physical Brownian — use episode Generator when provided so ablation arms
        # share the same env noise path (not architecture-global-RNG coupled).
        if self._ep_gen is not None:
            z = torch.randn(self.K, generator=self._ep_gen)
        else:
            z = torch.randn(self.K)
        self.spot = self.spot * torch.exp(
            (self.rate - 0.5 * atm * atm) * dt + atm * (dt**0.5) * z
        ).clamp(1.0, 1e4)

    def _greeks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spot = self.spot.detach().float()
        strike = spot.clone()  # ATM book
        tau = torch.full((self.K,), max(self.T - self.t, 1) / 252.0)
        rate = torch.full((self.K,), self.rate)
        vol_stack = self.surfaces[self.path, :, self.t]
        return get_portfolio_greeks(
            spot, strike, tau, rate, vol_stack, use_gpu=self.use_gpu
        )

    def _observe(
        self,
        reward: torch.Tensor,
        done: bool,
        *,
        truncated: bool = False,
        terminated: bool = False,
    ) -> StepResult:
        raw, iv_feat = self._build_raw_states()
        enriched = self.fe(raw, iv_feat)
        macro = self._macro_at()
        prices, deltas, vegas = self._greeks()
        return StepResult(
            enriched=enriched,
            macro=macro,
            deltas=deltas.unsqueeze(0),
            reward=reward,
            done=done,
            truncated=truncated,
            terminated=terminated,
            info={
                "prices": prices,
                "vegas": vegas,
                "t": self.t,
                "spot_mean": float(self.spot.mean()),
                # Instantaneous ATM local-vol for CMDP slack scaling (expert law).
                "atm_vol": float(
                    self.surfaces[self.path, :, self.t, self.S // 2, self.M // 2]
                    .mean()
                    .clamp_min(1e-4)
                ),
                "macro_start_idx": int(self.macro_start_idx),
            },
        )

    def step(self, w_exec: torch.Tensor) -> StepResult:
        # MTM on CMDP-projected weights; optional opt-in execution drag.
        prices, deltas, _ = self._greeks()
        if self.risk_guard is not None:
            self.risk_guard.check(  # type: ignore[attr-defined]
                w_exec, deltas.unsqueeze(0) if deltas.dim() == 1 else deltas
            )
        if self.prev_price is None:
            self.prev_price = prices.detach()
            pnl = torch.zeros(1)
            turn = torch.zeros(1)
        else:
            pnl = (w_exec.squeeze(0) * (prices - self.prev_price)).sum().view(1)
            turn = (w_exec - self.w).abs().sum().view(1)
            self.prev_price = prices.detach()
        # Opt-in drag (default 0): linear spread + square-root impact on ||Δw||₁.
        atm_vol = float(
            self.surfaces[self.path, :, self.t, self.S // 2, self.M // 2]
            .mean()
            .clamp_min(1e-4)
        )
        vol_mult = 1.0
        if self.execution_drag_mode == "vol_scaled":
            ref = max(self.execution_vol_ref, 1e-6)
            vol_mult = min(
                self.execution_vol_cap,
                max(self.execution_vol_floor, atm_vol / ref),
            )
        drag = torch.zeros(1)
        if self.execution_spread_bps > 0.0:
            drag = drag + (self.execution_spread_bps / 1e4) * vol_mult * turn
        if self.execution_impact_coef > 0.0:
            drag = drag + self.execution_impact_coef * vol_mult * torch.sqrt(
                turn.clamp_min(0.0)
            )
        funding_drag = torch.zeros(1)
        if self.funding is not None and getattr(self.funding, "enabled", False):
            macro = self._macro_at()
            funding_drag = self.funding(
                w_exec,
                prices=prices.unsqueeze(0) if prices.dim() == 1 else prices,
                deltas=deltas.unsqueeze(0) if deltas.dim() == 1 else deltas,
                macro=macro,
                spot=self.spot.unsqueeze(0),
            )
            if not torch.is_tensor(funding_drag):
                funding_drag = torch.tensor([float(funding_drag)])
            funding_drag = funding_drag.reshape(1).to(pnl.device)
        reward = pnl - drag - funding_drag
        self.w = w_exec.detach()
        self._evolve_spot()
        self.t += 1
        # Calendar/horizon cut of a continuing process = truncation (Pardo 2018).
        # No absorbing environmental failure → terminated always False.
        truncated = self.t >= self.T - 1
        terminated = False
        done = truncated or terminated
        out = self._observe(
            reward=reward, done=done, truncated=truncated, terminated=terminated
        )
        out.info["mtm_pnl"] = float(pnl.detach())
        out.info["exec_drag"] = float(drag.detach())
        out.info["funding_drag"] = float(funding_drag.detach())
        out.info["step_turnover"] = float(turn.detach())
        out.info["vol_drag_mult"] = float(vol_mult)
        out.info["truncated"] = truncated
        out.info["terminated"] = terminated
        return out

