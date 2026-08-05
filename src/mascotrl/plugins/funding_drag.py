"""Optional borrow / funding drag for CMDPEnv (default off = status quo)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def load_name_borrow_bps(
    path: str | Path | None,
    *,
    tickers: list[str] | None = None,
    n_assets: int | None = None,
) -> torch.Tensor | None:
    """
    Load per-name borrow fee in bps from CSV/parquet/JSON.

    Expected columns: ticker (or symbol), borrow_bps (or bps).
    Returns tensor (K,) aligned to ``tickers`` when provided.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    rows: dict[str, float] = {}
    if p.suffix.lower() in {".parquet", ".pq"}:
        import pandas as pd

        df = pd.read_parquet(p)
        tcol = "ticker" if "ticker" in df.columns else "symbol"
        bcol = "borrow_bps" if "borrow_bps" in df.columns else "bps"
        for _, r in df.iterrows():
            rows[str(r[tcol]).upper()] = float(r[bcol])
    elif p.suffix.lower() == ".json":
        import json

        raw = json.loads(p.read_text())
        if isinstance(raw, dict):
            rows = {str(k).upper(): float(v) for k, v in raw.items()}
        else:
            for r in raw:
                rows[str(r.get("ticker") or r.get("symbol")).upper()] = float(
                    r.get("borrow_bps", r.get("bps", 0.0))
                )
    else:
        import pandas as pd

        df = pd.read_csv(p)
        tcol = "ticker" if "ticker" in df.columns else "symbol"
        bcol = "borrow_bps" if "borrow_bps" in df.columns else "bps"
        for _, r in df.iterrows():
            rows[str(r[tcol]).upper()] = float(r[bcol])

    if tickers is not None:
        return torch.tensor(
            [float(rows.get(str(t).upper(), float("nan"))) for t in tickers],
            dtype=torch.float32,
        )
    if n_assets is not None:
        vals = list(rows.values())[: int(n_assets)]
        if len(vals) < int(n_assets):
            vals = vals + [float("nan")] * (int(n_assets) - len(vals))
        return torch.tensor(vals, dtype=torch.float32)
    return torch.tensor(list(rows.values()), dtype=torch.float32)


@dataclass
class FundingDrag:
    """
    MVP GC borrow on short notional proxy, with optional per-name borrow bps:

        drag += dt * rate_i * |w_i^-| * N_hat_i

    ``notional_proxy=abs_weight`` uses |w| as dimensionless exposure (documented
    research proxy — not equity share borrow). Optional SOFR margin term when
    ``margin_funding`` and an absolute SOFR level are supplied.

    Absolute funding rates are **not** read from z-scored macro columns.
    Pass ``sofr`` / ``sofr_level`` explicitly, or set ``sofr_level`` in YAML.
    """

    enabled: bool = False
    mode: str = "sofr_gc"
    gc_borrow_bps: float = 25.0
    margin_funding: bool = False
    margin_rate_spread_bps: float = 0.0
    notional_proxy: str = "abs_weight"
    dt_years: float = 1.0 / 252.0
    sofr_key: str = "sofr"
    sofr_level: float | None = None  # absolute decimal (e.g. 0.05), not z-score
    name_borrow_bps: torch.Tensor | None = None
    name_borrow_path: str | None = None

    def ensure_name_schedule(self, tickers: list[str] | None = None, n_assets: int | None = None) -> None:
        if self.name_borrow_bps is None and self.name_borrow_path:
            self.name_borrow_bps = load_name_borrow_bps(
                self.name_borrow_path, tickers=tickers, n_assets=n_assets
            )

    def coverage(self, w_exec: torch.Tensor) -> float:
        """
        Fraction of short names with finite name borrow bps.

        Returns NaN when there are no shorts (do not treat as full coverage).
        Shorts beyond the borrow schedule length count as uncovered.
        """
        if self.name_borrow_bps is None:
            return 0.0
        w = w_exec.reshape(-1)
        short = w < 0
        if not bool(short.any()):
            return float("nan")
        bps = self.name_borrow_bps.reshape(-1).to(device=w.device)
        if bps.numel() < w.numel():
            pad = torch.full(
                (w.numel() - bps.numel(),),
                float("nan"),
                device=w.device,
                dtype=bps.dtype,
            )
            bps = torch.cat([bps, pad], dim=0)
        return float(torch.isfinite(bps[short]).float().mean().item())

    def compute(
        self,
        w_exec: torch.Tensor,
        *,
        sofr: float | None = None,
        prices: torch.Tensor | None = None,
        deltas: torch.Tensor | None = None,
        spot: torch.Tensor | None = None,
        macro: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(1, device=w_exec.device, dtype=w_exec.dtype)
        w = w_exec.reshape(-1)
        short = (-w).clamp_min(0.0)
        if self.notional_proxy == "abs_weight":
            notional = short
        elif self.notional_proxy == "price_weight" and prices is not None:
            p = prices.reshape(-1).to(device=w.device, dtype=w.dtype)
            n = min(short.numel(), p.numel())
            notional = short.clone()
            notional[:n] = short[:n] * p[:n].abs().clamp_min(1e-6)
        elif self.notional_proxy == "spot_delta" and spot is not None and deltas is not None:
            s = spot.reshape(-1).to(device=w.device, dtype=w.dtype)
            d = deltas.reshape(-1).to(device=w.device, dtype=w.dtype)
            n = min(short.numel(), s.numel(), d.numel())
            notional = short.clone()
            notional[:n] = short[:n] * (s[:n] * d[:n]).abs().clamp_min(1e-6)
        else:
            notional = short

        if self.name_borrow_bps is not None:
            bps = self.name_borrow_bps.reshape(-1).to(device=w.device, dtype=w.dtype)
            if bps.numel() < notional.numel():
                pad = torch.full(
                    (notional.numel() - bps.numel(),),
                    float("nan"),
                    device=w.device,
                    dtype=w.dtype,
                )
                bps = torch.cat([bps, pad], dim=0)
            gc = self.gc_borrow_bps / 1e4
            rate = torch.where(
                torch.isfinite(bps[: notional.numel()]),
                bps[: notional.numel()] / 1e4,
                torch.full_like(notional, gc),
            )
            drag = (rate * self.dt_years * notional).sum().view(1)
        else:
            borrow_rate = self.gc_borrow_bps / 1e4
            drag = (borrow_rate * self.dt_years) * notional.sum().view(1)

        sofr_abs = sofr if sofr is not None else self.sofr_level
        if self.margin_funding and sofr_abs is not None:
            s = float(sofr_abs)
            if s > 1.0:
                s = s / 100.0
            fund = (s + self.margin_rate_spread_bps / 1e4) * self.dt_years
            drag = drag + fund * w.abs().sum().view(1)
        return drag

    def __call__(self, w_exec: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.compute(w_exec, **kwargs)
