"""Layer 2 Python interface for fused OpenCL/AVX2 pricing."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Tuple

import torch

from mascotrl._root import REPO_ROOT

_ROOT = REPO_ROOT
_LOAD_ERROR: Exception | None = None


def _load_polaris():
    global _LOAD_ERROR
    try:
        import polaris_pricer_cpp as mod

        return mod
    except ImportError:
        pass

    so = _ROOT / "build" / "polaris" / "polaris_pricer_cpp.so"
    if so.exists():
        spec = importlib.util.spec_from_file_location("polaris_pricer_cpp", so)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    try:
        from torch.utils.cpp_extension import load

        (_ROOT / "build" / "polaris").mkdir(parents=True, exist_ok=True)
        return load(
            name="polaris_pricer_cpp",
            sources=[str(_ROOT / "src" / "pricing" / "polaris_pricer.cpp")],
            extra_cflags=[
                "-O3",
                "-march=znver1",
                "-mavx2",
                "-mfma",
                "-fopenmp",
                "-ffast-math",
                "-std=c++20",
            ],
            extra_ldflags=[
                "-fopenmp",
                "-L/usr/lib/x86_64-linux-gnu",
                "-l:libOpenCL.so.1",
            ],
            extra_include_paths=[
                str(_ROOT / "third_party"),
                str(_ROOT / "third_party" / "OpenCL"),
                str(_ROOT / "src" / "pricing"),
            ],
            build_directory=str(_ROOT / "build" / "polaris"),
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover
        _LOAD_ERROR = exc
        return None


polaris_pricer_cpp = _load_polaris()


def opencl_available() -> bool:
    if polaris_pricer_cpp is None:
        return False
    fn = getattr(polaris_pricer_cpp, "opencl_available", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _atm_sigma_from_surface(vol_grid: torch.Tensor) -> torch.Tensor:
    """ATM local-vol scalar per option from a 2D or 3D surface grid."""
    if vol_grid.dim() == 2:
        s_mid = vol_grid.shape[0] // 2
        m_mid = vol_grid.shape[1] // 2
        return vol_grid[s_mid, m_mid].clamp_min(1e-6)
    if vol_grid.dim() == 3:
        s_mid = vol_grid.shape[1] // 2
        m_mid = vol_grid.shape[2] // 2
        return vol_grid[:, s_mid, m_mid].clamp_min(1e-6)
    raise ValueError("vol_grid must be 2D [S,M] or 3D [K,S,M]")


def _bs_greeks_torch(
    spot: torch.Tensor,
    strike: torch.Tensor,
    time_to_mat: torch.Tensor,
    rate: torch.Tensor,
    sigma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch ATM European call greeks (public-extract fallback)."""
    spot = spot.reshape(-1)
    strike = strike.reshape(-1)
    time_to_mat = time_to_mat.reshape(-1).clamp_min(1e-6)
    rate = rate.reshape(-1)
    sigma = sigma.reshape(-1).clamp_min(1e-6)
    if sigma.numel() == 1 and spot.numel() > 1:
        sigma = sigma.expand_as(spot)
    sqrt_t = torch.sqrt(time_to_mat)
    d1 = (torch.log(spot / strike.clamp_min(1e-6)) + (rate + 0.5 * sigma * sigma) * time_to_mat) / (
        sigma * sqrt_t
    )
    d2 = d1 - sigma * sqrt_t
    cdf = lambda x: 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    pdf = lambda x: torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    prices = spot * cdf(d1) - strike * torch.exp(-rate * time_to_mat) * cdf(d2)
    deltas = cdf(d1)
    vegas = spot * pdf(d1) * sqrt_t
    return prices, deltas, vegas


def _bs_greeks_torch_batch(
    spot: torch.Tensor,
    strike: torch.Tensor,
    time_to_mat: torch.Tensor,
    rate: torch.Tensor,
    vol_grid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if vol_grid.dim() == 3:
            n = spot.numel()
            if vol_grid.shape[0] != n:
                raise ValueError(f"vol_grid batch {vol_grid.shape[0]} != n_assets {n}")
            sigma = _atm_sigma_from_surface(vol_grid)
            return _bs_greeks_torch(spot, strike, time_to_mat, rate, sigma)
        if vol_grid.dim() != 2:
            raise ValueError("vol_grid must be 2D [S,M] or 3D [K,S,M]")
        sigma = _atm_sigma_from_surface(vol_grid)
        return _bs_greeks_torch(spot, strike, time_to_mat, rate, sigma)


def get_portfolio_greeks(
    spot: torch.Tensor,
    strike: torch.Tensor,
    time_to_mat: torch.Tensor,
    rate: torch.Tensor,
    vol_grid: torch.Tensor,
    use_gpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates options pricing in-register via OpenCL (RX 590) or AVX2 fallback.
    Returns Prices, Deltas, Vegas.

    ``vol_grid``:
      - 2D ``[S, M]`` — shared LV surface for all options in the batch
      - 3D ``[K, S, M]`` — per-asset stacked surfaces (one ATM option per asset);
        uses a single fused launch (``compute_greeks_fused_stacked``)
    """
    import os

    if os.environ.get("MASCOTRL_FORCE_CPU_PRICING", "").strip() in ("1", "true", "True"):
        use_gpu = False

    def _f32(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(dtype=torch.float32).contiguous()

    spot = _f32(spot).reshape(-1)
    strike = _f32(strike).reshape(-1)
    time_to_mat = _f32(time_to_mat).reshape(-1)
    rate = _f32(rate).reshape(-1)
    vol_grid = _f32(vol_grid)

    if polaris_pricer_cpp is None:
        return _bs_greeks_torch_batch(spot, strike, time_to_mat, rate, vol_grid)

    force_cpu = not use_gpu

    # Per-asset stacked surfaces: one OpenCL/AVX launch (preferred CMDP path).
    if vol_grid.dim() == 3:
        n = spot.numel()
        if vol_grid.shape[0] != n:
            raise ValueError(
                f"vol_grid batch {vol_grid.shape[0]} != n_assets {n}"
            )
        stacked = getattr(polaris_pricer_cpp, "compute_greeks_fused_stacked", None)
        if stacked is not None:
            with torch.no_grad():
                return stacked(spot, strike, time_to_mat, rate, vol_grid, force_cpu)
        # Legacy extension without stacked API — serial fallback (correct, slow).
        prices, deltas, vegas = [], [], []
        with torch.no_grad():
            for i in range(n):
                p, d, v = polaris_pricer_cpp.compute_greeks_fused(
                    spot[i : i + 1],
                    strike[i : i + 1],
                    time_to_mat[i : i + 1],
                    rate[i : i + 1],
                    vol_grid[i],
                    force_cpu,
                )
                prices.append(p)
                deltas.append(d)
                vegas.append(v)
        return torch.cat(prices), torch.cat(deltas), torch.cat(vegas)

    if vol_grid.dim() != 2:
        raise ValueError("vol_grid must be 2D [S,M] or 3D [K,S,M]")

    with torch.no_grad():
        return polaris_pricer_cpp.compute_greeks_fused(
            spot, strike, time_to_mat, rate, vol_grid, force_cpu
        )
