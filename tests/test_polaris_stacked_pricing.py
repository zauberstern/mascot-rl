"""Correctness: stacked multi-asset pricing vs serial per-asset reference."""
from __future__ import annotations

import os
import time

import pytest
import torch

from mascotrl.pricing.interface import get_portfolio_greeks, opencl_available, polaris_pricer_cpp


pytestmark = pytest.mark.skipif(
    polaris_pricer_cpp is None,
    reason="polaris_pricer_cpp extension not built",
)


def _make_book(k: int = 50, s: int = 15, m: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    spot = 80.0 + 40.0 * torch.rand(k, generator=g)
    strike = spot.clone()  # ATM book (matches CMDP env)
    tau = 0.05 + 0.4 * torch.rand(k, generator=g)
    rate = torch.full((k,), 0.02)
    # Distinct per-asset LV surfaces with mild smile structure.
    base = 0.12 + 0.25 * torch.rand(k, 1, 1, generator=g)
    smile = 0.02 * torch.linspace(-1.0, 1.0, s).view(1, s, 1) ** 2
    term = 0.01 * torch.linspace(0.0, 1.0, m).view(1, 1, m)
    vol = (base + smile + term).clamp_min(1e-3).expand(k, s, m).contiguous()
    return spot, strike, tau, rate, vol


def _serial_cpu(spot, strike, tau, rate, vol):
    """Legacy path: one fused call per asset, force CPU."""
    ps, ds, vs = [], [], []
    for i in range(spot.numel()):
        p, d, v = polaris_pricer_cpp.compute_greeks_fused(
            spot[i : i + 1].contiguous(),
            strike[i : i + 1].contiguous(),
            tau[i : i + 1].contiguous(),
            rate[i : i + 1].contiguous(),
            vol[i].contiguous(),
            True,
        )
        ps.append(p)
        ds.append(d)
        vs.append(v)
    return torch.cat(ps), torch.cat(ds), torch.cat(vs)


def test_stacked_cpu_matches_serial_reference():
    spot, strike, tau, rate, vol = _make_book(k=50)
    ref_p, ref_d, ref_v = _serial_cpu(spot, strike, tau, rate, vol)

    assert hasattr(polaris_pricer_cpp, "compute_greeks_fused_stacked")
    p, d, v = polaris_pricer_cpp.compute_greeks_fused_stacked(
        spot, strike, tau, rate, vol, True
    )
    assert torch.allclose(p, ref_p, rtol=0, atol=0)
    assert torch.allclose(d, ref_d, rtol=0, atol=0)
    assert torch.allclose(v, ref_v, rtol=0, atol=0)


def test_python_interface_stacked_matches_serial():
    spot, strike, tau, rate, vol = _make_book(k=50, seed=7)
    ref_p, ref_d, ref_v = _serial_cpu(spot, strike, tau, rate, vol)
    os.environ["MASCOTRL_FORCE_CPU_PRICING"] = "1"
    try:
        p, d, v = get_portfolio_greeks(spot, strike, tau, rate, vol, use_gpu=True)
    finally:
        os.environ.pop("MASCOTRL_FORCE_CPU_PRICING", None)
    assert torch.allclose(p, ref_p, rtol=0, atol=0)
    assert torch.allclose(d, ref_d, rtol=0, atol=0)
    assert torch.allclose(v, ref_v, rtol=0, atol=0)


def test_shared_grid_path_unchanged_cpu():
    """2D shared-surface API must still agree with itself under force_cpu."""
    k = 64
    g = torch.Generator().manual_seed(3)
    spot = 100.0 * torch.ones(k)
    strike = spot * (0.9 + 0.2 * torch.rand(k, generator=g))
    tau = torch.full((k,), 0.25)
    rate = torch.full((k,), 0.01)
    vol = (0.2 + 0.05 * torch.rand(15, 4, generator=g)).clamp_min(1e-3).contiguous()
    p1, d1, v1 = polaris_pricer_cpp.compute_greeks_fused(
        spot, strike, tau, rate, vol, True
    )
    p2, d2, v2 = get_portfolio_greeks(spot, strike, tau, rate, vol, use_gpu=False)
    assert torch.allclose(p1, p2, rtol=0, atol=0)
    assert torch.allclose(d1, d2, rtol=0, atol=0)
    assert torch.allclose(v1, v2, rtol=0, atol=0)


@pytest.mark.skipif(not opencl_available(), reason="OpenCL GPU unavailable")
def test_stacked_opencl_matches_cpu():
    """GPU stacked vs AVX stacked — float32 AS CDF, allow tight abs tol."""
    spot, strike, tau, rate, vol = _make_book(k=50, seed=11)
    # Ensure stacked threshold allows K=50 (default min is 8).
    os.environ.pop("MASCOTRL_OCL_MIN_N_STACKED", None)
    cpu_p, cpu_d, cpu_v = polaris_pricer_cpp.compute_greeks_fused_stacked(
        spot, strike, tau, rate, vol, True
    )
    gpu_p, gpu_d, gpu_v = polaris_pricer_cpp.compute_greeks_fused_stacked(
        spot, strike, tau, rate, vol, False
    )
    # If OpenCL failed over to AVX, tensors are bit-identical — still OK.
    assert torch.allclose(gpu_p, cpu_p, rtol=1e-5, atol=2e-5)
    assert torch.allclose(gpu_d, cpu_d, rtol=1e-5, atol=2e-5)
    assert torch.allclose(gpu_v, cpu_v, rtol=1e-5, atol=2e-5)
    assert torch.isfinite(gpu_p).all() and (gpu_d >= 0).all() and (gpu_d <= 1).all()


@pytest.mark.skipif(not opencl_available(), reason="OpenCL GPU unavailable")
def test_stacked_opencl_beats_serial_wall_time():
    """Sanity: one stacked launch should beat 50 serial C++ calls (warmup+measure)."""
    spot, strike, tau, rate, vol = _make_book(k=50, seed=21)
    # Warmup (JIT/context + buffers).
    for _ in range(3):
        polaris_pricer_cpp.compute_greeks_fused_stacked(
            spot, strike, tau, rate, vol, False
        )
        _serial_cpu(spot, strike, tau, rate, vol)

    t0 = time.perf_counter()
    for _ in range(20):
        polaris_pricer_cpp.compute_greeks_fused_stacked(
            spot, strike, tau, rate, vol, False
        )
    stacked_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        _serial_cpu(spot, strike, tau, rate, vol)
    serial_s = time.perf_counter() - t0

    # Soft check: stacked should not be dramatically slower; prefer speedup.
    # On RX 590 expect clear win; allow CI without GPU timing flake via ratio.
    assert stacked_s < serial_s * 1.25, (
        f"stacked={stacked_s:.4f}s serial={serial_s:.4f}s — unexpected regression"
    )


def test_shape_guard_rejects_mismatch():
    spot, strike, tau, rate, vol = _make_book(k=10)
    with pytest.raises(Exception):
        polaris_pricer_cpp.compute_greeks_fused_stacked(
            spot[:5].contiguous(),
            strike[:5].contiguous(),
            tau[:5].contiguous(),
            rate[:5].contiguous(),
            vol,  # still 10 assets
            True,
        )
