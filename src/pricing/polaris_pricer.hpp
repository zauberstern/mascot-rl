#pragma once
#include <torch/extension.h>
#include <tuple>

// Dispatcher: OpenCL GPU (RX 590) or AVX2 CPU fallback.
// vol_grid is a *shared* LV surface [n_strikes, n_maturities] for all options.
// Returns {Prices, Deltas, Vegas}
std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_greeks_fused(
    const at::Tensor& spot,
    const at::Tensor& strike,
    const at::Tensor& time_to_maturity,
    const at::Tensor& rate,
    const at::Tensor& vol_grid,
    bool force_cpu = false
);

// Multi-asset portfolio path: one ATM option per asset with *per-asset* LV
// surfaces packed as vol_grids [n_assets, n_strikes, n_maturities].
// Single OpenCL/AVX launch (no Python per-asset loop).
std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_greeks_fused_stacked(
    const at::Tensor& spot,
    const at::Tensor& strike,
    const at::Tensor& time_to_maturity,
    const at::Tensor& rate,
    const at::Tensor& vol_grids,
    bool force_cpu = false
);

// True once Mesa rusticl (or other) OpenCL GPU context initialized successfully.
bool polaris_opencl_available();
