#pragma once
#include <cstdint>
#include <tuple>
#include <pybind11/pybind11.h>

struct EngineConfig {
    int32_t n_paths = 20000;
    int32_t n_assets = 40;
    int32_t n_steps = 252;
    int32_t n_strikes = 21;
    int32_t n_maturities = 5;
    float hurst_exponent = 0.1f;
    // Reproducibility: the RNG stream is a deterministic function of
    // (seed, path, asset), never of the OpenMP thread id, so results are
    // invariant to thread count and to dynamic work scheduling.
    uint64_t seed = 42;
};

// Returns (ArrowSchema capsule, ArrowArray capsule) for flat float32 5D surface:
// [n_paths, n_assets, n_steps, n_strikes, n_maturities]
std::tuple<pybind11::capsule, pybind11::capsule> generate_surfaces(
    const EngineConfig& config,
    const float* cholesky_matrix  // row-major K x K lower Cholesky
);
