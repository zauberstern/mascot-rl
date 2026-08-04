#pragma once

#include "rbergomi_engine.hpp"

#include <cstdint>
#include <tuple>
#include <pybind11/pybind11.h>

// World ids: 0=rbergomi, 1=gbm, 2=heston, 3=garch, 4=sabr
enum class WorldId : int32_t {
    RBergomi = 0,
    GBM = 1,
    Heston = 2,
    Garch = 3,
    SABR = 4,
};

// Extended config for multi-world generation. Reuses EngineConfig fields for
// grid dimensions; world-specific params live here.
struct WorldConfig {
    EngineConfig base;
    int32_t world = 0;  // WorldId
    float rate = 0.0f;
    float div_q = 0.0f;
    float spot0 = 100.0f;

    // GBM
    float gbm_mu = 0.05f;
    float gbm_sigma = 0.20f;

    // Heston
    float heston_v0 = 0.04f;
    float heston_theta = 0.04f;
    float heston_kappa = 2.0f;
    float heston_xi = 0.30f;
    float heston_rho = -0.70f;
    // 0=full_truncation, 1=qe, 2=qe_martingale (default; Andersen QE-M)
    int32_t heston_scheme = 2;

    // GJR-GARCH
    float garch_mu = 0.0f;
    float garch_omega = 1e-6f;
    float garch_alpha = 0.02f;
    float garch_beta = 0.90f;
    float garch_gamma = 0.10f;
    float garch_lambda = 0.0f;
    int32_t garch_n_inner = 4096;  // reduced default for interactive builds; tests use higher

    // SABR beta=1
    float sabr_sigma0 = 0.20f;
    float sabr_nu = 0.6f;
    float sabr_rho = -0.4f;
};

// Returns six Arrow capsules:
// (surface_schema, surface_array, spot_schema, spot_array, iv_schema, iv_array)
// surface: [P, K, T, S, M]; spot/atm_iv: [P, K, T]
std::tuple<
    pybind11::capsule, pybind11::capsule,
    pybind11::capsule, pybind11::capsule,
    pybind11::capsule, pybind11::capsule>
generate_world(const WorldConfig& config, const float* cholesky_matrix);
