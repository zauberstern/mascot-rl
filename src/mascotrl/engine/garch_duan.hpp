#pragma once

// GJR-GARCH physical dynamics + Duan (1995) LRNVR option price table.
#include <cstdint>

struct GarchParams {
    float mu = 0.0f;
    float omega = 1e-6f;
    float alpha = 0.02f;
    float beta = 0.90f;
    float gamma = 0.10f;
    float lambda = 0.0f;  // unit risk premium; 0 => Q == P
};

// Fail closed if omega <= 0 or alpha+beta+gamma/2 >= 1.
bool garch_params_ok(const GarchParams& p);

float garch_long_run_variance(const GarchParams& p);

// Build a lookup table of call prices under Duan LRNVR.
// h_nodes: 32 log-spaced nodes; for each (h, tau_m, K_s) run n_inner antithetic paths.
// out_prices layout: [h_idx * n_maturities * n_strikes + m * n_strikes + s]
void garch_build_price_table(
    float spot0,
    float rate,
    float div_q,
    int n_strikes,
    int n_maturities,
    const float* strikes,     // length n_strikes (absolute K)
    const float* maturities,  // length n_maturities (years)
    const GarchParams& p,
    int n_inner,
    uint64_t seed,
    float* out_prices,  // length 32 * n_maturities * n_strikes
    float* out_h_nodes  // length 32
);

// Interpolate call price in log(h), then return price.
float garch_lookup_price(
    float h,
    int strike_idx,
    int mat_idx,
    int n_strikes,
    int n_maturities,
    const float* h_nodes,
    const float* prices
);
