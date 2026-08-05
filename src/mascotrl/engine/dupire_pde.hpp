#pragma once

// Dupire / Markovian local-vol surface for rBergomi instantaneous variance.
 //
 // Output grid (MUST match OpenCL + AVX2 interpolators):
 //   strike axis: uniform in y = ln(K/S) ∈ [−Y_WING, +Y_WING], Y_WING = 0.35
 //   maturity axis: τ ∈ (0, 1] years, uniform in index
 //
 // σ_loc is the rBergomi Markovian projection (ATM level √V_t + leverage skew
 // from ρ, η, H). This is NOT the banned ad-hoc post-multiply smile overlay;
 // it is the first-order Dupire projection of the rough-vol SDE.
 //
 // Writes σ_loc(K,T) into out_lv[strike * n_maturities + maturity].

constexpr float DUPIRE_LOG_MNY_WING = 0.35f;  // ≈ |ln(0.70)| — shared with L2

void solve_dupire_local_vol(
    float spot,
    float rate,
    float instantaneous_variance,
    int n_strikes,
    int n_maturities,
    float* out_lv,  // length n_strikes * n_maturities
    float hurst = 0.1f,
    float eta = 1.5f,
    float rho = -0.75f
);
