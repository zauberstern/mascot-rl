// Fused Black-Scholes + Greeks with Abramowitz–Stegun CDF and cubic local-vol interp.
// Tailored for AMD Polaris (gfx803): keep intermediates in-register; avoid HBM round-trips.
//
 // STRICT: do NOT clamp interpolated local vol to an artificial ceiling (legacy 5.0f).
 // Rough Bergomi (H≈0.1) legitimately spikes variance beyond classical GBM limits;
 // truncating vol blinds Vega / tail hedges. Use a floor + stable log / d1 saturation.

__constant float AS_P = 0.2316419f;
__constant float AS_B1 = 0.319381530f;
__constant float AS_B2 = -0.356563782f;
__constant float AS_B3 = 1.781477937f;
__constant float AS_B4 = -1.821255978f;
__constant float AS_B5 = 1.330274429f;
__constant float INV_SQRT_2PI = 0.3989422804014327f;

inline float norm_pdf(float x) {
    // Saturate |x| so exp(-0.5 x^2) does not underflow denorms wastefully.
    float ax = min(fabs(x), 12.0f);
    return INV_SQRT_2PI * exp(-0.5f * ax * ax);
}

inline float norm_cdf(float x) {
    float ax = fabs(x);
    // Beyond ~8 the AS rational is already 0/1 within float32; clamp argument.
    ax = min(ax, 12.0f);
    float t = 1.0f / (1.0f + AS_P * ax);
    float poly = ((((AS_B5 * t + AS_B4) * t + AS_B3) * t + AS_B2) * t + AS_B1) * t;
    float cdf = 1.0f - norm_pdf(ax) * poly;
    return (x >= 0.0f) ? cdf : (1.0f - cdf);
}

inline float cubic_hermite(float y0, float y1, float y2, float y3, float t) {
    // Catmull-Rom
    float a0 = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
    float a1 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
    float a2 = -0.5f * y0 + 0.5f * y2;
    float a3 = y1;
    return ((a0 * t + a1) * t + a2) * t + a3;
}

inline float interp_local_vol(
    __global const float* vol_grid,
    int n_strikes,
    int n_maturities,
    float moneyness,
    float tau,
    float spot
) {
    (void)spot;
    // L1 emits LV on uniform log-moneyness y=ln(K/S) ∈ [-0.35, +0.35]
    // (DUPIRE_LOG_MNY_WING). Do NOT treat K/S as linear [0.7, 1.3].
    float y = log(max(moneyness, 1e-4f));
    const float y_wing = 0.35f;
    float u = clamp((y + y_wing) / (2.0f * y_wing), 0.0f, 1.0f);
    float v = clamp(tau, 1e-4f, 1.0f);
    float sx = u * (float)(max(n_strikes - 1, 1));
    float sy = (v - 1e-4f) / (1.0f - 1e-4f) * (float)(max(n_maturities - 1, 1));
    int i = (int)floor(sx);
    int j = (int)floor(sy);
    i = clamp(i, 1, max(n_strikes - 3, 1));
    j = clamp(j, 0, max(n_maturities - 1, 0));
    float tx = sx - floor(sx);
    int j0 = j;
    float y0 = vol_grid[(i - 1) * n_maturities + j0];
    float y1 = vol_grid[i * n_maturities + j0];
    float y2 = vol_grid[(i + 1) * n_maturities + j0];
    float y3 = vol_grid[min(i + 2, n_strikes - 1) * n_maturities + j0];
    return cubic_hermite(y0, y1, y2, y3, tx);
}

__kernel void bs_greeks_fused(
    __global const float* spot,
    __global const float* strike,
    __global const float* tau,
    __global const float* rate,
    __global const float* vol_grid,
    __global float* price,
    __global float* delta,
    __global float* vega,
    const int n,
    const int n_strikes,
    const int n_maturities,
    // 0 => single shared LV surface for all gids (soak / dense book).
    // >0 => packed per-asset surfaces; gid uses vol_grid[gid * vol_stride + …]
    //      with vol_stride typically n_strikes * n_maturities.
    const int vol_stride
) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    float S = spot[gid];
    float K = strike[gid];
    float T = max(tau[gid], 1e-6f);
    float r = rate[gid];
    float mny = K / max(S, 1e-6f);

    __global const float* vg = vol_grid;
    if (vol_stride > 0) {
        vg = vol_grid + ((size_t)gid * (size_t)vol_stride);
    }
    float vol = interp_local_vol(vg, n_strikes, n_maturities, mny, T, S);
    // Floor only — no 5.0f ceiling (tail shocks from rough vol must pass through).
    vol = max(vol, 1e-4f);

    float sqrtT = sqrt(T);
    // Stable log for extreme spot/strike ratios; keep vol authentic.
    float d1 = (log(max(S / K, 1e-8f)) + (r + 0.5f * vol * vol) * T) / (vol * sqrtT);
    float d2 = d1 - vol * sqrtT;

    float Nd1 = norm_cdf(d1);
    float Nd2 = norm_cdf(d2);
    float pdf = norm_pdf(d1);
    float disc = exp(clamp(-r * T, -80.0f, 80.0f));

    price[gid] = S * Nd1 - K * disc * Nd2;
    delta[gid] = Nd1;
    vega[gid] = S * pdf * sqrtT;
}
