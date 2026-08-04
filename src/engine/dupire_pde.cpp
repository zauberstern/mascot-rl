#include "dupire_pde.hpp"

#include <algorithm>
#include <cmath>

void solve_dupire_local_vol(
    float spot,
    float rate,
    float instantaneous_variance,
    int n_strikes,
    int n_maturities,
    float* out_lv,
    float hurst,
    float eta,
    float rho
) {
    // rBergomi → local-vol Markovian projection.
    //
    // Prior bug: CN under flat σ=√V then every cell overwritten with √V —
    // guaranteeing zero strike/term gaps (flat-σ CN recovers flat LV).
    //
    // Projection (Bergomi / short-dated rough-vol skew):
    //   σ_loc(y, τ) = √V · exp( ½ η ρ y ψ(τ) + ⅛ η² y² τ )
    //   ψ(τ) = τ^{H−½} / (H + ½)
    // Output y-grid is uniform on [−Y_WING, Y_WING] (shared with L2 interp).

    (void)rate;
    (void)spot;

    const float V = std::max(instantaneous_variance, 1e-12f);
    const float sigma = std::sqrt(V);
    const int n_mat = std::max(n_maturities, 1);
    const int n_stk = std::max(n_strikes, 1);
    const float H = std::min(std::max(hurst, 0.01f), 0.49f);
    const float eta_c = std::max(eta, 0.0f);
    const float rho_c = std::min(std::max(rho, -0.999f), 0.999f);
    const float T_max = 1.0f;
    const float dT = T_max / static_cast<float>(n_mat);
    const float y_wing = DUPIRE_LOG_MNY_WING;
    const float Hp = H + 0.5f;

    for (int s = 0; s < n_stk; ++s) {
        const float u = (n_stk == 1)
            ? 0.5f
            : static_cast<float>(s) / static_cast<float>(n_stk - 1);
        const float y = -y_wing + 2.0f * y_wing * u;

        for (int m = 0; m < n_mat; ++m) {
            const float tau = dT * static_cast<float>(m + 1);
            const float psi = std::pow(tau, H - 0.5f) / Hp;
            const float skew = 0.5f * eta_c * rho_c * y * psi;
            const float smile = 0.125f * eta_c * eta_c * y * y * tau;
            float lv = sigma * std::exp(skew + smile);
            if (!std::isfinite(lv) || lv < 1e-4f) {
                lv = 1e-4f;
            }
            if (lv > 50.0f) {
                lv = 50.0f;
            }
            out_lv[s * n_mat + m] = lv;
        }
    }
}
