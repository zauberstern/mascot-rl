#include "garch_duan.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

bool garch_params_ok(const GarchParams& p) {
    if (!(p.omega > 0.0f)) return false;
    if (p.alpha < 0.0f || p.beta < 0.0f || p.gamma < 0.0f) return false;
    return (p.alpha + p.beta + 0.5f * p.gamma) < 1.0f;
}

float garch_long_run_variance(const GarchParams& p) {
    const float den = 1.0f - p.alpha - p.beta - 0.5f * p.gamma;
    if (!(den > 1e-12f)) return p.omega;
    return p.omega / den;
}

namespace {

struct Xoshiro {
    uint64_t s[4];
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }
    float uniform() { return (next() >> 11) * (1.0f / 9007199254740992.0f); }
    float gauss() {
        float u1 = std::max(uniform(), 1e-10f);
        float u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1))
             * std::cos(2.0f * 3.14159265358979323846f * u2);
    }
};

void seed_xoshiro(Xoshiro& rng, uint64_t seed) {
    auto splitmix = [](uint64_t& x) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    uint64_t x = seed;
    rng.s[0] = splitmix(x);
    rng.s[1] = splitmix(x);
    rng.s[2] = splitmix(x);
    rng.s[3] = splitmix(x);
}

float bs_call_cv(float spot, float strike, float tau, float sigma, float rate, float div_q) {
    if (tau <= 1e-8f || sigma <= 1e-8f) return std::max(spot - strike, 0.0f);
    const float sqrt_t = std::sqrt(tau);
    const float d1 =
        (std::log(spot / strike) + (rate - div_q + 0.5f * sigma * sigma) * tau)
        / (sigma * sqrt_t);
    const float d2 = d1 - sigma * sqrt_t;
    auto ncdf = [](float z) { return 0.5f * (1.0f + std::erf(z / std::sqrt(2.0f))); };
    return spot * std::exp(-div_q * tau) * ncdf(d1)
         - strike * std::exp(-rate * tau) * ncdf(d2);
}

}  // namespace

void garch_build_price_table(
    float spot0,
    float rate,
    float div_q,
    int n_strikes,
    int n_maturities,
    const float* strikes,
    const float* maturities,
    const GarchParams& p,
    int n_inner,
    uint64_t seed,
    float* out_prices,
    float* out_h_nodes
) {
    if (!garch_params_ok(p)) {
        throw std::invalid_argument("GJR-GARCH params violate stationarity");
    }
    const float omega_bar = garch_long_run_variance(p);
    const int n_h = 32;
    const float h_lo = 0.25f * omega_bar;
    const float h_hi = 16.0f * omega_bar;
    for (int i = 0; i < n_h; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(n_h - 1);
        out_h_nodes[i] = h_lo * std::exp(u * std::log(h_hi / h_lo));
    }

    const float persist = p.alpha + p.beta + 0.5f * p.gamma;
    const int half = std::max(n_inner / 2, 1);

    for (int hi = 0; hi < n_h; ++hi) {
        for (int m = 0; m < n_maturities; ++m) {
            const float tau = maturities[m];
            const int n_steps = std::max(1, static_cast<int>(std::lround(tau * 252.0f)));
            // Analytic E[h] path for BS control variance.
            float e_int = 0.0f;
            float h_exp = out_h_nodes[hi];
            for (int t = 0; t < n_steps; ++t) {
                e_int += h_exp;
                h_exp = omega_bar + persist * (h_exp - omega_bar);
            }
            const float sigma_cv = std::sqrt(std::max(e_int / static_cast<float>(n_steps), 1e-12f));

            for (int s = 0; s < n_strikes; ++s) {
                const float K = strikes[s];
                const float bs0 = bs_call_cv(spot0, K, tau, sigma_cv, rate, div_q);
                double acc = 0.0;
                for (int path = 0; path < half; ++path) {
                    for (int anti = 0; anti < 2; ++anti) {
                        Xoshiro rng;
                        seed_xoshiro(
                            rng,
                            seed
                                ^ (static_cast<uint64_t>(hi + 1) * 0x9E3779B97F4A7C15ULL)
                                ^ (static_cast<uint64_t>(m + 1) * 0xBF58476D1CE4E5B9ULL)
                                ^ (static_cast<uint64_t>(s + 1) * 0x94D049BB133111EBULL)
                                ^ (static_cast<uint64_t>(path + 1) * 0x2545F4914F6CDD1DULL)
                                ^ (static_cast<uint64_t>(anti) * 0xD6E8FEB86659FD93ULL)
                        );
                        float logS = std::log(spot0);
                        float h = out_h_nodes[hi];
                        float xi_prev = 0.0f;
                        for (int t = 0; t < n_steps; ++t) {
                            float z = rng.gauss();
                            if (anti == 1) z = -z;
                            // Duan LRNVR: log return = r - q - 0.5 h + sqrt(h) xi
                            // with h update using (xi - lambda).
                            const float xi = z;  // under Q after lambda shift in indicator
                            const float h_next =
                                p.omega
                                + p.beta * h
                                + (p.alpha
                                   + p.gamma * ((xi_prev - p.lambda) < 0.0f ? 1.0f : 0.0f))
                                      * h * (xi_prev - p.lambda) * (xi_prev - p.lambda);
                            h = std::max(h_next, 1e-12f);
                            logS += (rate - div_q - 0.5f * h) + std::sqrt(h) * xi;
                            xi_prev = xi;
                        }
                        const float ST = std::exp(logS);
                        const float payoff = std::max(ST - K, 0.0f) * std::exp(-rate * tau);
                        acc += static_cast<double>(payoff - bs0);
                    }
                }
                const float mc = bs0 + static_cast<float>(acc / static_cast<double>(2 * half));
                const int idx = hi * n_maturities * n_strikes + m * n_strikes + s;
                out_prices[idx] = std::max(mc, 0.0f);
            }
        }
    }
}

float garch_lookup_price(
    float h,
    int strike_idx,
    int mat_idx,
    int n_strikes,
    int n_maturities,
    const float* h_nodes,
    const float* prices
) {
    const int n_h = 32;
    if (h <= h_nodes[0]) {
        return prices[0 * n_maturities * n_strikes + mat_idx * n_strikes + strike_idx];
    }
    if (h >= h_nodes[n_h - 1]) {
        return prices[(n_h - 1) * n_maturities * n_strikes + mat_idx * n_strikes + strike_idx];
    }
    int lo = 0;
    for (int i = 0; i < n_h - 1; ++i) {
        if (h_nodes[i] <= h && h <= h_nodes[i + 1]) {
            lo = i;
            break;
        }
    }
    const float log_h = std::log(h);
    const float log_lo = std::log(h_nodes[lo]);
    const float log_hi = std::log(h_nodes[lo + 1]);
    const float w = (log_h - log_lo) / std::max(log_hi - log_lo, 1e-12f);
    const float p0 =
        prices[lo * n_maturities * n_strikes + mat_idx * n_strikes + strike_idx];
    const float p1 =
        prices[(lo + 1) * n_maturities * n_strikes + mat_idx * n_strikes + strike_idx];
    return (1.0f - w) * p0 + w * p1;
}
