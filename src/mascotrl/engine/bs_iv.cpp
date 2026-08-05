#include "bs_iv.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr float kPi = 3.14159265358979323846f;

inline float norm_cdf(float x) {
    return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

}  // namespace

float bs_call_price(
    float spot,
    float strike,
    float tau,
    float sigma,
    float rate,
    float div_q
) {
    if (!(spot > 0.0f) || !(strike > 0.0f) || !(sigma > 0.0f)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (tau <= 1e-8f) {
        return std::max(spot - strike, 0.0f);
    }
    const float sqrt_t = std::sqrt(tau);
    const float d1 =
        (std::log(spot / strike) + (rate - div_q + 0.5f * sigma * sigma) * tau)
        / (sigma * sqrt_t);
    const float d2 = d1 - sigma * sqrt_t;
    return spot * std::exp(-div_q * tau) * norm_cdf(d1)
         - strike * std::exp(-rate * tau) * norm_cdf(d2);
}

float brent_implied_vol(
    float price,
    float spot,
    float strike,
    float tau,
    float rate,
    float div_q,
    float tol,
    int max_iter
) {
    if (!(price > 0.0f) || !(spot > 0.0f) || !(strike > 0.0f) || tau <= 1e-8f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    // Intrinsic lower bound under dividends.
    const float disc_s = spot * std::exp(-div_q * tau);
    const float disc_k = strike * std::exp(-rate * tau);
    const float intrinsic = std::max(disc_s - disc_k, 0.0f);
    if (price < intrinsic - 1e-6f) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (price <= intrinsic + 1e-10f) {
        return 1e-4f;
    }

    float a = 1e-4f;
    float b = 5.0f;
    float fa = bs_call_price(spot, strike, tau, a, rate, div_q) - price;
    float fb = bs_call_price(spot, strike, tau, b, rate, div_q) - price;
    if (!(fa * fb < 0.0f)) {
        // Price outside BS range on [1e-4, 5] — fail closed for caller fallback.
        return std::numeric_limits<float>::quiet_NaN();
    }

    float c = a;
    float fc = fa;
    float d = 0.0f;
    float e = 0.0f;
    for (int iter = 0; iter < max_iter; ++iter) {
        if (fb * fc > 0.0f) {
            c = a;
            fc = fa;
            d = e = b - a;
        }
        if (std::fabs(fc) < std::fabs(fb)) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        const float tol1 = 2.0f * tol * std::fabs(b) + 0.5f * tol;
        const float xm = 0.5f * (c - b);
        if (std::fabs(xm) <= tol1 || fb == 0.0f) {
            return b;
        }
        if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
            const float s = fb / fa;
            float p, q;
            if (a == c) {
                p = 2.0f * xm * s;
                q = 1.0f - s;
            } else {
                q = fa / fc;
                const float r = fb / fc;
                p = s * (2.0f * xm * q * (q - r) - (b - a) * (r - 1.0f));
                q = (q - 1.0f) * (r - 1.0f) * (s - 1.0f);
            }
            if (p > 0.0f) q = -q;
            p = std::fabs(p);
            const float min1 = 3.0f * xm * q - std::fabs(tol1 * q);
            const float min2 = std::fabs(e * q);
            if (2.0f * p < std::min(min1, min2)) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if (std::fabs(d) > tol1) {
            b += d;
        } else {
            b += (xm > 0.0f ? tol1 : -tol1);
        }
        fb = bs_call_price(spot, strike, tau, b, rate, div_q) - price;
        (void)kPi;
    }
    return b;
}
