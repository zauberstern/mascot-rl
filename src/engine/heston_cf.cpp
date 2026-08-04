#include "heston_cf.hpp"

#include <cmath>
#include <complex>
#include <limits>

namespace {

constexpr float kPi = 3.14159265358979323846f;
using cplx = std::complex<float>;

cplx heston_phi(cplx u, float tau, float logF, const HestonParams& p) {
    const cplx i(0.0f, 1.0f);
    const float kappa = p.kappa;
    const float theta = p.theta;
    const float xi = p.xi;
    const float rho = p.rho;
    const float v0 = p.v0;

    const cplx a = cplx(kappa, 0.0f) - rho * xi * i * u;
    const cplx d = std::sqrt(a * a + xi * xi * (i * u + u * u));
    // Little-trap (Albrecher): g = (a - d)/(a + d)
    const cplx g = (a - d) / (a + d);
    const cplx exp_mdt = std::exp(-d * tau);
    const cplx one(1.0f, 0.0f);
    const cplx C =
        (kappa * theta / (xi * xi))
        * ((a - d) * tau - float(2.0f) * std::log((one - g * exp_mdt) / (one - g)));
    const cplx D = ((a - d) / (xi * xi)) * ((one - exp_mdt) / (one - g * exp_mdt));
    return std::exp(i * u * logF + C + D * v0);
}

float bs_call_atm_proxy(float spot, float strike, float tau, float sigma, float rate, float div_q) {
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

float heston_call_price(
    float spot,
    float strike,
    float tau,
    float rate,
    float div_q,
    const HestonParams& p
) {
    if (!(spot > 0.0f) || !(strike > 0.0f)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (tau <= 1e-8f) {
        return std::max(spot - strike, 0.0f);
    }
    // Lewis (2001) formula with adaptive trapezoid on u ∈ (0, U].
    const float logF = std::log(spot) + (rate - div_q) * tau;
    const float k = logF - std::log(strike);
    const cplx i(0.0f, 1.0f);
    const float U = 100.0f;
    const int N = 256;
    const float du = U / static_cast<float>(N);
    float integ = 0.0f;
    for (int n = 1; n <= N; ++n) {
        const float u = du * static_cast<float>(n);
        const cplx u_shift(u, -0.5f);
        const cplx phi = heston_phi(u_shift, tau, logF, p);
        if (!std::isfinite(phi.real()) || !std::isfinite(phi.imag())) {
            continue;
        }
        const cplx numer = std::exp(i * u * k) * phi;
        const float denom = u * u + 0.25f;
        const float w = (n == N) ? 0.5f : 1.0f;
        integ += w * numer.real() / denom;
    }
    integ *= du;
    float call =
        spot * std::exp(-div_q * tau)
        - (strike * std::exp(-rate * tau) / kPi) * integ;
    if (!std::isfinite(call) || call < 0.0f) {
        // Fallback: BS at instantaneous vol.
        const float sig = std::sqrt(std::max(p.v0, 1e-8f));
        call = bs_call_atm_proxy(spot, strike, tau, sig, rate, div_q);
    }
    // Intrinsic floor under dividends.
    const float intrinsic = std::max(
        spot * std::exp(-div_q * tau) - strike * std::exp(-rate * tau), 0.0f
    );
    return std::max(call, intrinsic);
}
