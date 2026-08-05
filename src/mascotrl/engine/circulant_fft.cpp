#include "circulant_fft.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>

void fft_inplace(std::vector<std::complex<float>>& a, bool inverse) {
    const std::size_t n = a.size();
    // bit-reverse
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const float ang = 2.0f * static_cast<float>(std::numbers::pi_v<float>) / static_cast<float>(len)
                          * (inverse ? 1.0f : -1.0f);
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (std::size_t j = 0; j < len / 2; ++j) {
                auto u = a[i + j];
                auto v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        const float inv = 1.0f / static_cast<float>(n);
        for (auto& z : a) z *= inv;
    }
}

void build_volterra_circulant_row(int n_steps, float hurst, std::vector<float>& row) {
    // Kernel κ(Δt) = √(2H) (t)^{H-1/2} on physical time with Δt = 1/252.
    // Embedding this (not the unit-grid power) keeps Var(W^H_t) ≈ t^{2H}
    // so rBergomi needs no ad-hoc σ ceiling for float stability.
    const std::size_t m = next_pow2(static_cast<std::size_t>(2 * n_steps));
    row.assign(m, 0.0f);
    const float H = hurst;
    const float dt = 1.0f / 252.0f;
    const float pref = std::sqrt(2.0f * H);
    auto gamma = [](float x) { return std::tgamma(x); };
    const float c = pref / gamma(H + 0.5f);
    for (int k = 0; k < n_steps; ++k) {
        const float t = (static_cast<float>(k) + 1.0f) * dt;
        // κ(t) ΔW with ΔW ~ N(0,dt) absorbed as √dt factor on the kernel row.
        row[static_cast<std::size_t>(k)] = c * std::pow(t, H - 0.5f) * std::sqrt(dt);
    }
    for (std::size_t k = 1; k < static_cast<std::size_t>(n_steps); ++k) {
        row[m - k] = row[k];
    }
}

void simulate_fractional_increments(
    int n_steps,
    float hurst,
    const float* gaussian_n,
    float* out_increments
) {
    std::vector<float> circ;
    build_volterra_circulant_row(n_steps, hurst, circ);
    const std::size_t m = circ.size();
    std::vector<std::complex<float>> lam(m);
    for (std::size_t i = 0; i < m; ++i) lam[i] = {circ[i], 0.0f};
    fft_inplace(lam, false);

    // Eigenvalues must be non-negative for a valid embedding; clamp tiny negatives.
    for (auto& z : lam) {
        float re = std::max(z.real(), 0.0f);
        z = {std::sqrt(re), 0.0f};
    }

    std::vector<std::complex<float>> z(m);
    // Pair gaussians into complex white noise of length m
    for (std::size_t i = 0; i < m; ++i) {
        // gaussian_n expected length >= 2*m for real/imag, but we use m pairs from 2*m floats
        const float g0 = gaussian_n[2 * i];
        const float g1 = gaussian_n[2 * i + 1];
        z[i] = {g0, g1};
    }
    fft_inplace(z, false);
    for (std::size_t i = 0; i < m; ++i) z[i] *= lam[i];
    fft_inplace(z, true);

    for (int t = 0; t < n_steps; ++t) {
        out_increments[t] = z[static_cast<std::size_t>(t)].real();
    }
}
