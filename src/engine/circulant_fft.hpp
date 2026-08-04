#pragma once
#include <cstddef>
#include <vector>
#include <complex>
#include <cmath>

// In-place radix-2 Cooley–Tukey FFT (float complex). n must be power of 2.
void fft_inplace(std::vector<std::complex<float>>& a, bool inverse);

// Next power of two >= n
inline std::size_t next_pow2(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Davies–Harte circulant embedding: simulate fBM increments via Volterra kernel
// (t-s)^{H-1/2} using FFT of the circulant covariance/kernel row.
// Out: path of length n_steps (integrated fractional noise W^H increments).
void simulate_fractional_increments(
    int n_steps,
    float hurst,
    const float* gaussian_n,           // length m = 2 * next_pow2(n_steps) real Gaussians (paired as complex)
    float* out_increments              // length n_steps
);

// Build first row of circulant embedding for kernel weights.
void build_volterra_circulant_row(int n_steps, float hurst, std::vector<float>& row);
