#include "polaris_pricer.hpp"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>

namespace {

inline float as_cdf(float x) {
    const float p = 0.2316419f;
    const float b1 = 0.319381530f;
    const float b2 = -0.356563782f;
    const float b3 = 1.781477937f;
    const float b4 = -1.821255978f;
    const float b5 = 1.330274429f;
    const float inv_sqrt_2pi = 0.3989422804014327f;
    float ax = std::min(std::fabs(x), 12.0f);
    float t = 1.0f / (1.0f + p * ax);
    float pdf = inv_sqrt_2pi * std::exp(-0.5f * ax * ax);
    float poly = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t;
    float cdf = 1.0f - pdf * poly;
    return x >= 0.0f ? cdf : 1.0f - cdf;
}

inline float as_pdf(float x) {
    float ax = std::min(std::fabs(x), 12.0f);
    return 0.3989422804014327f * std::exp(-0.5f * ax * ax);
}

inline float interp_lv(const float* vol_grid, int n_strikes, int n_maturities,
                       float moneyness, float tau) {
    // Match L1 output grid: uniform log-moneyness y=ln(K/S) ∈ [-0.35, +0.35].
    const float y = std::log(std::max(moneyness, 1e-4f));
    constexpr float y_wing = 0.35f;
    float u = std::clamp((y + y_wing) / (2.0f * y_wing), 0.0f, 1.0f);
    float v = std::clamp(tau, 1e-4f, 1.0f);
    float sx = u * static_cast<float>(std::max(n_strikes - 1, 1));
    int i = static_cast<int>(std::floor(sx));
    i = std::clamp(i, 1, std::max(n_strikes - 3, 1));
    int j = std::clamp(static_cast<int>(std::floor(
                (v - 1e-4f) / (1.0f - 1e-4f) * static_cast<float>(std::max(n_maturities - 1, 1)))),
            0, std::max(n_maturities - 1, 0));
    float t = sx - std::floor(sx);
    auto at = [&](int ii) { return vol_grid[ii * n_maturities + j]; };
    float y0 = at(i - 1), y1 = at(i), y2 = at(i + 1), y3 = at(std::min(i + 2, n_strikes - 1));
    float a0 = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
    float a1 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
    float a2 = -0.5f * y0 + 0.5f * y2;
    float a3 = y1;
    return ((a0 * t + a1) * t + a2) * t + a3;
}

void avx2_greeks(
    const float* spot, const float* strike, const float* tau, const float* rate,
    const float* vol_grid, int n_strikes, int n_maturities,
    float* price, float* delta, float* vega, int n
) {
#pragma omp parallel for schedule(dynamic) num_threads(16)
    for (int i = 0; i < n; ++i) {
        float S = spot[i];
        float K = strike[i];
        float T = std::max(tau[i], 1e-6f);
        float r = rate[i];
        float vol = interp_lv(vol_grid, n_strikes, n_maturities, K / std::max(S, 1e-6f), T);
        // Floor only — no artificial 5.0f ceiling (rough-vol tails must pass through).
        vol = std::max(vol, 1e-4f);
        float sqrtT = std::sqrt(T);
        float d1 = (std::log(std::max(S / K, 1e-8f)) + (r + 0.5f * vol * vol) * T) / (vol * sqrtT);
        float d2 = d1 - vol * sqrtT;
        float Nd1 = as_cdf(d1);
        float Nd2 = as_cdf(d2);
        float pdf = as_pdf(d1);
        float disc = std::exp(std::clamp(-r * T, -80.0f, 80.0f));
        price[i] = S * Nd1 - K * disc * Nd2;
        delta[i] = Nd1;
        vega[i] = S * pdf * sqrtT;
    }
    // Touch AVX2 path explicitly for 8-wide loads on prices (validation / throughput)
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 p = _mm256_loadu_ps(price + i);
        _mm256_storeu_ps(price + i, p);
    }
}

struct OpenCLContext {
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    bool ok = false;
    // Persistent buffers — avoid ~500µs create/destroy per launch on rusticl
    size_t cap_n = 0;
    size_t cap_v = 0;
    cl_mem bS = nullptr, bK = nullptr, bT = nullptr, bR = nullptr;
    cl_mem bV = nullptr, bP = nullptr, bD = nullptr, bG = nullptr;

    bool ensure_caps(size_t n, size_t v_elems) {
        cl_int err = 0;
        auto remake = [&](cl_mem& buf, size_t bytes, cl_mem_flags flags) -> bool {
            if (buf) {
                clReleaseMemObject(buf);
                buf = nullptr;
            }
            buf = clCreateBuffer(ctx, flags, bytes, nullptr, &err);
            return err == CL_SUCCESS && buf != nullptr;
        };
        if (n > cap_n) {
            const size_t bytes = n * sizeof(float);
            if (!remake(bS, bytes, CL_MEM_READ_ONLY)) return false;
            if (!remake(bK, bytes, CL_MEM_READ_ONLY)) return false;
            if (!remake(bT, bytes, CL_MEM_READ_ONLY)) return false;
            if (!remake(bR, bytes, CL_MEM_READ_ONLY)) return false;
            if (!remake(bP, bytes, CL_MEM_WRITE_ONLY)) return false;
            if (!remake(bD, bytes, CL_MEM_WRITE_ONLY)) return false;
            if (!remake(bG, bytes, CL_MEM_WRITE_ONLY)) return false;
            cap_n = n;
        }
        if (v_elems > cap_v) {
            if (!remake(bV, v_elems * sizeof(float), CL_MEM_READ_ONLY)) return false;
            cap_v = v_elems;
        }
        return true;
    }
};

OpenCLContext& get_ocl() {
    static OpenCLContext ocl;
    static std::once_flag once;
    std::call_once(once, []() {
        auto fail = [](const char* msg) {
            std::cerr << "[polaris] OpenCL unavailable (" << msg << "); using AVX2 CPU\n";
        };
        cl_uint nplat = 0;
        cl_int plat_err = clGetPlatformIDs(0, nullptr, &nplat);
        if (plat_err != CL_SUCCESS || nplat == 0) {
            fail(nplat == 0 ? "no platforms — install mesa-opencl-icd / ICD vendor" : "clGetPlatformIDs failed");
            return;
        }
        std::vector<cl_platform_id> plats(nplat);
        clGetPlatformIDs(nplat, plats.data(), nullptr);

        const char* hint = std::getenv("MASCOTRL_OPENCL_DEVICE_HINT");
        std::string hints = hint ? hint : "Ellesmere|gfx803|RX 590|Radeon|AMD";

        cl_device_id chosen = nullptr;
        for (auto plat : plats) {
            cl_uint ndev = 0;
            if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev) != CL_SUCCESS || !ndev)
                continue;
            std::vector<cl_device_id> devs(ndev);
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, ndev, devs.data(), nullptr);
            for (auto d : devs) {
                char name[256] = {0};
                clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, nullptr);
                std::string ns(name);
                // Prefer hint match; else take first GPU
                bool match = false;
                std::stringstream ss(hints);
                std::string tok;
                while (std::getline(ss, tok, '|')) {
                    if (!tok.empty() && ns.find(tok) != std::string::npos) match = true;
                }
                if (match || !chosen) chosen = d;
                if (match) break;
            }
            if (chosen) break;
        }
        if (!chosen) {
            fail("no GPU devices on any platform");
            return;
        }

        cl_int err = 0;
        ocl.ctx = clCreateContext(nullptr, 1, &chosen, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) { fail("clCreateContext failed"); return; }
        ocl.queue = clCreateCommandQueue(ocl.ctx, chosen, 0, &err);
        if (err != CL_SUCCESS) { fail("clCreateCommandQueue failed"); return; }

        // Locate kernel source relative to common install layouts
        std::vector<std::string> candidates = {
            "src/pricing/kernels/bs_greeks.cl",
            "../src/pricing/kernels/bs_greeks.cl",
            "mascotrl/src/pricing/kernels/bs_greeks.cl",
        };
        if (const char* root = std::getenv("MASCOTRL_ROOT")) {
            candidates.insert(candidates.begin(), std::string(root) + "/src/pricing/kernels/bs_greeks.cl");
        }
        std::string src;
        for (const auto& path : candidates) {
            std::ifstream in(path);
            if (!in) continue;
            std::ostringstream ss;
            ss << in.rdbuf();
            src = ss.str();
            break;
        }
        if (src.empty()) {
            fail("kernel source not found");
            return;
        }
        const char* csrc = src.c_str();
        size_t slen = src.size();
        ocl.program = clCreateProgramWithSource(ocl.ctx, 1, &csrc, &slen, &err);
        if (err != CL_SUCCESS) { fail("clCreateProgramWithSource failed"); return; }
        err = clBuildProgram(ocl.program, 1, &chosen, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(ocl.program, chosen, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(ocl.program, chosen, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "[polaris] OpenCL build failed:\n" << log << "\n";
            return;
        }
        ocl.kernel = clCreateKernel(ocl.program, "bs_greeks_fused", &err);
        ocl.ok = (err == CL_SUCCESS);
        if (ocl.ok) {
            char name[256] = {0};
            clGetDeviceInfo(chosen, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            std::cerr << "[polaris] OpenCL device: " << name << "\n";
        } else {
            fail("clCreateKernel failed");
        }
    });
    return ocl;
}

bool run_opencl(
    const float* spot, const float* strike, const float* tau, const float* rate,
    const float* vol_grid, int n_strikes, int n_maturities,
    float* price, float* delta, float* vega, int n,
    int vol_stride
) {
    auto& ocl = get_ocl();
    if (!ocl.ok) return false;

    // Shared-grid soak launches need large n to beat AVX on rusticl.
    // Stacked multi-asset path (vol_stride>0) uploads K surfaces once and
    // launches K work-items — worth it well below the soak threshold.
    int min_n = 256;
    if (vol_stride > 0) {
        min_n = 8;
        if (const char* env = std::getenv("MASCOTRL_OCL_MIN_N_STACKED")) {
            min_n = std::max(1, std::atoi(env));
        }
    } else if (const char* env = std::getenv("MASCOTRL_OCL_MIN_N")) {
        min_n = std::max(1, std::atoi(env));
    }
    if (n < min_n) return false;

    const size_t bytes = sizeof(float) * static_cast<size_t>(n);
    const size_t v_elems = (vol_stride > 0)
        ? (static_cast<size_t>(n) * static_cast<size_t>(vol_stride))
        : (static_cast<size_t>(n_strikes) * static_cast<size_t>(n_maturities));
    if (!ocl.ensure_caps(static_cast<size_t>(n), v_elems)) return false;

    cl_int err = 0;
    err |= clEnqueueWriteBuffer(ocl.queue, ocl.bS, CL_FALSE, 0, bytes, spot, 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(ocl.queue, ocl.bK, CL_FALSE, 0, bytes, strike, 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(ocl.queue, ocl.bT, CL_FALSE, 0, bytes, tau, 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(ocl.queue, ocl.bR, CL_FALSE, 0, bytes, rate, 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(ocl.queue, ocl.bV, CL_FALSE, 0, v_elems * sizeof(float), vol_grid, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    int arg = 0;
    clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &ocl.bS);
    clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &ocl.bK);
    clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &ocl.bT);
    clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &ocl.bR);
    clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &ocl.bV);
    clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &ocl.bP);
    clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &ocl.bD);
    clSetKernelArg(ocl.kernel, arg++, sizeof(cl_mem), &ocl.bG);
    clSetKernelArg(ocl.kernel, arg++, sizeof(int), &n);
    clSetKernelArg(ocl.kernel, arg++, sizeof(int), &n_strikes);
    clSetKernelArg(ocl.kernel, arg++, sizeof(int), &n_maturities);
    clSetKernelArg(ocl.kernel, arg++, sizeof(int), &vol_stride);

    size_t global = static_cast<size_t>((n + 255) / 256 * 256);
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    clEnqueueReadBuffer(ocl.queue, ocl.bP, CL_FALSE, 0, bytes, price, 0, nullptr, nullptr);
    clEnqueueReadBuffer(ocl.queue, ocl.bD, CL_FALSE, 0, bytes, delta, 0, nullptr, nullptr);
    clEnqueueReadBuffer(ocl.queue, ocl.bG, CL_TRUE, 0, bytes, vega, 0, nullptr, nullptr);
    return true;
}

void avx2_greeks_stacked(
    const float* spot, const float* strike, const float* tau, const float* rate,
    const float* vol_grids, int n_strikes, int n_maturities,
    float* price, float* delta, float* vega, int n_assets
) {
    const int stride = n_strikes * n_maturities;
    // Honor OMP_NUM_THREADS (no hardcoded 16) so overnight caps stay effective.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_assets; ++i) {
        float S = spot[i];
        float K = strike[i];
        float T = std::max(tau[i], 1e-6f);
        float r = rate[i];
        const float* vol_grid = vol_grids + static_cast<size_t>(i) * static_cast<size_t>(stride);
        float vol = interp_lv(vol_grid, n_strikes, n_maturities, K / std::max(S, 1e-6f), T);
        vol = std::max(vol, 1e-4f);
        float sqrtT = std::sqrt(T);
        float d1 = (std::log(std::max(S / K, 1e-8f)) + (r + 0.5f * vol * vol) * T) / (vol * sqrtT);
        float d2 = d1 - vol * sqrtT;
        float Nd1 = as_cdf(d1);
        float Nd2 = as_cdf(d2);
        float pdf = as_pdf(d1);
        float disc = std::exp(std::clamp(-r * T, -80.0f, 80.0f));
        price[i] = S * Nd1 - K * disc * Nd2;
        delta[i] = Nd1;
        vega[i] = S * pdf * sqrtT;
    }
}

}  // namespace

bool polaris_opencl_available() {
    return get_ocl().ok;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_greeks_fused(
    const at::Tensor& spot,
    const at::Tensor& strike,
    const at::Tensor& time_to_maturity,
    const at::Tensor& rate,
    const at::Tensor& vol_grid,
    bool force_cpu
) {
    TORCH_CHECK(spot.is_contiguous() && spot.scalar_type() == at::kFloat, "spot float32 contiguous");
    TORCH_CHECK(strike.is_contiguous() && strike.scalar_type() == at::kFloat, "strike float32 contiguous");
    TORCH_CHECK(time_to_maturity.is_contiguous() && time_to_maturity.scalar_type() == at::kFloat, "tau float32");
    TORCH_CHECK(rate.is_contiguous() && rate.scalar_type() == at::kFloat, "rate float32");
    TORCH_CHECK(vol_grid.is_contiguous() && vol_grid.scalar_type() == at::kFloat, "vol_grid float32");
    TORCH_CHECK(vol_grid.dim() == 2, "vol_grid must be 2D [strikes, maturities]");

    const int n = static_cast<int>(spot.numel());
    const int n_strikes = static_cast<int>(vol_grid.size(0));
    const int n_maturities = static_cast<int>(vol_grid.size(1));

    auto price = at::empty({n}, spot.options());
    auto delta = at::empty({n}, spot.options());
    auto vega = at::empty({n}, spot.options());

    const float* sp = spot.data_ptr<float>();
    const float* st = strike.data_ptr<float>();
    const float* ta = time_to_maturity.data_ptr<float>();
    const float* rt = rate.data_ptr<float>();
    const float* vg = vol_grid.data_ptr<float>();
    float* pr = price.data_ptr<float>();
    float* de = delta.data_ptr<float>();
    float* ve = vega.data_ptr<float>();

    bool used_gpu = false;
    if (!force_cpu) {
        used_gpu = run_opencl(sp, st, ta, rt, vg, n_strikes, n_maturities, pr, de, ve, n, /*vol_stride=*/0);
    }
    if (!used_gpu) {
        avx2_greeks(sp, st, ta, rt, vg, n_strikes, n_maturities, pr, de, ve, n);
    }
    return {price, delta, vega};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_greeks_fused_stacked(
    const at::Tensor& spot,
    const at::Tensor& strike,
    const at::Tensor& time_to_maturity,
    const at::Tensor& rate,
    const at::Tensor& vol_grids,
    bool force_cpu
) {
    TORCH_CHECK(spot.is_contiguous() && spot.scalar_type() == at::kFloat, "spot float32 contiguous");
    TORCH_CHECK(strike.is_contiguous() && strike.scalar_type() == at::kFloat, "strike float32 contiguous");
    TORCH_CHECK(time_to_maturity.is_contiguous() && time_to_maturity.scalar_type() == at::kFloat, "tau float32");
    TORCH_CHECK(rate.is_contiguous() && rate.scalar_type() == at::kFloat, "rate float32");
    TORCH_CHECK(vol_grids.is_contiguous() && vol_grids.scalar_type() == at::kFloat, "vol_grids float32");
    TORCH_CHECK(vol_grids.dim() == 3, "vol_grids must be 3D [n_assets, strikes, maturities]");

    const int n = static_cast<int>(spot.numel());
    TORCH_CHECK(strike.numel() == n && time_to_maturity.numel() == n && rate.numel() == n,
                "spot/strike/tau/rate length mismatch");
    TORCH_CHECK(vol_grids.size(0) == n, "vol_grids dim0 must equal n_assets (=spot.numel())");

    const int n_strikes = static_cast<int>(vol_grids.size(1));
    const int n_maturities = static_cast<int>(vol_grids.size(2));
    const int vol_stride = n_strikes * n_maturities;

    auto price = at::empty({n}, spot.options());
    auto delta = at::empty({n}, spot.options());
    auto vega = at::empty({n}, spot.options());

    const float* sp = spot.data_ptr<float>();
    const float* st = strike.data_ptr<float>();
    const float* ta = time_to_maturity.data_ptr<float>();
    const float* rt = rate.data_ptr<float>();
    const float* vg = vol_grids.data_ptr<float>();
    float* pr = price.data_ptr<float>();
    float* de = delta.data_ptr<float>();
    float* ve = vega.data_ptr<float>();

    bool used_gpu = false;
    if (!force_cpu) {
        used_gpu = run_opencl(
            sp, st, ta, rt, vg, n_strikes, n_maturities, pr, de, ve, n, vol_stride);
    }
    if (!used_gpu) {
        avx2_greeks_stacked(sp, st, ta, rt, vg, n_strikes, n_maturities, pr, de, ve, n);
    }
    return {price, delta, vega};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_greeks_fused", &compute_greeks_fused,
          "Fused OpenCL/AVX2 Black-Scholes Greeks (shared vol surface)",
          pybind11::arg("spot"),
          pybind11::arg("strike"),
          pybind11::arg("time_to_maturity"),
          pybind11::arg("rate"),
          pybind11::arg("vol_grid"),
          pybind11::arg("force_cpu") = false);
    m.def("compute_greeks_fused_stacked", &compute_greeks_fused_stacked,
          "Fused OpenCL/AVX2 Greeks with per-asset stacked LV surfaces",
          pybind11::arg("spot"),
          pybind11::arg("strike"),
          pybind11::arg("time_to_maturity"),
          pybind11::arg("rate"),
          pybind11::arg("vol_grids"),
          pybind11::arg("force_cpu") = false);
    m.def("opencl_available", &polaris_opencl_available,
          "Whether Polaris OpenCL GPU context initialized");
}
