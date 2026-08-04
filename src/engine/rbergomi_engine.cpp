#include "rbergomi_engine.hpp"
#include "circulant_fft.hpp"
#include "dupire_pde.hpp"

#include <arrow/c/abi.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include <cmath>

namespace {

struct AlignedBuffer {
    float* data = nullptr;
    std::size_t n = 0;
    explicit AlignedBuffer(std::size_t count) : n(count) {
        data = static_cast<float*>(_mm_malloc(count * sizeof(float), 32));
        if (!data) throw std::bad_alloc();
        std::memset(data, 0, count * sizeof(float));
    }
    ~AlignedBuffer() {
        if (data) _mm_free(data);
    }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

// Xoshiro256++
struct Xoshiro256pp {
    uint64_t s[4];
    static uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
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
    float uniform() {
        return (next() >> 11) * (1.0f / 9007199254740992.0f);
    }
    float gauss() {
        // Box-Muller
        float u1 = std::max(uniform(), 1e-10f);
        float u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265358979323846f * u2);
    }
};

void seed_xoshiro(Xoshiro256pp& rng, uint64_t seed) {
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

inline std::size_t idx5(int p, int k, int t, int s, int m,
                        int K, int T, int S, int M) {
    return static_cast<std::size_t>(p) * (K * T * S * M)
         + static_cast<std::size_t>(k) * (T * S * M)
         + static_cast<std::size_t>(t) * (S * M)
         + static_cast<std::size_t>(s) * M
         + static_cast<std::size_t>(m);
}

struct ArrowOwner {
    ArrowSchema schema{};
    ArrowArray array{};
    std::shared_ptr<AlignedBuffer> buf;
};

void release_schema(ArrowSchema* schema) {
    if (!schema) return;
    if (schema->name) {
        std::free(const_cast<char*>(schema->name));
        schema->name = nullptr;
    }
    if (schema->format) {
        std::free(const_cast<char*>(schema->format));
        schema->format = nullptr;
    }
    schema->release = nullptr;
}

void release_array(ArrowArray* array) {
    if (!array) return;
    auto** private_data = reinterpret_cast<std::shared_ptr<AlignedBuffer>**>(array->private_data);
    if (private_data && *private_data) {
        delete *private_data;
        delete private_data;
        array->private_data = nullptr;
    }
    if (array->buffers) {
        delete[] array->buffers;
        array->buffers = nullptr;
    }
    array->release = nullptr;
}

}  // namespace

std::tuple<pybind11::capsule, pybind11::capsule> generate_surfaces(
    const EngineConfig& config,
    const float* cholesky_matrix
) {
    const int P = config.n_paths;
    const int K = config.n_assets;
    const int T = config.n_steps;
    const int S = config.n_strikes;
    const int M = config.n_maturities;
    const float H = config.hurst_exponent;

    if (P <= 0 || K <= 0 || T <= 0 || S <= 0 || M <= 0) {
        throw std::invalid_argument("EngineConfig dimensions must be positive");
    }

    const std::size_t total = static_cast<std::size_t>(P) * K * T * S * M;
    auto buffer = std::make_shared<AlignedBuffer>(total);

    const std::size_t m_fft = next_pow2(static_cast<std::size_t>(2 * T));
    const float dt = 1.0f / 252.0f;
    const float rate = 0.02f;
    const float xi0 = 0.04f;
    const float eta = 1.5f;

    const uint64_t base_seed = config.seed;

#pragma omp parallel num_threads(16)
    {
        // Deliberately NOT seeded per thread. Seeding by omp_get_thread_num()
        // combined with schedule(dynamic) makes the random stream a given path
        // receives depend on which thread happens to pick it up, so identical
        // configs produce different surfaces run to run. The stream is instead
        // re-keyed per (path, asset) inside the loop below.
        Xoshiro256pp rng;

        std::vector<float> Z(K);
        std::vector<float> Zc(K);
        std::vector<float> g_noise(2 * m_fft);
        std::vector<float> fbm_inc(T);
        std::vector<float> var_path(T);
        std::vector<float> spot_path(T);
        std::vector<float> lv(S * M);

#pragma omp for schedule(dynamic)
        for (int p = 0; p < P; ++p) {
            for (int a = 0; a < K; ++a) {
                // Re-key the stream from (seed, path, asset). SplitMix-style
                // mixing keeps neighbouring work items well separated while
                // making the draw sequence independent of thread scheduling.
                {
                    uint64_t s = base_seed;
                    s ^= (static_cast<uint64_t>(p) + 1ULL) * 0x9E3779B97F4A7C15ULL;
                    s ^= (static_cast<uint64_t>(a) + 1ULL) * 0xBF58476D1CE4E5B9ULL;
                    seed_xoshiro(rng, s);
                }
                // Fractional driver via circulant embedding
                for (std::size_t i = 0; i < 2 * m_fft; ++i) g_noise[i] = rng.gauss();
                simulate_fractional_increments(T, H, g_noise.data(), fbm_inc.data());

                // Correlated Brownian for spot (Cholesky across assets, fresh draws)
                for (int j = 0; j < K; ++j) Z[j] = rng.gauss();
                for (int i = 0; i < K; ++i) {
                    float acc = 0.0f;
                    for (int j = 0; j <= i; ++j) {
                        acc += cholesky_matrix[i * K + j] * Z[j];
                    }
                    Zc[i] = acc;
                }

                const float rho = -0.75f;
                const float rho_perp = std::sqrt(std::max(0.0f, 1.0f - rho * rho));

                float logS = std::log(100.0f);
                // rBergomi: V_t = ξ0 exp( η W^H_t − ½ η² t^{2H} )
                // W^H from circulant embedding already carries √(2H)(t)^{H-½}√dt.
                for (int t = 0; t < T; ++t) {
                    const float tau = dt * static_cast<float>(t + 1);
                    const float wh = fbm_inc[t];
                    const float ito = 0.5f * eta * eta * std::pow(tau, 2.0f * H);
                    float logV = std::log(xi0) + eta * wh - ito;
                    logV = std::max(logV, -23.0f);
                    float v = std::exp(logV);
                    if (!std::isfinite(v) || v <= 0.0f) {
                        v = (t > 0 && std::isfinite(var_path[t - 1])) ? var_path[t - 1] : xi0;
                    }
                    var_path[t] = v;
                    for (int j = 0; j < K; ++j) Z[j] = rng.gauss();
                    for (int i = 0; i < K; ++i) {
                        float acc = 0.0f;
                        for (int j = 0; j <= i; ++j) {
                            acc += cholesky_matrix[i * K + j] * Z[j];
                        }
                        Zc[i] = acc;
                    }
                    // Leverage: correlate spot Brownian with the Volterra driver
                    // (contemporaneous white noise that feeds circulant embedding).
                    const float dW2 = g_noise[2 * static_cast<std::size_t>(t)];
                    const float dW_spot = rho * dW2 + rho_perp * Zc[a];
                    const float v_euler = std::min(v, 1.0e4f);
                    logS += (rate - 0.5f * v_euler) * dt + std::sqrt(v_euler * dt) * dW_spot;
                    logS = std::min(std::max(logS, std::log(1.0f)), std::log(1.0e4f));
                    spot_path[t] = std::exp(logS);

                    solve_dupire_local_vol(
                        spot_path[t], rate, v, S, M, lv.data(), H, eta, rho
                    );
                    for (int s = 0; s < S; ++s) {
                        for (int m = 0; m < M; ++m) {
                            float x = lv[s * M + m];
                            if (!std::isfinite(x) || x < 1e-4f) {
                                x = 1e-4f;
                            }
                            buffer->data[idx5(p, a, t, s, m, K, T, S, M)] = x;
                        }
                    }
                }
            }
        }
    }

    auto owner = new ArrowOwner();
    owner->buf = buffer;

    // Schema: large binary / fixed-size float list — expose as dense float32 array via format "+w:N" is awkward.
    // Use primitive float32 array of length total (Python reshapes).
    owner->schema.format = strdup("f");  // float32
    owner->schema.name = strdup("surface");
    owner->schema.metadata = nullptr;
    owner->schema.flags = 0;
    owner->schema.n_children = 0;
    owner->schema.children = nullptr;
    owner->schema.dictionary = nullptr;
    owner->schema.release = &release_schema;
    owner->schema.private_data = nullptr;

    owner->array.length = static_cast<int64_t>(total);
    owner->array.null_count = 0;
    owner->array.offset = 0;
    owner->array.n_buffers = 2;
    owner->array.n_children = 0;
    owner->array.children = nullptr;
    owner->array.dictionary = nullptr;
    owner->array.release = &release_array;
    auto** holder = new std::shared_ptr<AlignedBuffer>*;
    *holder = new std::shared_ptr<AlignedBuffer>(buffer);
    owner->array.private_data = holder;
    owner->array.buffers = new const void*[2];
    owner->array.buffers[0] = nullptr;  // validity
    owner->array.buffers[1] = buffer->data;

    auto schema_capsule = pybind11::capsule(&owner->schema, "arrow_schema", [](void* p) {
        auto* sch = static_cast<ArrowSchema*>(p);
        if (sch && sch->release) sch->release(sch);
    });
    auto array_capsule = pybind11::capsule(&owner->array, "arrow_array", [](void* p) {
        auto* arr = static_cast<ArrowArray*>(p);
        if (arr && arr->release) arr->release(arr);
        // ArrowOwner leaks schema/array structs intentionally tied to capsules;
        // buffer lifetime held via private_data shared_ptr.
    });

    // Keep ArrowOwner alive by attaching to array capsule context via a second shared holder.
    // Simpler: leak the ArrowOwner shell (schema/array) — buffer freed via release_array.
    (void)owner;

    return {schema_capsule, array_capsule};
}
