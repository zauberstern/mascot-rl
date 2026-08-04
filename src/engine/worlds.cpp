#include "worlds.hpp"
#include "bs_iv.hpp"
#include "dupire_pde.hpp"
#include "garch_duan.hpp"
#include "heston_cf.hpp"
#include "sabr_hagan.hpp"
#include "circulant_fft.hpp"

#include <arrow/c/abi.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <immintrin.h>
#include <omp.h>

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
        float u1 = std::max(uniform(), 1e-10f);
        float u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1))
             * std::cos(2.0f * 3.14159265358979323846f * u2);
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

// Andersen (2008) QE / QE-M. Schemes: 0=full truncation Euler, 1=QE, 2=QE-M.
constexpr float kHestonPsiC = 1.5f;
constexpr float kHestonG1 = 0.5f;
constexpr float kHestonG2 = 0.5f;

inline float heston_qe_next_v(
    float v,
    float kappa,
    float theta,
    float xi,
    float dt,
    float u,
    float z
) {
    const float ekt = std::exp(-kappa * dt);
    const float m = theta + (v - theta) * ekt;
    const float s2 =
        v * xi * xi * ekt * (1.0f - ekt) / kappa
        + theta * xi * xi * (1.0f - ekt) * (1.0f - ekt) / (2.0f * kappa);
    const float psi = s2 / std::max(m * m, 1e-16f);
    if (psi <= kHestonPsiC) {
        const float inv = 2.0f / std::max(psi, 1e-16f);
        const float b2 =
            inv - 1.0f
            + std::sqrt(std::max(inv, 0.0f)) * std::sqrt(std::max(inv - 1.0f, 0.0f));
        const float a = m / (1.0f + b2);
        const float t = std::sqrt(std::max(b2, 0.0f)) + z;
        return std::max(a * t * t, 0.0f);
    }
    const float p = (psi - 1.0f) / (psi + 1.0f);
    const float beta = (1.0f - p) / std::max(m, 1e-16f);
    if (u <= p) {
        return 0.0f;
    }
    return std::log(std::max((1.0f - p) / std::max(1.0f - u, 1e-16f), 1e-16f)) / beta;
}

inline float heston_qe_k0(
    float v,
    float kappa,
    float theta,
    float xi,
    float rho,
    float dt,
    bool martingale,
    float& k1,
    float& k2,
    float& k3,
    float& k4
) {
    const float g1 = kHestonG1;
    const float g2 = kHestonG2;
    float k0 = -rho * kappa * theta * dt / xi;
    k1 = g1 * dt * (kappa * rho / xi - 0.5f) - rho / xi;
    k2 = g2 * dt * (kappa * rho / xi - 0.5f) + rho / xi;
    k3 = g1 * dt * (1.0f - rho * rho);
    k4 = g2 * dt * (1.0f - rho * rho);
    if (!martingale) {
        return k0;
    }
    const float A = k2 + 0.5f * k4;
    const float ekt = std::exp(-kappa * dt);
    const float m = theta + (v - theta) * ekt;
    const float s2 =
        v * xi * xi * ekt * (1.0f - ekt) / kappa
        + theta * xi * xi * (1.0f - ekt) * (1.0f - ekt) / (2.0f * kappa);
    const float psi = s2 / std::max(m * m, 1e-16f);
    float M = 1.0f;
    if (psi <= kHestonPsiC) {
        const float inv = 2.0f / std::max(psi, 1e-16f);
        const float b2 =
            inv - 1.0f
            + std::sqrt(std::max(inv, 0.0f)) * std::sqrt(std::max(inv - 1.0f, 0.0f));
        const float a = m / (1.0f + b2);
        const float denom = std::max(1.0f - 2.0f * A * a, 1e-12f);
        M = std::exp(A * b2 * a / denom) / std::sqrt(denom);
    } else {
        const float p = (psi - 1.0f) / (psi + 1.0f);
        const float beta = (1.0f - p) / std::max(m, 1e-16f);
        M = p + (1.0f - p) * beta / std::max(beta - A, 1e-12f);
    }
    if (!std::isfinite(M) || M <= 1e-16f) {
        M = 1.0f;
    }
    return -(k1 + 0.5f * k3) * v - std::log(M);
}

inline std::size_t idx5(int p, int k, int t, int s, int m, int K, int T, int S, int M) {
    return static_cast<std::size_t>(p) * (K * T * S * M)
         + static_cast<std::size_t>(k) * (T * S * M)
         + static_cast<std::size_t>(t) * (S * M)
         + static_cast<std::size_t>(s) * M
         + static_cast<std::size_t>(m);
}

inline std::size_t idx3(int p, int k, int t, int K, int T) {
    return static_cast<std::size_t>(p) * (K * T)
         + static_cast<std::size_t>(k) * T
         + static_cast<std::size_t>(t);
}

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
    auto** private_data =
        reinterpret_cast<std::shared_ptr<AlignedBuffer>**>(array->private_data);
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

struct ArrowOwner {
    ArrowSchema schema{};
    ArrowArray array{};
    std::shared_ptr<AlignedBuffer> buf;
};

std::tuple<pybind11::capsule, pybind11::capsule> make_arrow_pair(
    std::shared_ptr<AlignedBuffer> buffer, std::size_t total, const char* name
) {
    auto owner = new ArrowOwner();
    owner->buf = buffer;
    owner->schema.format = strdup("f");
    owner->schema.name = strdup(name);
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
    owner->array.buffers[0] = nullptr;
    owner->array.buffers[1] = buffer->data;

    auto schema_capsule = pybind11::capsule(&owner->schema, "arrow_schema", [](void* p) {
        auto* sch = static_cast<ArrowSchema*>(p);
        if (sch && sch->release) sch->release(sch);
    });
    auto array_capsule = pybind11::capsule(&owner->array, "arrow_array", [](void* p) {
        auto* arr = static_cast<ArrowArray*>(p);
        if (arr && arr->release) arr->release(arr);
    });
    (void)owner;
    return {schema_capsule, array_capsule};
}

void fill_strike_maturity_grid(
    float spot,
    int S,
    int M,
    std::vector<float>& strikes,
    std::vector<float>& mats
) {
    strikes.resize(static_cast<std::size_t>(S));
    mats.resize(static_cast<std::size_t>(M));
    for (int s = 0; s < S; ++s) {
        const float u = (S == 1) ? 0.5f : static_cast<float>(s) / static_cast<float>(S - 1);
        const float y = -DUPIRE_LOG_MNY_WING + u * (2.0f * DUPIRE_LOG_MNY_WING);
        strikes[static_cast<std::size_t>(s)] = spot * std::exp(y);
    }
    for (int m = 0; m < M; ++m) {
        const float u = (M == 1) ? 1.0f : static_cast<float>(m + 1) / static_cast<float>(M);
        mats[static_cast<std::size_t>(m)] = u;  // years in (0, 1]
    }
}

}  // namespace

std::tuple<
    pybind11::capsule, pybind11::capsule,
    pybind11::capsule, pybind11::capsule,
    pybind11::capsule, pybind11::capsule>
generate_world(const WorldConfig& config, const float* cholesky_matrix) {
    const EngineConfig& cfg = config.base;
    const int P = cfg.n_paths;
    const int K = cfg.n_assets;
    const int T = cfg.n_steps;
    const int S = cfg.n_strikes;
    const int M = cfg.n_maturities;
    if (P <= 0 || K <= 0 || T <= 0 || S <= 0 || M <= 0) {
        throw std::invalid_argument("WorldConfig dimensions must be positive");
    }

    const std::size_t total_surf = static_cast<std::size_t>(P) * K * T * S * M;
    const std::size_t total_path = static_cast<std::size_t>(P) * K * T;
    auto surf_buf = std::make_shared<AlignedBuffer>(total_surf);
    auto spot_buf = std::make_shared<AlignedBuffer>(total_path);
    auto iv_buf = std::make_shared<AlignedBuffer>(total_path);

    const float dt = 1.0f / 252.0f;
    const float rate = config.rate;
    const float div_q = config.div_q;
    const float spot0 = config.spot0;
    const uint64_t base_seed = cfg.seed;
    const int world = config.world;

    // Optional GARCH price table (built once if world==3).
    std::vector<float> garch_prices;
    std::vector<float> garch_h_nodes(32);
    std::vector<float> grid_strikes;
    std::vector<float> grid_mats;
    fill_strike_maturity_grid(spot0, S, M, grid_strikes, grid_mats);
    if (world == static_cast<int>(WorldId::Garch)) {
        GarchParams gp;
        gp.mu = config.garch_mu;
        gp.omega = config.garch_omega;
        gp.alpha = config.garch_alpha;
        gp.beta = config.garch_beta;
        gp.gamma = config.garch_gamma;
        gp.lambda = config.garch_lambda;
        if (!garch_params_ok(gp)) {
            throw std::invalid_argument("GJR-GARCH params violate stationarity");
        }
        garch_prices.assign(static_cast<std::size_t>(32 * M * S), 0.0f);
        garch_build_price_table(
            spot0,
            rate,
            div_q,
            S,
            M,
            grid_strikes.data(),
            grid_mats.data(),
            gp,
            std::max(config.garch_n_inner, 256),
            base_seed,
            garch_prices.data(),
            garch_h_nodes.data()
        );
    }

#pragma omp parallel
    {
        Xoshiro256pp rng;
        std::vector<float> Z(static_cast<std::size_t>(K));
        std::vector<float> Zc(static_cast<std::size_t>(K));

#pragma omp for schedule(dynamic)
        for (int p = 0; p < P; ++p) {
            for (int a = 0; a < K; ++a) {
                {
                    uint64_t s = base_seed;
                    s ^= (static_cast<uint64_t>(p) + 1ULL) * 0x9E3779B97F4A7C15ULL;
                    s ^= (static_cast<uint64_t>(a) + 1ULL) * 0xBF58476D1CE4E5B9ULL;
                    s ^= (static_cast<uint64_t>(world) + 1ULL) * 0x94D049BB133111EBULL;
                    seed_xoshiro(rng, s);
                }

                float logS = std::log(spot0);
                float v = 0.04f;          // heston / atm iv state
                float sigma_sabr = config.sabr_sigma0;
                float h_garch = 0.0f;
                float eps_prev = 0.0f;

                if (world == static_cast<int>(WorldId::Heston)) {
                    v = config.heston_v0;
                } else if (world == static_cast<int>(WorldId::Garch)) {
                    h_garch = garch_long_run_variance(GarchParams{
                        config.garch_mu,
                        config.garch_omega,
                        config.garch_alpha,
                        config.garch_beta,
                        config.garch_gamma,
                        config.garch_lambda});
                } else if (world == static_cast<int>(WorldId::GBM)) {
                    v = config.gbm_sigma * config.gbm_sigma;
                }

                for (int t = 0; t < T; ++t) {
                    float atm_iv = 0.20f;
                    float spot = std::exp(logS);

                    if (world == static_cast<int>(WorldId::GBM)) {
                        const float sig = config.gbm_sigma;
                        const float mu = config.gbm_mu;
                        const float z = rng.gauss();
                        // Correlated multi-asset noise
                        for (int j = 0; j < K; ++j) Z[static_cast<std::size_t>(j)] = rng.gauss();
                        float zc = 0.0f;
                        for (int j = 0; j <= a; ++j) {
                            zc += cholesky_matrix[a * K + j] * Z[static_cast<std::size_t>(j)];
                        }
                        (void)z;
                        logS += (mu - 0.5f * sig * sig) * dt + sig * std::sqrt(dt) * zc;
                        spot = std::exp(logS);
                        atm_iv = sig;
                        // Flat IV surface
                        for (int s = 0; s < S; ++s) {
                            for (int m = 0; m < M; ++m) {
                                surf_buf->data[idx5(p, a, t, s, m, K, T, S, M)] = sig;
                            }
                        }
                    } else if (world == static_cast<int>(WorldId::Heston)) {
                        const float kappa = config.heston_kappa;
                        const float theta = config.heston_theta;
                        const float xi = config.heston_xi;
                        const float rho = config.heston_rho;
                        const int scheme = config.heston_scheme;
                        for (int j = 0; j < K; ++j) Z[static_cast<std::size_t>(j)] = rng.gauss();
                        float zc = 0.0f;
                        for (int j = 0; j <= a; ++j) {
                            zc += cholesky_matrix[a * K + j] * Z[static_cast<std::size_t>(j)];
                        }
                        const float v_plus = std::max(v, 0.0f);
                        float v_next = v_plus;
                        if (scheme <= 0) {
                            // Legacy full-truncation Euler (QuantLib FullTruncation).
                            float z2 = rng.gauss();
                            float z1 = rho * z2
                                       + std::sqrt(std::max(0.0f, 1.0f - rho * rho)) * zc;
                            v_next = v + kappa * (theta - v_plus) * dt
                                   + xi * std::sqrt(v_plus * dt) * z2;
                            logS += (rate - div_q - 0.5f * v_plus) * dt
                                  + std::sqrt(v_plus * dt) * z1;
                        } else {
                            // Andersen QE (1) or QE-M (2, default).
                            const float u = rng.uniform();
                            const float zv = rng.gauss();
                            v_next = heston_qe_next_v(v_plus, kappa, theta, xi, dt, u, zv);
                            float k1 = 0.0f, k2 = 0.0f, k3 = 0.0f, k4 = 0.0f;
                            const float k0 = heston_qe_k0(
                                v_plus, kappa, theta, xi, rho, dt,
                                /*martingale=*/scheme >= 2, k1, k2, k3, k4
                            );
                            const float vol = std::sqrt(
                                std::max(k3 * v_plus + k4 * std::max(v_next, 0.0f), 0.0f)
                            );
                            // Cross-asset correlation rides the orthogonal shock zc.
                            logS += (rate - div_q) * dt + k0 + k1 * v_plus
                                  + k2 * std::max(v_next, 0.0f) + vol * zc;
                        }
                        v = v_next;
                        spot = std::exp(std::min(std::max(logS, std::log(1.0f)), std::log(1.0e4f)));
                        logS = std::log(spot);
                        atm_iv = std::sqrt(std::max(std::max(v, 0.0f), 1e-8f));

                        HestonParams hp;
                        hp.v0 = std::max(std::max(v, 0.0f), 1e-8f);
                        hp.theta = theta;
                        hp.kappa = kappa;
                        hp.xi = xi;
                        hp.rho = rho;
                        for (int s = 0; s < S; ++s) {
                            for (int m = 0; m < M; ++m) {
                                const float Ks = grid_strikes[static_cast<std::size_t>(s)];
                                const float tau = grid_mats[static_cast<std::size_t>(m)];
                                const float px = heston_call_price(spot, Ks, tau, rate, div_q, hp);
                                float iv = brent_implied_vol(px, spot, Ks, tau, rate, div_q);
                                if (!std::isfinite(iv) || iv < 1e-4f || iv > 5.0f) {
                                    iv = atm_iv;
                                }
                                surf_buf->data[idx5(p, a, t, s, m, K, T, S, M)] = iv;
                            }
                        }
                    } else if (world == static_cast<int>(WorldId::Garch)) {
                        const float z = rng.gauss();
                        for (int j = 0; j < K; ++j) Z[static_cast<std::size_t>(j)] = rng.gauss();
                        float zc = 0.0f;
                        for (int j = 0; j <= a; ++j) {
                            zc += cholesky_matrix[a * K + j] * Z[static_cast<std::size_t>(j)];
                        }
                        (void)z;
                        const float eps = std::sqrt(std::max(h_garch, 1e-12f)) * zc;
                        const float r_t = config.garch_mu + eps;
                        logS += r_t;
                        spot = std::exp(logS);
                        const float h_next =
                            config.garch_omega
                            + (config.garch_alpha
                               + config.garch_gamma * (eps_prev < 0.0f ? 1.0f : 0.0f))
                                  * eps_prev * eps_prev
                            + config.garch_beta * h_garch;
                        h_garch = std::max(h_next, 1e-12f);
                        eps_prev = eps;
                        atm_iv = std::sqrt(h_garch * 252.0f);  // annualized from daily var
                        for (int s = 0; s < S; ++s) {
                            for (int m = 0; m < M; ++m) {
                                const float px = garch_lookup_price(
                                    h_garch, s, m, S, M, garch_h_nodes.data(), garch_prices.data()
                                );
                                // Scale price by spot ratio vs spot0 table.
                                const float px_adj = px * (spot / spot0);
                                const float Ks = grid_strikes[static_cast<std::size_t>(s)]
                                               * (spot / spot0);
                                const float tau = grid_mats[static_cast<std::size_t>(m)];
                                float iv = brent_implied_vol(px_adj, spot, Ks, tau, rate, div_q);
                                if (!std::isfinite(iv) || iv < 1e-4f || iv > 5.0f) {
                                    iv = atm_iv;
                                }
                                surf_buf->data[idx5(p, a, t, s, m, K, T, S, M)] = iv;
                            }
                        }
                    } else if (world == static_cast<int>(WorldId::SABR)) {
                        const float nu = config.sabr_nu;
                        const float rho = config.sabr_rho;
                        float z1 = rng.gauss();
                        float z2 = rng.gauss();
                        for (int j = 0; j < K; ++j) Z[static_cast<std::size_t>(j)] = rng.gauss();
                        float zc = 0.0f;
                        for (int j = 0; j <= a; ++j) {
                            zc += cholesky_matrix[a * K + j] * Z[static_cast<std::size_t>(j)];
                        }
                        z1 = rho * z2 + std::sqrt(std::max(0.0f, 1.0f - rho * rho)) * zc;
                        const float sig = std::max(sigma_sabr, 1e-8f);
                        logS += (rate - div_q - 0.5f * sig * sig) * dt + sig * std::sqrt(dt) * z1;
                        sigma_sabr = sig * std::exp(-0.5f * nu * nu * dt + nu * std::sqrt(dt) * z2);
                        spot = std::exp(std::min(std::max(logS, std::log(1.0f)), std::log(1.0e4f)));
                        logS = std::log(spot);
                        atm_iv = sigma_sabr;
                        const float fwd = spot * std::exp((rate - div_q) * 0.0f);
                        for (int s = 0; s < S; ++s) {
                            for (int m = 0; m < M; ++m) {
                                const float Ks = grid_strikes[static_cast<std::size_t>(s)];
                                const float tau = grid_mats[static_cast<std::size_t>(m)];
                                float iv = sabr_hagan_iv_beta1(
                                    fwd, Ks, tau, sigma_sabr, nu, rho
                                );
                                if (!std::isfinite(iv) || iv < 1e-4f || iv > 5.0f) {
                                    iv = atm_iv;
                                }
                                surf_buf->data[idx5(p, a, t, s, m, K, T, S, M)] = iv;
                            }
                        }
                    } else {
                        // rBergomi fallback: use Dupire local-vol surface on a GBM-like
                        // spot with rough vol proxy. Full rBergomi path remains in
                        // generate_surfaces; here we emit a consistent world bundle.
                        const float H = cfg.hurst_exponent;
                        const float eta = 1.5f;
                        const float rho = -0.75f;
                        const float xi0 = 0.04f;
                        float z = rng.gauss();
                        for (int j = 0; j < K; ++j) Z[static_cast<std::size_t>(j)] = rng.gauss();
                        float zc = 0.0f;
                        for (int j = 0; j <= a; ++j) {
                            zc += cholesky_matrix[a * K + j] * Z[static_cast<std::size_t>(j)];
                        }
                        (void)z;
                        const float tau = dt * static_cast<float>(t + 1);
                        const float wh = zc * std::pow(tau, H - 0.5f) * std::sqrt(dt);
                        float logV = std::log(xi0) + eta * wh - 0.5f * eta * eta * std::pow(tau, 2.0f * H);
                        v = std::exp(std::max(logV, -23.0f));
                        const float v_e = std::min(v, 1.0e4f);
                        logS += (rate - 0.5f * v_e) * dt + std::sqrt(v_e * dt) * zc;
                        spot = std::exp(std::min(std::max(logS, std::log(1.0f)), std::log(1.0e4f)));
                        logS = std::log(spot);
                        atm_iv = std::sqrt(std::max(v_e, 1e-8f));
                        std::vector<float> lv(static_cast<std::size_t>(S * M));
                        solve_dupire_local_vol(spot, rate, v_e, S, M, lv.data(), H, eta, rho);
                        for (int s = 0; s < S; ++s) {
                            for (int m = 0; m < M; ++m) {
                                float x = lv[static_cast<std::size_t>(s * M + m)];
                                if (!std::isfinite(x) || x < 1e-4f) x = 1e-4f;
                                surf_buf->data[idx5(p, a, t, s, m, K, T, S, M)] = x;
                            }
                        }
                    }

                    spot_buf->data[idx3(p, a, t, K, T)] = spot;
                    iv_buf->data[idx3(p, a, t, K, T)] = atm_iv;
                }
            }
        }
    }

    auto [sch_s, arr_s] = make_arrow_pair(surf_buf, total_surf, "surface");
    auto [sch_p, arr_p] = make_arrow_pair(spot_buf, total_path, "spot_path");
    auto [sch_i, arr_i] = make_arrow_pair(iv_buf, total_path, "atm_iv_path");
    return {sch_s, arr_s, sch_p, arr_p, sch_i, arr_i};
}
