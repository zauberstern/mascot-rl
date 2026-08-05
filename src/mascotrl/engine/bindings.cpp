#include "rbergomi_engine.hpp"
#include "worlds.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_rbergomi, m) {
    m.doc() = "Layer 1: multi-world surface + path generators (rBergomi/GBM/Heston/GARCH/SABR)";

    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("n_paths", &EngineConfig::n_paths)
        .def_readwrite("n_assets", &EngineConfig::n_assets)
        .def_readwrite("n_steps", &EngineConfig::n_steps)
        .def_readwrite("n_strikes", &EngineConfig::n_strikes)
        .def_readwrite("n_maturities", &EngineConfig::n_maturities)
        .def_readwrite("hurst_exponent", &EngineConfig::hurst_exponent)
        .def_readwrite("seed", &EngineConfig::seed);

    py::class_<WorldConfig>(m, "WorldConfig")
        .def(py::init<>())
        .def_readwrite("base", &WorldConfig::base)
        .def_readwrite("world", &WorldConfig::world)
        .def_readwrite("rate", &WorldConfig::rate)
        .def_readwrite("div_q", &WorldConfig::div_q)
        .def_readwrite("spot0", &WorldConfig::spot0)
        .def_readwrite("gbm_mu", &WorldConfig::gbm_mu)
        .def_readwrite("gbm_sigma", &WorldConfig::gbm_sigma)
        .def_readwrite("heston_v0", &WorldConfig::heston_v0)
        .def_readwrite("heston_theta", &WorldConfig::heston_theta)
        .def_readwrite("heston_kappa", &WorldConfig::heston_kappa)
        .def_readwrite("heston_xi", &WorldConfig::heston_xi)
        .def_readwrite("heston_rho", &WorldConfig::heston_rho)
        .def_readwrite("heston_scheme", &WorldConfig::heston_scheme)
        .def_readwrite("garch_mu", &WorldConfig::garch_mu)
        .def_readwrite("garch_omega", &WorldConfig::garch_omega)
        .def_readwrite("garch_alpha", &WorldConfig::garch_alpha)
        .def_readwrite("garch_beta", &WorldConfig::garch_beta)
        .def_readwrite("garch_gamma", &WorldConfig::garch_gamma)
        .def_readwrite("garch_lambda", &WorldConfig::garch_lambda)
        .def_readwrite("garch_n_inner", &WorldConfig::garch_n_inner)
        .def_readwrite("sabr_sigma0", &WorldConfig::sabr_sigma0)
        .def_readwrite("sabr_nu", &WorldConfig::sabr_nu)
        .def_readwrite("sabr_rho", &WorldConfig::sabr_rho);

    m.def(
        "generate_surfaces",
        [](const EngineConfig& config, py::array_t<float, py::array::c_style | py::array::forcecast> chol) {
            if (chol.ndim() != 2) {
                throw std::invalid_argument("cholesky_matrix must be 2D");
            }
            if (chol.shape(0) != config.n_assets || chol.shape(1) != config.n_assets) {
                throw std::invalid_argument("cholesky_matrix must be (n_assets, n_assets)");
            }
            return generate_surfaces(config, chol.data());
        },
        py::arg("config"),
        py::arg("cholesky_matrix"),
        "Generate 5D local-vol surfaces; returns (arrow_schema, arrow_array) capsules");

    m.def(
        "generate_world",
        [](const WorldConfig& config, py::array_t<float, py::array::c_style | py::array::forcecast> chol) {
            if (chol.ndim() != 2) {
                throw std::invalid_argument("cholesky_matrix must be 2D");
            }
            const int n = config.base.n_assets;
            if (chol.shape(0) != n || chol.shape(1) != n) {
                throw std::invalid_argument("cholesky_matrix must be (n_assets, n_assets)");
            }
            return generate_world(config, chol.data());
        },
        py::arg("config"),
        py::arg("cholesky_matrix"),
        "Generate world surfaces+spots+atm_iv; returns six Arrow capsules");
}
