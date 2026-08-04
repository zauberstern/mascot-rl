"""Layer 1 Python bridge: Arrow capsules → torch tensor (zero-copy)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import torch

from src.logging_utils import get_logger, log_span, log_tensor

log = get_logger("mascotrl.l1")

try:
    import cpp_rbergomi
except ImportError as exc:  # pragma: no cover
    cpp_rbergomi = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

WORLD_IDS = {
    "rbergomi": 0,
    "gbm": 1,
    "heston": 2,
    "garch": 3,
    "sabr": 4,
}


def _require_engine() -> Any:
    if cpp_rbergomi is None:
        raise ImportError(
            "cpp_rbergomi not built. Run: bash scripts/build_extensions.sh"
        ) from _IMPORT_ERROR
    return cpp_rbergomi


def make_identity_cholesky(n_assets: int) -> torch.Tensor:
    return torch.eye(n_assets, dtype=torch.float32)


def _arrow_to_numpy(schema_capsule: Any, array_capsule: Any) -> np.ndarray:
    if hasattr(pa.Array, "_import_from_c_capsule"):
        arrow_array = pa.Array._import_from_c_capsule(schema_capsule, array_capsule)
    else:
        arrow_array = pa.Array._import_from_c(array_capsule, schema_capsule)
    try:
        return arrow_array.to_numpy(zero_copy_only=True)
    except (pa.ArrowInvalid, ValueError):
        return arrow_array.to_numpy(zero_copy_only=False)


def _fill_world_config(eng: Any, config: dict) -> Any:
    wcfg = eng.WorldConfig()
    base = wcfg.base
    base.n_paths = int(config["n_paths"])
    base.n_assets = int(config["n_assets"])
    base.n_steps = int(config["n_steps"])
    base.n_strikes = int(config["n_strikes"])
    base.n_maturities = int(config["n_maturities"])
    base.hurst_exponent = float(config.get("hurst_exponent", 0.1))
    if hasattr(base, "seed"):
        base.seed = int(config.get("seed", 42))
    wcfg.base = base

    world_name = str(
        config.get("train_world")
        or config.get("train_distribution")
        or config.get("world")
        or "rbergomi"
    ).lower()
    if world_name in ("synthetic", "sim"):
        world_name = "rbergomi"
    if world_name not in WORLD_IDS:
        raise ValueError(f"unknown train_world={world_name!r}; expected one of {sorted(WORLD_IDS)}")
    wcfg.world = int(WORLD_IDS[world_name])
    wcfg.rate = float(config.get("rate", 0.0))
    wcfg.div_q = float(config.get("div_q", config.get("q", 0.0)))
    wcfg.spot0 = float(config.get("spot0", 100.0))

    wcfg.gbm_mu = float(config.get("gbm_mu", 0.05))
    wcfg.gbm_sigma = float(config.get("gbm_sigma", 0.20))
    wcfg.heston_v0 = float(config.get("heston_v0", 0.04))
    wcfg.heston_theta = float(config.get("heston_theta", 0.04))
    wcfg.heston_kappa = float(config.get("heston_kappa", 2.0))
    wcfg.heston_xi = float(config.get("heston_xi", 0.30))
    wcfg.heston_rho = float(config.get("heston_rho", -0.70))
    # 0=full_truncation, 1=qe, 2=qe_martingale (Andersen QE-M; default)
    _scheme_raw = config.get("heston_scheme", "qe_martingale")
    if isinstance(_scheme_raw, (int, float)):
        wcfg.heston_scheme = int(_scheme_raw)
    else:
        _s = str(_scheme_raw).strip().lower().replace("-", "_")
        _scheme_map = {
            "full_truncation": 0,
            "ft": 0,
            "euler": 0,
            "qe": 1,
            "quadratic_exponential": 1,
            "qe_martingale": 2,
            "qe_m": 2,
            "quadratic_exponential_martingale": 2,
        }
        if _s not in _scheme_map:
            raise ValueError(
                f"unknown heston_scheme={_scheme_raw!r}; expected one of "
                f"{sorted(_scheme_map)}"
            )
        wcfg.heston_scheme = int(_scheme_map[_s])
    # Feller is diagnostic only — never force-fit parameters for RL worlds.
    from src.sim.heston_qe import feller_gap

    _fg = feller_gap(
        kappa=wcfg.heston_kappa, theta=wcfg.heston_theta, xi=wcfg.heston_xi
    )
    if _fg < 0.0 and wcfg.heston_scheme == 0:
        log.warning(
            "heston Feller violated (2κθ-ξ²=%.4g) with full_truncation; "
            "prefer heston_scheme=qe_martingale",
            _fg,
        )
    wcfg.garch_mu = float(config.get("garch_mu", 0.0))
    wcfg.garch_omega = float(config.get("garch_omega", 1e-6))
    wcfg.garch_alpha = float(config.get("garch_alpha", 0.02))
    wcfg.garch_beta = float(config.get("garch_beta", 0.90))
    wcfg.garch_gamma = float(config.get("garch_gamma", 0.10))
    wcfg.garch_lambda = float(config.get("garch_lambda", 0.0))
    wcfg.garch_n_inner = int(config.get("garch_n_inner", 4096))
    wcfg.sabr_sigma0 = float(config.get("sabr_sigma0", 0.20))
    wcfg.sabr_nu = float(config.get("sabr_nu", 0.6))
    wcfg.sabr_rho = float(config.get("sabr_rho", -0.4))
    return wcfg


def get_surface_tensor(
    config: dict,
    cholesky_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate 5D local-vol / IV surfaces via C++ multi-world engine.
    Returns torch float32 tensor shaped
    [n_paths, n_assets, n_steps, n_strikes, n_maturities].
    """
    bundle = get_world_bundle(config, cholesky_matrix=cholesky_matrix)
    return bundle["surfaces"]


def get_world_bundle(
    config: dict,
    cholesky_matrix: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | str]:
    """Generate surfaces + spot paths + ATM IV paths for the configured world."""
    eng = _require_engine()
    if cholesky_matrix is None:
        cholesky_matrix = make_identity_cholesky(int(config["n_assets"]))
    chol = cholesky_matrix.detach().cpu().contiguous().numpy().astype(np.float32)

    world_name = str(
        config.get("train_world")
        or config.get("train_distribution")
        or config.get("world")
        or "rbergomi"
    ).lower()
    if world_name in ("synthetic", "sim"):
        world_name = "rbergomi"

    # Legacy path: pure rBergomi without world dispatch (no spot/iv bundles).
    if world_name == "rbergomi" and not bool(config.get("force_world_bundle", False)):
        cfg = eng.EngineConfig()
        cfg.n_paths = int(config["n_paths"])
        cfg.n_assets = int(config["n_assets"])
        cfg.n_steps = int(config["n_steps"])
        cfg.n_strikes = int(config["n_strikes"])
        cfg.n_maturities = int(config["n_maturities"])
        cfg.hurst_exponent = float(config.get("hurst_exponent", 0.1))
        if hasattr(cfg, "seed"):
            cfg.seed = int(config.get("seed", 42))
        with log_span(
            log,
            "L1.generate_surfaces",
            paths=cfg.n_paths,
            assets=cfg.n_assets,
            steps=cfg.n_steps,
            strikes=cfg.n_strikes,
            mats=cfg.n_maturities,
            H=cfg.hurst_exponent,
            seed=getattr(cfg, "seed", None),
            world="rbergomi",
        ) as m:
            schema_capsule, array_capsule = eng.generate_surfaces(cfg, chol)
            np_view = _arrow_to_numpy(schema_capsule, array_capsule)
            shape = (
                cfg.n_paths,
                cfg.n_assets,
                cfg.n_steps,
                cfg.n_strikes,
                cfg.n_maturities,
            )
            arr = np.asarray(np_view, dtype=np.float32).reshape(shape)
            tensor = torch.from_numpy(np.array(arr, copy=True))
            m["numel"] = int(tensor.numel())
            log_tensor(log, "surface", tensor)
            return {
                "surfaces": tensor,
                "spot_paths": None,
                "atm_iv_paths": None,
                "world": "rbergomi",
            }

    wcfg = _fill_world_config(eng, config)
    with log_span(
        log,
        "L1.generate_world",
        paths=int(config["n_paths"]),
        assets=int(config["n_assets"]),
        steps=int(config["n_steps"]),
        world=world_name,
        seed=int(config.get("seed", 42)),
    ) as m:
        sch_s, arr_s, sch_p, arr_p, sch_i, arr_i = eng.generate_world(wcfg, chol)
        surf = np.asarray(_arrow_to_numpy(sch_s, arr_s), dtype=np.float32).reshape(
            (
                int(config["n_paths"]),
                int(config["n_assets"]),
                int(config["n_steps"]),
                int(config["n_strikes"]),
                int(config["n_maturities"]),
            )
        )
        spots = np.asarray(_arrow_to_numpy(sch_p, arr_p), dtype=np.float32).reshape(
            (int(config["n_paths"]), int(config["n_assets"]), int(config["n_steps"]))
        )
        ivs = np.asarray(_arrow_to_numpy(sch_i, arr_i), dtype=np.float32).reshape(
            (int(config["n_paths"]), int(config["n_assets"]), int(config["n_steps"]))
        )
        surfaces = torch.from_numpy(np.array(surf, copy=True))
        spot_paths = torch.from_numpy(np.array(spots, copy=True))
        atm_iv_paths = torch.from_numpy(np.array(ivs, copy=True))
        m["numel"] = int(surfaces.numel())
        log_tensor(log, "surface", surfaces)
        return {
            "surfaces": surfaces,
            "spot_paths": spot_paths,
            "atm_iv_paths": atm_iv_paths,
            "world": world_name,
        }
