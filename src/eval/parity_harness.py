"""Estimand parity harness: identical economic object for policy and peers.

Every strategy (RL policy, industry benchmark, OLPS) is scored by stepping
``HistoricalArmEnv`` so friction, borrow, RF, and FF4 residualization are
byte-identical by construction rather than by convention.

Dual scorecard (user decision 2026-08-07):
- ``total_net`` = gross - cost - borrow - rf   (headline)
- ``residual``  = total_net - beta · factors   (co-primary)

Every scored series carries an ``estimand_hash`` (and ``estimand_hash_residual``)
that binds friction, cadence, universe, residualizer, rebalance mask, and the
scorecard column itself, so two series with the same hash are guaranteed to be
comparable and two series compared by a statistic (SPA, Romano-Wolf, ...) must
carry the same scorecard label (see :func:`assert_same_scorecard`).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.arms import ArmSpec, default_arm_spec
from src.env.historical_env import HistoricalArmEnv
from src.eval.friction import FrictionSpec
from src.eval.residualization import ResidualizerState, fit_ff4_residualizer, freeze_residualizer

WeightFn = Callable[..., np.ndarray]

ESTIMAND_VERSION = "parity_v2"
ESTIMAND_FIELDS = (
    "version",
    "friction_spec_id",
    "equity_bps",
    "impact_c_eq",
    "borrow_floor_bps_annual",
    "cost_multiplier",
    "residualize",
    "cadence",
    "scorecard",
    "universe_fingerprint",
    "residualizer_fold_id",
    "residualizer_model",
    "mask_fingerprint",
    "reward",
    "feature_channels_fingerprint",
)


def _fingerprint(items: Sequence[Any] | None) -> str:
    """Order-independent short fingerprint of an iterable of hashables."""
    if not items:
        return ""
    blob = ",".join(sorted(str(x) for x in items)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _mask_fingerprint(mask: np.ndarray | None) -> str:
    if mask is None:
        return ""
    m = np.asarray(mask, dtype=bool)
    return hashlib.sha256(m.tobytes()).hexdigest()[:16]


def estimand_hash(
    *,
    friction: FrictionSpec,
    cadence: str,
    residualize: bool = True,
    scorecard: str = "total_net",
    universe: Sequence[Any] | None = None,
    residualizer: ResidualizerState | None = None,
    rebalance_mask: np.ndarray | None = None,
    reward: str | None = None,
    feature_channels: Sequence[str] | None = None,
) -> str:
    """Stable SHA-256 of the locked estimand contract.

    ``cadence`` is mandatory: callers must resolve an explicit cadence label
    rather than let it be inferred from mask density (see A7 in the surface
    alpha rebuild dossier). ``scorecard`` distinguishes ``total_net`` from
    ``residual`` so the two columns are never silently comparable.
    """
    if not cadence:
        raise ValueError("estimand_hash requires an explicit non-empty cadence")
    # The residualizer only enters the "residual" scorecard's computation
    # (total_net = gross - cost - borrow - rf never touches factor betas), so
    # binding its identity into the total_net hash would make an otherwise
    # identical total_net series across a policy and a benchmark panel
    # non-uniform purely because they happen to freeze differently labeled
    # residualizer fits. Scope residualizer identity to scorecard=="residual".
    resid_fold_id = ""
    resid_model = ""
    if str(scorecard) == "residual" and residualizer is not None:
        resid_fold_id = str(residualizer.fold_id)
        resid_model = str(residualizer.model)
    payload = {
        "version": ESTIMAND_VERSION,
        "friction_spec_id": str(friction.spec_id),
        "equity_bps": float(friction.equity_bps),
        "impact_c_eq": float(friction.impact_c_eq),
        "borrow_floor_bps_annual": float(friction.borrow_floor_bps_annual),
        "cost_multiplier": float(friction.cost_multiplier),
        "residualize": bool(residualize),
        "cadence": str(cadence),
        "scorecard": str(scorecard),
        "universe_fingerprint": _fingerprint(universe),
        "residualizer_fold_id": resid_fold_id,
        "residualizer_model": resid_model,
        "mask_fingerprint": _mask_fingerprint(rebalance_mask),
        "reward": str(reward or ""),
        "feature_channels_fingerprint": _fingerprint(feature_channels),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def assert_estimand_hash(
    got: str,
    *,
    friction: FrictionSpec,
    cadence: str,
    residualize: bool = True,
    scorecard: str = "total_net",
    universe: Sequence[Any] | None = None,
    residualizer: ResidualizerState | None = None,
    rebalance_mask: np.ndarray | None = None,
    reward: str | None = None,
    feature_channels: Sequence[str] | None = None,
) -> None:
    expected = estimand_hash(
        friction=friction,
        cadence=cadence,
        residualize=residualize,
        scorecard=scorecard,
        universe=universe,
        residualizer=residualizer,
        rebalance_mask=rebalance_mask,
        reward=reward,
        feature_channels=feature_channels,
    )
    if str(got) != expected:
        raise AssertionError(
            f"estimand_hash mismatch: got={got!r} expected={expected!r}"
        )


def require_uniform_estimand_hashes(
    entries: Mapping[str, Mapping[str, Any]], *, field: str = "estimand_hash"
) -> str:
    """Fail closed if scored strategies carry different estimand hashes.

    Returns the common hash. Used before writing ``stats_table.json``.
    """
    hashes: dict[str, str] = {}
    for name, row in entries.items():
        h = row.get(field)
        if not h:
            raise AssertionError(f"{field} missing for {name!r}")
        hashes[str(name)] = str(h)
    uniq = set(hashes.values())
    if len(uniq) != 1:
        raise AssertionError(
            f"{field} mismatch across strategies: {hashes}"
        )
    return next(iter(uniq))


def assert_same_scorecard(policy_scorecard: str, peer_scorecard: str) -> None:
    """Fail closed when a statistic would compare mismatched economic objects.

    ``total_net`` (gross - cost - borrow - rf) and ``residual`` (total_net minus
    factor exposure) are different quantities; SPA / Romano-Wolf / DSR must
    never mix them across the policy and its challengers.
    """
    if str(policy_scorecard) != str(peer_scorecard):
        raise AssertionError(
            f"scorecard mismatch: policy={policy_scorecard!r} peer={peer_scorecard!r}; "
            "statistics must compare identical economic objects (total_net vs total_net, "
            "or residual vs residual)"
        )


def _l1_normalize(w: np.ndarray) -> np.ndarray:
    """L1-normalize to a signed simplex; preserve an intentional zero vector.

    A strategy that returns exactly zero weights (``no_trade``, or a momentum
    sleeve with no signal yet) means "hold no position", not "equal weight".
    Silently substituting equal weight here previously destroyed that
    semantic and inflated warm-up-period Sharpes (A4 in the rebuild dossier).
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    denom = float(np.sum(np.abs(w)))
    if denom <= 1e-12:
        return np.zeros(w.size, dtype=np.float64)
    return w / denom


def score_strategy(
    weight_fn: WeightFn,
    returns: np.ndarray,
    *,
    factors: np.ndarray,
    arm: ArmSpec | None = None,
    friction: FrictionSpec | None = None,
    residualizer: ResidualizerState | None = None,
    rebalance_mask: np.ndarray | None = None,
    mktcap: np.ndarray | None = None,
    project_fn: Callable[..., np.ndarray] | None = None,
    cadence: str | None = None,
    universe: Sequence[Any] | None = None,
    reward: str | None = None,
    feature_channels: Sequence[str] | None = None,
    slot_valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Walk ``weight_fn`` through ``HistoricalArmEnv``; return dual PnL series.

    Parameters
    ----------
    weight_fn
        ``(returns_hist, *, t, w_prev, mktcap=...) -> (K,)`` weights.
        Called with history ``returns[:t]`` (no look-ahead).
    cadence
        Explicit rebalance-cadence label (``"daily"``, ``"weekly"``,
        ``"monthly"``, ...). Defaults to ``"daily"`` only when no
        ``rebalance_mask`` is supplied; when a mask is supplied the caller
        must state the cadence explicitly (no density-based inference).

    Returned ``total_net`` / ``residual`` / ``turnover`` / ``cost`` / ``gross``
    / ``weights`` arrays cover only the steps the environment actually took
    (``env`` never scores day 0, the pre-trade day, or the terminal day); their
    length is given by ``t_index.size`` and ``t_index`` gives the absolute
    row index in ``returns`` each entry corresponds to. Pre-allocating a
    ``T``-length array and leaving index 0 at its zero-initialized value
    silently injected a phantom zero-return day into every Sharpe (A3).
    """
    rets = np.asarray(returns, dtype=np.float64)
    fac = np.asarray(factors, dtype=np.float64)
    if rets.ndim != 2:
        raise ValueError("returns must be (T, K)")
    if fac.ndim != 2 or fac.shape[0] != rets.shape[0]:
        raise ValueError("factors must be (T, F) aligned with returns")
    t_len, k = rets.shape
    if arm is None:
        arm = default_arm_spec(k)
        # Force eq arm for equity panel scoring.
        arm = ArmSpec(
            id="eq",
            option_slots=0,
            equity_slots=k,
            delta_mode="off",
        )
    if int(arm.n_slots) != k:
        raise ValueError(f"arm.n_slots={arm.n_slots} != K={k}")
    if friction is None:
        friction = FrictionSpec()
    if residualizer is None:
        y = np.nanmean(rets, axis=1)
        residualizer = freeze_residualizer(
            fit_ff4_residualizer(y, fac, fold_id="parity_harness"),
            "parity_harness",
        )

    if cadence is None:
        if rebalance_mask is None:
            cadence = "daily"
        else:
            raise ValueError(
                "score_strategy: rebalance_mask supplied but cadence is None; "
                "cadence must be stated explicitly (no density inference)"
            )

    env = HistoricalArmEnv(
        returns=rets,
        factors=fac,
        arm=arm,
        friction=friction,
        residualizer=residualizer,
        project_fn=project_fn,
        rebalance_mask=rebalance_mask,
        slot_valid_mask=slot_valid_mask,
    )
    obs, _ = env.reset()
    del obs

    t_index_list: list[int] = []
    total_net_list: list[float] = []
    residual_list: list[float] = []
    turnover_list: list[float] = []
    cost_list: list[float] = []
    gross_list: list[float] = []
    weights_list: list[np.ndarray] = []

    terminated = truncated = False
    while not (terminated or truncated):
        t = int(env.t)
        hist = rets[:t]
        w_prev = env.w.copy()
        kw: dict[str, Any] = {"t": t, "w_prev": w_prev}
        if mktcap is not None:
            kw["mktcap"] = mktcap
        w = np.asarray(weight_fn(hist, **kw), dtype=np.float64).reshape(-1)
        if w.size != k:
            raise ValueError(f"weight_fn returned size {w.size} != K={k}")
        w = _l1_normalize(w)
        _obs, _reward, terminated, truncated, info = env.step(w)
        # total_net = gross - cost - borrow - rf (factor NOT removed)
        g = float(info.get("gross", 0.0))
        c = float(info.get("cost", 0.0))
        b = float(info.get("borrow", 0.0))
        rf = float(info.get("rf", 0.0))
        tot = g - c - b - rf
        t_index_list.append(t)
        total_net_list.append(tot)
        residual_list.append(float(info.get("residual", tot)))
        turnover_list.append(float(info.get("turnover", 0.0)))
        cost_list.append(c + b)
        gross_list.append(g)
        weights_list.append(np.asarray(info.get("post_fill_w", w), dtype=np.float64))

    t_index = np.asarray(t_index_list, dtype=np.int64)
    total_net = np.asarray(total_net_list, dtype=np.float64)
    residual = np.asarray(residual_list, dtype=np.float64)
    turnover = np.asarray(turnover_list, dtype=np.float64)
    cost = np.asarray(cost_list, dtype=np.float64)
    gross = np.asarray(gross_list, dtype=np.float64)
    weights = (
        np.stack(weights_list, axis=0)
        if weights_list
        else np.zeros((0, k), dtype=np.float64)
    )

    common_kw = dict(
        friction=friction,
        cadence=str(cadence),
        residualize=True,
        universe=universe,
        residualizer=residualizer,
        rebalance_mask=rebalance_mask,
        reward=reward,
        feature_channels=feature_channels,
    )
    ehash_total_net = estimand_hash(scorecard="total_net", **common_kw)
    ehash_residual = estimand_hash(scorecard="residual", **common_kw)
    return {
        "total_net": total_net,
        "residual": residual,
        "turnover": turnover,
        "cost": cost,
        "gross": gross,
        "weights": weights,
        "t_index": t_index,
        "scorecard": "total_net",
        "estimand_hash": ehash_total_net,
        "estimand_hash_residual": ehash_residual,
        "cadence": str(cadence),
        "friction_spec_id": str(friction.spec_id),
    }


def score_equal_weight(
    returns: np.ndarray,
    *,
    factors: np.ndarray,
    arm: ArmSpec | None = None,
    friction: FrictionSpec | None = None,
    residualizer: ResidualizerState | None = None,
    rebalance_mask: np.ndarray | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Equal-weight peer scored through the same harness as the policy."""
    rets = np.asarray(returns, dtype=np.float64)
    k = int(rets.shape[1])

    def _ew(returns_hist: np.ndarray, *, t: int, w_prev: np.ndarray, **_kw: Any) -> np.ndarray:
        del returns_hist, t, w_prev
        return np.full(k, 1.0 / max(k, 1), dtype=np.float64)

    return score_strategy(
        _ew,
        rets,
        factors=factors,
        arm=arm,
        friction=friction,
        residualizer=residualizer,
        rebalance_mask=rebalance_mask,
        **kwargs,
    )


def score_benchmark_panel(
    names: list[str] | tuple[str, ...],
    returns: np.ndarray,
    *,
    factors: np.ndarray,
    arm: ArmSpec | None = None,
    friction: FrictionSpec | None = None,
    residualizer: ResidualizerState | None = None,
    rebalance_mask: np.ndarray | None = None,
    mktcap: np.ndarray | None = None,
    cadence: str | None = None,
    universe: Sequence[Any] | None = None,
    turnover_cap: float | None = None,
    slot_valid_mask: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Score every named benchmark through the parity harness."""
    from src.eval.benchmark_panel import get_weight_fn
    from src.eval.research_alpha_train import _turnover_cap_project

    out: dict[str, dict[str, Any]] = {}
    for name in names:
        fn = get_weight_fn(str(name))
        project_fn = None
        if str(name) == "equal_weight_tau_matched":
            tau = float(turnover_cap) if turnover_cap is not None else 0.15

            def project_fn(
                w: np.ndarray,
                *,
                t: int | None = None,
                w_prev: np.ndarray | None = None,
                _tau: float = tau,
                **_kw: Any,
            ) -> np.ndarray:
                return _turnover_cap_project(w, t=t, w_prev=w_prev, tau=_tau)

        out[str(name)] = score_strategy(
            fn,
            returns,
            factors=factors,
            arm=arm,
            friction=friction,
            residualizer=residualizer,
            rebalance_mask=rebalance_mask,
            mktcap=mktcap,
            cadence=cadence,
            universe=universe,
            project_fn=project_fn,
            slot_valid_mask=slot_valid_mask,
        )
    return out
