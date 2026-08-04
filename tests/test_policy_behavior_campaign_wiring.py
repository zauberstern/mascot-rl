"""Wave 2: campaign must feed sleeves/macros into policy_behavior; full v2 persist."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.reporting.policy_behavior import (
    ARCHETYPE_IDS,
    build_policy_behavior,
    extract_crucible_behaviour_inputs,
    load_behaviour_macro_context,
    pack_policy_behavior_campaign_record,
)


def test_build_policy_behavior_non_nan_tilts_and_finite_macro_coefs():
    """Synthetic sleeve matrix + macro series → non-NaN tilts, finite macro coefs."""
    rng = np.random.default_rng(0)
    T, K = 80, 7
    # One name per sleeve (identity sleeve matrix).
    S = np.eye(K, 7)
    W = np.full((T, K), 1.0 / K)
    # Plant a defensive tilt correlated with lagged VIX z.
    vix = rng.standard_normal(T) * 0.5
    hy = rng.standard_normal(T) * 0.1
    term = rng.standard_normal(T) * 0.1
    W[:, 3] = np.clip(0.25 + 0.15 * np.r_[0.0, vix[:-1]], 0.05, 0.55)
    rem = 1.0 - W[:, 3]
    W[:, [0, 1, 2, 4, 5, 6]] = (rem / 6.0)[:, None]
    R = rng.normal(0.0, 0.01, size=(T, K))
    regimes = np.array(
        ["calm"] * 30 + ["inflationary"] * 25 + ["crisis"] * 25, dtype=object
    )

    payload = build_policy_behavior(
        algo="ppo",
        architecture="mlp",
        objective="differential_sharpe",
        train_world="historical",
        policy_mode="balanced",
        universe_fingerprint="abc123",
        weights=W,
        asset_returns=R,
        sleeve_matrix=S,
        regimes=regimes,
        vix_z=vix,
        hy_oas_z=hy,
        term_spread=term,
        n_null_shuffles=10,
    )

    tilts = payload["sleeve_tilt_series"]
    assert set(tilts) >= {
        "trend",
        "reversal",
        "carry",
        "defensive",
        "lottery",
        "illiquid",
        "core",
    }
    for sid, series in tilts.items():
        assert len(series) == T
        assert np.all(np.isfinite(series)), f"NaN/Inf tilt in {sid}"

    macro = payload["macro_tilt_sensitivity"]
    assert macro, "macro_tilt_sensitivity empty"
    for sleeve, regs in macro.items():
        for name, stats in regs.items():
            assert np.isfinite(stats["coef"]), f"{sleeve}.{name}.coef not finite"
            assert np.isfinite(stats["se"]), f"{sleeve}.{name}.se not finite"


def test_extract_crucible_behaviour_inputs_from_results_and_cfg():
    S = np.eye(4, 7).tolist()
    membership = {"trend": [0], "core": [1, 2, 3]}
    results = {
        "crucible": {
            "sleeve_matrix": S,
            "sleeve_membership": membership,
            "fingerprint": "fp_from_results",
            "sleeve_primary": {"0": "trend"},
        }
    }
    out = extract_crucible_behaviour_inputs(results=results, cfg={})
    assert out["crucible_block_found"] is True
    assert out["universe_fingerprint"] == "fp_from_results"
    assert out["sleeve_membership"] == membership
    assert np.asarray(out["sleeve_matrix"]).shape == (4, 7)

    cfg_only = {
        "_crucible_result": {
            "sleeve_matrix": S,
            "fingerprint": "fp_cfg",
        },
        "_crucible_universe_fingerprint": "fp_cfg_alt",
    }
    out2 = extract_crucible_behaviour_inputs(results={}, cfg=cfg_only)
    assert out2["universe_fingerprint"] == "fp_cfg"
    assert out2["sleeve_matrix"] is not None


def test_pack_policy_behavior_campaign_record_keeps_full_v2():
    w = np.full((12, 4), 0.25)
    s = np.eye(4, 7)
    behavior = build_policy_behavior(
        algo="ppo",
        weights=w,
        sleeve_matrix=s,
        policy_mode="balanced",
        universe_fingerprint="deadbeef",
        n_null_shuffles=5,
    )
    packed = pack_policy_behavior_campaign_record(
        behavior,
        path="/tmp/policy_behavior.json",
        figures=["/tmp/a.png"],
        macro_status={"status": "ok", "reason": ""},
    )
    assert packed["path"] == "/tmp/policy_behavior.json"
    assert packed["figures"] == ["/tmp/a.png"]
    assert packed["archetype_primary"] in set(ARCHETYPE_IDS) | {"mixed"}
    assert packed["archetype_scores"]
    assert set(packed["archetype_scores"]) == set(ARCHETYPE_IDS)
    assert packed["schema_version"] == 2
    assert "behaviour" in packed
    assert "sleeve_tilt_series" in packed
    assert (packed.get("extras") or {}).get("macro_context_status", {}).get(
        "status"
    ) == "ok"


def test_load_behaviour_macro_context_missing_lake_stamps_status(tmp_path: Path):
    dates = ["2018-01-02", "2018-01-03", "2018-01-04"]
    # Use a nonexistent subdir so the repo-local fioracle fallback cannot rescue.
    ctx = load_behaviour_macro_context(
        dates,
        lake_root=tmp_path / "no_such_lake",
        lake_subdir="macro/fioracle_missing_wave2_test",
    )
    assert ctx["status"]["status"] == "missing_lake"
    assert ctx["regimes"] is None
    assert ctx["vix_z"] is None
    assert ctx["hy_oas_z"] is None
    assert ctx["term_spread"] is None
    assert ctx["status"]["reason"]


def test_load_behaviour_macro_context_from_real_lake_when_present():
    lake = Path(__file__).resolve().parents[1] / "lake"
    fioracle = lake / "macro" / "fioracle"
    if not (fioracle / "vix.parquet").is_file():
        pytest.skip("fioracle lake not present")
    dates = pd_bdates("2018-01-02", periods=40)
    ctx = load_behaviour_macro_context(
        dates, lake_root=lake, lake_subdir="macro/fioracle", min_history_days=100
    )
    assert ctx["status"]["status"] == "ok"
    assert ctx["regimes"] is not None
    assert len(ctx["regimes"]) == len(dates)
    assert ctx["vix_z"] is not None
    assert np.isfinite(ctx["vix_z"]).sum() > 0
    assert ctx["hy_oas_z"] is not None
    assert ctx["term_spread"] is not None


def pd_bdates(start: str, periods: int):
    import pandas as pd

    return list(pd.bdate_range(start, periods=periods))
