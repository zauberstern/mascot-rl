"""Tests for scripts/assemble_regime_desk.py (Scenario A rescue Fix5)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.assemble_regime_desk import (
    assemble_regime_desk,
    select_experts,
    write_regime_desk,
)


def _write_cell(
    cell_dir: Path,
    *,
    stem: str,
    archetype: str,
    head: str,
    confidence: float,
    returns: np.ndarray,
    dates: list[str],
    panel: np.ndarray,
    designed: str | None = None,
    mandate_in_stem: bool = False,
) -> None:
    art = {
        "dates": dates,
        "panel_returns": panel.tolist(),
        "weights": np.ones((len(dates), panel.shape[1])).tolist(),
        "runner_artifact": {
            "paths": {"0": {"pnl": returns.tolist()}},
            "panel_returns": panel.tolist(),
        },
    }
    (cell_dir / f"{stem}.json").write_text(json.dumps(art), encoding="utf-8")
    beh = {
        "archetype_primary": archetype,
        "archetype_confidence": confidence,
        "weight_head": head,
        "l1_vs_ew_mean": 1.5 if head == "sparse_tilt" else 0.05,
        "sharpe": 0.8 if head == "softmax" else 0.2,
    }
    if designed:
        beh["designed_personality"] = designed
    (cell_dir / f"{stem}_policy_behavior.json").write_text(
        json.dumps(beh), encoding="utf-8"
    )
    if mandate_in_stem:
        assert "archetype_" in stem


def test_select_experts_picks_highest_confidence_sparse(tmp_path: Path) -> None:
    t, n = 80, 4
    dates = [f"2020-01-{i+1:02d}" for i in range(t)]
    # Use ISO-ish dates that sort: build properly
    import datetime as dt

    start = dt.date(2018, 1, 2)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(t)]
    panel = np.random.default_rng(0).normal(0, 0.01, size=(t, n))
    rng = np.random.default_rng(1)

    cells = [
        ("eq_a_sparse_tilt_cvar", "risk_manager", "sparse_tilt", 0.4),
        ("eq_b_sparse_tilt_cvar", "risk_manager", "sparse_tilt", 0.9),
        ("eq_c_sparse_tilt_mtm", "contrarian", "sparse_tilt", 0.7),
        ("eq_d_softmax_mean", "mixed", "softmax", 0.5),
    ]
    cands = []
    for stem, arch, head, conf in cells:
        rets = rng.normal(0.0002, 0.01, size=t)
        _write_cell(
            tmp_path,
            stem=stem,
            archetype=arch,
            head=head,
            confidence=conf,
            returns=rets,
            dates=dates,
            panel=panel,
        )
        from scripts.assemble_regime_desk import load_candidate_cells

        # load after all writes
    cands = __import__(
        "scripts.assemble_regime_desk", fromlist=["load_candidate_cells"]
    ).load_candidate_cells(tmp_path)
    selected = select_experts(cands)
    assert selected["risk_manager"]["stem"] == "eq_b_sparse_tilt_cvar"
    assert selected["contrarian"]["stem"] == "eq_c_sparse_tilt_mtm"
    assert "owl" in selected
    assert selected["owl"]["head"] == "softmax"


def test_assemble_mixers_and_primary_with_dominant_expert(tmp_path: Path) -> None:
    """Synthetic cells: mixers present; dominant expert makes FTL/EG > FS."""
    import datetime as dt

    t, n = 300, 6
    start = dt.date(2017, 1, 3)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(t)]
    rng = np.random.default_rng(11)
    panel = rng.normal(0, 0.01, size=(t, n))
    specs = [
        ("eq_a_sparse_tilt_cvar", "risk_manager", 0.8),
        ("eq_b_sparse_tilt_mtm", "contrarian", 0.75),
        ("eq_c_sparse_tilt_smse", "trend_follower", 0.7),
        ("eq_d_sparse_tilt_mik", "speculator", 0.65),
        ("eq_e_softmax_mean", "mixed", 0.5),
    ]
    for stem, arch, conf in specs:
        head = "sparse_tilt" if "sparse_tilt" in stem else "softmax"
        base = rng.normal(0.0002, 0.01, size=t)
        if arch == "mixed":
            base = base + 0.002  # Owl-like dominant so EG/FTL concentrate
        _write_cell(
            tmp_path,
            stem=stem,
            archetype=arch,
            head=head,
            confidence=conf,
            returns=base,
            dates=dates,
            panel=panel,
        )
    payload = assemble_regime_desk(
        cell_dir=tmp_path,
        k_switches=3,
        turb_window=40,
        prefer_seal_timeline=False,
        sleeping_experts=True,
    )
    assert "mixers" in payload
    assert "fixed_share" in payload["mixers"]
    assert "eg_experts" in payload["mixers"]
    for key in (
        "follow_the_leader",
        "hold_leader_annual",
        "hold_leader_quarter",
        "owl_hysteresis",
        "page_hinkley",
        "performance_sleeping",
        "rolling_leader_126",
    ):
        assert key in payload["mixers"]
    assert isinstance(payload["primary_mixer"], str)
    assert payload["honesty"]["loss_baseline"] == "-R"
    assert payload["honesty"]["alpha_turbulence_gated"] is False
    assert payload["honesty"]["wake_lag"] == 1
    assert payload["honesty"]["switcher_family"] == "piecewise_constant_plus_mixers"
    fs_s = float(payload["mixers"]["fixed_share"]["sharpe"])
    eg_s = float(payload["mixers"]["eg_experts"]["sharpe"])
    assert eg_s > fs_s
    assert float(payload["mixers"]["follow_the_leader"]["mean_max_weight"]) > float(
        payload["mixers"]["fixed_share"]["mean_max_weight"]
    )
    assert float(payload["mixers"]["hold_leader_quarter"]["mean_max_weight"]) > 0.45
    assert "primary_wealth" in payload
    assert len(payload["primary_dominant_expert"]) == len(payload["dates"])


def test_assemble_writes_required_keys_and_fs_sanity(tmp_path: Path) -> None:
    import datetime as dt

    t, n = 400, 8  # enough for turb window after asset truncate
    start = dt.date(2015, 1, 2)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(t)]
    # Skip weekends roughly by using business-day-ish spacing already as consecutive labels
    rng = np.random.default_rng(42)
    panel = rng.normal(0, 0.01, size=(t, n))

    specs = [
        ("eq_K100_single_ppo_mlp_sparse_tilt_cvar_ru", "risk_manager", 0.8),
        ("eq_K100_single_ppo_mlp_sparse_tilt_mtm_pnl", "contrarian", 0.75),
        ("eq_K100_single_ppo_mlp_sparse_tilt_differential_sharpe", "trend_follower", 0.7),
        ("eq_K100_single_ppo_mlp_sparse_tilt_mikkila_asym", "speculator", 0.65),
        ("eq_K100_single_ppo_mlp_softmax_mean_std_cao", "mixed", 0.5),
    ]
    for stem, arch, conf in specs:
        head = "sparse_tilt" if "sparse_tilt" in stem else "softmax"
        # Give tortoise (risk_manager) an edge in high-vol days so FS can track.
        base = rng.normal(0.0001, 0.01, size=t)
        if arch == "risk_manager":
            base = base + 0.0005
        if arch == "mixed":
            base = rng.normal(0.00015, 0.008, size=t)
        _write_cell(
            tmp_path,
            stem=stem,
            archetype=arch,
            head=head,
            confidence=conf,
            returns=base,
            dates=dates,
            panel=panel,
        )

    payload = assemble_regime_desk(
        cell_dir=tmp_path, k_switches=5, turb_window=50, prefer_seal_timeline=False
    )
    required = {
        "dates",
        "turbulence",
        "threshold",
        "turbulent",
        "wealth",
        "fixed_share",
        "equal_weight",
        "event_marks",
        "expert_names",
        "expert_returns",
        "oracle_best_k_shift",
        "regret_gap",
        "regret_bound",
        "alpha",
        "k_switches",
        "eta",
        "synthetic",
    }
    assert required.issubset(payload.keys())
    assert payload["synthetic"] is False
    assert payload["n_experts_active"] >= 2
    for m in ("cheetah", "fox", "tortoise", "magpie", "hummingbird", "owl"):
        assert m in payload["wealth"]
        assert len(payload["wealth"][m]) == len(payload["dates"])
    assert len(payload["fixed_share"]) == len(payload["dates"])
    # Sanity: Fixed-Share Sharpe should be finite (not a capital claim).
    fs_s = payload["diagnostics"]["fixed_share_sharpe"]
    ew_s = payload["diagnostics"]["equal_weight_sharpe"]
    assert np.isfinite(fs_s)
    assert np.isfinite(ew_s)
    assert payload["honesty"]["alpha_turbulence_gated"] is False
    assert "fixed_share_weights" in payload
    assert "dominant_expert" in payload
    assert len(payload["dominant_expert"]) == len(payload["dates"])
    W = np.asarray(payload["fixed_share_weights"], dtype=np.float64)
    assert W.shape[0] == len(payload["dates"])
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-8)
    assert set(payload["dominant_expert"]).issubset(set(payload["expert_names"]))
    assert payload["diagnostics"]["alpha_grid_preregistered"] is True
    assert "by_regime" in payload["diagnostics"]
    assert "table" in payload["diagnostics"]
    assert payload["honesty"]["switcher"] == "fixed_share_herbster_warmuth"

    out = write_regime_desk(payload, tmp_path / "regime_desk_series.json")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["synthetic"] is False
    assert "regret_gap" in reloaded


def test_roster_lock_roundtrip_and_require(tmp_path: Path) -> None:
    import datetime as dt

    from scripts.assemble_regime_desk import (
        build_roster_lock,
        write_roster_lock,
    )

    t, n = 200, 6
    start = dt.date(2016, 1, 4)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(t)]
    rng = np.random.default_rng(3)
    panel = rng.normal(0, 0.01, size=(t, n))
    specs = [
        ("eq_a_sparse_tilt_cvar", "risk_manager", 0.8),
        ("eq_b_sparse_tilt_mtm", "contrarian", 0.75),
        ("eq_c_sparse_tilt_smse", "trend_follower", 0.7),
        ("eq_d_sparse_tilt_mik", "speculator", 0.65),
        ("eq_e_softmax_mean", "mixed", 0.5),
    ]
    for stem, arch, conf in specs:
        head = "sparse_tilt" if "sparse_tilt" in stem else "softmax"
        _write_cell(
            tmp_path,
            stem=stem,
            archetype=arch,
            head=head,
            confidence=conf,
            returns=rng.normal(0.0002, 0.01, size=t),
            dates=dates,
            panel=panel,
        )
    payload = assemble_regime_desk(
        cell_dir=tmp_path, k_switches=3, turb_window=40, prefer_seal_timeline=False
    )
    lock = build_roster_lock(payload, cell_dir=tmp_path)
    lock_path = tmp_path / "expert_roster_lock.json"
    write_roster_lock(lock, lock_path)
    # Matching lock succeeds
    assemble_regime_desk(
        cell_dir=tmp_path,
        k_switches=3,
        turb_window=40,
        prefer_seal_timeline=False,
        require_roster_lock=lock_path,
    )
    # Tampered lock fails
    lock["seats"]["fox"]["stem"] = "wrong_stem"
    write_roster_lock(lock, lock_path)
    with pytest.raises(RuntimeError, match="roster lock mismatch"):
        assemble_regime_desk(
            cell_dir=tmp_path,
            k_switches=3,
            turb_window=40,
            prefer_seal_timeline=False,
            require_roster_lock=lock_path,
        )


def test_assemble_with_synthetic_seal_timeline(tmp_path: Path) -> None:
    import datetime as dt

    from src.eval.regime_desk_seal import align_sealed_operational_mask

    t, n = 220, 6
    start = dt.date(2018, 1, 2)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(t)]
    rng = np.random.default_rng(9)
    panel = rng.normal(0, 0.01, size=(t, n))
    specs = [
        ("eq_a_sparse_tilt_cvar", "risk_manager", 0.8),
        ("eq_b_sparse_tilt_mtm", "contrarian", 0.75),
        ("eq_c_sparse_tilt_smse", "trend_follower", 0.7),
        ("eq_d_softmax_mean", "mixed", 0.5),
    ]
    for stem, arch, conf in specs:
        head = "sparse_tilt" if "sparse_tilt" in stem else "softmax"
        _write_cell(
            tmp_path,
            stem=stem,
            archetype=arch,
            head=head,
            confidence=conf,
            returns=rng.normal(0.0002, 0.01, size=t),
            dates=dates,
            panel=panel,
        )
    # Write a SCHEMA 3 seal covering the desk calendar
    seal_dir = tmp_path / "sealed" / "usb_kpt10_v3"
    seal_dir.mkdir(parents=True)
    import pandas as pd

    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    turb_mask = np.zeros(t, dtype=bool)
    turb_mask[t // 2 :] = True
    frame = pd.DataFrame(
        {
            "turbulence": np.linspace(1.0, 4.0, t),
            "turbulent": turb_mask,
            "turbulent_q75": turb_mask,
            "hmm_p_highvol": np.linspace(0.2, 0.8, t),
        },
        index=idx,
    )
    frame.to_parquet(seal_dir / "regime_series.parquet")
    (seal_dir / "manifest.json").write_text(
        json.dumps(
            {
                "name": "usb_kpt10_v3",
                "schema_version": 3,
                "hyperparams": {
                    "operational_label": "markov_filtered_p05",
                    "returns_source": "kpt10_gics",
                },
            }
        ),
        encoding="utf-8",
    )
    aligned = align_sealed_operational_mask(seal_dir, dates)
    assert aligned["status"] == "ok"

    payload = assemble_regime_desk(
        cell_dir=tmp_path,
        k_switches=3,
        turb_window=40,
        seal_path=seal_dir,
        prefer_seal_timeline=True,
    )
    assert payload["timeline_source"].startswith("seal:")
    assert payload["operational_label"] == "markov_filtered_p05"
    assert payload["honesty"]["seal_name"] == "usb_kpt10_v3"
    assert payload["turbulent"][-1] is True
    assert payload["turbulent"][0] is False


def test_hummingbird_mandate_proxy_selected(tmp_path: Path) -> None:
    import datetime as dt

    t, n = 120, 4
    start = dt.date(2019, 1, 2)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(t)]
    rng = np.random.default_rng(7)
    panel = rng.normal(0, 0.01, size=(t, n))

    _write_cell(
        tmp_path,
        stem="eq_K100_single_ppo_mlp_sparse_tilt_mean_std_cao_pm-archetype_carry",
        archetype="mixed",
        head="sparse_tilt",
        confidence=0.55,
        returns=rng.normal(0.0002, 0.01, size=t),
        dates=dates,
        panel=panel,
        designed="tactical_rotator",
        mandate_in_stem=True,
    )
    _write_cell(
        tmp_path,
        stem="eq_K100_single_ppo_mlp_sparse_tilt_cvar_ru",
        archetype="risk_manager",
        head="sparse_tilt",
        confidence=0.8,
        returns=rng.normal(0.0002, 0.01, size=t),
        dates=dates,
        panel=panel,
    )
    _write_cell(
        tmp_path,
        stem="eq_K100_single_ppo_mlp_softmax_mean_std_cao",
        archetype="mixed",
        head="softmax",
        confidence=0.4,
        returns=rng.normal(0.00015, 0.008, size=t),
        dates=dates,
        panel=panel,
    )

    from scripts.assemble_regime_desk import load_candidate_cells

    selected = select_experts(load_candidate_cells(tmp_path))
    assert "tactical_rotator" in selected
    assert selected["tactical_rotator"].get("hummingbird_proxy") is True
