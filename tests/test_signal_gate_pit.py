"""Phase F: signal IC gate — PIT selection window + fail-closed allowlist."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tests.conftest import FLOAT_TOL

from mascotrl.eval.signal_gate import (
    assert_allowlist_valid,
    assert_geometry_pack_valid,
    decile_long_short,
    effective_breadth,
    fama_macbeth,
    ff_alpha,
    ic_decay,
    ic_series,
    load_obs_pack,
    load_signal_allowlist,
    orthogonalize_signals,
    run_signal_gate,
    run_signal_gate_v2,
    write_signal_allowlist,
)


def _predictive_panel(
    t: int = 120,
    k: int = 20,
    *,
    noise_scale: float = 0.01,
    signal_strength: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Signal at t predicts cross-section of returns at t+1."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 1.0, size=(t, k))
    noise = rng.normal(0.0, noise_scale, size=(t, k))
    returns = np.zeros((t, k), dtype=np.float64)
    returns[1:] = signal_strength * signal[:-1] + noise[1:]
    returns[0] = noise[0]
    return signal, returns


def test_fama_macbeth_predictive_positive_t():
    signal, returns = _predictive_panel()
    out = fama_macbeth(signal, returns, lags=1)
    assert set(out) >= {"mean_coef", "t_stat", "n_dates"}
    assert out["n_dates"] > 10
    assert out["t_stat"] > 2.0
    assert out["mean_coef"] > 0.0


def test_noise_signal_weak_t_stat():
    rng = np.random.default_rng(1)
    t, k = 120, 20
    signal = rng.normal(size=(t, k))
    returns = rng.normal(0.0, 0.01, size=(t, k))
    out = fama_macbeth(signal, returns, lags=1)
    assert abs(out["t_stat"]) < 2.0


def test_ic_series_and_decay_shapes():
    signal, returns = _predictive_panel(t=60, k=15)
    ics = ic_series(signal, returns)
    assert ics.ndim == 1
    assert ics.size == returns.shape[0] - 1
    assert float(np.nanmean(ics)) > 0.1

    decay = ic_decay(signal, returns, horizons=(1, 3, 6))
    assert set(decay.keys()) == {1, 3, 6}
    assert decay[1] > decay[6] or np.isfinite(decay[1])


def test_decile_long_short_positive_on_predictive():
    signal, returns = _predictive_panel()
    out = decile_long_short(signal, returns, n_deciles=10)
    assert set(out) >= {"mean_return", "sharpe"}
    assert out["mean_return"] > 0.0


def test_orthogonalize_and_effective_breadth():
    rng = np.random.default_rng(2)
    t, k = 40, 10
    a = rng.normal(size=(t, k))
    b = a + 0.1 * rng.normal(size=(t, k))
    c = rng.normal(size=(t, k))
    ortho = orthogonalize_signals({"a": a, "b": b, "c": c})
    assert list(ortho.keys()) == ["a", "b", "c"]
    # Residualized b should be nearly orthogonal to a (flattened).
    corr = np.corrcoef(
        np.nan_to_num(ortho["a"], nan=0.0).ravel(),
        np.nan_to_num(ortho["b"], nan=0.0).ravel(),
    )[0, 1]
    assert abs(corr) < 0.15

    cm = np.eye(3)
    assert effective_breadth(cm) == pytest.approx(3.0)
    ones = np.ones((3, 3))
    assert effective_breadth(ones) == pytest.approx(1.0)


def test_ff_alpha_recovers_known_alpha_with_significant_t():
    rng = np.random.default_rng(7)
    n = 200
    factors = rng.normal(0.0, 0.02, size=(n, 5))
    true_alpha = 0.01
    true_beta = np.array([0.5, -0.2, 0.1, 0.3, 0.0])
    y = true_alpha + factors @ true_beta + rng.normal(0.0, 0.001, size=n)
    out = ff_alpha(y, factors)
    assert set(out) >= {"alpha", "t_stat", "n", "lags"}
    assert out["alpha"] == pytest.approx(true_alpha, abs=0.005)
    assert abs(out["t_stat"]) > 2.0


def test_ff_alpha_zero_alpha_weak_t_stat():
    rng = np.random.default_rng(8)
    n = 150
    factors = rng.normal(0.0, 0.02, size=(n, 5))
    y = factors @ np.array([0.4, 0.1, -0.1, 0.2, 0.0]) + rng.normal(0.0, 0.02, size=n)
    out = ff_alpha(y, factors)
    assert abs(out["t_stat"]) < 2.0


def test_ff_alpha_handles_nan_rows_and_short_series():
    factors = np.zeros((3, 5))
    y = np.array([np.nan, 0.01, 0.02])
    out = ff_alpha(y, factors)
    assert np.isnan(out["t_stat"])


def test_run_signal_gate_admits_predictive_rejects_noise(tmp_path: Path):
    t, k = 100, 16
    good, returns = _predictive_panel(t=t, k=k, seed=3)
    rng = np.random.default_rng(4)
    noise = rng.normal(size=(t, k))
    dates = [f"2008-{(1 + i % 12):02d}-{1 + (i % 28):02d}" for i in range(t)]

    result = run_signal_gate(
        {"good": good, "noise": noise},
        returns,
        dates=dates,
        selection_end="2012-12-31",
        t_min=2.0,
    )
    assert "good" in result["allowlist"]
    assert "noise" not in result["allowlist"]
    assert result["selection_end"] == "2012-12-31"
    assert "effective_breadth" in result
    assert "estimand" in result
    assert abs(result["stats"]["good"]["t_stat"]) > 2.0
    assert abs(result["stats"]["noise"]["t_stat"]) <= 2.0

    path = tmp_path / "signal_allowlist.json"
    write_signal_allowlist(result, path)
    loaded = load_signal_allowlist(path)
    assert loaded["allowlist"] == result["allowlist"]
    assert_allowlist_valid(path, selection_end="2012-12-31")


def test_gate_refuses_dates_after_selection_end():
    t, k = 30, 8
    signal, returns = _predictive_panel(t=t, k=k, seed=5)
    dates = [f"2010-01-{1 + i:02d}" for i in range(t - 1)] + ["2015-06-01"]
    with pytest.raises((ValueError, AssertionError)):
        run_signal_gate(
            {"good": signal},
            returns,
            dates=dates,
            selection_end="2012-12-31",
            t_min=2.0,
        )


def test_empty_allowlist_raises(tmp_path: Path):
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps(
            {
                "allowlist": [],
                "status": "pending_gate",
                "selection_end": "2012-12-31",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises((ValueError, AssertionError)):
        assert_allowlist_valid(path, selection_end="2012-12-31")


def test_live_allowlist_config_is_well_formed():
    """B2: config/signal_allowlist.json is produced by a real gate run
    (scripts/run_signal_gate.py) over a PIT-restricted secid pool and
    window, not left as the hand-written empty ``pending_gate`` stub.

    This checks the recipe was actually executed (real per-signal stats
    keyed by name, a real ``estimand``, ``selection_end`` inside the PIT
    window) without asserting a specific admission outcome: whether any
    signal clears ``|t| > t_min`` is a property of the data, and a test
    that requires a non-empty allowlist would incentivize re-running the
    gate until something clears the bar (factor-shopping), which is
    explicitly out of bounds for this project.
    """
    root = Path(__file__).resolve().parents[1]
    path = root / "config" / "signal_allowlist.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selection_end"] <= "2012-12-31"
    assert isinstance(payload.get("allowlist"), list)
    assert payload.get("status") != "pending_gate", (
        "allowlist is still the unrun hand-written stub"
    )
    assert payload.get("stats"), "gate produced no per-signal stats"
    assert payload.get("estimand") in {
        "signal_ic_gate_v1",
        "signal_ic_gate_v2",
    }
    if payload["allowlist"]:
        assert_allowlist_valid(path)
    else:
        with pytest.raises((ValueError, AssertionError)):
            assert_allowlist_valid(path)


def test_run_signal_gate_v2_fdr_stricter_than_raw_t_min():
    """Mild |t|>2 family fails BH FDR q=0.05; one strong signal admits."""
    t, k = 120, 20
    rng = np.random.default_rng(11)
    latent = rng.normal(0.0, 1.0, size=(t, k))
    returns = np.zeros((t, k), dtype=np.float64)
    returns[1:] = 0.06 * latent[:-1] + rng.normal(0.0, 0.01, size=(t - 1, k))
    strong = latent + rng.normal(0.0, 0.05, size=(t, k))
    mild = {
        f"mild_{i}": (
            0.025 * latent
            + rng.normal(0.0, 1.0, size=(t, k))
        )
        for i in range(20)
    }
    dates = [f"2009-{(1 + i % 12):02d}-{1 + (i % 28):02d}" for i in range(t)]
    signals = {"strong": strong, **mild}
    legacy = run_signal_gate(
        signals, returns, dates=dates, selection_end="2012-12-31", t_min=2.0
    )
    v2 = run_signal_gate_v2(
        signals,
        returns,
        dates=dates,
        selection_end="2012-12-31",
        fdr_q=0.05,
        hlz_t=3.0,
    )
    assert v2["estimand"] == "signal_ic_gate_v2"
    assert v2["fdr_q"] == pytest.approx(0.05, **FLOAT_TOL)
    assert int(v2["n_family"]) == len(
        [n for n, r in v2["stats"].items() if r.get("status") == "scored"]
    )
    assert "strong" in v2["allowlist"]
    mild_legacy = [n for n in legacy["allowlist"] if n.startswith("mild_")]
    mild_v2 = [n for n in v2["allowlist"] if n.startswith("mild_")]
    assert len(legacy["allowlist"]) >= 1
    assert len(v2["allowlist"]) <= len(legacy["allowlist"])
    if mild_legacy:
        assert len(mild_v2) < len(mild_legacy)
    row = v2["stats"]["strong"]
    assert row["admitted"] is True
    assert row["status"] == "scored"
    assert "t_stat_nw" in row
    assert "p_value" in row
    assert "discovery_hlz" in row
    assert set(v2["allowlist"]).issubset(set(legacy["allowlist"]))


def test_run_signal_gate_v2_quarantines_all_nan():
    t, k = 40, 8
    good, returns = _predictive_panel(t=t, k=k, seed=13)
    blank = np.full((t, k), np.nan)
    dates = [f"2010-01-{1 + (i % 28):02d}" for i in range(t)]
    v2 = run_signal_gate_v2(
        {"good": good, "blank": blank},
        returns,
        dates=dates,
        selection_end="2012-12-31",
    )
    assert v2["stats"]["blank"]["status"] == "unscored"
    assert v2["stats"]["blank"]["admitted"] is False
    assert "blank" not in v2["allowlist"]
    assert v2["stats"]["good"]["status"] == "scored"


def test_run_signal_gate_v2_pit_refuse_unchanged():
    t, k = 30, 8
    signal, returns = _predictive_panel(t=t, k=k, seed=14)
    dates = [f"2010-01-{1 + i:02d}" for i in range(t - 1)] + ["2015-06-01"]
    with pytest.raises(ValueError, match="refuses dates after"):
        run_signal_gate_v2(
            {"good": signal},
            returns,
            dates=dates,
            selection_end="2012-12-31",
        )


def test_geometry_pack_valid_and_unknown_name_raises(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    lite = root / "config" / "obs_packs" / "surf_geometry_lite.yaml"
    assert lite.is_file()
    pack = assert_geometry_pack_valid(lite)
    assert pack["pack_id"] == "surf_geometry_lite"
    assert pack["estimand"] == "signal_obs_geometry_v1"
    assert set(pack["channels"]) == {"mfiv_30", "iv_term_slope", "iv_skew_30d"}

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "pack_id: bad\nestimand: signal_obs_geometry_v1\nchannels: [not_a_signal]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown"):
        assert_geometry_pack_valid(bad)


def test_geometry_pack_surf_off_empty_and_cs_admit_resolve():
    root = Path(__file__).resolve().parents[1]
    off = load_obs_pack(root / "config" / "obs_packs" / "surf_off.yaml")
    assert off["pack_id"] == "surf_off"
    assert list(off.get("channels") or []) == []
    cs = load_obs_pack(root / "config" / "obs_packs" / "surf_cs_admit.yaml")
    assert cs["pack_id"] == "surf_cs_admit"
    assert cs.get("resolve_from") == "signal_allowlist"
    # Empty geometry_lite is refused (non-off packs need channels or resolve).
    with pytest.raises(ValueError):
        assert_geometry_pack_valid(root / "config" / "obs_packs" / "surf_off.yaml")
