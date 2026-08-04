"""Guards the resumable campaign manifest against silently reusing a stale
CPCV seed artifact computed under a different training config.

Regression for a real incident: relaunching scripts/run_eq_alloc_campaign.py
with a different --k / --train-env-steps / surface-signal allowlist silently
loaded a prior run's cpcv_seed_N.json (a stale, different-estimand artifact)
because the manifest only keyed on (fold_id, seed, arm), with no check that
the cached artifact was produced under the current config.
"""
from __future__ import annotations

from scripts.run_eq_alloc_campaign import _campaign_config_fingerprint


def _base_cfg() -> dict:
    return {
        "cpcv_n_splits": 6,
        "cpcv_n_test_groups": 2,
        "cpcv_purge_days": 21,
        "cpcv_embargo_days": 21,
        "train_env_steps": 100000,
        "train_epochs": 10,
        "min_optimizer_steps_total": 200,
        "min_optimizer_steps": 40,
        "use_surface_signals": True,
        "feature_extras": {"iv_surface": {"iv_term_slope": None, "mfis_365": None}},
        "turnover_limit": 0.15,
        "projection_mode": "soft",
        "rebalance_cadence": "monthly",
        "equity_bps": 5.0,
        "impact_c_eq": 0.5,
        "ppo_hidden": 64,
        "lr": 3e-4,
        "entropy_coef": 0.02,
        "dii_epochs": 80,
        "max_pool": 250,
        "weight_head": "softmax",
        "actor_final_gain": 0.01,
        "weight_head_temperature": 1.0,
        "train_updates_per_fold": 1,
        "universe_mode": "ROLLING_TRAILING_PIT",
    }


def test_fingerprint_stable_for_identical_config() -> None:
    cfg = _base_cfg()
    h1 = _campaign_config_fingerprint(cfg, realized_k=100)
    h2 = _campaign_config_fingerprint(dict(cfg), realized_k=100)
    assert h1 == h2


def test_fingerprint_changes_with_realized_k() -> None:
    cfg = _base_cfg()
    h_k100 = _campaign_config_fingerprint(cfg, realized_k=100)
    h_k40 = _campaign_config_fingerprint(cfg, realized_k=40)
    assert h_k100 != h_k40


def test_fingerprint_changes_with_train_env_steps() -> None:
    cfg = _base_cfg()
    h_full = _campaign_config_fingerprint(cfg, realized_k=40)
    cfg["train_env_steps"] = 3000
    h_smoke = _campaign_config_fingerprint(cfg, realized_k=40)
    assert h_full != h_smoke


def test_fingerprint_changes_with_surface_allowlist() -> None:
    cfg = _base_cfg()
    h_two_signals = _campaign_config_fingerprint(cfg, realized_k=40)
    cfg["feature_extras"] = {"iv_surface": {"iv_term_slope": None}}
    h_one_signal = _campaign_config_fingerprint(cfg, realized_k=40)
    assert h_two_signals != h_one_signal

    cfg["use_surface_signals"] = False
    cfg["feature_extras"] = {}
    h_no_signals = _campaign_config_fingerprint(cfg, realized_k=40)
    assert h_no_signals not in (h_two_signals, h_one_signal)


def test_fingerprint_changes_with_w31_training_knobs() -> None:
    """W3.1: ppo_hidden/lr/entropy_coef/dii_epochs/max_pool/weight_head/
    actor_final_gain/weight_head_temperature/train_updates_per_fold/
    universe_mode must all be fingerprinted so a resumed campaign retrains
    instead of silently reusing a seed artifact from a different config.
    """
    base = _base_cfg()
    base_hash = _campaign_config_fingerprint(base, realized_k=40)
    for key, changed in (
        ("ppo_hidden", 128),
        ("lr", 1e-3),
        ("entropy_coef", 0.05),
        ("dii_epochs", 40),
        ("max_pool", 100),
        ("weight_head", "tanh_l1"),
        ("actor_final_gain", 0.5),
        ("weight_head_temperature", 2.0),
        ("train_updates_per_fold", 3),
        ("universe_mode", "densest_subgraph"),
    ):
        cfg = _base_cfg()
        cfg[key] = changed
        h = _campaign_config_fingerprint(cfg, realized_k=40)
        assert h != base_hash, f"fingerprint did not change for key={key!r}"


def test_fingerprint_ignores_key_order() -> None:
    cfg1 = _base_cfg()
    cfg2 = dict(reversed(list(_base_cfg().items())))
    assert _campaign_config_fingerprint(cfg1, realized_k=40) == _campaign_config_fingerprint(
        cfg2, realized_k=40
    )


def test_run_config_hash_stamp_matches_fingerprint() -> None:
    """Headline summary must carry the same hash the resume fingerprint uses.

    ``build_report_numbers --require-config-hash`` reads ``results['run_config_hash']``
    (or confirmatory); if the campaign never stamps it, seal refusal is permanent.
    """
    cfg = _base_cfg()
    h = _campaign_config_fingerprint(cfg, realized_k=100)
    results = {
        "run_config_hash": h,
        "confirmatory": {"run_config_hash": h},
    }
    assert results["run_config_hash"] == h
    assert results["confirmatory"]["run_config_hash"] == h
    assert len(h) >= 8

