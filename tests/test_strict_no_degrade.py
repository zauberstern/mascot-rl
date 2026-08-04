"""Strict mode: feature-net / seed / fallback degrades never promote."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_run_cell_hoists_feature_net_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts import run_spectrum_campaign as camp

    cfg_path = tmp_path / "cell.yaml"
    cfg_path.write_text(
        "spectrum_cell_id: t1\nalgo: ppo\nportfolio_arm: eq\nclaim_tier: research\n"
    )

    def _fake_arm(cfg, arm, **kwargs):
        return (
            {
                "path_summary": {"mean_sharpe": 0.1, "n_paths": 1},
                "real_reference_arm_present": True,
                "panel_source": "toy",
                "toy_panel": True,
                "claim_tier": "research",
                "_feature_net_errors": ["attach boom"],
                "spectrum_seed_errors": ["seed=0: x"],
                "collapse_guard": {"ok": True},
            },
            None,
        )

    monkeypatch.setattr(camp, "_run_research_arm", _fake_arm)
    monkeypatch.setattr(camp, "load_cell_yaml", lambda _p: {
        "spectrum_cell_id": "t1",
        "algo": "ppo",
        "portfolio_arm": "eq",
        "claim_tier": "research",
    })
    monkeypatch.setattr(camp, "validate_cfg", lambda c: dict(c))
    monkeypatch.setattr(
        camp,
        "resolve_spectrum_budget",
        lambda _c: {
            "claim_tier": "research",
            "n_episodes": 1,
            "horizon": 8,
            "seeds": [0],
            "cpcv_n_splits": 2,
            "cpcv_n_test_groups": 1,
            "dispatch_only": False,
        },
    )
    monkeypatch.setattr(
        camp,
        "_transfer_from_runner",
        lambda **_k: {"real_reference_arm_present": True},
    )
    monkeypatch.setattr(
        camp,
        "_collapse_from_runner",
        lambda *_a, **_k: {"ok": True},
    )
    monkeypatch.setattr(
        camp,
        "_gates_from_runner",
        lambda *_a, **_k: {"gate1": {"pass": True}, "gate2": {"pass": True}, "gate3": {"pass": True}},
    )
    monkeypatch.setattr(camp, "_hoist_runner_weights", lambda *_a, **_k: {})

    art = camp.run_cell(cfg_path, dry_run=False, strict=True)
    assert art["feature_net_errors"] == ["attach boom"]
    assert art["spectrum_seed_errors"] == ["seed=0: x"]
    assert art["promotable"] is False
    assert art.get("strict_degraded") is True


def test_run_cell_strict_false_still_hoists_but_may_promote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import run_spectrum_campaign as camp

    cfg_path = tmp_path / "cell.yaml"
    cfg_path.write_text("spectrum_cell_id: t2\nalgo: ppo\n")

    monkeypatch.setattr(
        camp,
        "_run_research_arm",
        lambda *_a, **_k: (
            {
                "path_summary": {"mean_sharpe": 0.1, "n_paths": 1},
                "real_reference_arm_present": True,
                "claim_tier": "research",
                "_feature_net_errors": ["soft"],
                "collapse_guard": {"ok": True},
            },
            None,
        ),
    )
    monkeypatch.setattr(camp, "load_cell_yaml", lambda _p: {
        "spectrum_cell_id": "t2",
        "algo": "ppo",
        "portfolio_arm": "eq",
        "claim_tier": "research",
    })
    monkeypatch.setattr(camp, "validate_cfg", lambda c: dict(c))
    monkeypatch.setattr(
        camp,
        "resolve_spectrum_budget",
        lambda _c: {
            "claim_tier": "research",
            "n_episodes": 1,
            "horizon": 8,
            "seeds": [0],
            "cpcv_n_splits": 2,
            "cpcv_n_test_groups": 1,
            "dispatch_only": False,
        },
    )
    monkeypatch.setattr(
        camp,
        "_transfer_from_runner",
        lambda **_k: {"real_reference_arm_present": True},
    )
    monkeypatch.setattr(camp, "_collapse_from_runner", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(
        camp,
        "_gates_from_runner",
        lambda *_a, **_k: {"gate1": {"pass": True}, "gate2": {"pass": True}, "gate3": {"pass": True}},
    )
    monkeypatch.setattr(camp, "_hoist_runner_weights", lambda *_a, **_k: {"weights": [[0.5, 0.5]]})

    art = camp.run_cell(cfg_path, dry_run=False, strict=False)
    assert art["feature_net_errors"] == ["soft"]
    # Without strict, feature_net_errors alone do not force promotable=False
    # (fallback_reason / dry still do). Collapse+ref present => promotable True.
    assert art["promotable"] is True
    assert art.get("strict_degraded") is not True


def test_artifact_is_strict_degraded() -> None:
    from scripts.aws_pull_artifacts import artifact_is_strict_degraded

    assert artifact_is_strict_degraded({"dry_run": True}) is True
    assert artifact_is_strict_degraded({"feature_net_errors": ["x"]}) is True
    assert artifact_is_strict_degraded({"spectrum_seed_errors": ["s"]}) is True
    assert artifact_is_strict_degraded({"strict_degraded": True}) is True
    assert artifact_is_strict_degraded({"dry_run": False, "promotable": True}) is False


def test_pull_rejects_feature_net_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import aws_pull_artifacts as pull

    monkeypatch.setattr(pull, "armed_profiles", lambda _r: [{"profile": "p0", "shard": 0}])
    monkeypatch.setattr(pull, "_expected_stems", lambda *_a, **_k: {"good", "bad"})

    class _Client:
        def account_id(self):
            return "1"

        def download_prefix(self, bucket, prefix, dest: Path, **_kwargs):
            import hashlib

            dest.mkdir(parents=True, exist_ok=True)
            good = {
                "dry_run": False,
                "compute_host": "remote",
                "instance_type": "c",
                "container_digest": "sha256:x",
            }
            bad = {
                "dry_run": False,
                "feature_net_errors": ["missing ohlc"],
                "compute_host": "remote",
                "instance_type": "c",
                "container_digest": "sha256:x",
            }
            for name, payload in (("good.json", good), ("bad.json", bad)):
                path = dest / name
                blob = json.dumps(payload).encode("utf-8")
                path.write_bytes(blob)
                (dest / f"{name}.sha256").write_text(
                    hashlib.sha256(blob).hexdigest() + "\n", encoding="utf-8"
                )

    monkeypatch.setattr(pull, "BurstClient", lambda *_a, **_k: _Client())
    monkeypatch.setattr(pull, "artifact_bucket", lambda *_a, **_k: "b")
    monkeypatch.setattr(
        pull,
        "validate_remote_cell",
        lambda art, **_k: {"ok": True} if not art.get("feature_net_errors") else {"ok": False, "errors": ["feat"]},
    )
    # Force our degrade check to fire before validate for bad
    dest = tmp_path / "out"
    with pytest.raises(SystemExit, match="pull_incomplete"):
        pull.pull_wave(tmp_path, "PICK", dest=dest, require_complete=True)
    index = json.loads((dest / "index.json").read_text())
    reasons = [r.get("reason") for r in index["rejected"]]
    assert any(r == "feature_net_errors" or (isinstance(r, list) and r) for r in reasons) or any(
        "feature_net" in str(r) for r in reasons
    )


def test_cell_runner_refuses_strict_degraded_artifact() -> None:
    from deploy.aws_burst.docker.cell_runner import is_strict_degraded_artifact

    assert is_strict_degraded_artifact({"dry_run": True}) is True
    assert is_strict_degraded_artifact({"feature_net_errors": ["x"]}) is True
    assert is_strict_degraded_artifact({"spectrum_seed_errors": ["e"]}) is True
    assert is_strict_degraded_artifact({"promotable": True, "dry_run": False}) is False
