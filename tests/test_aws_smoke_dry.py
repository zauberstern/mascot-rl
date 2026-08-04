"""Moto/dry coverage for scripts/aws_smoke.py compose path."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

from pathlib import Path

from scripts.aws_smoke import assert_smoke_artifacts, compose_smoke_plan, run_smoke


def test_compose_smoke_plan_ordered_steps() -> None:
    plan = compose_smoke_plan(profile="volsurf-burst-1", wave="PICK_SMOKE")
    ops = [s["op"] for s in plan["steps"]]
    assert ops[0] == "deploy_stack"
    assert "arm_budget_action" in ops
    assert "submit_wave" in ops
    assert "wait_for_array" in ops
    assert "pull_artifacts" in ops
    assert plan["region"] == "eu-central-1"
    templates = [s.get("template") for s in plan["steps"] if s["op"] == "deploy_stack"]
    assert any(t and t.endswith("00_guardrails.yaml") for t in templates)
    assert any(t and t.endswith("10_batch_spot.yaml") for t in templates)


def test_run_smoke_dry_run_no_aws() -> None:
    out = run_smoke(profile="volsurf-burst-1", dry_run=True)
    assert out["dry_run"] is True
    assert out["plan"]["wave"] == "PICK_SMOKE"
    assert len(out["plan"]["steps"]) >= 6


def test_assert_smoke_artifacts_requires_accepted() -> None:
    try:
        assert_smoke_artifacts({"n_accepted": 0, "rejected": [{"reason": "x"}]})
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "smoke_" in str(exc)
    assert raised
    assert_smoke_artifacts({"n_accepted": 1, "rejected": []})


def test_smoke_cli_dry(tmp_path: Path, monkeypatch) -> None:
    import scripts.aws_smoke as smoke

    # Ensure main returns 0 on dry-run
    rc = smoke.main(["--dry-run", "--profile", "volsurf-burst-1"])
    assert rc == 0
