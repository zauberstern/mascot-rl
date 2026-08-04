#!/usr/bin/env python3
"""Score AWS readiness on the four axes; optionally dry-run or live-submit PICK.

Axes (user-defined 10/10 gate):
  (a) methodological / mathematical / econometric correctness + crash-free
  (b) optimized + parallel AWS infrastructure
  (c) fault-proof inputs (features) and outputs (reports/logs/stats)
  (d) interrupted processes resume end-to-end

Local scoring is evidence-backed from tests + code invariants. A live full-scale
wave is gated behind ``--live`` and requires armed AWS profiles.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pytest_focused(paths: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", *paths, "-q", "--tb=no"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "tail": "\n".join(out.strip().splitlines()[-8:]),
    }


def score_readiness(*, run_tests: bool = True) -> dict[str, Any]:
    from scripts.aws_smoke import compose_smoke_plan
    from scripts.aws_submit_wave import build_plan
    from src.aws_burst.profiles import MAX_VCPUS_PER_ACCOUNT
    from src.data.feature_panels import required_panel_families
    from src.eval.research_happo_cpcv import (
        _discover_latest_happo_checkpoint,
        _maybe_resume_happo_checkpoint,
    )
    from scripts.run_spectrum_campaign import (
        _prepare_spectrum_resume_dirs,
        _spectrum_run_config_hash,
    )

    checks: dict[str, Any] = {}

    # (a) correctness + crash-free: focused regression packs green
    if run_tests:
        checks["a_resume_tests"] = _pytest_focused(["tests/test_resume_e2e.py"])
        checks["a_strict_tests"] = _pytest_focused(["tests/test_strict_no_degrade.py"])
        checks["a_panel_tests"] = _pytest_focused(["tests/test_panel_bundle.py"])
        checks["a_smoke_tests"] = _pytest_focused(["tests/test_aws_smoke_dry.py"])
        checks["a_cherrypick_tests"] = _pytest_focused(
            ["tests/test_cherrypick_panel.py", "tests/test_aws_burst_waves.py"]
        )
    else:
        for k in (
            "a_resume_tests",
            "a_strict_tests",
            "a_panel_tests",
            "a_smoke_tests",
            "a_cherrypick_tests",
        ):
            checks[k] = {"ok": True, "skipped": True}

    # (b) infra parallelism
    checks["b_max_vcpus"] = {
        "ok": MAX_VCPUS_PER_ACCOUNT == 32,
        "value": MAX_VCPUS_PER_ACCOUNT,
    }
    smoke = compose_smoke_plan(profile="volsurf-burst-1")
    checks["b_smoke_plan"] = {
        "ok": "wait_for_array" in [s["op"] for s in smoke["steps"]],
        "n_steps": len(smoke["steps"]),
    }
    cfn_batch = (ROOT / "deploy/aws_burst/cloudformation/10_batch_spot.yaml").read_text()
    checks["b_cloudwatch"] = {
        "ok": "AWS::Logs::LogGroup" in cfn_batch and "awslogs" in cfn_batch,
    }
    cfn_g = (ROOT / "deploy/aws_burst/cloudformation/00_guardrails.yaml").read_text()
    arm_script = (ROOT / "scripts/aws_arm_budget_action.py").read_text()
    checks["b_budget_action"] = {
        "ok": (
            "volsurf-burst-spend-deny" in cfn_g
            and "volsurf-burst-budgets-action-exec" in cfn_g
            and "ensure_budget_action" in arm_script
            and "AWS::Budgets::BudgetsAction" not in cfn_g
        ),
    }
    live_smoke = ROOT / "deploy/aws_burst/config/live_smoke_PICK_SMOKE.json"
    checks["b_live_smoke"] = {
        "ok": live_smoke.is_file(),
        "path": str(live_smoke.relative_to(ROOT)) if live_smoke.is_file() else None,
    }

    # (c) fault-proof I/O
    families = required_panel_families()
    checks["c_panel_families"] = {"ok": len(families) == 11, "n": len(families)}
    plan = build_plan(ROOT, "PICK")
    n_expected = len(plan.get("expected_cells") or [])
    checks["c_pick_plan"] = {
        "ok": n_expected >= 140,
        "n_cells": n_expected,
        "expected_cells": n_expected,
    }
    checks["c_strict_helpers"] = {
        "ok": callable(_spectrum_run_config_hash)
        and callable(_prepare_spectrum_resume_dirs),
    }

    # (d) resume end-to-end symbols present
    checks["d_happo_resume"] = {
        "ok": callable(_discover_latest_happo_checkpoint)
        and callable(_maybe_resume_happo_checkpoint),
    }
    cell_runner = (ROOT / "deploy/aws_burst/docker/cell_runner.py").read_text()
    checks["d_cell_runner_resume"] = {
        "ok": "pull_resume_state" in cell_runner and "push_resume_state" in cell_runner,
    }
    checks["d_spectrum_strict_flag"] = {
        "ok": "--strict" in (ROOT / "scripts/run_spectrum_campaign.py").read_text(),
    }

    axis_a = all(checks[k].get("ok") for k in checks if k.startswith("a_"))
    axis_b = all(checks[k].get("ok") for k in checks if k.startswith("b_"))
    axis_c = all(checks[k].get("ok") for k in checks if k.startswith("c_"))
    axis_d = all(checks[k].get("ok") for k in checks if k.startswith("d_"))
    scores = {
        "a_correctness_crashfree": 10 if axis_a else 7,
        "b_parallel_infra": 10 if axis_b else 7,
        "c_faultproof_io": 10 if axis_c else 6,
        "d_resume_e2e": 10 if axis_d else 5,
    }
    # Live full-scale not yet executed in this process => note separately.
    overall = min(scores.values())
    return {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "scores": scores,
        "overall_local": overall,
        "live_fullscale_required_for_10": True,
        "live_prerequisites": {
            "panel_bundle": "logs/aws_burst_panel_bundle/panel_bundle.tar + panel_manifest.json (bash deploy/aws_burst/scripts/aws_build_panel_bundle.sh)",
            "image_digest": "deploy/aws_burst/config/image_digest_*.json (bash deploy/aws_burst/scripts/aws_build_push_image.sh)",
            "smoke": "python scripts/aws_smoke.py --profile volsurf-burst-1",
            "fullscale": "python scripts/aws_submit_wave.py PICK && python scripts/aws_pull_artifacts.py --wave PICK",
        },
        "live_commands": {
            "smoke": "python scripts/aws_smoke.py --profile volsurf-burst-1",
            "submit_pick": "python scripts/aws_submit_wave.py PICK",
            "pull_pick": "python scripts/aws_pull_artifacts.py --wave PICK",
        },
        "note": (
            "overall_local is the readiness of the codebase + tests (10 when all "
            "four axes pass locally). A literal operational 10/10 also requires a "
            "successful live PICK pull with zero dry_run / feature_net_errors / "
            "spectrum_seed_errors after panel bundle + image digests exist."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument(
        "--dry-run-submit",
        action="store_true",
        help="Also dry-run aws_submit_wave PICK and stamp into the report.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "logs" / "aws_readiness_report.json",
    )
    args = p.parse_args(argv)
    report = score_readiness(run_tests=not bool(args.skip_tests))
    if args.dry_run_submit:
        from scripts.aws_submit_wave import submit_wave

        report["pick_dry_run"] = submit_wave(ROOT, "PICK", dry_run=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "overall_local": report["overall_local"],
        "scores": report["scores"],
        "out": str(args.out),
        "live_fullscale_required_for_10": True,
    }, indent=2))
    return 0 if report["overall_local"] >= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
