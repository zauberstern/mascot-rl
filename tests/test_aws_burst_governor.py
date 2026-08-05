"""AWS burst governor and deploy script guards."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.plumbing

from mascotrl.aws_burst.governor import check_submit_allowed, projected_wave_cost
from mascotrl.aws_burst.profiles import BURST_PROFILES, BUDGET_USD, CREDIT_USD, SPEND_CAP_FRAC, armed_profiles


ROOT = Path(__file__).resolve().parents[1]
THREE = [{"profile": p["profile"]} for p in BURST_PROFILES]


def test_projected_wave_cost() -> None:
    assert projected_wave_cost(
        n_cells=140, hours_per_cell=1.91, usd_per_vcpu_hour=0.022, vcpus=1
    ) == pytest.approx(140 * 1.91 * 0.022)


def test_check_submit_allowed_under_cap() -> None:
    cap = CREDIT_USD * SPEND_CAP_FRAC
    check_submit_allowed(
        armed_profiles=THREE,
        projected_usd=cap - 1.0,
    )


def test_check_submit_refuses_over_cap() -> None:
    with pytest.raises(ValueError, match="spend_cap_exceeded"):
        check_submit_allowed(
            armed_profiles=THREE,
            projected_usd=BUDGET_USD + 10.0,
        )


def test_check_submit_refuses_incomplete_armed() -> None:
    with pytest.raises(ValueError, match="incomplete_armed_profiles"):
        check_submit_allowed(
            armed_profiles=[{"profile": "volsurf-burst-1"}],
            projected_usd=1.0,
        )


def test_check_submit_allows_partial_when_flagged() -> None:
    check_submit_allowed(
        armed_profiles=[{"profile": "volsurf-burst-1"}],
        projected_usd=1.0,
        allow_partial_profiles=True,
    )


def test_armed_profiles_requires_all_three(tmp_path: Path) -> None:
    cfg = tmp_path / "deploy" / "aws_burst" / "config"
    cfg.mkdir(parents=True)
    (cfg / "budget_armed_volsurf-burst-1.json").write_text(
        '{"verified": true, "action_id": "x", "armed": true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incomplete_armed_profiles"):
        armed_profiles(tmp_path)


def test_aws_deploy_batch_requires_digest_pin() -> None:
    script = (ROOT / "deploy/aws_burst/scripts/aws_deploy_batch.sh").read_text(
        encoding="utf-8"
    )
    assert "ImageUri=$URI" in script
    assert "pinned_image_uri" in script
    assert "image_digest_" in script
    assert "@sha256:" in script
    assert "image_uri.txt" not in script
    assert "public.ecr.aws/docker/library/python" in script