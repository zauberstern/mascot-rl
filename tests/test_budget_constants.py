"""Budget constants and burst profile identity locks."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.plumbing
from tests.conftest import FLOAT_TOL

from mascotrl.aws_burst.cost_model import affordable_frontier, refuse_submit_if_unsafe
from mascotrl.aws_burst.profiles import (
    BUDGET_USD,
    BURST_PROFILES,
    CREDIT_USD,
    SPEND_CAP_FRAC,
    SPOT_VCPU_REQUEST,
)


ROOT = Path(__file__).resolve().parents[1]


def test_budget_sot_values() -> None:
    assert CREDIT_USD == pytest.approx(200.0, **FLOAT_TOL)
    assert BUDGET_USD == pytest.approx(180.0, **FLOAT_TOL)
    assert SPEND_CAP_FRAC == pytest.approx(0.90, **FLOAT_TOL)
    assert SPOT_VCPU_REQUEST == 64


def test_exactly_four_verified_profiles() -> None:
    assert len(BURST_PROFILES) == 4
    ids = {p["account_id"] for p in BURST_PROFILES}
    assert ids == {
        "000000000001",
        "000000000002",
        "000000000003",
        "000000000004",
    }

def test_cost_model_defaults_resolve_from_profiles() -> None:
    out = affordable_frontier(
        n_cells=1,
        hours_per_cell_by_vcpu={1: 0.01},
        usd_per_vcpu_hour=0.022,
    )
    assert out["ok"] is True
    assert out["cap_usd"] == min(BUDGET_USD, CREDIT_USD * SPEND_CAP_FRAC)


def test_refuse_over_default_credit_cap() -> None:
    with pytest.raises(ValueError, match="spend_cap_exceeded"):
        refuse_submit_if_unsafe(budget_action_armed=True, projected_usd=185.0)


def test_common_sh_has_no_literal_90_budget() -> None:
    text = (ROOT / "deploy/aws_burst/scripts/_common.sh").read_text(encoding="utf-8")
    assert "BUDGET_USD=90" not in text
    assert "SPOT_QUOTA_REQUEST=512" not in text


def test_guardrails_default_budget_180() -> None:
    text = (ROOT / "deploy/aws_burst/cloudformation/00_guardrails.yaml").read_text(
        encoding="utf-8"
    )
    assert "Default: 180" in text
    assert "Default: 90" not in text


def test_deploy_batch_default_maxvcpus_matches_sot() -> None:
    from mascotrl.aws_burst.profiles import MAX_VCPUS_PER_ACCOUNT

    text = (ROOT / "deploy/aws_burst/scripts/aws_deploy_batch.sh").read_text(
        encoding="utf-8"
    )
    assert (
        f'MAXV="${{1:-{MAX_VCPUS_PER_ACCOUNT}}}"' in text
        or f'MAXV="${{1:-32}}"' in text
    )
    frontier_path = ROOT / "deploy/aws_burst/config/cost_frontier.json"
    if frontier_path.is_file():
        frontier = frontier_path.read_text(encoding="utf-8")
        assert '"cap_usd": 180.0' in frontier or '"cap_usd": 180' in frontier
