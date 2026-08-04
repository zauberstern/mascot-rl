"""AWS-7 cost governor unit tests."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

from src.aws_burst.cost_model import affordable_frontier, refuse_submit_if_unsafe


def test_affordable_frontier_picks_cheapest() -> None:
    out = affordable_frontier(
        n_cells=10,
        hours_per_cell_by_vcpu={1: 2.0, 2: 1.1, 4: 0.7},
        usd_per_vcpu_hour=0.022,
        budget_usd=90.0,
    )
    assert out["ok"] is True
    assert out["chosen_vcpus"] in (1, 2, 4)


def test_refuse_without_budget_action() -> None:
    with pytest.raises(ValueError, match="budget_action_not_armed"):
        refuse_submit_if_unsafe(
            budget_action_armed=False, projected_usd=10.0
        )


def test_refuse_over_credit_cap() -> None:
    with pytest.raises(ValueError, match="spend_cap_exceeded"):
        refuse_submit_if_unsafe(
            budget_action_armed=True, projected_usd=185.0, credit_usd=100.0
        )
