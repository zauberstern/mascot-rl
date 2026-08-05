"""Budget action + live-spend governor tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import boto3
import pytest

pytestmark = pytest.mark.plumbing
from moto import mock_aws

from mascotrl.aws_burst.budget_action import (
    ensure_budget_action,
    read_actual_spend,
    spend_headroom,
    stamp_armed_flag,
)
from mascotrl.aws_burst.governor import check_submit_allowed
from mascotrl.aws_burst.profiles import BURST_PROFILES, CREDIT_USD, SPEND_CAP_FRAC


class _FakeBudgets:
    def __init__(self) -> None:
        self.actions: list[dict[str, Any]] = []
        self.budget = {
            "BudgetName": "volsurf-burst-180",
            "CalculatedSpend": {"ActualSpend": {"Amount": "10.5", "Unit": "USD"}},
        }

    def describe_budget(self, **kwargs):
        return {"Budget": self.budget}

    def describe_budget_actions_for_budget(self, **kwargs):
        return {"Actions": list(self.actions)}

    def create_budget_action(self, **kwargs):
        action_id = f"action-{len(self.actions)+1}"
        self.actions.append(
            {
                "ActionId": action_id,
                "ActionType": kwargs["ActionType"],
                "ActionThreshold": {
                    "ActionThresholdValue": kwargs["ActionThreshold"]["ActionThresholdValue"]
                },
            }
        )
        return {"ActionId": action_id}


class _FakeClient:
    profile = "volsurf-burst-1"

    def __init__(self) -> None:
        self._fb = _FakeBudgets()

    def account_id(self) -> str:
        return "000000000001"

    def _budgets(self):
        return self._fb


def test_read_actual_spend_parses_amount() -> None:
    assert read_actual_spend(_FakeClient()) == pytest.approx(10.5)


def test_ensure_budget_action_idempotent() -> None:
    client = _FakeClient()
    first = ensure_budget_action(
        client,
        execution_role_arn="arn:aws:iam::000000000001:role/exec",
        target_role_name="volsurf-burst-BatchJobRole",
        deny_policy_arn="arn:aws:iam::000000000001:policy/deny",
    )
    assert first["created"] is True
    second = ensure_budget_action(
        client,
        execution_role_arn="arn:aws:iam::000000000001:role/exec",
        target_role_name="volsurf-burst-BatchJobRole",
        deny_policy_arn="arn:aws:iam::000000000001:policy/deny",
    )
    assert second["created"] is False
    assert second["action_id"] == first["action_id"]


def test_spend_headroom_and_live_refuse() -> None:
    assert spend_headroom(actual_usd=10.0, projected_usd=5.0) == pytest.approx(
        CREDIT_USD * SPEND_CAP_FRAC - 15.0
    )
    client = _FakeClient()
    client._fb.budget["CalculatedSpend"]["ActualSpend"]["Amount"] = "170"
    three = [{"profile": p["profile"]} for p in BURST_PROFILES]
    with pytest.raises(ValueError, match="spend_cap_exceeded_live"):
        check_submit_allowed(
            armed_profiles=three,
            projected_usd=20.0,
            clients=[client],
            offline=False,
        )


def test_offline_skips_live_spend() -> None:
    three = [{"profile": p["profile"]} for p in BURST_PROFILES]
    check_submit_allowed(
        armed_profiles=three,
        projected_usd=1.0,
        clients=[_FakeClient()],
        offline=True,
    )


def test_stamp_armed_flag(tmp_path: Path) -> None:
    path = stamp_armed_flag(tmp_path, "volsurf-burst-1", action_id="a1", verified=True)
    assert path.is_file()
    payload = path.read_text(encoding="utf-8")
    assert "action_id" in payload
    assert "verified_at" in payload


@mock_aws
def test_moto_budgets_describe_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    budgets = boto3.client("budgets", region_name="us-east-1")
    # moto budgets create may be limited; smoke that client constructs.
    assert budgets.meta.service_model.service_name == "budgets"
