"""AWS Budgets hard-deny action + live spend readback for burst governor."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from mascotrl.aws_burst.profiles import BUDGET_USD, CREDIT_USD, SPEND_CAP_FRAC

log = logging.getLogger(__name__)

DEFAULT_BUDGET_NAME = "volsurf-burst-180"
DEFAULT_THRESHOLD_PCT = 95.0
# Managed policy document applied by Budgets Action to the Batch job role.
DENY_POLICY_DOCUMENT = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyBatchAndEC2SpendAboveBudget",
            "Effect": "Deny",
            "Action": [
                "batch:SubmitJob",
                "ec2:RunInstances",
                "ec2:StartInstances",
                "ecs:RunTask",
                "ecs:StartTask",
            ],
            "Resource": "*",
        }
    ],
}


def _budgets_client_and_account(client: Any) -> tuple[Any, str]:
    if hasattr(client, "_budgets") and hasattr(client, "account_id"):
        return client._budgets(), client.account_id()
    raise ValueError("expected BurstClient with _budgets() and account_id()")


def read_actual_spend(client: Any, budget_name: str = DEFAULT_BUDGET_NAME) -> float:
    """Return CalculatedSpend.ActualSpend.Amount for the named COST budget."""
    budgets, account_id = _budgets_client_and_account(client)
    resp = budgets.describe_budget(AccountId=account_id, BudgetName=budget_name)
    budget = resp.get("Budget") or {}
    calc = budget.get("CalculatedSpend") or {}
    actual = calc.get("ActualSpend") or {}
    amount = actual.get("Amount")
    if amount is None:
        return 0.0
    return float(amount)


def _list_actions(budgets_client: Any, account_id: str, budget_name: str) -> list[dict]:
    try:
        resp = budgets_client.describe_budget_actions_for_budget(
            AccountId=account_id,
            BudgetName=budget_name,
        )
        return list(resp.get("Actions") or [])
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"NotFoundException", "ResourceNotFoundException"}:
            return []
        raise


def ensure_budget_action(
    client: Any,
    *,
    budget_name: str = DEFAULT_BUDGET_NAME,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    execution_role_arn: str,
    target_role_name: str,
    deny_policy_arn: str,
    notification_topic_arn: str | None = None,
) -> dict[str, Any]:
    """Idempotently create an APPLY_IAM_POLICY Budget Action at threshold_pct.

    Uses boto3 budgets.create_budget_action with ApprovalModel=AUTOMATIC so the
    deny attaches without manual approval when actual spend crosses the gate.
    See: docs.aws.amazon.com/aws-cost-management/latest/APIReference/API_budgets_CreateBudgetAction.html
    """
    budgets, account_id = _budgets_client_and_account(client)

    existing = _list_actions(budgets, account_id, budget_name)
    for action in existing:
        thr = float(
            ((action.get("ActionThreshold") or {}).get("ActionThresholdValue") or 0)
        )
        if (
            abs(thr - float(threshold_pct)) < 1e-6
            and action.get("ActionType") == "APPLY_IAM_POLICY"
        ):
            # Fail closed on soft-matched actions that drifted from operator policy.
            definition = action.get("Definition") or {}
            iam = definition.get("IamActionDefinition") or {}
            roles = list(iam.get("Roles") or [])
            policy_arn = str(iam.get("PolicyArn") or "")
            approval = str(action.get("ApprovalModel") or "")
            if deny_policy_arn and policy_arn and policy_arn != deny_policy_arn:
                raise ValueError(
                    f"budget_action_policy_arn_mismatch: got={policy_arn!r} "
                    f"expected={deny_policy_arn!r}"
                )
            if target_role_name and roles and target_role_name not in roles:
                raise ValueError(
                    f"budget_action_role_mismatch: roles={roles!r} "
                    f"expected={target_role_name!r}"
                )
            if approval and approval != "AUTOMATIC":
                raise ValueError(
                    f"budget_action_approval_not_automatic: {approval!r}"
                )
            return {
                "created": False,
                "action_id": action.get("ActionId"),
                "budget_name": budget_name,
                "threshold_pct": threshold_pct,
            }

    kwargs: dict[str, Any] = {
        "AccountId": account_id,
        "BudgetName": budget_name,
        "NotificationType": "ACTUAL",
        "ActionType": "APPLY_IAM_POLICY",
        "ActionThreshold": {
            "ActionThresholdValue": float(threshold_pct),
            "ActionThresholdType": "PERCENTAGE",
        },
        "Definition": {
            "IamActionDefinition": {
                "PolicyArn": deny_policy_arn,
                "Roles": [target_role_name],
            }
        },
        "ExecutionRoleArn": execution_role_arn,
        "ApprovalModel": "AUTOMATIC",
    }
    if notification_topic_arn:
        kwargs["Subscribers"] = [
            {"SubscriptionType": "SNS", "Address": notification_topic_arn}
        ]

    resp = budgets.create_budget_action(**kwargs)
    return {
        "created": True,
        "action_id": resp.get("ActionId"),
        "budget_name": budget_name,
        "threshold_pct": threshold_pct,
    }


def stamp_armed_flag(
    root: Path,
    profile: str,
    *,
    action_id: str | None,
    verified: bool = True,
) -> Path:
    cfg_dir = root / "deploy" / "aws_burst" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"budget_armed_{profile}.json"
    payload = {
        "profile": profile,
        "budget_action_armed": True,
        "armed": True,
        "action_id": action_id,
        "verified": bool(verified),
        "verified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "budget_usd": BUDGET_USD,
        "credit_usd": CREDIT_USD,
        "spend_cap_frac": SPEND_CAP_FRAC,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def spend_headroom(
    *,
    actual_usd: float,
    projected_usd: float,
    credit_usd: float | None = None,
    spend_cap_frac: float | None = None,
) -> float:
    cred = float(CREDIT_USD if credit_usd is None else credit_usd)
    frac = float(SPEND_CAP_FRAC if spend_cap_frac is None else spend_cap_frac)
    return cred * frac - float(actual_usd) - float(projected_usd)
