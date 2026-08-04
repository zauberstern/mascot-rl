"""CloudFormation template lint tests."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFN = ROOT / "deploy/aws_burst/cloudformation"


def _text(name: str) -> str:
    return (CFN / name).read_text(encoding="utf-8")


def test_batch_template_has_job_definition_and_queue_name() -> None:
    text = _text("10_batch_spot.yaml")
    assert re.search(r"MaxvCpus:\s*\n\s*Type: Number\s*\n\s*Default: 32", text)
    assert "CellJobDefinition:" in text
    assert "JobQueueName: volsurf-burst-queue" in text
    assert "AttemptDurationSeconds: 43200" in text
    assert "Attempts: 3" in text
    assert "AWS::Logs::LogGroup" in text
    assert "LogDriver: awslogs" in text
    assert "FailedJobsAlarm:" in text


def test_batch_job_role_not_s3_readonly_only() -> None:
    text = _text("10_batch_spot.yaml")
    assert "AmazonS3ReadOnlyAccess" not in text
    assert "BatchJobRole:" in text
    assert "RoleName: volsurf-burst-BatchJobRole" in text


def test_guardrails_budget_180() -> None:
    text = _text("00_guardrails.yaml")
    assert "Default: 180" in text
    assert "Default: 90" not in text
    assert "AWS::Budgets::Budget" in text
    assert "volsurf-burst-spend-deny" in text
    assert "volsurf-burst-budgets-action-exec" in text


def test_batch_spot_instance_types_are_free_tier_eligible() -> None:
    """New Free-plan accounts reject Spot of non-eligible types at ASG launch."""
    text = _text("10_batch_spot.yaml")
    assert "m7i-flex.large" in text
    assert "m5.large" not in text
    assert "c7i-flex.large" not in text
    """BudgetsAction is created by aws_arm_budget_action.py after Batch is up.

    CFN Early Validation refuses APPLY_IAM_POLICY when the target role is
    created in the same changeset, so the action is intentionally not in
    either CFN template.
    """
    batch = _text("10_batch_spot.yaml")
    guard = _text("00_guardrails.yaml")
    assert "AWS::Budgets::BudgetsAction" not in batch
    assert "AWS::Budgets::BudgetsAction" not in guard
    assert "volsurf-burst-spend-deny" in guard
    assert "volsurf-burst-budgets-action-exec" in guard
    assert "RoleName: volsurf-burst-BatchJobRole" in batch

