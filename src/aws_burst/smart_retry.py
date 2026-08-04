"""Smart retry for AWS Batch cell failures (OOM / spot interruption)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RetryAction = Literal["retry_himem", "retry_same", "no_retry"]


@dataclass(frozen=True)
class RetryDecision:
    action: RetryAction
    reason: str
    max_retries: int = 2


def classify_batch_failure(
    *,
    exit_code: int | None,
    status_reason: str | None = None,
    attempt: int = 0,
    max_retries: int = 2,
) -> RetryDecision:
    """Decide whether/how to retry a failed Batch cell.

    - Exit 137 (SIGKILL / OOM): retry on himem queue.
    - Spot interruption signal: retry on same queue.
    - Other non-zero exit: log and do not retry.
    - Max 2 retries per cell.
    """
    reason = str(status_reason or "").lower()
    if attempt >= int(max_retries):
        return RetryDecision("no_retry", "max_retries_exhausted", max_retries)
    if exit_code == 137 or "out of memory" in reason or "oom" in reason:
        return RetryDecision("retry_himem", "oom_exit_137", max_retries)
    if (
        "spot" in reason
        or "host ec2" in reason
        or "task interrupted" in reason
        or "cannotpullcontainer" in reason
    ):
        return RetryDecision("retry_same", "spot_or_infra_interruption", max_retries)
    if exit_code is not None and int(exit_code) != 0:
        return RetryDecision("no_retry", f"application_error_exit_{exit_code}", max_retries)
    return RetryDecision("no_retry", "unknown_failure", max_retries)
