"""Tests for OOM estimate and smart retry."""
from __future__ import annotations

from mascotrl.aws_burst.job_routing import estimate_peak_memory
from mascotrl.aws_burst.smart_retry import classify_batch_failure


def test_estimate_peak_memory_gru_higher_than_mlp() -> None:
    mlp = estimate_peak_memory({"n_assets": 100, "architecture": "mlp"})
    gru = estimate_peak_memory({"n_assets": 100, "architecture": "gru"})
    assert gru > mlp
    assert mlp > 4000


def test_smart_retry_oom_and_app_error() -> None:
    oom = classify_batch_failure(exit_code=137, attempt=0)
    assert oom.action == "retry_himem"
    app = classify_batch_failure(exit_code=1, attempt=0)
    assert app.action == "no_retry"
    exhausted = classify_batch_failure(exit_code=137, attempt=2)
    assert exhausted.action == "no_retry"
    spot = classify_batch_failure(
        exit_code=None, status_reason="Host EC2 Spot interruption", attempt=0
    )
    assert spot.action == "retry_same"
