"""Job definition builder tests."""
from __future__ import annotations

import pytest

from mascotrl.aws_burst.jobdef import (
    build_container_env,
    build_job_definition,
    build_retry_strategy,
)

pytestmark = pytest.mark.plumbing


def test_retry_strategy_evaluate_on_exit() -> None:
    rs = build_retry_strategy(attempts=3)
    assert rs["attempts"] == 3
    actions = {e["action"] for e in rs["evaluateOnExit"]}
    assert actions == {"RETRY", "EXIT"}


def test_job_definition_timeout_and_resources() -> None:
    jd = build_job_definition(
        image_uri="123.dkr.ecr.eu-central-1.amazonaws.com/volsurf:latest",
        job_role_arn="arn:aws:iam::1:role/job",
        timeout_s=14400,
    )
    assert jd["timeout"]["attemptDurationSeconds"] == 14400
    assert jd["containerProperties"]["vcpus"] == 1
    assert jd["containerProperties"]["memory"] == 6912


def test_job_memory_sanity_locked_to_free_tier() -> None:
    from mascotrl.aws_burst.jobdef import assert_job_memory_fits_instance
    import pytest

    assert_job_memory_fits_instance(6912)
    with pytest.raises(ValueError, match="job_memory_exceeds_instance"):
        assert_job_memory_fits_instance(7000)


def test_container_env_requires_all_keys() -> None:
    env = build_container_env(
        {
            "MASCOTRL_WAVE": "PICK",
            "MASCOTRL_SHARD_MANIFEST_URI": "s3://b/m.json",
            "MASCOTRL_PANEL_URI": "s3://b/panel.tar",
            "MASCOTRL_PANEL_SHA256": "abc",
            "MASCOTRL_OUT_URI": "s3://b/out/",
            "MASCOTRL_CONTAINER_DIGEST": "sha256:dead",
            "MASCOTRL_COMPUTE_HOST": "remote",
        }
    )
    assert len(env) == 7
