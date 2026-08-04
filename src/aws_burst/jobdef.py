"""AWS Batch job definition builders for burst cell runs."""
from __future__ import annotations

from typing import Any

# Free-plan Spot CE: m7i-flex.large only (8 GiB). ECS agent + Docker reserve
# roughly 1.5 GiB; refuse job defs that would OOM-kill at launch.
DEFAULT_INSTANCE_MEM_MIB = 8192
DEFAULT_ECS_OVERHEAD_MIB = 1280


def assert_job_memory_fits_instance(
    memory_mb: int,
    *,
    instance_mem_mib: int = DEFAULT_INSTANCE_MEM_MIB,
    ecs_overhead_mib: int = DEFAULT_ECS_OVERHEAD_MIB,
) -> None:
    """Refuse when job memory exceeds (instance_mem - ECS overhead)."""
    budget = int(instance_mem_mib) - int(ecs_overhead_mib)
    if int(memory_mb) > budget:
        raise ValueError(
            f"job_memory_exceeds_instance: memory={memory_mb}MiB > "
            f"budget={budget}MiB (instance={instance_mem_mib}MiB "
            f"ecs_overhead={ecs_overhead_mib}MiB); keep Free-plan Spot on "
            f"m7i-flex.large or lower job Memory"
        )


def build_retry_strategy(*, attempts: int = 3) -> dict[str, Any]:
    return {
        "attempts": int(attempts),
        "evaluateOnExit": [
            {"onStatusReason": "Host EC2*", "action": "RETRY"},
            {"onReason": "*", "action": "EXIT"},
        ],
    }


def build_job_definition(
    *,
    image_uri: str,
    job_role_arn: str,
    vcpus: int = 1,
    memory_mb: int = 6912,
    timeout_s: int = 43200,
    attempts: int = 3,
    instance_mem_mib: int = DEFAULT_INSTANCE_MEM_MIB,
    ecs_overhead_mib: int = DEFAULT_ECS_OVERHEAD_MIB,
) -> dict[str, Any]:
    assert_job_memory_fits_instance(
        int(memory_mb),
        instance_mem_mib=int(instance_mem_mib),
        ecs_overhead_mib=int(ecs_overhead_mib),
    )
    return {
        "type": "container",
        "containerProperties": {
            "image": image_uri,
            "vcpus": int(vcpus),
            "memory": int(memory_mb),
            "jobRoleArn": job_role_arn,
            "command": ["python", "deploy/aws_burst/docker/cell_runner.py"],
            "environment": [],
        },
        "retryStrategy": build_retry_strategy(attempts=attempts),
        "timeout": {"attemptDurationSeconds": int(timeout_s)},
    }


def build_container_env(env: dict[str, str]) -> list[dict[str, str]]:
    required = (
        "MASCOTRL_WAVE",
        "MASCOTRL_SHARD_MANIFEST_URI",
        "MASCOTRL_PANEL_URI",
        "MASCOTRL_PANEL_SHA256",
        "MASCOTRL_OUT_URI",
        "MASCOTRL_CONTAINER_DIGEST",
        "MASCOTRL_COMPUTE_HOST",
    )
    missing = [k for k in required if not str(env.get(k) or "").strip()]
    if missing:
        raise ValueError(f"container_env_incomplete: missing {missing}")
    return [{"name": k, "value": str(env[k])} for k in required]
