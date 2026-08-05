"""N-account Frankfurt burst profiles (no secrets)."""
from __future__ import annotations

import json
from pathlib import Path

REGION = "eu-central-1"

# Production waves must never use --offline (live spend + CE preflight required).
PRODUCTION_WAVES = frozenset(
    {
        "PICK",
        "PICK2",
        "PICK_SMOKE",
        "PICK_CANARY",
        "VAL",
        "RC6",
        "RC6_CANARY",
        "RC6_HEADS",
        "RC6_HAPPO",
    }
)
# Explicitly named waves allowed to skip live Budgets/CE under --offline.
OFFLINE_ALLOWED_WAVES = frozenset({"CAL", "H0"})

# Placeholder burst profiles for public extract (no real account ids or emails).
BURST_PROFILES: tuple[dict[str, str], ...] = (
    {
        "profile": "volsurf-burst-1",
        "email": "burst-1@example.invalid",
        "account_id": "000000000001",
        "shard": "0",
    },
    {
        "profile": "volsurf-burst-2",
        "email": "burst-2@example.invalid",
        "account_id": "000000000002",
        "shard": "1",
    },
    {
        "profile": "volsurf-burst-3",
        "email": "burst-3@example.invalid",
        "account_id": "000000000003",
        "shard": "2",
    },
    {
        "profile": "volsurf-burst-4",
        "email": "burst-4@example.invalid",
        "account_id": "000000000004",
        "shard": "3",
    },
)

SPOT_VCPU_QUOTA_CODE = "L-34B43A08"
SPOT_VCPU_REQUEST = 64
# User premise: $200 credit / account, 0.90 coded spend fraction -> $180
# governor / account. Fleet cap scales as $180 * N armed accounts.
BUDGET_USD = 180.0
CREDIT_USD = 200.0
SPEND_CAP_FRAC = 0.90
MAX_VCPUS_PER_ACCOUNT = 32

# Default free-tier shape (reproduced when account_shape.json absent or profile unlisted).
DEFAULT_ACCOUNT_SHAPE: dict[str, object] = {
    "max_vcpus": 32,
    "allowed_instance_types": ["m7i-flex.large"],
    "job_memory_mib": 6912,
    "himem_job_memory_mib": 6912,
    "instance_mem_mib": 8192,
    "credit_usd": 200.0,
}
FREE_PLAN_INSTANCE_TYPES = frozenset({"m7i-flex.large"})


def account_shape_path(root: Path) -> Path:
    return root / "deploy" / "aws_burst" / "config" / "account_shape.json"


def load_account_shapes(root: Path | None = None) -> dict[str, dict[str, object]]:
    """Per-profile CE/job/governor shape overrides."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    path = account_shape_path(base)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise ValueError("account_shape profiles must be an object")
    return {str(k): dict(v) for k, v in profiles.items() if isinstance(v, dict)}


def profile_shape(root: Path | None, profile: str) -> dict[str, object]:
    """Merged shape for one profile (defaults + overrides)."""
    shape = dict(DEFAULT_ACCOUNT_SHAPE)
    overrides = load_account_shapes(root).get(profile) or {}
    shape.update(overrides)
    types = shape.get("allowed_instance_types") or ["m7i-flex.large"]
    shape["allowed_instance_types"] = [str(t) for t in types]
    return shape


def max_vcpus_for_profile(root: Path | None, profile: str) -> int:
    return int(profile_shape(root, profile)["max_vcpus"])


def credit_usd_for_profile(root: Path | None, profile: str) -> float:
    return float(profile_shape(root, profile)["credit_usd"])


def allowed_instance_types_for_profile(root: Path | None, profile: str) -> frozenset[str]:
    return frozenset(profile_shape(root, profile)["allowed_instance_types"])


def job_memory_mib_for_profile(root: Path | None, profile: str, *, himem: bool = False) -> int:
    shape = profile_shape(root, profile)
    if himem:
        return int(shape.get("himem_job_memory_mib") or shape["job_memory_mib"])
    return int(shape["job_memory_mib"])


def instance_mem_mib_for_profile(root: Path | None, profile: str) -> int:
    return int(profile_shape(root, profile)["instance_mem_mib"])


def panel_bucket(account_id: str) -> str:
    return f"volsurf-burst-{account_id}-panels"


def artifact_bucket(account_id: str) -> str:
    return f"volsurf-burst-{account_id}-artifacts"


def _validate_armed_payload(profile: str, payload: dict) -> None:
    """Fail closed unless the armed stamp is verified with an action_id."""
    if not bool(payload.get("verified")):
        raise ValueError(f"armed_not_verified: profile={profile}")
    action_id = str(payload.get("action_id") or "").strip()
    if not action_id:
        raise ValueError(f"armed_missing_action_id: profile={profile}")
    if not (
        bool(payload.get("armed"))
        or bool(payload.get("budget_action_armed"))
    ):
        raise ValueError(f"armed_flag_false: profile={profile}")


def armed_profiles(root: Path) -> list[dict[str, str]]:
    """Return profiles with a validated budget_armed_{profile}.json (all 3)."""
    cfg_dir = root / "deploy" / "aws_burst" / "config"
    armed: list[dict[str, str]] = []
    for p in BURST_PROFILES:
        path = cfg_dir / f"budget_armed_{p['profile']}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"armed_corrupt: profile={p['profile']}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"armed_not_object: profile={p['profile']}")
        _validate_armed_payload(p["profile"], payload)
        armed.append(dict(p))
    if len(armed) != len(BURST_PROFILES):
        raise ValueError(
            f"incomplete_armed_profiles: need all {len(BURST_PROFILES)}, got {len(armed)}"
        )
    return armed


def assert_offline_allowed(wave: str) -> None:
    """Refuse --offline for production waves; allow only named test waves."""
    if wave in PRODUCTION_WAVES:
        raise ValueError(
            f"offline_refused_for_production_wave: {wave} "
            f"(allowed={sorted(OFFLINE_ALLOWED_WAVES)})"
        )
    if wave not in OFFLINE_ALLOWED_WAVES:
        raise ValueError(
            f"offline_refused_unnamed_wave: {wave} "
            f"(allowed={sorted(OFFLINE_ALLOWED_WAVES)})"
        )
