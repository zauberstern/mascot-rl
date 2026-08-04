#!/usr/bin/env python3
"""Validate a remotely produced spectrum cell artifact before promotion.

Hard-rejects mismatched fingerprints and missing provenance fields.
Accepts ``universe_fingerprint`` (+ kind) when CRUCIBLE is not applicable
(dyn_hrp / DII spines) without inventing a fake crucible_fingerprint (WP-R2).
Does not modify deploy/aws_free_tier.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_REMOTE_FIELDS = (
    "compute_host",
    "instance_type",
    "container_digest",
    "requirements_lock_sha256",
)


def validate_remote_cell(
    artifact: dict[str, Any],
    *,
    expected_fingerprint: str | None = None,
    expected_universe_fingerprint: str | None = None,
    expected_container_digest: str | None = None,
    allow_universe_fingerprint: bool = True,
) -> dict[str, Any]:
    """Return ``{ok, errors, artifact}``. Rejects on any hard failure."""
    errors: list[str] = []
    host = artifact.get("compute_host")
    if host != "remote":
        errors.append(f"compute_host must be 'remote' (got {host!r})")
    for field in REQUIRED_REMOTE_FIELDS:
        if field == "compute_host":
            continue
        if not artifact.get(field):
            errors.append(f"missing required provenance field: {field}")

    got_crucible = str(artifact.get("crucible_fingerprint") or "")
    got_universe = str(artifact.get("universe_fingerprint") or "")
    kind = str(artifact.get("universe_fingerprint_kind") or "")
    got_digest = str(artifact.get("container_digest") or "").strip()

    exp_c = str(expected_fingerprint or "").strip()
    exp_u = str(expected_universe_fingerprint or "").strip()
    exp_d = str(expected_container_digest or "").strip()
    if exp_d:
        if not got_digest:
            errors.append("missing required provenance field: container_digest")
        elif got_digest != exp_d and not got_digest.endswith(exp_d):
            # Accept either bare sha256:... or repo@sha256:...
            if exp_d not in got_digest:
                errors.append(
                    f"container_digest mismatch: got={got_digest!r} expected={exp_d!r}"
                )

    if exp_c:
        if not got_crucible:
            errors.append("missing required provenance field: crucible_fingerprint")
        elif got_crucible != exp_c:
            errors.append(
                f"crucible_fingerprint mismatch: got={got_crucible!r} expected={exp_c!r}"
            )
    elif exp_u:
        if not allow_universe_fingerprint:
            errors.append("universe_fingerprint not allowed for this validation mode")
        elif not got_universe:
            errors.append("missing required provenance field: universe_fingerprint")
        elif got_universe != exp_u:
            errors.append(
                f"universe_fingerprint mismatch: got={got_universe!r} expected={exp_u!r}"
            )
        elif not kind:
            errors.append("universe_fingerprint_kind required with universe_fingerprint")
    else:
        # Need at least one fingerprint family present on the artifact.
        if not got_crucible and not (got_universe and kind):
            errors.append(
                "expected_fingerprint is empty and artifact lacks "
                "crucible_fingerprint or universe_fingerprint+kind"
            )

    ok = not errors
    return {"ok": ok, "errors": errors, "artifact": artifact if ok else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--expected-fingerprint", type=str, default="")
    p.add_argument("--expected-universe-fingerprint", type=str, default="")
    args = p.parse_args()
    art = json.loads(args.artifact.read_text(encoding="utf-8"))
    result = validate_remote_cell(
        art,
        expected_fingerprint=args.expected_fingerprint or None,
        expected_universe_fingerprint=args.expected_universe_fingerprint or None,
    )
    if not result["ok"]:
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    print(json.dumps({"ok": True, "errors": []}, indent=2))


if __name__ == "__main__":
    main()
