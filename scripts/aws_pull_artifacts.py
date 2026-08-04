#!/usr/bin/env python3
"""Pull and validate remote spectrum artifacts from S3."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_remote_cell import validate_remote_cell
from src.aws_burst.aws_client import BurstClient
from src.aws_burst.image_digest import load_digest_record
from src.aws_burst.profiles import REGION, armed_profiles, artifact_bucket
from src.aws_burst.waves import WAVES, discover_wave_cells
from src.eval.universe_fingerprint import EQ_BURST_WAVES, read_panel_bundle_sha256
from src.reporting.policy_behavior import validate_policy_behavior_payload


def _default_dest(wave: str) -> str:
    spec = WAVES.get(wave)
    if spec is None:
        raise ValueError(f"unknown wave {wave!r}")
    return f"logs/artifacts/spectrum/{spec.out_subdir}"


def _expected_stems(root: Path, wave: str) -> list[str]:
    manifest_path = root / f"deploy/aws_burst/config/wave_{wave}_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = payload.get("expected_cells")
        if expected:
            return [str(s) for s in expected]
        cells = (payload.get("cells") or [])
        if cells:
            return [Path(c).stem for c in cells]
    cells = discover_wave_cells(root, wave)
    return [Path(c).stem for c in cells]


def _panel_fingerprint(root: Path) -> str | None:
    return read_panel_bundle_sha256(root)


def resolve_expected_universe_fingerprint(
    root: Path,
    wave: str,
    *,
    cli_fingerprint: str = "",
    allow_missing: bool = False,
) -> str | None:
    """Resolve universe fingerprint for pull validation (C-2).

    Equity burst waves require an explicit ``--expected-universe-fingerprint``
    or the on-disk panel bundle sha256 unless ``allow_missing`` is set.
    """
    explicit = str(cli_fingerprint or "").strip()
    if explicit:
        return explicit
    if wave in EQ_BURST_WAVES and not allow_missing:
        panel_fp = _panel_fingerprint(root)
        if not panel_fp:
            raise ValueError(
                f"wave {wave!r} requires --expected-universe-fingerprint "
                "(or logs/aws_burst_panel_bundle/panel_bundle.sha256 on disk); "
                "pass --allow-missing-universe-fingerprint only for offline smoke"
            )
        return panel_fp
    return _panel_fingerprint(root)


def artifact_is_strict_degraded(art: dict) -> bool:
    """True when an artifact must not count as a complete promotable cell."""
    if bool(art.get("dry_run")):
        return True
    if bool(art.get("strict_degraded")):
        return True
    if art.get("feature_net_errors"):
        return True
    if art.get("spectrum_seed_errors"):
        return True
    return False


def pull_wave(
    root: Path,
    wave: str,
    *,
    dest: Path | None = None,
    expected_universe_fingerprint: str | None = None,
    expected_container_digest: str | None = None,
    require_complete: bool = True,
    profiles: list[str] | None = None,
    allow_digest_drift: bool = False,
) -> dict:
    dest = dest or (root / _default_dest(wave))
    dest.mkdir(parents=True, exist_ok=True)
    rejected_dir = dest / "_rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[str] = []
    rejected: list[dict] = []
    accepted_stems: set[str] = set()
    prefix = f"{wave}/"
    expected = _expected_stems(root, wave)
    panel_fp = expected_universe_fingerprint or _panel_fingerprint(root)

    armed = armed_profiles(root)
    if profiles:
        want = set(profiles)
        armed = [p for p in armed if p["profile"] in want]
        if not armed:
            raise ValueError(f"no_armed_profiles_match: {sorted(want)}")

    for profile_info in armed:
        profile = profile_info["profile"]
        digest = str(expected_container_digest or "").strip()
        if allow_digest_drift:
            digest = ""
        elif not digest:
            try:
                digest = load_digest_record(root, profile)["digest"]
            except ValueError:
                digest = ""
        client = BurstClient(profile, REGION)
        bucket = artifact_bucket(client.account_id())
        tmp = dest / "_pull_tmp" / profile
        # Skip resume ckpts and archives: pull only accepts wave-root *.json finals.
        client.download_prefix(
            bucket,
            prefix,
            tmp,
            skip_rel_prefixes=("resume/", "_archive", "_archive/"),
        )
        for art_path in tmp.rglob("*.json"):
            if art_path.name.endswith(".sha256") or art_path.name == "index.json":
                continue
            # Only wave-root finals count; ignore resume/ and _archive_*/ trees.
            try:
                rel = art_path.relative_to(tmp)
            except ValueError:
                continue
            if len(rel.parts) != 1:
                continue
            if art_path.name.endswith(".error.json"):
                rejected.append({"path": str(art_path), "reason": "cell_error_artifact"})
                (rejected_dir / art_path.name).write_bytes(art_path.read_bytes())
                continue

            sha_path = Path(str(art_path) + ".sha256")
            if not sha_path.is_file():
                # also accept stem.json.sha256 via with_suffix style
                alt = art_path.with_suffix(art_path.suffix + ".sha256")
                sha_path = alt if alt.is_file() else sha_path
            if not sha_path.is_file():
                rejected.append({"path": str(art_path), "reason": "missing_sha256_sidecar"})
                (rejected_dir / art_path.name).write_bytes(art_path.read_bytes())
                continue
            expected_sha = sha_path.read_text(encoding="utf-8").strip()
            got = hashlib.sha256(art_path.read_bytes()).hexdigest()
            if got != expected_sha:
                rejected.append({"path": str(art_path), "reason": "sha256_mismatch"})
                continue

            if art_path.name.endswith("_policy_behavior.json"):
                try:
                    beh = json.loads(art_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    rejected.append(
                        {"path": str(art_path), "reason": f"behavior_json_error: {exc}"}
                    )
                    continue
                beh_val = validate_policy_behavior_payload(beh)
                if not beh_val.get("ok"):
                    rejected.append(
                        {"path": str(art_path), "reason": beh_val.get("errors")}
                    )
                    (rejected_dir / art_path.name).write_bytes(art_path.read_bytes())
                    continue
                out = dest / art_path.name
                out.write_bytes(art_path.read_bytes())
                accepted.append(str(out))
                continue

            try:
                art = json.loads(art_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                rejected.append({"path": str(art_path), "reason": f"json_error: {exc}"})
                continue
            if art.get("dry_run"):
                rejected.append({"path": str(art_path), "reason": "dry_run"})
                continue
            if art.get("feature_net_errors"):
                rejected.append({"path": str(art_path), "reason": "feature_net_errors"})
                (rejected_dir / art_path.name).write_bytes(art_path.read_bytes())
                continue
            if art.get("spectrum_seed_errors"):
                rejected.append({"path": str(art_path), "reason": "spectrum_seed_errors"})
                (rejected_dir / art_path.name).write_bytes(art_path.read_bytes())
                continue
            if art.get("strict_degraded"):
                rejected.append({"path": str(art_path), "reason": "strict_degraded"})
                (rejected_dir / art_path.name).write_bytes(art_path.read_bytes())
                continue
            if art.get("toy_panel"):
                rejected.append({"path": str(art_path), "reason": "toy_panel"})
                continue
            val = validate_remote_cell(
                art,
                expected_universe_fingerprint=panel_fp,
                expected_container_digest=digest or None,
            )
            if not val.get("ok"):
                rejected.append({"path": str(art_path), "reason": val.get("errors")})
                (rejected_dir / art_path.name).write_bytes(art_path.read_bytes())
                continue
            out = dest / art_path.name
            out.write_bytes(art_path.read_bytes())
            accepted.append(str(out))
            accepted_stems.add(Path(art_path.name).stem)
            # Promote decision-trace sidecar when present (logging / audit path).
            stem = Path(art_path.name).stem
            for side in (
                f"{stem}_decision_trace.jsonl",
                f"{stem}_decision_trace.jsonl.sha256",
                f"{stem}_policy_behavior.json",
                f"{stem}_policy_behavior.json.sha256",
                f"{art_path.name}.sha256",
            ):
                src_side = tmp / side
                if src_side.is_file():
                    (dest / side).write_bytes(src_side.read_bytes())

    missing = sorted(set(expected) - accepted_stems)
    index = {
        "wave": wave,
        "accepted": accepted,
        "rejected": rejected,
        "n_accepted": len(accepted_stems),
        "n_rejected": len(rejected),
        "n_expected": len(expected),
        "missing_cells": missing,
        "complete": len(missing) == 0,
    }
    (dest / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    if require_complete and missing:
        raise SystemExit(
            f"pull_incomplete: accepted={len(accepted_stems)} expected={len(expected)} "
            f"missing={missing[:20]}{'...' if len(missing) > 20 else ''}"
        )
    return index


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wave", required=True)
    p.add_argument("--dest", type=Path, default=None)
    p.add_argument("--expected-universe-fingerprint", default="")
    p.add_argument("--expected-container-digest", default="")
    p.add_argument(
        "--allow-missing-universe-fingerprint",
        action="store_true",
        help=(
            "Skip the eq-wave requirement for an explicit universe fingerprint "
            "(offline smoke only; do not use for production pulls)."
        ),
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Do not exit non-zero when accepted < expected (partial pull).",
    )
    p.add_argument(
        "--allow-digest-drift",
        action="store_true",
        help=(
            "Do not require artifacts to match the current image pin "
            "(use when pulling sealed historical waves after a later rebuild)."
        ),
    )
    p.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile filter (e.g. volsurf-burst-1 for smoke).",
    )
    args = p.parse_args(argv)
    try:
        panel_fp = resolve_expected_universe_fingerprint(
            ROOT,
            args.wave,
            cli_fingerprint=args.expected_universe_fingerprint,
            allow_missing=bool(args.allow_missing_universe_fingerprint),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    profiles = [x.strip() for x in str(args.profiles).split(",") if x.strip()] or None
    result = pull_wave(
        ROOT,
        args.wave,
        dest=args.dest,
        expected_universe_fingerprint=panel_fp,
        expected_container_digest=args.expected_container_digest or None,
        require_complete=not bool(args.allow_incomplete),
        profiles=profiles,
        allow_digest_drift=bool(args.allow_digest_drift),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
