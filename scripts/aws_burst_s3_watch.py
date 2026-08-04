#!/usr/bin/env python3
"""Poll S3 for wave-root final artifacts until expected stems validate."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_remote_cell import validate_remote_cell
from scripts.validate_val_subset import HYBRID_STEM, validate_hybrid_error
from src.aws_burst.profiles import BURST_PROFILES
from src.aws_burst.waves import discover_wave_cells

PROFILE_ACCOUNT_IDS = {p["profile"]: p["account_id"] for p in BURST_PROFILES}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expected_stems(root: Path, wave: str) -> list[str]:
    manifest_path = root / f"deploy/aws_burst/config/wave_{wave}_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = payload.get("expected_cells")
        if expected:
            return [str(s) for s in expected]
        cells = payload.get("cells") or []
        if cells:
            return [Path(c).stem for c in cells]
    return [Path(c).stem for c in discover_wave_cells(root, wave)]


def count_wave_root_finals(
    wave: str,
    profiles: list[str] | None = None,
    *,
    root: Path | None = None,
    expected_stems: list[str] | None = None,
    validate: bool = True,
) -> dict:
    want_profiles = profiles or list(PROFILE_ACCOUNT_IDS)
    base = root or ROOT
    expected = list(expected_stems) if expected_stems is not None else _expected_stems(base, wave)
    want = set(expected)
    found_stems: set[str] = set()
    found: list[dict] = []
    errors: list[dict] = []
    extras: list[dict] = []
    invalid: list[dict] = []
    allowed_error_stems: set[str] = set()
    for p in want_profiles:
        acct = PROFILE_ACCOUNT_IDS[p]
        s3 = boto3.Session(profile_name=p, region_name="eu-central-1").client("s3")
        bucket = f"volsurf-burst-{acct}-artifacts"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{wave}/"):
            for obj in page.get("Contents") or []:
                key = str(obj["Key"])
                rel = key[len(wave) + 1 :]
                if "/" in rel:
                    continue
                if rel.endswith(".error.json"):
                    stem = Path(rel).name[: -len(".error.json")]
                    entry = {"profile": p, "key": key, "stem": stem}
                    if wave == "VAL" and stem == HYBRID_STEM:
                        try:
                            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                            err_doc = json.loads(body.decode("utf-8"))
                            if validate_hybrid_error(err_doc).get("ok"):
                                allowed_error_stems.add(stem)
                                continue
                        except Exception:
                            pass
                    # hybrid_heston NaN quarantine lifted: inactive-slot zero-fill in
                    # HistoricalArmEnv + Andersen QE-M Heston default (worlds.cpp).
                    # FEATNET edge quarantines (ablation-only; not headline evidence):
                    # LSTM+feature_net hits torch oneDNN "could not create a primitive"
                    # on all CPCV folds; GRU featnet succeeds. Heston+feature_net refuses
                    # without _universe_secids on the synthetic train panel (often wrapped
                    # as CalledProcessError in the cell_runner error.json message).
                    if wave == "FEATNET" and (
                        stem.endswith("_lstm_softmax_mean_std_cao_featnet")
                        or stem.endswith("_tw-heston_featnet")
                    ):
                        try:
                            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                            err_doc = json.loads(body.decode("utf-8"))
                            blob = " ".join(
                                str(err_doc.get(k) or "")
                                for k in ("message", "error", "reason", "traceback")
                            )
                            if (
                                "could not create a primitive" in blob
                                or "_universe_secids missing" in blob
                                or (
                                    stem.endswith("_tw-heston_featnet")
                                    and "CalledProcessError" in blob
                                )
                            ):
                                allowed_error_stems.add(stem)
                                continue
                        except Exception:
                            pass
                    errors.append(entry)
                    continue
                if (
                    not rel.endswith(".json")
                    or rel.endswith(".sha256")
                    or rel.endswith("_policy_behavior.json")
                ):
                    continue
                stem = Path(rel).stem
                entry = {
                    "profile": p,
                    "key": key,
                    "stem": stem,
                    "size": int(obj["Size"]),
                }
                if stem not in want:
                    extras.append(entry)
                    continue
                if validate:
                    try:
                        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                        art = json.loads(body.decode("utf-8"))
                        val = validate_remote_cell(art)
                        if not val.get("ok"):
                            invalid.append({**entry, "errors": val.get("errors")})
                            continue
                    except Exception as exc:  # noqa: BLE001
                        invalid.append({**entry, "errors": [str(exc)]})
                        continue
                found_stems.add(stem)
                found.append(entry)
    missing = sorted(want - found_stems - allowed_error_stems)
    complete = (
        len(missing) == 0
        and len(extras) == 0
        and len(invalid) == 0
        and len(errors) == 0
        and len(want) > 0
    )
    return {
        "polled_at": _utc(),
        "wave": wave,
        "expected_stems": expected,
        "n_expected": len(expected),
        "n_found": len(found_stems),
        "n_errors": len(errors),
        "n_allowed_errors": len(allowed_error_stems),
        "n_extras": len(extras),
        "n_invalid": len(invalid),
        "allowed_error_stems": sorted(allowed_error_stems),
        "found": found,
        "errors": errors,
        "extras": extras,
        "invalid": invalid,
        "missing": missing,
        "complete": complete,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wave")
    p.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Optional count check; defaults to manifest expected_cells length.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--timeout-seconds", type=float, default=14400.0)
    p.add_argument("--profiles", default="")
    args = p.parse_args(argv)
    profiles = [x.strip() for x in args.profiles.split(",") if x.strip()] or None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    expected_stems = _expected_stems(ROOT, args.wave)
    if args.expected is not None and int(args.expected) != len(expected_stems):
        print(
            f"watcher_expected_count_mismatch: --expected={args.expected} "
            f"manifest_n={len(expected_stems)}",
            flush=True,
        )
        return 3
    t0 = time.monotonic()
    while True:
        snap = count_wave_root_finals(
            args.wave, profiles, root=ROOT, expected_stems=expected_stems
        )
        snap["expected"] = len(expected_stems)
        args.out.write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "polled_at": snap["polled_at"],
                    "n_found": snap["n_found"],
                    "n_errors": snap["n_errors"],
                    "n_extras": snap["n_extras"],
                    "complete": snap["complete"],
                }
            ),
            flush=True,
        )
        if snap["complete"]:
            return 0
        if (time.monotonic() - t0) > float(args.timeout_seconds):
            print("timeout", flush=True)
            return 2
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
