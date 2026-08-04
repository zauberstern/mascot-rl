#!/usr/bin/env python3
"""AWS Batch array child: download manifest, run one spectrum cell, upload artifact."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RETRY = Config(
    retries={"mode": "standard", "max_attempts": 5},
    connect_timeout=10,
    read_timeout=60,
)
_S3 = None


def _parse_s3(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"expected s3 uri, got {uri!r}")
    return p.netloc, p.path.lstrip("/")


def _required_env() -> dict[str, str]:
    keys = (
        "MASCOTRL_WAVE",
        "MASCOTRL_SHARD_MANIFEST_URI",
        "MASCOTRL_PANEL_URI",
        "MASCOTRL_PANEL_SHA256",
        "MASCOTRL_OUT_URI",
        "MASCOTRL_CONTAINER_DIGEST",
        "MASCOTRL_COMPUTE_HOST",
    )
    out: dict[str, str] = {}
    missing = []
    for k in keys:
        v = str(os.environ.get(k) or "").strip()
        if not v:
            missing.append(k)
        else:
            out[k] = v
    if missing:
        raise SystemExit(f"cell_runner_env_missing: {missing}")
    # Array jobs set AWS_BATCH_JOB_ARRAY_INDEX. Non-array (size-1) submits
    # cannot inject that reserved name; use MASCOTRL_ARRAY_INDEX or default 0.
    idx = (
        str(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX") or "").strip()
        or str(os.environ.get("MASCOTRL_ARRAY_INDEX") or "").strip()
        or "0"
    )
    out["AWS_BATCH_JOB_ARRAY_INDEX"] = idx
    return out


def _s3_client():
    global _S3
    if _S3 is None:
        _S3 = boto3.client("s3", config=_RETRY)
    return _S3


def _download_file(bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _s3_client().download_file(bucket, key, str(dest))


def _upload_bytes(bucket: str, key: str, body: bytes, content_type: str | None = None) -> None:
    kwargs: dict = {"Bucket": bucket, "Key": key, "Body": body}
    if content_type:
        kwargs["ContentType"] = content_type
    _s3_client().put_object(**kwargs)


def _upload_file_with_sha(bucket: str, key: str, path: Path) -> str:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    _upload_bytes(bucket, key, data)
    if not str(key).endswith(".sha256"):
        _upload_bytes(bucket, f"{key}.sha256", (digest + "\n").encode("utf-8"), "text/plain")
    return digest


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract tar members under dest only (blocks path traversal)."""
    dest = dest.resolve()
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
        return
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise RuntimeError(f"tar_path_traversal_blocked: {member.name}")
        tar.extract(member, dest)


def _upload_error(
    env: dict[str, str],
    stem: str,
    exc: BaseException,
    *,
    reason: str | None = None,
) -> None:
    try:
        ob, oprefix = _parse_s3(env["MASCOTRL_OUT_URI"].rstrip("/") + "/")
        key = f"{oprefix}{stem}.error.json"
        parent_job = str(os.environ.get("AWS_BATCH_JOB_ID") or "").split(":")[0]
        n_attempts = 1
        prev: dict = {}
        try:
            obj = _s3_client().get_object(Bucket=ob, Key=key)
            prev = json.loads(obj["Body"].read().decode("utf-8"))
            prev_parent = str(prev.get("last_parent_job_id") or "")
            prev_n = int(prev.get("n_attempts") or prev.get("n_logical_attempts") or 0)
            # Batch Attempts:3 reuses the same parent job id; count once per submit.
            if parent_job and prev_parent and parent_job == prev_parent:
                n_attempts = max(prev_n, 1)
            else:
                n_attempts = prev_n + 1 if prev_n else 1
        except Exception:
            n_attempts = 1
        payload = {
            "error": type(exc).__name__,
            "message": str(exc),
            "reason": reason or getattr(exc, "reason", None) or type(exc).__name__,
            "traceback": traceback.format_exc(),
            "wave": env.get("MASCOTRL_WAVE"),
            "array_index": env.get("AWS_BATCH_JOB_ARRAY_INDEX"),
            "compute_host": env.get("MASCOTRL_COMPUTE_HOST"),
            "container_digest": env.get("MASCOTRL_CONTAINER_DIGEST"),
            "n_attempts": n_attempts,
            "n_logical_attempts": n_attempts,
            "last_parent_job_id": parent_job or prev.get("last_parent_job_id"),
        }
        _upload_bytes(
            ob,
            key,
            (json.dumps(payload, indent=2) + "\n").encode("utf-8"),
            "application/json",
        )
    except Exception as upload_exc:  # noqa: BLE001 - best-effort error upload
        print(f"error_upload_failed: {upload_exc}", file=sys.stderr)


class DigestMismatchError(RuntimeError):
    """Resume state was written under a different container digest."""

    reason = "digest_mismatch"


def resume_prefix(oprefix: str, stem: str) -> str:
    """S3 key prefix for per-cell CPCV/ckpt resume state."""
    base = oprefix.rstrip("/") + "/" if oprefix else ""
    return f"{base}resume/{stem}/"


def resume_digest_key(oprefix: str, stem: str) -> str:
    return f"{resume_prefix(oprefix, stem)}image_digest.json"


def assert_resume_digest_compatible(
    bucket: str,
    oprefix: str,
    stem: str,
    expected_digest: str,
) -> None:
    """Refuse resume when stored digest differs from the running container."""
    expected = str(expected_digest or "").strip()
    if not expected or expected == "unknown":
        raise DigestMismatchError(
            f"digest_mismatch: running digest invalid ({expected!r})"
        )
    key = resume_digest_key(oprefix, stem)
    try:
        obj = _s3_client().get_object(Bucket=bucket, Key=key)
        stored = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        # No prior marker: allow (fresh cell or pre-gate resume state).
        return
    got = str(stored.get("image_digest") or stored.get("container_digest") or "").strip()
    if got and got != expected and expected not in got and got not in expected:
        raise DigestMismatchError(
            f"digest_mismatch: stored={got!r} running={expected!r}"
        )


def write_resume_digest_marker(
    bucket: str, oprefix: str, stem: str, digest: str
) -> None:
    key = resume_digest_key(oprefix, stem)
    body = (
        json.dumps({"image_digest": digest, "container_digest": digest}, indent=2)
        + "\n"
    ).encode("utf-8")
    _upload_bytes(bucket, key, body, "application/json")


def artifact_missing_provenance(art: dict) -> str | None:
    """Return a reason string when a remote artifact must not be published."""
    if art.get("provenance_stamp_error"):
        return f"provenance_stamp_failure: {art.get('provenance_stamp_error')}"
    required = (
        "compute_host",
        "instance_type",
        "container_digest",
        "requirements_lock_sha256",
    )
    missing = [k for k in required if not art.get(k)]
    if missing:
        return f"provenance_incomplete: missing={missing}"
    if str(art.get("compute_host")) != "remote":
        return f"provenance_host_not_remote: {art.get('compute_host')!r}"
    digest = str(art.get("container_digest") or "").strip()
    if not digest or digest == "unknown":
        return f"provenance_digest_invalid: {digest!r}"
    return None


def assert_panel_manifest(extract_root: Path) -> None:
    """Fail closed unless panel_manifest.json exists and listed files match."""
    man_path = Path(extract_root) / "panel_manifest.json"
    if not man_path.is_file():
        raise RuntimeError("panel_manifest_missing")
    try:
        man = json.loads(man_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"panel_manifest_corrupt: {exc}") from exc
    files = man.get("files") or []
    if not files:
        raise RuntimeError("panel_manifest_empty_files")
    for row in files:
        arc = str(row.get("arcname") or "")
        if not arc:
            continue
        path = Path(extract_root) / arc
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"panel_manifest_file_missing: {arc}")
        expected = str(row.get("sha256") or "")
        if expected:
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            if got != expected:
                raise RuntimeError(f"panel_manifest_sha_mismatch: {arc}")


def is_strict_degraded_artifact(art: dict) -> bool:
    if bool(art.get("dry_run")) or bool(art.get("strict_degraded")):
        return True
    if art.get("feature_net_errors") or art.get("spectrum_seed_errors"):
        return True
    return False


def pull_resume_state(bucket: str, oprefix: str, stem: str, dest: Path) -> int:
    """Download existing cpcv/ + ckpt/ resume objects into dest. Returns file count."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    prefix = resume_prefix(oprefix, stem)
    s3 = _s3_client()
    written = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents") or []:
            key = str(item["Key"])
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(out))
            written += 1
    return written


def push_resume_state(bucket: str, oprefix: str, stem: str, local_out: Path) -> int:
    """Upload local cpcv/ + ckpt/ trees under resume/{stem}/. Best-effort."""
    local_out = Path(local_out)
    prefix = resume_prefix(oprefix, stem)
    uploaded = 0
    for sub in ("cpcv", "ckpt"):
        root = local_out / sub
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_out).as_posix()
            key = f"{prefix}{rel}"
            try:
                _upload_file_with_sha(bucket, key, path)
                uploaded += 1
            except Exception as exc:  # noqa: BLE001
                print(f"resume_push_failed: {key}: {exc}", file=sys.stderr)
    return uploaded


def main() -> int:
    env = _required_env()
    idx = int(env["AWS_BATCH_JOB_ARRAY_INDEX"])
    stem = "unknown"
    try:
        mb, mk = _parse_s3(env["MASCOTRL_SHARD_MANIFEST_URI"])
        manifest = json.loads(_s3_client().get_object(Bucket=mb, Key=mk)["Body"].read())
        cells = manifest.get("cells") or []
        if idx >= len(cells):
            print(
                f"array_index_out_of_range: idx={idx} n_cells={len(cells)}",
                file=sys.stderr,
            )
            return 3
        cell_rel = str(cells[idx])
        stem = Path(cell_rel).stem
        cell_path = ROOT / cell_rel
        if not cell_path.is_file():
            raise SystemExit(f"cell_yaml_missing: {cell_rel}")
        print(f"phase=cell_start stem={stem} idx={idx}", flush=True)

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            panel_tar = tdir / "panel_bundle.tar"
            pb, pk = _parse_s3(env["MASCOTRL_PANEL_URI"])
            print("phase=panel_download start", flush=True)
            _download_file(pb, pk, panel_tar)
            got = hashlib.sha256(panel_tar.read_bytes()).hexdigest()
            if got != env["MASCOTRL_PANEL_SHA256"]:
                print(f"panel_bundle_hash_mismatch: got={got}", file=sys.stderr)
                return 2
            extract = tdir / "panel"
            extract.mkdir()
            with tarfile.open(panel_tar, "r") as tar:
                _safe_extractall(tar, extract)
            assert_panel_manifest(extract)
            print("phase=panel_download done", flush=True)
            os.environ["MASCOTRL_ARCTIC_DIR"] = str(extract / "volsurf_arcticdb")
            os.environ["MASCOTRL_LAKE_BASE"] = str(extract)
            # Alias: LAKE_ROOT reads MASCOTRL_LAKE_DIR (parity with H0).
            os.environ["MASCOTRL_LAKE_DIR"] = str(extract)
            os.environ["MASCOTRL_COMPUTE_HOST"] = env["MASCOTRL_COMPUTE_HOST"]
            os.environ["MASCOTRL_CONTAINER_DIGEST"] = env["MASCOTRL_CONTAINER_DIGEST"]

            out_dir = tdir / "out"
            out_dir.mkdir()
            # Pull prior CPCV manifest + checkpoints so resume=True finds them.
            ob, oprefix = _parse_s3(env["MASCOTRL_OUT_URI"].rstrip("/") + "/")
            try:
                assert_resume_digest_compatible(
                    ob, oprefix, stem, env["MASCOTRL_CONTAINER_DIGEST"]
                )
                n_pulled = pull_resume_state(ob, oprefix, stem, out_dir)
                if n_pulled:
                    print(f"resume_state_pulled: n={n_pulled} stem={stem}", file=sys.stderr)
            except DigestMismatchError as dig_exc:
                _upload_error(env, stem, dig_exc, reason="digest_mismatch")
                print(f"digest_mismatch_refused: stem={stem} {dig_exc}", file=sys.stderr)
                return 5
            except Exception as pull_exc:  # noqa: BLE001
                print(f"resume_pull_failed: {pull_exc}", file=sys.stderr)

            cmd = [
                sys.executable,
                str(ROOT / "scripts/run_spectrum_campaign.py"),
                "--config-dir",
                str(cell_path.parent),
                "--config-glob",
                cell_path.name,
                "--no-dry-run",
                "--strict",
                "--out-dir",
                str(out_dir),
            ]
            print(f"phase=train start stem={stem}", flush=True)
            # Cap BLAS/thread arenas on 2-vCPU m7i-flex.large to avoid fold-transition OOM.
            os.environ.setdefault("TORCH_NUM_THREADS", "1")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MASCOTRL_THREADS_PER_WORKER", "1")
            try:
                subprocess.check_call(cmd, cwd=str(ROOT))
            finally:
                # Always push resume state (even on crash) so the next attempt
                # can continue mid-fold / mid-campaign.
                try:
                    write_resume_digest_marker(
                        ob, oprefix, stem, env["MASCOTRL_CONTAINER_DIGEST"]
                    )
                    n_pushed = push_resume_state(ob, oprefix, stem, out_dir)
                    if n_pushed:
                        print(f"resume_state_pushed: n={n_pushed} stem={stem}", file=sys.stderr)
                except Exception as push_exc:  # noqa: BLE001
                    print(f"resume_push_failed: {push_exc}", file=sys.stderr)

            artifacts = [
                art
                for art in out_dir.glob("*.json")
                if art.name != "index.json" and not art.name.endswith("_policy_behavior.json")
            ]
            # Prefer the cell-stem artifact; fall back to any non-index JSON.
            preferred = out_dir / f"{stem}.json"
            if preferred.is_file():
                artifacts = [preferred]
            if not artifacts:
                raise RuntimeError(f"cell_artifact_missing: expected {stem}.json")
            art = artifacts[0]
            if art.stat().st_size <= 0:
                raise RuntimeError(f"cell_artifact_empty: {art.name}")

            try:
                art_payload = json.loads(art.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"cell_artifact_json_error: {exc}") from exc
            if is_strict_degraded_artifact(art_payload):
                # Do not publish as a final stem.json (would look "complete").
                reason = (
                    f"strict_degraded: dry_run={art_payload.get('dry_run')} "
                    f"fallback={art_payload.get('fallback_reason')} "
                    f"feature_net_errors={art_payload.get('feature_net_errors')} "
                    f"spectrum_seed_errors={art_payload.get('spectrum_seed_errors')}"
                )
                _upload_error(env, stem, RuntimeError(reason))
                print(f"strict_degraded_refused: stem={stem} {reason}", file=sys.stderr)
                return 4

            stamp_fail = artifact_missing_provenance(art_payload)
            if stamp_fail:
                # Never upload a final {stem}.json on provenance failure.
                _upload_error(env, stem, RuntimeError(stamp_fail), reason="provenance_stamp_failure")
                print(
                    f"provenance_stamp_refused: stem={stem} {stamp_fail}",
                    file=sys.stderr,
                )
                return 4

            # Stamp panel-bundle fingerprint so pull provenance accepts remote cells.
            # Prefer env (submit-time panel_bundle.sha256). The copy of
            # panel_manifest.json inside the tar is written before the tar digest
            # exists, so it often lacks sha256.
            fp = str(env.get("MASCOTRL_PANEL_SHA256") or "").strip()
            if not fp:
                manifest_path = extract / "panel_manifest.json"
                if manifest_path.is_file():
                    try:
                        man = json.loads(manifest_path.read_text(encoding="utf-8"))
                        fp = str(man.get("sha256") or "").strip()
                    except Exception as fp_exc:  # noqa: BLE001
                        print(f"fingerprint_stamp_failed: {fp_exc}", file=sys.stderr)
            if fp:
                art_payload.setdefault("universe_fingerprint", fp)
                art_payload.setdefault(
                    "universe_fingerprint_kind", "panel_bundle_sha256"
                )
                art.write_text(
                    json.dumps(art_payload, indent=2) + "\n", encoding="utf-8"
                )
            else:
                print("fingerprint_stamp_missing: no panel sha available", file=sys.stderr)

            _upload_file_with_sha(ob, f"{oprefix}{art.name}", art)
            # Also upload companion behavior / training files if present.
            for companion in out_dir.glob(f"{stem}_*"):
                if companion.suffix in {".json", ".jsonl"}:
                    _upload_file_with_sha(ob, f"{oprefix}{companion.name}", companion)
            # Clear stale error marker so the S3 watcher reaches complete.
            try:
                _s3_client().delete_object(Bucket=ob, Key=f"{oprefix}{stem}.error.json")
            except Exception as del_exc:  # noqa: BLE001
                print(f"error_marker_cleanup_failed: {del_exc}", file=sys.stderr)
            print(f"phase=artifact_upload done stem={stem}", flush=True)
        return 0
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or (
            isinstance(exc, SystemExit) and exc.code not in (0, None)
        ):
            _upload_error(env, stem, exc if not isinstance(exc, SystemExit) else RuntimeError(str(exc)))
        if isinstance(exc, SystemExit):
            raise
        print(f"cell_runner_failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
