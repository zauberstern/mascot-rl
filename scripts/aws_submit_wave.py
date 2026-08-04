#!/usr/bin/env python3
"""Submit a burst wave as AWS Batch array jobs (idempotent, dry-run capable)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aws_burst.aws_client import BurstClient
from src.aws_burst.governor import check_submit_allowed, projected_wave_cost
from src.aws_burst.image_digest import load_digest_record, pinned_image_uri
from src.aws_burst.job_routing import (
    JOB_DEFINITION_HIMEM,
    JOB_DEFINITION_HIMEM56,
    JOB_DEFINITION_STANDARD,
    cell_requires_himem,
    partition_cells_for_submit,
)
from src.aws_burst.jobdef import (
    DEFAULT_ECS_OVERHEAD_MIB,
    DEFAULT_INSTANCE_MEM_MIB,
    assert_job_memory_fits_instance,
    build_container_env,
)
from src.aws_burst.manifest import build_manifest, manifest_sha256
from src.aws_burst.profiles import (
    MAX_VCPUS_PER_ACCOUNT,
    REGION,
    allowed_instance_types_for_profile,
    armed_profiles,
    artifact_bucket,
    assert_offline_allowed,
    instance_mem_mib_for_profile,
    job_memory_mib_for_profile,
    max_vcpus_for_profile,
    panel_bucket,
    profile_shape,
)
from src.aws_burst.waves import WAVES, _local_equivalent_for, discover_wave_cells, shard_cells

COMPUTE_ENV_NAME = "volsurf-burst-spot"  # logical; match by JobQueue CE order
JOB_DEFINITION_NAME = JOB_DEFINITION_STANDARD
# Legacy alias; per-profile shapes live in account_shape.json.
FREE_PLAN_INSTANCE_TYPES = frozenset({"m7i-flex.large"})
# Logical submit attempts (Batch may retry 3x per submit; count once per parent job).
DEFAULT_MAX_ERROR_RETRIES = 3
# Campaign hard stop Mon 31 Aug 2026 18:00 CEST == 16:00 UTC.
FIGURE_DEADLINE_UTC = datetime(2026, 8, 31, 16, 0, 0, tzinfo=timezone.utc)
# Cap any new attempt below a 48h fantasy timeout; wall is the scarce resource.
MAX_ATTEMPT_SECONDS = 21 * 3600
DEFAULT_WALL_BUFFER_SECONDS = 20 * 60


def remaining_wall_attempt_seconds(
    *,
    now: datetime | None = None,
    deadline_utc: datetime = FIGURE_DEADLINE_UTC,
    buffer_seconds: int = DEFAULT_WALL_BUFFER_SECONDS,
    max_seconds: int = MAX_ATTEMPT_SECONDS,
) -> int:
    """Seconds until campaign deadline minus buffer, capped (never 48h)."""
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    remaining = int((deadline_utc - clock).total_seconds()) - int(buffer_seconds)
    if remaining <= 0:
        raise ValueError(
            f"deadline_passed: now={clock.isoformat()} deadline={deadline_utc.isoformat()}"
        )
    return min(int(remaining), int(max_seconds))


def job_definition_with_revision(job_definition: str, revision: int | None) -> str:
    """Pin a Batch job-definition name to an ACTIVE revision when requested."""
    name = str(job_definition)
    if revision is None:
        return name
    rev = int(revision)
    if rev <= 0:
        raise ValueError(f"job_definition_revision_invalid: {rev}")
    # Caller may already pass name:rev; do not double-suffix.
    if ":" in name:
        return name
    return f"{name}:{rev}"



def resolve_job_definition_revision_for_digest(
    client: BurstClient,
    job_definition: str,
    digest: str,
) -> int:
    """Return highest ACTIVE revision whose image URI embeds ``digest``."""
    want = str(digest or "").strip()
    if not want:
        raise ValueError("pin_digest_empty")
    bare = want.split(":")[-1]
    batch = client._batch()  # noqa: SLF001
    name = job_definition.split(":")[0]
    defs = batch.describe_job_definitions(jobDefinitionName=name, status="ACTIVE").get(
        "jobDefinitions"
    ) or []
    defs = sorted(defs, key=lambda d: int(d.get("revision") or 0), reverse=True)
    for d in defs:
        img = str((d.get("containerProperties") or {}).get("image") or "")
        if want in img or bare in img:
            return int(d["revision"])
    raise ValueError(
        f"pin_digest_no_revision: profile={client.profile} jd={name} digest={want}"
    )


def _filter_cells_by_stems(cells: list[str], stems: list[str] | None) -> list[str]:
    if not stems:
        return cells
    want = {str(s).strip() for s in stems if str(s).strip()}
    if not want:
        return cells
    filtered = [c for c in cells if Path(c).stem in want]
    found = {Path(c).stem for c in filtered}
    missing = sorted(want - found)
    if missing:
        raise ValueError(f"stems_not_in_wave: {missing}")
    # Preserve stem order from the operator list when possible.
    by_stem = {Path(c).stem: c for c in filtered}
    return [by_stem[s] for s in stems if s in by_stem]


def _load_calibration(root: Path) -> dict:
    path = root / "deploy/aws_burst/config/calibration.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "hours_per_cell_by_vcpu": {"1": 1.91},
        "usd_per_vcpu_hour": 0.022,
    }


def _load_capabilities(root: Path) -> dict | None:
    path = root / "deploy/aws_burst/config/account_capabilities.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _incomplete_cells(
    client: BurstClient,
    wave: str,
    shard_cells_list: list[str],
    *,
    force: bool = False,
    max_error_retries: int = DEFAULT_MAX_ERROR_RETRIES,
) -> list[str]:
    """Return cells still needing work.

    A cell is complete only when its final ``{stem}.json`` artifact exists
    at the wave root (``{wave}/{stem}.json``). Nested objects under
    ``{wave}/resume/``, sibling ``_archive/{wave}_*/``, legacy
    ``{wave}/_archive_*/``, etc. do **not** mark a cell complete.
    ``.error.json`` companions never count as done.

    F4 / Phase-4: unless ``force``, skip stems whose wave-root ``.error.json``
    reports ``n_attempts`` (logical submits, not Batch child retries) >=
    ``max_error_retries``. Use ``--force`` to resubmit exhausted stems.
    """
    prefix = f"{wave}/"
    done: set[str] = set()
    error_counts: dict[str, int] = {}
    allowed_errors: set[str] = set()
    bucket = artifact_bucket(client.account_id())
    for key in client.list_keys(bucket, prefix):
        # Only wave-root finals: wave/{stem}.json (exactly one path segment).
        rel = key[len(prefix) :] if key.startswith(prefix) else key
        if "/" in rel:
            continue
        name = Path(rel).name
        if name.endswith(".error.json"):
            stem = name[: -len(".error.json")]
            n_att = 1
            try:
                payload = client.get_json(bucket, key)
                # Prefer logical attempt counter (parent job keyed).
                n_att = int(
                    payload.get("n_attempts")
                    or payload.get("n_logical_attempts")
                    or payload.get("attempts")
                    or 1
                )
                if wave == "VAL":
                    from scripts.validate_val_subset import HYBRID_STEM, validate_hybrid_error

                    if stem == HYBRID_STEM and validate_hybrid_error(payload).get("ok"):
                        allowed_errors.add(stem)
            except Exception:
                n_att = 1
            error_counts[stem] = max(error_counts.get(stem, 0), n_att)
            continue
        if name.endswith(".json") and not name.endswith(".sha256"):
            done.add(name)
    out: list[str] = []
    for c in shard_cells_list:
        stem = Path(c).stem
        if stem in allowed_errors:
            continue
        if f"{stem}.json" in done:
            continue
        if (
            not force
            and int(error_counts.get(stem, 0)) >= int(max_error_retries)
        ):
            continue
        out.append(c)
    return out


def assert_wave_panel_families_present(
    cell_paths: list[str],
    panel_manifest: dict,
    *,
    root: Path | None = None,
) -> None:
    """Refuse submit when any feature-cube cell lacks a shipped family.

    Cells with ``use_equity_feature_cube: true`` require every family in
    ``required_panel_families()`` plus ``gics`` to appear in
    ``panel_manifest['families_present']``.
    """
    from src.data.feature_panels import required_panel_families
    from src.spectrum.yaml_loader import load_cell_yaml

    needs_cube = False
    base = Path(root) if root is not None else ROOT
    for rel in cell_paths:
        path = Path(rel)
        if not path.is_file():
            path = base / rel
        if not path.is_file():
            continue
        try:
            cfg = load_cell_yaml(path)
        except Exception:
            continue
        if bool(cfg.get("use_equity_feature_cube", False)):
            needs_cube = True
            break
    if not needs_cube:
        return
    present = {str(x) for x in (panel_manifest.get("families_present") or [])}
    required = set(required_panel_families()) | {"gics"}
    missing = sorted(required - present)
    if missing:
        raise ValueError(
            "panel_bundle_missing_required_families: " + ",".join(missing)
        )


def preflight_compute_environment(client: BurstClient, *, root: Path) -> None:
    """Refuse submit unless Spot CE is ENABLED/VALID at profile max_vcpus."""
    profile = client.profile
    want_max = max_vcpus_for_profile(root, profile)
    want_types = allowed_instance_types_for_profile(root, profile)
    envs = client.describe_compute_environments()
    if not envs:
        raise ValueError("compute_environment_missing: no Batch CEs in account")
    chosen = None
    for env in envs:
        name = str(env.get("computeEnvironmentName") or "")
        cr = env.get("computeResources") or {}
        if cr.get("type") == "SPOT" or "volsurf-burst" in name:
            chosen = env
            break
    if chosen is None:
        chosen = envs[0]
    status = str(chosen.get("status") or "")
    state = str(chosen.get("state") or "")
    if state != "ENABLED" or status != "VALID":
        raise ValueError(
            f"compute_environment_not_ready: state={state} status={status} "
            f"name={chosen.get('computeEnvironmentName')}"
        )
    max_v = int((chosen.get("computeResources") or {}).get("maxvCpus") or 0)
    if max_v != want_max:
        raise ValueError(
            f"maxvcpus_drift: ce={chosen.get('computeEnvironmentName')} "
            f"maxvCpus={max_v} != configured {want_max} for {profile}"
        )
    cr = chosen.get("computeResources") or {}
    types = {str(t) for t in (cr.get("instanceTypes") or [])}
    if types and not types.issubset(want_types):
        raise ValueError(
            f"instance_type_drift: {sorted(types)} "
            f"(allowed={sorted(want_types)} for {profile})"
        )


def preflight_job_definition_resources(
    client: BurstClient,
    *,
    root: Path,
    job_definition_name: str = JOB_DEFINITION_NAME,
    himem: bool = False,
) -> None:
    """Refuse submit when job Memory exceeds instance RAM minus ECS overhead."""
    batch = client._batch()  # noqa: SLF001 - intentional Batch API access
    resp = batch.describe_job_definitions(
        jobDefinitionName=job_definition_name,
        status="ACTIVE",
    )
    defs = resp.get("jobDefinitions") or []
    if not defs:
        raise ValueError(f"job_definition_missing: {job_definition_name}")
    # Highest revision wins.
    defs = sorted(defs, key=lambda d: int(d.get("revision") or 0), reverse=True)
    props = defs[0].get("containerProperties") or {}
    memory = int(props.get("memory") or 0)
    inst_mem = instance_mem_mib_for_profile(root, client.profile)
    assert_job_memory_fits_instance(
        memory,
        instance_mem_mib=inst_mem,
        ecs_overhead_mib=DEFAULT_ECS_OVERHEAD_MIB,
    )
    want_mem = job_memory_mib_for_profile(root, client.profile, himem=himem)
    if memory != want_mem:
        raise ValueError(
            f"job_memory_drift: {job_definition_name} memory={memory}MiB "
            f"expected={want_mem}MiB for {client.profile}"
        )


def preflight_job_definition_image(
    client: BurstClient,
    *,
    expected_digest: str,
    expected_image_uri: str | None = None,
    job_definition_name: str = JOB_DEFINITION_NAME,
) -> None:
    """Refuse submit when ACTIVE job-def image does not end with the digest pin."""
    digest = str(expected_digest or "").strip()
    if not digest or digest == "unknown" or not digest.startswith("sha256:"):
        raise ValueError(f"container_digest_unknown_or_invalid: {digest!r}")
    batch = client._batch()  # noqa: SLF001
    resp = batch.describe_job_definitions(
        jobDefinitionName=job_definition_name,
        status="ACTIVE",
    )
    defs = resp.get("jobDefinitions") or []
    if not defs:
        raise ValueError(f"job_definition_missing: {job_definition_name}")
    defs = sorted(defs, key=lambda d: int(d.get("revision") or 0), reverse=True)
    image = str((defs[0].get("containerProperties") or {}).get("image") or "")
    if not image.endswith(digest) and digest not in image:
        raise ValueError(
            f"job_definition_image_mismatch: image={image!r} "
            f"expected_digest={digest!r}"
        )
    if expected_image_uri and "@sha256:" in expected_image_uri:
        # Prefer exact pin match when caller has the full URI.
        if image != expected_image_uri and not image.endswith("@" + digest):
            raise ValueError(
                f"job_definition_image_mismatch: image={image!r} "
                f"expected_uri={expected_image_uri!r}"
            )


def served_pick_stems_for_pick2_gate(root: Path) -> list[str]:
    """PICK stems the final cherrypick manifest actually served.

    Delegates to ``waves._cherrypick_final_served_stems`` so the PICK2 gate
    and Batch discovery share one eq-only source of truth. Falls back to the
    raw PICK glob when the manifest is absent.
    """
    from src.aws_burst.waves import _cherrypick_final_served_stems

    served = _cherrypick_final_served_stems(root, "PICK")
    if served is None:
        return [Path(c).stem for c in discover_wave_cells(root, "PICK")]
    return sorted(served)


def assert_pick_clean_for_pick2(
    root: Path,
    clients: list[BurstClient],
    *,
    expected_stems: list[str] | None = None,
) -> None:
    """Refuse PICK2 unless every expected PICK stem has a validated final."""
    from scripts.validate_remote_cell import validate_remote_cell

    stems = expected_stems
    if stems is None:
        stems = served_pick_stems_for_pick2_gate(root)
    if not stems:
        raise ValueError("pick2_gate_no_pick_cells")
    want = set(stems)
    found: set[str] = set()
    pending_errors: list[str] = []
    for client in clients:
        bucket = artifact_bucket(client.account_id())
        prefix = "PICK/"
        for key in client.list_keys(bucket, prefix):
            rel = key[len(prefix) :] if key.startswith(prefix) else key
            if "/" in rel:
                continue
            name = Path(rel).name
            if name.endswith(".error.json"):
                stem = name[: -len(".error.json")]
                if stem in want:
                    pending_errors.append(f"{client.profile}:{name}")
                continue
            if not name.endswith(".json") or name.endswith(".sha256"):
                continue
            if name.endswith("_policy_behavior.json"):
                continue
            stem = Path(name).stem
            if stem not in want:
                continue
            try:
                art = client.get_json(bucket, key)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"pick2_gate_pick_unreadable: {client.profile}:{key}: {exc}"
                ) from exc
            val = validate_remote_cell(art)
            if not val.get("ok"):
                raise ValueError(
                    f"pick2_gate_pick_invalid: {client.profile}:{stem} "
                    f"errors={val.get('errors')}"
                )
            found.add(stem)
    if pending_errors:
        raise ValueError(
            "pick2_gate_pending_pick_errors: " + ",".join(pending_errors[:20])
        )
    missing = sorted(want - found)
    if missing:
        raise ValueError(
            f"pick2_gate_missing_pick_cells: {missing[:20]}"
            f"{'...' if len(missing) > 20 else ''}"
        )


def assert_deskorg_priors_complete(root: Path) -> None:
    """Refuse DESKORG unless prior prior wave indexes/watches are complete.

    Accepts either a sealed local ``index.json`` or an S3 watch snapshot.
    Incomplete stale indexes must not mask a complete watch (and vice versa).
    """
    from src.aws_burst.waves import WAVES

    required = ("PICK_SMOKE", "PICK", "PICK2", "K200", "FEATNET", "HYBRID")
    for wave_name in required:
        spec = WAVES.get(wave_name)
        if spec is None:
            raise ValueError(f"deskorg_gate_unknown_wave: {wave_name}")
        idx = root / "logs" / "artifacts" / "spectrum" / spec.out_subdir / "index.json"
        watch = root / f"logs/aws_burst_watch_{wave_name}.json"
        ok = False
        details: list[str] = []
        for path in (idx, watch):
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if "polled_at" in data or path.name.startswith("aws_burst_watch"):
                path_ok = bool(data.get("complete")) and int(data.get("n_errors") or 0) == 0
                details.append(
                    f"{path}: complete={data.get('complete')} "
                    f"n_errors={data.get('n_errors')}"
                )
            else:
                path_ok = bool(data.get("complete")) and int(data.get("n_accepted") or 0) >= int(
                    data.get("n_expected") or 0
                )
                details.append(
                    f"{path}: complete={data.get('complete')} "
                    f"accepted={data.get('n_accepted')}/{data.get('n_expected')}"
                )
            if path_ok:
                ok = True
                break
        if not ok:
            detail = "; ".join(details) if details else f"no_index_for_{wave_name}"
            raise ValueError(f"deskorg_gate_prior_incomplete: {wave_name}: {detail}")


def build_plan(
    root: Path,
    wave: str,
    *,
    profiles: list[str] | None = None,
    stems: list[str] | None = None,
) -> dict:
    spec = WAVES.get(wave)
    if spec is not None and spec.glob is None:
        raise ValueError(
            f"{wave} is local-only (validate_headline_packs.py); refusing Batch submit"
        )
    armed = armed_profiles(root)
    if profiles:
        want = set(profiles)
        armed = [p for p in armed if p["profile"] in want]
        if not armed:
            raise ValueError(f"no_armed_profiles_match: {sorted(want)}")
    cells = _filter_cells_by_stems(discover_wave_cells(root, wave), stems)
    if not cells:
        raise ValueError(f"wave {wave!r} has no cells to submit")
    n_shards = len(armed)
    shards_meta = [{"profile": p["profile"], "shard": int(p["shard"])} for p in armed]
    manifest = build_manifest(
        wave=wave,
        region=REGION,
        cells=cells,
        shards=shards_meta,
        local_equivalent=_local_equivalent_for(wave),
    )
    cal = _load_calibration(root)
    hours = float(cal.get("hours_per_cell_by_vcpu", {}).get("1", 1.91))
    rate = float(cal.get("usd_per_vcpu_hour", 0.022))
    projected = projected_wave_cost(
        n_cells=len(cells), hours_per_cell=hours, usd_per_vcpu_hour=rate, vcpus=1
    )
    shard_weights = [float(max_vcpus_for_profile(root, p["profile"])) for p in armed]
    return {
        "wave": wave,
        "manifest": manifest,
        "manifest_sha": manifest_sha256(manifest),
        "armed_profiles": armed,
        "n_shards": n_shards,
        "projected_usd": projected,
        "shard_weights": shard_weights,
        "shard_plans": shard_cells(cells, n_shards, weights=shard_weights),
        "expected_cells": [str(Path(c).stem) for c in cells],
    }


def emit_wave_manifest(
    root: Path,
    wave: str,
    *,
    profiles: list[str] | None = None,
    stems: list[str] | None = None,
) -> Path:
    """Write deploy/aws_burst/config/wave_{wave}_manifest.json without submitting."""
    plan = build_plan(root, wave, profiles=profiles, stems=stems)
    manifest_path = root / f"deploy/aws_burst/config/wave_{wave}_manifest.json"
    full_manifest = dict(plan["manifest"])
    full_manifest["expected_cells"] = plan["expected_cells"]
    full_manifest["n_expected"] = len(plan["expected_cells"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(full_manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def submit_wave(
    root: Path,
    wave: str,
    *,
    dry_run: bool = False,
    offline: bool = False,
    profiles: list[str] | None = None,
    force: bool = False,
    allow_incomplete_priors: bool = False,
    stems: list[str] | None = None,
    skip_spend_gate: bool = False,
    attempt_seconds: int | None = None,
    job_definition_revision: int | None = None,
    pin_digest: str | None = None,
) -> dict:
    if offline:
        assert_offline_allowed(wave)
    plan = build_plan(root, wave, profiles=profiles, stems=stems)
    caps = _load_capabilities(root)
    if caps:
        for acct in caps.get("accounts", []):
            prof = str(acct.get("profile") or "")
            spot = acct.get("spot_vcpu_quota")
            need = max_vcpus_for_profile(root, prof) if prof else MAX_VCPUS_PER_ACCOUNT
            if spot is not None and float(spot) < need:
                raise ValueError(
                    f"spot_quota_drift: {prof} spot={spot} < configured {need}"
                )

    clients: list[BurstClient] = []
    if not dry_run and not offline:
        for profile_info in plan["armed_profiles"]:
            clients.append(BurstClient(profile_info["profile"], REGION))

    if not skip_spend_gate:
        check_submit_allowed(
            armed_profiles=plan["armed_profiles"],
            projected_usd=plan["projected_usd"],
            clients=clients or None,
            offline=offline or dry_run,
            allow_partial_profiles=bool(profiles),
            root=root,
        )
    manifest_path = emit_wave_manifest(root, wave, profiles=profiles, stems=stems)
    submitted: list[dict] = []
    if dry_run:
        return {"dry_run": True, "plan": plan, "manifest_path": str(manifest_path)}

    if wave == "PICK2" and not offline:
        if not allow_incomplete_priors:
            gate_clients = clients or [
                BurstClient(p["profile"], REGION) for p in plan["armed_profiles"]
            ]
            assert_pick_clean_for_pick2(root, gate_clients)
        else:
            import logging

            logging.getLogger("mascotrl.aws_submit_wave").warning(
                "PICK2 submit with --allow-incomplete-priors: PICK-seal gate skipped by operator"
            )

    if wave == "DESKORG" and not offline:
        if not allow_incomplete_priors:
            assert_deskorg_priors_complete(root)
        else:
            import logging

            logging.getLogger("mascotrl.aws_submit_wave").warning(
                "DESKORG submit with --allow-incomplete-priors: priors gate skipped by operator"
            )

    panel_bundle = root / "logs/aws_burst_panel_bundle/panel_bundle.tar"
    panel_sha_path = root / "logs/aws_burst_panel_bundle/panel_bundle.sha256"
    panel_manifest_path = root / "logs/aws_burst_panel_bundle/panel_manifest.json"
    panel_sha = panel_sha_path.read_text(encoding="utf-8").strip() if panel_sha_path.is_file() else ""
    all_cells = [c for shard in plan["shard_plans"] for c in shard]
    if panel_manifest_path.is_file():
        panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    else:
        panel_manifest = {"families_present": []}
    # Fail closed when any feature-cube cell is in the wave.
    assert_wave_panel_families_present(all_cells, panel_manifest, root=root)

    wave_needs_himem = any(cell_requires_himem(root, c) for c in all_cells)
    wave_needs_himem56 = any(
        p.job_definition == JOB_DEFINITION_HIMEM56
        for shard in plan["shard_plans"]
        for p in partition_cells_for_submit(root, shard)
    )

    for profile_info, shard_cells_list in zip(plan["armed_profiles"], plan["shard_plans"]):
        profile = profile_info["profile"]
        client = BurstClient(profile, REGION)
        digest_rec = load_digest_record(root, profile)
        # Resume-compatible pin: container env + JD image must match pin_digest,
        # not the latest image_digest_*.json stamp.
        digest = str(pin_digest).strip() if pin_digest else digest_rec["digest"]
        image_uri = pinned_image_uri(root, profile) if not pin_digest else None
        if not offline:
            preflight_compute_environment(client, root=root)
            preflight_job_definition_resources(client, root=root)
            # pin_digest uses an older ACTIVE revision; latest-JD image preflight
            # would false-fail against image_digest_*.json. Revision resolve below
            # is the pin check.
            if not pin_digest:
                preflight_job_definition_image(
                    client, expected_digest=digest, expected_image_uri=image_uri
                )
            if wave_needs_himem:
                preflight_job_definition_resources(
                    client,
                    root=root,
                    job_definition_name=JOB_DEFINITION_HIMEM,
                    himem=True,
                )
                if not pin_digest:
                    preflight_job_definition_image(
                        client,
                        expected_digest=digest,
                        expected_image_uri=image_uri,
                        job_definition_name=JOB_DEFINITION_HIMEM,
                    )
            if wave_needs_himem56 and not pin_digest:
                # himem56 is a custom JD (57344 MiB); memory is pinned on the JD.
                preflight_job_definition_image(
                    client,
                    expected_digest=digest,
                    expected_image_uri=image_uri,
                    job_definition_name=JOB_DEFINITION_HIMEM56,
                )
        acct = client.account_id()
        pb = panel_bucket(acct)
        ab = artifact_bucket(acct)
        client.ensure_bucket(pb)
        client.ensure_bucket(ab)
        if panel_bundle.is_file() and panel_sha:
            client.put_file_with_sha(pb, "panel_bundle.tar", panel_bundle)
        incomplete = _incomplete_cells(
            client, wave, shard_cells_list, force=bool(force)
        )
        shard_manifest = {
            "wave": wave,
            "profile": profile,
            "cells": incomplete,
            "manifest_sha": plan["manifest_sha"],
        }
        sm_key = f"manifests/{wave}/{plan['manifest_sha'][:8]}_{profile}.json"
        client.put_json(pb, sm_key, shard_manifest)
        if not incomplete:
            submitted.append({"profile": profile, "ok": True, "n": 0, "skipped": "complete"})
            continue
        partitions = partition_cells_for_submit(root, incomplete)
        profile_jobs: list[dict] = []
        for part in partitions:
            job_def = part.job_definition
            part_cells = part.cells
            part_manifest = {
                "wave": wave,
                "profile": profile,
                "cells": part_cells,
                "manifest_sha": plan["manifest_sha"],
                "job_definition": job_def,
                "memory_mib": part.memory_mib,
            }
            part_key = (
                f"manifests/{wave}/{plan['manifest_sha'][:8]}_{profile}_"
                f"{job_def.split('-')[-1]}"
                f"{'' if part.memory_mib is None else f'_{part.memory_mib}'}.json"
            )
            client.put_json(pb, part_key, part_manifest)
            part_env = build_container_env(
                {
                    "MASCOTRL_WAVE": wave,
                    "MASCOTRL_SHARD_MANIFEST_URI": f"s3://{pb}/{part_key}",
                    "MASCOTRL_PANEL_URI": f"s3://{pb}/panel_bundle.tar",
                    "MASCOTRL_PANEL_SHA256": panel_sha,
                    "MASCOTRL_OUT_URI": f"s3://{ab}/{wave}/",
                    "MASCOTRL_CONTAINER_DIGEST": digest,
                    "MASCOTRL_COMPUTE_HOST": "remote",
                }
            )
            overrides: dict = {"environment": part_env}
            if part.memory_mib is not None:
                inst_mem = instance_mem_mib_for_profile(root, profile)
                assert_job_memory_fits_instance(
                    int(part.memory_mib),
                    instance_mem_mib=inst_mem,
                    ecs_overhead_mib=DEFAULT_ECS_OVERHEAD_MIB,
                )
                overrides["memory"] = int(part.memory_mib)
            rev = job_definition_revision
            if rev is None and pin_digest:
                rev = resolve_job_definition_revision_for_digest(
                    client, job_def, pin_digest
                )
            pinned_jd = job_definition_with_revision(job_def, rev)
            attempt_s = (
                int(attempt_seconds)
                if attempt_seconds is not None
                else remaining_wall_attempt_seconds()
            )
            job = client.submit_array_job(
                job_name=(
                    f"mascotrl-{wave}-{plan['manifest_sha'][:8]}-{profile}-"
                    f"{job_def.split('-')[-1]}"
                ),
                job_queue="volsurf-burst-queue",
                job_definition=pinned_jd,
                array_size=len(part_cells),
                client_token=manifest_sha256(part_manifest),
                container_overrides=overrides,
                attempt_duration_seconds=attempt_s,
            )
            profile_jobs.append(
                {
                    "job_definition": pinned_jd,
                    "job_id": job["jobId"],
                    "job_arn": job.get("jobArn"),
                    "n": len(part_cells),
                    "memory_mib": part.memory_mib,
                    "attempt_seconds": attempt_s,
                }
            )
        submitted.append(
            {
                "profile": profile,
                "ok": True,
                "jobs": profile_jobs,
                "n": len(incomplete),
                "image_digest": digest,
            }
        )
    out_path = root / f"deploy/aws_burst/config/wave_{wave}_submit.json"
    payload = {"manifest": str(manifest_path), "submitted": submitted, "plan": plan}
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wave")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Skip live Budgets spend readback and CE preflight (local/dev only).",
    )
    p.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile filter (e.g. volsurf-burst-1 for smoke).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Resubmit stems even when error.json n_attempts >= max_error_retries "
            "(logical submits; Batch child retries count as one). Documented in "
            "AWS_BURST_MODE.md."
        ),
    )
    p.add_argument(
        "--allow-incomplete-priors",
        action="store_true",
        help=(
            "PICK2/DESKORG: skip prior-wave-complete gates (PICK seal for PICK2; "
            "PICK2/FEATNET/HYBRID for DESKORG). Use when operator accepts early "
            "submit under spare fleet capacity."
        ),
    )
    p.add_argument(
        "--emit-manifest-only",
        action="store_true",
        help="Write deploy/aws_burst/config/wave_{wave}_manifest.json and exit (no submit).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Staggered submit batch size per account (reduces Spot rejection).",
    )
    p.add_argument(
        "--batch-delay",
        type=int,
        default=60,
        help="Seconds between staggered batches (default 60).",
    )
    p.add_argument(
        "--stems",
        default="",
        help=(
            "Comma-separated cell stems to submit (filter before 4-account sharding). "
            "Keeps MASCOTRL_OUT_URI on the real wave name."
        ),
    )
    p.add_argument(
        "--skip-spend-gate",
        action="store_true",
        help="Skip check_submit_allowed / live Budgets read (operator $150/account cap).",
    )
    p.add_argument(
        "--attempt-seconds",
        type=int,
        default=None,
        help=(
            "Batch attemptDurationSeconds override. Default: remaining seconds to "
            "Mon 18:00 CEST minus 20m, capped at 21h."
        ),
    )
    p.add_argument(
        "--job-definition-revision",
        type=int,
        default=None,
        help=(
            "Pin every partition to jobDefinition:revision (e.g. 28 for digest "
            "2fb020 himem on burst-1). Omit to use ACTIVE latest."
        ),
    )
    p.add_argument(
        "--pin-digest",
        default="",
        help=(
            "Per-account: resolve ACTIVE job-definition revision whose image "
            "embeds this digest (e.g. sha256:2fb020...). Overrides latest pin "
            "for resume-compatible submits."
        ),
    )
    args = p.parse_args(argv)
    profiles = [x.strip() for x in str(args.profiles).split(",") if x.strip()] or None
    stems = [x.strip() for x in str(args.stems).split(",") if x.strip()] or None
    if args.emit_manifest_only:
        path = emit_wave_manifest(ROOT, args.wave, profiles=profiles, stems=stems)
        print(json.dumps({"emit_manifest_only": True, "manifest_path": str(path)}, indent=2))
        return 0
    result = submit_wave(
        ROOT,
        args.wave,
        dry_run=bool(args.dry_run),
        offline=bool(args.offline),
        profiles=profiles,
        force=bool(args.force),
        allow_incomplete_priors=bool(args.allow_incomplete_priors),
        stems=stems,
        skip_spend_gate=bool(args.skip_spend_gate),
        attempt_seconds=args.attempt_seconds,
        job_definition_revision=args.job_definition_revision,
        pin_digest=(str(args.pin_digest).strip() or None),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
