#!/usr/bin/env python3
"""Serial local runner for the three excluded mamba cherrypick cells.

Default ``--dry-run``: resolve cells, env snapshot, per-cell estimates; exit 0
without training.  Real mode (``--no-dry-run``) dispatches one
``run_spectrum_campaign.py`` subprocess per cell, strictly serial.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDIT_LEDGER = ROOT / "logs" / "campaign_sprint" / "AUDIT_LEDGER.md"
CAMPAIGN_LEDGER = ROOT / "logs" / "artifacts" / "spectrum" / "local_excluded_campaign.jsonl"
G0_SENTINEL = "G0_STATUS=FIXED"
MIN_MEM_GIB = 12.0
SUBSTRATE_FINGERPRINT = "G0_eq_obs_26ch_geometry_lite"

ARCTIC_OPT_LIBRARY = "hyper_volanet_features_opt100"
ARCTIC_MIX_LIBRARY = "hyper_volanet_features_mix100"


@dataclass(frozen=True)
class ExcludedCell:
    stem: str
    config_relpath: str
    config_dir_relpath: str
    config_glob: str
    out_dir_relpath: str
    warn: str | None = None


EXCLUDED_CELLS: tuple[ExcludedCell, ...] = (
    ExcludedCell(
        stem="eq_K100_single_ppo_mamba_softmax_mean_std_cao",
        config_relpath=(
            "config/spectrum/cherrypick/_dropped_mamba/"
            "eq_K100_single_ppo_mamba_softmax_mean_std_cao.yaml"
        ),
        config_dir_relpath="config/spectrum/cherrypick/_dropped_mamba",
        config_glob="eq_K100_single_ppo_mamba_softmax_mean_std_cao.yaml",
        out_dir_relpath="logs/artifacts/spectrum/cherrypick/",
    ),
    ExcludedCell(
        stem="eq_K200_single_ppo_mamba_softmax_mean_std_cao",
        config_relpath=(
            "config/spectrum/cherrypick/k200/_dropped_mamba/"
            "eq_K200_single_ppo_mamba_softmax_mean_std_cao.yaml"
        ),
        config_dir_relpath="config/spectrum/cherrypick/k200/_dropped_mamba",
        config_glob="eq_K200_single_ppo_mamba_softmax_mean_std_cao.yaml",
        out_dir_relpath="logs/artifacts/spectrum/cherrypick/k200/",
    ),
    ExcludedCell(
        stem="eq_K100_single_ppo_mamba_dirichlet_tilt_cvar_ru",
        config_relpath=(
            "config/spectrum/cherrypick/narrative/_dropped_mamba/"
            "eq_K100_single_ppo_mamba_dirichlet_tilt_cvar_ru.yaml"
        ),
        config_dir_relpath="config/spectrum/cherrypick/narrative/_dropped_mamba",
        config_glob="eq_K100_single_ppo_mamba_dirichlet_tilt_cvar_ru.yaml",
        out_dir_relpath="logs/artifacts/spectrum/cherrypick/narrative/",
        warn="LAST cell: 10 seeds x 100k train_env_steps; expect long runtime and high RAM",
    ),
)


def _peak_rss_mb() -> float:
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return 0.0


def read_mem_available_gib(meminfo_path: Path | None = None) -> float:
    path = meminfo_path or Path("/proc/meminfo")
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            kb = int(line.split()[1])
            return kb / (1024.0 * 1024.0)
    raise ValueError(f"MemAvailable not found in {path}")


def check_mem_available(
    *,
    min_gib: float = MIN_MEM_GIB,
    meminfo_path: Path | None = None,
) -> float:
    avail = read_mem_available_gib(meminfo_path)
    if avail < min_gib:
        raise SystemExit(
            f"mem_guard: MemAvailable={avail:.2f} GiB < required {min_gib:.0f} GiB"
        )
    return avail


def check_g0_sentinel(ledger_path: Path | None = None) -> None:
    path = ledger_path or AUDIT_LEDGER
    if not path.is_file():
        print(f"g0_sentinel: missing audit ledger: {path}", file=sys.stderr)
        raise SystemExit(2)
    text = path.read_text(encoding="utf-8")
    if G0_SENTINEL not in text:
        print(
            f"g0_sentinel: {path} lacks {G0_SENTINEL!r}; refusing excluded-cell runner",
            file=sys.stderr,
        )
        raise SystemExit(2)


def check_lake_mounted() -> Path:
    from src.data.paths import LAKE_ROOT, assert_lake_mounted

    assert_lake_mounted()
    return Path(LAKE_ROOT)


def arctic_opt_mix_symbol_counts() -> dict[str, int]:
    """Return symbol counts for deferred opt/mix Arctic libraries (0 on failure)."""
    counts: dict[str, int] = {"opt100": 0, "mix100": 0}
    try:
        from src.data.arctic_store import ArcticStateStore

        opt_store = ArcticStateStore(library_name=ARCTIC_OPT_LIBRARY)
        counts["opt100"] = len(opt_store.list_available_features())
        mix_store = ArcticStateStore(library_name=ARCTIC_MIX_LIBRARY)
        counts["mix100"] = len(mix_store.list_available_features())
    except Exception:
        pass
    return counts


def check_deferred_opt_mix_allowed(
    *,
    allow_deferred_opt_mix: bool,
    arctic_counts: dict[str, int] | None = None,
) -> None:
    if not allow_deferred_opt_mix:
        return
    counts = arctic_counts if arctic_counts is not None else arctic_opt_mix_symbol_counts()
    if counts.get("opt100", 0) == 0 and counts.get("mix100", 0) == 0:
        raise SystemExit(
            "deferred_opt_mix_refused: --allow-deferred-opt-mix set but Arctic "
            f"{ARCTIC_OPT_LIBRARY!r} and {ARCTIC_MIX_LIBRARY!r} report 0 symbols"
        )


def artifact_looks_complete(artifact_path: Path) -> bool:
    if not artifact_path.is_file():
        return False
    try:
        art = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(art, dict):
        return False
    if bool(art.get("dry_run")):
        return False
    if bool(art.get("strict_degraded")):
        return False
    if art.get("feature_net_errors") or art.get("spectrum_seed_errors"):
        return False
    if art.get("fallback_reason"):
        return False
    return bool(art.get("spectrum_cell_id") or art.get("promotable") is not None)


def collect_env_snapshot(
    *,
    meminfo_path: Path | None = None,
) -> dict[str, Any]:
    from src.data.paths import CANONICAL_LAKE, LAKE_ROOT

    lake_root = Path(LAKE_ROOT)
    snapshot: dict[str, Any] = {
        "lake_root": str(lake_root),
        "canonical_lake": str(CANONICAL_LAKE),
        "lake_exists": lake_root.exists(),
        "canonical_lake_exists": CANONICAL_LAKE.exists(),
        "substrate_fingerprint": SUBSTRATE_FINGERPRINT,
    }
    try:
        snapshot["mem_available_gib"] = round(read_mem_available_gib(meminfo_path), 3)
    except (OSError, ValueError) as exc:
        snapshot["mem_available_gib"] = None
        snapshot["mem_available_error"] = str(exc)
    snapshot["arctic_symbol_counts"] = arctic_opt_mix_symbol_counts()
    return snapshot


def estimate_cell(cell: ExcludedCell, *, root: Path = ROOT) -> dict[str, Any]:
    from src.spectrum.yaml_loader import load_cell_yaml

    cfg_path = root / cell.config_relpath
    cfg = load_cell_yaml(cfg_path)
    seeds = list(cfg.get("seeds") or [0])
    train_steps = int(cfg.get("train_env_steps") or 0)
    n_assets = int(cfg.get("n_assets") or 0)
    # Rough wall-clock hint for mamba on CPU (not a budget gate).
    step_factor = 4.0e-5
    est_s = len(seeds) * max(train_steps, 1) * step_factor * max(n_assets / 100.0, 1.0)
    return {
        "stem": cell.stem,
        "n_seeds": len(seeds),
        "train_env_steps": train_steps,
        "n_assets": n_assets,
        "estimated_wall_s": round(est_s, 1),
        "estimated_wall_h": round(est_s / 3600.0, 3),
    }


def resolve_cells(*, root: Path = ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in EXCLUDED_CELLS:
        cfg_path = root / cell.config_relpath
        out_dir = root / cell.out_dir_relpath
        if not cfg_path.is_file():
            raise SystemExit(f"excluded cell config missing: {cfg_path}")
        rows.append(
            {
                "stem": cell.stem,
                "config_path": str(cfg_path),
                "config_dir": str(root / cell.config_dir_relpath),
                "config_glob": cell.config_glob,
                "out_dir": str(out_dir),
                "artifact_path": str(out_dir / f"{cell.stem}.json"),
                "warn": cell.warn,
                "estimate": estimate_cell(cell, root=root),
            }
        )
    return rows


def append_campaign_ledger(
    row: dict[str, Any],
    *,
    ledger_path: Path | None = None,
) -> None:
    path = ledger_path or CAMPAIGN_LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _python_bin(root: Path = ROOT) -> str:
    venv = root / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def run_cell_subprocess(
    cell: dict[str, Any],
    *,
    root: Path = ROOT,
    env: dict[str, str] | None = None,
) -> tuple[int, float, float]:
    cmd = [
        _python_bin(root),
        str(root / "scripts" / "run_spectrum_campaign.py"),
        "--config-dir",
        cell["config_dir"],
        "--config-glob",
        cell["config_glob"],
        "--no-dry-run",
        "--strict",
        "--out-dir",
        cell["out_dir"],
    ]
    run_env = dict(os.environ)
    run_env["PYTHONPATH"] = str(root)
    run_env["TORCH_NUM_THREADS"] = "1"
    run_env["OMP_NUM_THREADS"] = "1"
    if env:
        run_env.update(env)
    t0 = time.perf_counter()
    rss0 = _peak_rss_mb()
    proc = subprocess.run(cmd, cwd=str(root), env=run_env)
    elapsed = time.perf_counter() - t0
    peak = max(_peak_rss_mb(), rss0)
    return int(proc.returncode), float(elapsed), float(peak)


def dry_run_report(
    *,
    root: Path = ROOT,
    meminfo_path: Path | None = None,
) -> dict[str, Any]:
    check_g0_sentinel()
    lake = check_lake_mounted()
    cells = resolve_cells(root=root)
    env = collect_env_snapshot(meminfo_path=meminfo_path)
    env["lake_root_resolved"] = str(lake)
    return {"mode": "dry_run", "env": env, "cells": cells}


def run_campaign(
    *,
    root: Path = ROOT,
    force: bool = False,
    allow_deferred_opt_mix: bool = False,
    meminfo_path: Path | None = None,
    ledger_path: Path | None = None,
) -> int:
    check_g0_sentinel()
    check_lake_mounted()
    check_deferred_opt_mix_allowed(allow_deferred_opt_mix=allow_deferred_opt_mix)
    cells = resolve_cells(root=root)
    worst_exit = 0
    for idx, cell in enumerate(cells):
        if cell.get("warn"):
            warnings.warn(cell["warn"], stacklevel=1)
            print(f"WARNING: {cell['warn']}", file=sys.stderr, flush=True)
        if idx == len(cells) - 1 and cell["stem"].endswith("dirichlet_tilt_cvar_ru"):
            msg = (
                "NARRATIVE MAMBA LAST: 10 seeds x 100k train_env_steps — "
                "serial only; ensure >=12 GiB MemAvailable before this cell"
            )
            warnings.warn(msg, stacklevel=1)
            print(f"WARNING: {msg}", file=sys.stderr, flush=True)

        artifact_path = Path(cell["artifact_path"])
        if not force and artifact_looks_complete(artifact_path):
            print(f"skip complete: {artifact_path}", flush=True)
            append_campaign_ledger(
                {
                    "stem": cell["stem"],
                    "exit_code": 0,
                    "elapsed_s": 0.0,
                    "peak_rss_mb": 0.0,
                    "substrate_fingerprint": SUBSTRATE_FINGERPRINT,
                    "skipped": "complete",
                },
                ledger_path=ledger_path,
            )
            continue

        check_mem_available(meminfo_path=meminfo_path)
        exit_code, elapsed_s, peak_rss_mb = run_cell_subprocess(cell, root=root)
        append_campaign_ledger(
            {
                "stem": cell["stem"],
                "exit_code": exit_code,
                "elapsed_s": round(elapsed_s, 3),
                "peak_rss_mb": round(peak_rss_mb, 2),
                "substrate_fingerprint": SUBSTRATE_FINGERPRINT,
            },
            ledger_path=ledger_path,
        )
        print(
            f"cell={cell['stem']} exit={exit_code} elapsed_s={elapsed_s:.1f} "
            f"peak_rss_mb={peak_rss_mb:.1f}",
            flush=True,
        )
        if exit_code != 0:
            worst_exit = max(worst_exit, exit_code)
    return worst_exit


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="List resolved cells and env snapshot (default).",
    )
    p.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Run real spectrum campaign subprocesses (serial).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when a complete artifact already exists.",
    )
    p.add_argument(
        "--allow-deferred-opt-mix",
        action="store_true",
        help=(
            "Permit deferred opt/mix path only when Arctic opt100/mix100 have symbols "
            "(refused while both report 0)."
        ),
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Serial only for now; values >1 are refused.",
    )
    args = p.parse_args(argv)

    if int(args.max_parallel) > 1:
        raise SystemExit("--max-parallel > 1 is not supported; excluded cells are serial only")

    dry = not bool(args.no_dry_run)

    if dry:
        report = dry_run_report()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    return run_campaign(
        force=bool(args.force),
        allow_deferred_opt_mix=bool(args.allow_deferred_opt_mix),
    )


if __name__ == "__main__":
    raise SystemExit(main())
