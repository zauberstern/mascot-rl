#!/usr/bin/env python3
"""Full-scale equity allocation campaign (Phase K).

Confirmatory arm: dyn_liquidity / dyn_hrp universe K=100 + research_alpha CPCV
+ benchmark panel. Densest-subgraph metrics are diagnostic only.

Resumable via logs/artifacts/eq_alloc/campaign_manifest.json.
This repo makes no capital-allocation claim fields.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from mascotrl.eval.equity_substrate import (
    _wide_field,
    _wide_returns,
    _wide_returns_with_availability,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "artifacts" / "eq_alloc"
OUT.mkdir(parents=True, exist_ok=True)

_JSONL = OUT / "campaign.jsonl"


def _log_event(phase: str, **fields: Any) -> None:
    """Append one structured JSONL record for campaign diagnostics (W5)."""
    import logging
    from datetime import datetime, timezone

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": str(phase),
        **fields,
    }
    try:
        with _JSONL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass
    logging.getLogger("mascotrl.eq_alloc").info("%s %s", phase, fields)


def _assert_safe_to_write_summary(
    out_dir: Path, k: int, *, force_overwrite: bool = False
) -> None:
    """Refuse to replace a larger-K headline summary without an explicit override."""
    summary_path = Path(out_dir) / "cpcv_path_summary.json"
    if not summary_path.exists() or force_overwrite:
        return

    try:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_k = int(existing["k"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"refusing to overwrite unreadable headline summary: {summary_path}; "
            "pass --force-overwrite to replace it"
        ) from exc

    if existing_k > int(k):
        raise SystemExit(
            f"refusing to overwrite headline summary K={existing_k} with smaller "
            f"K={k}; pass --force-overwrite to replace it"
        )


ACCEPTED_UNIVERSE_ARMS = ("dyn_hrp", "dyn_liquidity", "dyn_crucible")


def resolve_universe_arm(*, cli_arm: str | None, cfg: dict) -> str:
    """CLI > YAML > dyn_liquidity. YAML is authoritative when CLI is omitted."""
    if cli_arm is not None and str(cli_arm).strip():
        return str(cli_arm).strip()
    yaml_arm = cfg.get("universe_arm")
    if yaml_arm is not None and str(yaml_arm).strip():
        return str(yaml_arm).strip()
    return "dyn_liquidity"


def resolve_campaign_k(args: argparse.Namespace, cfg: Mapping[str, Any]) -> int:
    """CLI ``--k`` > YAML ``k`` > 100.

    Always touches ``cfg['k']`` when present so tracked YAML honesty does not
    orphan crucible workflows that declare ``k:`` while bakeoff passes ``--k``.
    """
    yaml_k = cfg.get("k")
    if getattr(args, "k", None) is not None:
        return int(args.k)
    if yaml_k is not None:
        return int(yaml_k)
    return 100


def _aggregate_decision_fields(
    seed_arts: list[dict],
    sharpes: list[float],
) -> dict:
    """Promote per-seed collapse / turnover / L1 metrics into PREREG decision fields."""
    collapse_flags = [
        bool(a.get("equal_weight_collapse_detected")) for a in seed_arts
    ]
    l1_vals: list[float] = []
    bind_vals: list[float] = []
    for a in seed_arts:
        diag = a.get("policy_diagnostics") or {}
        if diag.get("l1_vs_ew_mean") is not None:
            l1_vals.append(float(diag["l1_vs_ew_mean"]))
        if a.get("turnover_cap_binding_fraction") is not None:
            bind_vals.append(float(a["turnover_cap_binding_fraction"]))
    return {
        "equal_weight_collapse_detected_any": bool(any(collapse_flags)),
        "equal_weight_collapse_detected_per_seed": collapse_flags,
        "turnover_cap_binding_fraction_mean": (
            float(np.nanmean(bind_vals)) if bind_vals else None
        ),
        "l1_vs_ew_mean": float(np.nanmean(l1_vals)) if l1_vals else None,
        "sharpe_std_across_seeds": float(np.nanstd(sharpes)) if sharpes else None,
        "sharpe_mean_across_seeds": float(np.nanmean(sharpes)) if sharpes else None,
        "n_seeds": len(sharpes),
    }


def _code_identity_payload() -> dict:
    """Git + estimand-defining source hash for resume fingerprinting."""
    import subprocess

    commit = "unknown"
    dirty = True
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.call(
                ["git", "diff", "--quiet"], cwd=str(ROOT), stderr=subprocess.DEVNULL
            )
            != 0
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    sources = [
        "src/mascotrl/eval/parity_harness.py",
        "src/mascotrl/eval/research_alpha_cpcv.py",
        "src/mascotrl/eval/research_alpha_train.py",
        "src/mascotrl/env/historical_env.py",
        "src/mascotrl/eval/stats_rigor.py",
        "src/mascotrl/eval/calendar_scaling.py",
    ]
    h = hashlib.sha256()
    for rel in sources:
        p = ROOT / rel
        if p.is_file():
            h.update(p.read_bytes())
        else:
            h.update(f"MISSING:{rel}".encode())
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "estimand_sources_sha256": h.hexdigest()[:16],
    }


def _assert_disk_budget(*, out_dir: str | None, k: int) -> None:
    """Refuse to start when free space cannot cover a conservative checkpoint budget."""
    import shutil

    target = Path(out_dir) if out_dir else (ROOT / "logs" / "artifacts" / "eq_alloc")
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / (1024**3)
    env_min = os.environ.get("MASCOTRL_MIN_FREE_GB")
    if env_min is not None:
        need = float(env_min)
    else:
        need = 40.0 if int(k) >= 40 else 8.0
    if free_gb < need:
        raise SystemExit(
            f"disk preflight failed: {free_gb:.1f} GiB free on {target} "
            f"< required {need:.1f} GiB (set MASCOTRL_MIN_FREE_GB to override)"
        )


def _write_heartbeat(out: Path, **fields: Any) -> None:
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
    path = out / "heartbeat.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    # Mirror to overnight path for systemd monitoring.
    overnight = ROOT / "logs" / "overnight_heartbeat.txt"
    overnight.parent.mkdir(parents=True, exist_ok=True)
    overnight.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _install_diagnostic_signals(out: Path) -> None:
    """Map SIGUSR1 to a heartbeat dump instead of the default fatal exit.

    Multi-hour campaigns must not die when an operator (or tooling) sends
    SIGUSR1 for diagnostics. SIGTERM/SIGINT keep their normal graceful path.
    """
    import signal
    import traceback

    def _on_usr1(signum: int, frame: Any) -> None:  # noqa: ARG001
        stack = "".join(traceback.format_stack(frame)) if frame is not None else ""
        _write_heartbeat(
            out,
            phase="sigusr1",
            stack_tail=stack[-4000:],
        )
        _log_event("sigusr1", stack_chars=len(stack))

    try:
        signal.signal(signal.SIGUSR1, _on_usr1)
    except Exception:
        pass


def _heartbeat_loop(out: Path, stop_event: Any, *, phase: str, interval_s: float = 300.0, **fields: Any) -> None:
    """Daemon helper: emit heartbeat.json every ``interval_s`` until stopped."""
    while not stop_event.wait(timeout=float(interval_s)):
        _write_heartbeat(out, phase=phase, **fields)

def _campaign_config_fingerprint(cfg: dict, *, realized_k: int) -> str:
    """Stable hash of the training-relevant config knobs for a CPCV seed cell.

    Guards the resumable manifest (``campaign_manifest.json``) against
    silently reusing a ``cpcv_seed_N.json`` produced under a *different*
    config (different realized universe size, CPCV geometry, training
    budget, or surface-signal allowlist) when the caller reruns the
    campaign with new CLI flags. A changed fingerprint forces a fresh
    training run for that seed instead of loading the stale artifact.
    """
    iv_surface = (cfg.get("feature_extras") or {}).get("iv_surface")
    allowlist_names = sorted(iv_surface.keys()) if isinstance(iv_surface, dict) else []
    payload = {
        "realized_k": int(realized_k),
        "universe_arm": cfg.get("universe_arm", "dyn_liquidity"),
        "crucible": cfg.get("crucible") or {},
        "cpcv_n_splits": cfg.get("cpcv_n_splits"),
        "cpcv_n_test_groups": cfg.get("cpcv_n_test_groups"),
        "cpcv_purge_days": cfg.get("cpcv_purge_days"),
        "cpcv_embargo_days": cfg.get("cpcv_embargo_days"),
        "train_env_steps": cfg.get("train_env_steps"),
        "train_epochs": cfg.get("train_epochs"),
        "train_episodes": cfg.get("train_episodes"),
        "min_optimizer_steps_total": cfg.get("min_optimizer_steps_total"),
        "min_optimizer_steps": cfg.get("min_optimizer_steps"),
        "use_surface_signals": bool(cfg.get("use_surface_signals", False)),
        "surface_obs_lane": str(cfg.get("surface_obs_lane") or "geometry_lite"),
        "obs_pack_id": cfg.get("_obs_pack_id"),
        "surface_allowlist_names": allowlist_names,
        "turnover_limit": cfg.get("turnover_limit"),
        "projection_mode": cfg.get("projection_mode"),
        "architecture": cfg.get("architecture"),
        "objective": cfg.get("objective"),
        "algo": cfg.get("algo"),
        "reward": cfg.get("reward"),
        "rebalance_cadence": cfg.get("rebalance_cadence"),
        "equity_bps": cfg.get("equity_bps"),
        "impact_c_eq": cfg.get("impact_c_eq"),
        "ppo_hidden": cfg.get("ppo_hidden"),
        "lr": cfg.get("lr"),
        "entropy_coef": cfg.get("entropy_coef"),
        "dii_epochs": cfg.get("dii_epochs"),
        "max_pool": cfg.get("max_pool"),
        "weight_head": cfg.get("weight_head"),
        "actor_final_gain": cfg.get("actor_final_gain"),
        "weight_head_temperature": cfg.get("weight_head_temperature"),
        "train_updates_per_fold": cfg.get("train_updates_per_fold"),
        "universe_mode": cfg.get("universe_mode"),
        "_crucible_universe_fingerprint": cfg.get("_crucible_universe_fingerprint"),
        "_crucible_schedule_fingerprint": cfg.get("_crucible_schedule_fingerprint"),
        "code_identity": _code_identity_payload(),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


_SEED_PACK_SKIP_KEYS = frozenset(
    {
        "_crucible_panels",
        # TrackingDict bookkeeping if ever materialised as a key.
        "accessed_keys",
    }
)
_NPY_MARKER = "__npy__"


def _at_least_one_int(value: str) -> int:
    """argparse type: positive int (min 1)."""
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected int, got {value!r}") from exc
    if n < 1:
        raise argparse.ArgumentTypeError("--seed-workers must be >= 1")
    return n


def _default_seed_workers() -> int:
    raw = os.environ.get("MASCOTRL_SEED_WORKERS", "1") or "1"
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _pack_cfg_value(key_path: str, value: Any, pack_dir: Path) -> Any:
    """Turn cfg values into JSON-safe form; dump ndarrays to pack_dir."""
    if isinstance(value, np.ndarray):
        safe = (
            key_path.replace("/", "_")
            .replace(".", "__")
            .replace("[", "_")
            .replace("]", "_")
        )
        fname = f"{safe}.npy"
        np.save(pack_dir / fname, value)
        return {_NPY_MARKER: fname}
    if isinstance(value, Mapping):
        return {
            str(k): _pack_cfg_value(f"{key_path}.{k}", v, pack_dir)
            for k, v in value.items()
            if str(k) not in _SEED_PACK_SKIP_KEYS
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        out = []
        for i, item in enumerate(value):
            if isinstance(item, np.ndarray):
                out.append(_pack_cfg_value(f"{key_path}.{i}", item, pack_dir))
            else:
                try:
                    json.dumps(item, default=str)
                    out.append(item)
                except (TypeError, ValueError):
                    out.append(str(item))
        return out
    try:
        json.dumps(value, default=str)
        return value
    except (TypeError, ValueError):
        return str(value)


def _unpack_cfg_value(value: Any, pack_dir: Path) -> Any:
    if isinstance(value, Mapping):
        if set(value.keys()) == {_NPY_MARKER}:
            return np.load(pack_dir / str(value[_NPY_MARKER]), allow_pickle=False)
        return {k: _unpack_cfg_value(v, pack_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack_cfg_value(v, pack_dir) for v in value]
    return value


def _serialize_cfg_runtime(cfg: Mapping[str, Any], pack_dir: Path) -> dict[str, Any]:
    packed: dict[str, Any] = {}
    for key, value in dict(cfg).items():
        if str(key) in _SEED_PACK_SKIP_KEYS:
            continue
        packed[str(key)] = _pack_cfg_value(str(key), value, pack_dir)
    return packed


def _load_cfg_runtime(pack_dir: Path) -> dict[str, Any]:
    blob = json.loads((pack_dir / "cfg_runtime.json").read_text(encoding="utf-8"))
    return _unpack_cfg_value(blob, pack_dir)


def _write_seed_pack(
    pack_dir: Path | str,
    *,
    panel: np.ndarray,
    factors: np.ndarray,
    dates: Sequence[Any],
    cfg: Mapping[str, Any],
    cpcv: Any,
    run_config_hash: str,
    realized_k: int,
) -> dict[str, Any]:
    """Persist arrays + JSON meta so spawn workers can reconstruct a seed job."""
    pack_dir = Path(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    np.save(pack_dir / "panel.npy", np.asarray(panel, dtype=np.float64))
    np.save(pack_dir / "factors.npy", np.asarray(factors, dtype=np.float64))
    cfg_runtime = _serialize_cfg_runtime(cfg, pack_dir)
    (pack_dir / "cfg_runtime.json").write_text(
        json.dumps(cfg_runtime, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    meta = {
        "dates": [str(pd.Timestamp(d).date()) for d in dates],
        "run_config_hash": str(run_config_hash),
        "realized_k": int(realized_k),
        "cpcv": {
            "n_splits": int(getattr(cpcv, "n_splits", 6)),
            "n_test_groups": int(getattr(cpcv, "n_test_groups", 2)),
            "purge_days": int(getattr(cpcv, "purge_days", 21)),
            "embargo_days": int(getattr(cpcv, "embargo_days", 21)),
        },
    }
    (pack_dir / "seed_pack_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"pack_dir": str(pack_dir)}


def _locked_mark_seed_complete(
    out_dir: Path,
    *,
    seed: int,
    run_config_hash: str,
    accessed_keys: Sequence[str] | None = None,
) -> None:
    """Update campaign_manifest.json under an exclusive flock (multi-worker safe)."""
    from mascotrl.eval.campaign_manifest import (
        load_manifest,
        mark_cell_complete,
        save_manifest,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / "campaign_manifest.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            manifest = load_manifest(out_dir)
            mark_cell_complete(
                manifest,
                fold_id=-1,
                seed=int(seed),
                arm="eq_dii",
                extra={
                    "artifact": str(out_dir / f"cpcv_seed_{int(seed)}.json"),
                    "run_config_hash": str(run_config_hash),
                    "accessed_keys": list(accessed_keys or ()),
                },
            )
            save_manifest(out_dir, manifest)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _run_one_seed_cpcv(
    seed: int,
    dates: Sequence[Any],
    panel: np.ndarray,
    fac: np.ndarray,
    cfg_dict: Mapping[str, Any],
    cpcv: Any,
    out_dir: Path | str,
    run_config_hash: str,
    *,
    panel_source: str = "equity_sp500",
    resume: bool = True,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Train one CPCV seed cell; write ``cpcv_seed_{seed}.json`` + locked manifest."""
    from mascotrl.eval.campaign_manifest import atomic_write_json
    from mascotrl.eval.research_alpha_cpcv import run_research_alpha_cpcv
    from mascotrl.eval.yaml_honesty import TrackingDict
    from mascotrl.eval.pbo_appendix import append_trial_ledger_entry

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(cfg_dict, TrackingDict):
        # Share the parent access set so serial campaign honesty stays live.
        cfg = TrackingDict(cfg_dict, _accessed=cfg_dict._accessed)
    else:
        cfg = TrackingDict(dict(cfg_dict))
    cfg["_checkpoint_dir"] = str(out_dir / "checkpoints" / f"seed{int(seed)}")
    cfg["_run_config_hash"] = str(run_config_hash)
    cfg["_out_dir"] = str(out_dir)

    t0 = time.perf_counter()
    art = run_research_alpha_cpcv(
        dates,
        panel,
        fac,
        cfg,
        cpcv=cpcv,
        seed=int(seed),
        panel_source=panel_source,
        out_dir=out_dir,
        resume=resume,
    )
    art["wall_s"] = time.perf_counter() - t0
    art["seed"] = int(seed)
    atomic_write_json(out_dir / f"cpcv_seed_{int(seed)}.json", art)
    accessed = sorted(getattr(cfg, "accessed_keys", ()) or ())
    _locked_mark_seed_complete(
        out_dir,
        seed=int(seed),
        run_config_hash=str(run_config_hash),
        accessed_keys=accessed,
    )
    root = Path(repo_root) if repo_root is not None else ROOT
    sh_mean = (art.get("path_summary") or {}).get("sharpe_mean")
    append_trial_ledger_entry(
        root / "logs" / "trial_ledger.json",
        source="eq_alloc_cpcv",
        trial_id=f"eq_dii_seed{int(seed)}",
        sharpe=float(sh_mean) if sh_mean is not None else None,
        status="ok",
        config_sha=str(run_config_hash),
        extra={
            "seed": int(seed),
            "arm": "eq_dii",
            "k": int(cfg.get("n_assets") or panel.shape[1]),
        },
    )
    return art


def _seed_worker_main(payload: dict) -> dict:
    """Top-level spawn entry: reconstruct pack, run one seed CPCV cell."""
    threads_per = int(
        payload.get("threads_per")
        or os.environ.get("MASCOTRL_THREADS_PER_WORKER", "4")
        or "4"
    )
    threads_per = max(1, threads_per)
    os.environ["TORCH_NUM_THREADS"] = str(threads_per)
    os.environ["OMP_NUM_THREADS"] = str(threads_per)

    from mascotrl.eval.cpcv import CPCVConfig

    pack_dir = Path(payload["pack_dir"])
    panel = np.load(pack_dir / "panel.npy", allow_pickle=False)
    fac = np.load(pack_dir / "factors.npy", allow_pickle=False)
    meta = json.loads((pack_dir / "seed_pack_meta.json").read_text(encoding="utf-8"))
    cfg = _load_cfg_runtime(pack_dir)
    dates = list(pd.to_datetime(meta["dates"]))
    cpcv = CPCVConfig(**dict(meta.get("cpcv") or {}))
    return _run_one_seed_cpcv(
        int(payload["seed"]),
        dates,
        panel,
        fac,
        cfg,
        cpcv,
        Path(payload["out_dir"]),
        str(meta["run_config_hash"]),
        panel_source=str(payload.get("panel_source") or "equity_sp500"),
        resume=bool(payload.get("resume", True)),
        repo_root=payload.get("repo_root") or str(ROOT),
    )


def _collect_seed_art_paths(
    seed: int,
    art: Mapping[str, Any],
    path_pnls_all: dict[str, Any],
    path_dates_all: dict[str, Any],
) -> None:
    for pid, path in (art.get("paths") or {}).items():
        if isinstance(path, dict) and "pnl" in path:
            path_pnls_all[f"{seed}:{pid}"] = path["pnl"]
            path_dates_all[f"{seed}:{pid}"] = path.get("dates") or []


def _load_cfg(path: Path) -> "TrackingDict":
    from mascotrl.eval.yaml_honesty import TrackingDict

    return TrackingDict(yaml.safe_load(path.read_text()) or {})


def _estimate_campaign_dsr_trials(
    ledger_path: Path, *, cfg: dict
) -> tuple[int, dict[str, Any]]:
    """Estimate DSR N from the persisted executed-trial ledger."""
    from mascotrl.eval.publication import estimate_n_trials

    try:
        ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        ledger = {}
    return estimate_n_trials({"trial_ledger": ledger}, cfg)


def _corr_adjacency(returns: np.ndarray, *, thr: float = 0.35) -> np.ndarray:
    x = np.nan_to_num(returns, nan=0.0)
    c = np.corrcoef(x, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    a = (np.abs(c) >= thr).astype(np.float64)
    np.fill_diagonal(a, 1.0)
    return a


def _densest_k(returns: np.ndarray, k: int) -> np.ndarray:
    """Diagnostic densest-subgraph control; never used as the selection arm."""
    from mascotrl.reporting.figures.graph_helpers import densest_subgraph_greedy

    x = np.nan_to_num(returns, nan=0.0)
    c = np.abs(np.corrcoef(x, rowvar=False))
    c = np.nan_to_num(c, nan=0.0)
    np.fill_diagonal(c, 0.0)
    return np.asarray(densest_subgraph_greedy(c, k), dtype=int)


def _set_universe_secids(results: dict, secids: list) -> list:
    """Primary key ``universe_secids``; keep deprecated ``dii_secids`` alias."""
    cleaned = [int(s) if not isinstance(s, str) else s for s in secids]
    results["universe_secids"] = cleaned
    results["dii_secids"] = cleaned
    return cleaned


def _crucible_artifact_dir(cfg: dict) -> Path:
    """Prefer OUT / cfg['_out_dir']; fall back to logs/artifacts/eq_alloc."""
    for cand in (cfg.get("_out_dir"), OUT):
        if cand is None:
            continue
        p = Path(cand)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            continue
    fallback = ROOT / "logs" / "artifacts" / "eq_alloc"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _rolling_market_beta(
    returns: pd.DataFrame,
    mkt_rf: pd.Series,
    *,
    lookback: int = 63,
    min_periods: int = 21,
) -> pd.DataFrame:
    """Per-name rolling OLS beta of returns on market excess return."""
    mkt = pd.Series(mkt_rf, index=returns.index, dtype=np.float64)
    var = mkt.rolling(lookback, min_periods=min_periods).var()
    betas = {}
    for col in returns.columns:
        cov = returns[col].rolling(lookback, min_periods=min_periods).cov(mkt)
        betas[col] = cov / var.replace(0.0, np.nan)
    return pd.DataFrame(betas, index=returns.index)


def _surface_wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Wide ``(secid, date, signal...)`` -> long ``date, secid, signal, value``."""
    if wide is None or len(wide) == 0:
        return pd.DataFrame(columns=["date", "secid", "signal", "value"])
    d = wide.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["secid"] = d["secid"].astype(int)
    id_vars = ["date", "secid"]
    value_vars = [c for c in d.columns if c not in id_vars]
    if not value_vars:
        return pd.DataFrame(columns=["date", "secid", "signal", "value"])
    long = d.melt(id_vars=id_vars, value_vars=value_vars, var_name="signal", value_name="value")
    long = long[np.isfinite(pd.to_numeric(long["value"], errors="coerce"))]
    # Sleeve-score aliases used by crucible.sleeve_scores
    alias = {
        "iv_skew_30d": "skew",
        "iv_term_slope": "term",
        "mfiv_30": "iv_30",
    }
    extra_rows = []
    for src, dst in alias.items():
        sub = long[long["signal"] == src]
        if sub.empty:
            continue
        aliased = sub.copy()
        aliased["signal"] = dst
        extra_rows.append(aliased)
    if extra_rows:
        long = pd.concat([long, *extra_rows], ignore_index=True)
    return long.reset_index(drop=True)


def _eligible_from_surface(
    surface: pd.DataFrame,
    *,
    required_signals: tuple[str, ...] = ("mfis_30", "mfis_365"),
    min_obs: int = 2,
) -> list[int]:
    """Option-eligible names from surface coverage (not return coverage)."""
    if surface is None or len(surface) == 0:
        return []
    ok: list[int] = []
    for sid, g in surface.groupby("secid"):
        good = True
        for sig in required_signals:
            n = int(np.isfinite(g.loc[g["signal"] == sig, "value"].to_numpy(dtype=float)).sum())
            if n < int(min_obs):
                good = False
                break
        if good:
            ok.append(int(sid))
    return ok


def _load_crucible_adv_panel(
    *,
    lake_root,
    secids: list[int],
    dates_idx: pd.DatetimeIndex,
    cfg: dict,
) -> tuple[pd.DataFrame | None, str]:
    """Load ADV wide panel from lake or equity dollar_volume.

    Returns ``(panel, source)``. ``panel is None`` only when
    ``crucible_allow_proxy_panels`` is True and real ADV is unavailable.
    """
    lake = Path(lake_root)
    adv_path = lake / "macro" / "crsp_om_adv.parquet"
    if adv_path.is_file():
        try:
            df = pd.read_parquet(adv_path, columns=["date", "secid", "adv"])
            df["date"] = pd.to_datetime(df["date"])
            df["secid"] = df["secid"].astype(int)
            pad_start = dates_idx.min() - pd.Timedelta(days=400)
            df = df[(df["date"] >= pad_start) & (df["date"] <= dates_idx.max())]
            df = df[df["secid"].isin(secids)]
            if not df.empty and df["adv"].notna().any():
                wide = df.pivot_table(
                    index="date", columns="secid", values="adv", aggfunc="last"
                )
                wide = wide.reindex(index=dates_idx, columns=secids).ffill(limit=5)
                if wide.notna().any().any():
                    return wide, "crsp_om_adv"
        except Exception as exc:  # noqa: BLE001
            _log_event("crucible_adv_load_error", path=str(adv_path), error=str(exc)[:200])

    # Equity dollar_volume fallback (not the return-vol proxy).
    try:
        from mascotrl.data.equity_panel import load_sp500_security_returns

        pad_start = str((dates_idx.min() - pd.Timedelta(days=400)).date())
        raw = load_sp500_security_returns(
            lake, start=pad_start, end=str(dates_idx.max().date())
        )
        if (
            raw is not None
            and "volume" in raw.columns
            and "close" in raw.columns
        ):
            tmp = raw.copy()
            tmp["dollar_volume"] = pd.to_numeric(tmp["volume"], errors="coerce") * pd.to_numeric(
                tmp["close"], errors="coerce"
            ).abs()
            arr = _wide_field(
                tmp, secids=secids, dates=list(dates_idx), value_col="dollar_volume"
            )
            if arr is not None:
                wide = pd.DataFrame(arr, index=dates_idx, columns=secids)
                if wide.notna().any().any():
                    return wide, "equity_dollar_volume"
    except Exception as exc:  # noqa: BLE001
        _log_event("crucible_adv_dollar_volume_error", error=str(exc)[:200])

    if bool(cfg.get("crucible_allow_proxy_panels")):
        return None, "proxy_allowed"
    raise SystemExit(
        "CRUCIBLE fail-closed: ADV panel missing. Expected lake "
        f"{adv_path} or equity dollar_volume (volume*close). "
        "Set crucible_allow_proxy_panels=true only for smoke tests that "
        "accept coverage_proxy_no_surface; do not silently use return-vol ADV."
    )


def _load_crucible_surface_panel(
    *,
    lake_root,
    secids: list[int],
    dates_idx: pd.DatetimeIndex,
    cfg: dict,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, list[int] | None, str]:
    """Materialize surface signals; fail closed unless proxy panels allowed."""
    allow_proxy = bool(cfg.get("crucible_allow_proxy_panels"))
    try:
        from mascotrl.data.surface_signals import materialize_surface_signals_from_lake

        start = str((dates_idx.min() - pd.DateOffset(months=2)).date())
        end = str(dates_idx.max().date())
        # Equity volume optional for os_ratio; keep light for selection path.
        wide = materialize_surface_signals_from_lake(
            lake_root,
            secids=secids,
            start=start,
            end=end,
            cache_path=artifact_dir / "crucible_surface_signals.parquet",
            hv=_load_hv_table(lake_root, secids, start, end),
            option_volume=_load_option_volume_table(lake_root, secids, start, end),
            equity_volume=None,
            borrow=_load_borrow_table(lake_root, secids, start, end),
            month_end_only=True,
        )
    except Exception as exc:
        if allow_proxy:
            return (
                pd.DataFrame(columns=["date", "secid", "signal", "value"]),
                None,
                "coverage_proxy_no_surface",
            )
        raise SystemExit(
            f"CRUCIBLE fail-closed: surface materialization failed: {exc}"
        ) from exc

    for req in ("mfis_30", "mfis_365"):
        if req not in wide.columns or not np.isfinite(
            pd.to_numeric(wide[req], errors="coerce")
        ).any():
            if allow_proxy:
                return (
                    pd.DataFrame(columns=["date", "secid", "signal", "value"]),
                    None,
                    "coverage_proxy_no_surface",
                )
            raise SystemExit(
                f"CRUCIBLE fail-closed: required surface signal {req!r} missing "
                "from lake materialization"
            )

    long = _surface_wide_to_long(wide)
    eligible = _eligible_from_surface(long)
    if not eligible and not allow_proxy:
        raise SystemExit(
            "CRUCIBLE fail-closed: no secids with mfis_30/mfis_365 coverage"
        )
    return long, eligible, "lake_panels"


def _write_crucible_selection_json(cfg: dict, as_of, diagnostics: dict) -> Path:
    out_dir = _crucible_artifact_dir(cfg)
    stamp = pd.Timestamp(as_of).date().isoformat()
    path = out_dir / f"crucible_selection_{stamp}.json"
    payload = json.dumps(diagnostics, default=str, indent=2)
    path.write_text(payload, encoding="utf-8")
    # Canonical name for figure_payloads gatherer (M1-M3).
    canonical = out_dir / "crucible_selection.json"
    canonical.write_text(payload, encoding="utf-8")
    return path


def _stamp_selection_vs_sizing(
    decision_fields: dict,
    *,
    confirmatory: dict,
    path_pnls_all: dict | None = None,
) -> dict:
    """Attach selection vs sizing attribution when return series exist."""
    from mascotrl.eval.policy_diagnostics import selection_vs_sizing_attribution

    path_summary = dict(confirmatory.get("path_summary") or {})
    pol = path_summary.get("policy_returns")
    ew_c = path_summary.get("ew_crucible_returns")
    ew_p = path_summary.get("ew_parent_returns")
    if pol is None and path_pnls_all:
        first = next(iter(path_pnls_all.values()), None)
        if first is not None:
            pol = first
    if ew_c is None:
        ew_c = confirmatory.get("_ew_total_net")
    if ew_p is None:
        ew_p = confirmatory.get("_ew_parent_total_net")
    if pol is not None and ew_c is not None and ew_p is not None:
        decision_fields["selection_vs_sizing_attribution"] = (
            selection_vs_sizing_attribution(pol, ew_c, ew_p)
        )
    else:
        decision_fields["selection_vs_sizing_attribution"] = {
            "status": "pending_returns"
        }
    return decision_fields


def _build_crucible_universe(
    *,
    rets_e: np.ndarray,
    secids_e: list,
    dates: list,
    rb_mask: np.ndarray | None,
    k: int,
    cfg: dict,
    lake_root,
    rets_hist: np.ndarray | None = None,
    secids_hist: list | None = None,
    dates_hist: list | None = None,
) -> tuple[np.ndarray, list, dict]:
    """CRUCIBLE slow-reselect universe with behavioural sleeves.

    Prefers lake ADV / surface / rolling FF4 beta panels. Return-vol ADV proxy
    is fail-closed unless ``cfg['crucible_allow_proxy_panels']`` is True.
    Injected ``cfg['_crucible_panels']`` remains the unit-test escape hatch.

    ``rets_hist`` / ``dates_hist`` (selection window) are prepended so the first
    eval reselect has a full FF4 residual lookback — eval-only returns make
    residual_communities fail closed at EVAL_START.
    """
    from mascotrl.data.crucible import CrucibleSpec, select_universe_crucible
    from mascotrl.data.dynamic_universe import (
        build_slotted_panel,
        selection_turnover,
    )
    from mascotrl.eval.cadence import (
        assert_universe_subset_of_policy,
        build_rebalance_mask,
        build_universe_cadence_mask,
    )
    from mascotrl.eval.friction import friction_spec_from_cfg
    from mascotrl.policy.cmdp_projector import make_cmdp_projector

    cruc_cfg = dict(cfg.get("crucible") or {})
    quotas = cruc_cfg.get("quotas")
    spec = CrucibleSpec(
        k=int(k),
        max_pool=int(cfg.get("max_pool") or cruc_cfg.get("max_pool") or 511),
        lookback_days=int(cruc_cfg.get("lookback_days", 252)),
        reselect_every_days=int(cruc_cfg.get("reselect_every_days", 63)),
        reselect_churn_cap=float(cruc_cfg.get("reselect_churn_cap", 0.25)),
        adv_participation_floor=float(cruc_cfg.get("adv_participation_floor", 0.10)),
        amihud_drop_pct=float(cruc_cfg.get("amihud_drop_pct", 95.0)),
        n_communities=int(cruc_cfg.get("n_communities", 20)),
        max_per_community=int(cruc_cfg.get("max_per_community", 3)),
        lottery_resid_var_share_cap=float(
            cruc_cfg.get("lottery_resid_var_share_cap", 0.20)
        ),
        g1_l1_floor=float(cruc_cfg.get("g1_l1_floor", 0.08)),
        g1_entropy_gap_floor=float(cruc_cfg.get("g1_entropy_gap_floor", 0.60)),
        g2_tc_floor=float(cruc_cfg.get("g2_tc_floor", 0.35)),
        g3_sharpe_floor=float(cruc_cfg.get("g3_sharpe_floor", 0.10)),
        max_repair_passes=int(cruc_cfg.get("max_repair_passes", 5)),
    )
    if quotas:
        spec.quotas = {str(kk): int(vv) for kk, vv in dict(quotas).items()}

    dates_idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    secids = [int(s) for s in secids_e]
    returns_eval = pd.DataFrame(
        np.asarray(rets_e, dtype=np.float64), index=dates_idx, columns=secids
    )
    returns = returns_eval
    if (
        rets_hist is not None
        and secids_hist is not None
        and dates_hist is not None
        and len(dates_hist) > 0
    ):
        hist_dates = pd.DatetimeIndex(pd.to_datetime(list(dates_hist)))
        hist_secids = [int(s) for s in secids_hist]
        col_map = {s: i for i, s in enumerate(hist_secids)}
        keep = [s for s in secids if s in col_map]
        if keep:
            arr = np.asarray(rets_hist, dtype=np.float64)[
                :, [col_map[s] for s in keep]
            ]
            returns_hist = pd.DataFrame(arr, index=hist_dates, columns=keep)
            returns_hist = returns_hist.loc[returns_hist.index < dates_idx[0]]
            returns = pd.concat(
                [returns_hist.reindex(columns=secids), returns_eval], axis=0
            ).sort_index()
            returns = returns[~returns.index.duplicated(keep="last")]
    dates_full = pd.DatetimeIndex(returns.index)
    artifact_dir = _crucible_artifact_dir(cfg)
    proxy_panels = False

    pre = cfg.get("_crucible_panels")
    if isinstance(pre, dict) and pre.get("returns") is not None:
        returns = pre["returns"]
        ff4 = pre["ff4_factors"]
        adv = pre["adv_panel"]
        amihud = pre["amihud_panel"]
        surface = pre.get("surface_panel")
        beta = pre["beta_panel"]
        eligible_override = pre.get("eligible_secids")
        surface_mode = "cfg_panels"
        dates_full = pd.DatetimeIndex(pd.to_datetime(returns.index))
    else:
        fac = _load_ff4(list(dates_full), lake_root)
        ff4 = pd.DataFrame(
            fac, index=dates_full, columns=["mkt_rf", "smb", "hml", "umd"]
        )
        adv, adv_source = _load_crucible_adv_panel(
            lake_root=lake_root,
            secids=secids,
            dates_idx=dates_full,
            cfg=cfg,
        )
        if adv is None:
            # Explicit smoke-only proxy; stamped below.
            vol = returns.rolling(63, min_periods=21).std()
            adv = (1.0 / vol.replace(0.0, np.nan)).clip(lower=1e5, upper=1e9)
            adv = adv.fillna(adv.median().median())
            proxy_panels = True
            surface_mode = "coverage_proxy_no_surface"
            amihud = returns.abs() / adv.replace(0.0, np.nan)
            amihud = amihud.fillna(amihud.median().median())
            beta = pd.DataFrame(1.0, index=dates_full, columns=secids)
            surface = pd.DataFrame(columns=["date", "secid", "signal", "value"])
            cov = returns.notna().rolling(63, min_periods=1).sum().iloc[-1]
            eligible_override = [int(s) for s, c in cov.items() if float(c) >= 21]
        else:
            amihud = returns.abs() / adv.replace(0.0, np.nan)
            amihud = amihud.fillna(amihud.median().median())
            beta = _rolling_market_beta(
                returns,
                ff4["mkt_rf"],
                lookback=int(cruc_cfg.get("lookback_days", 252)),
                min_periods=21,
            )
            beta = beta.fillna(1.0)
            surface, eligible_override, surface_mode = _load_crucible_surface_panel(
                lake_root=lake_root,
                secids=secids[: int(spec.max_pool)],
                dates_idx=dates_full,
                cfg=cfg,
                artifact_dir=artifact_dir,
            )
            if surface_mode == "coverage_proxy_no_surface":
                proxy_panels = True
                if eligible_override is None:
                    cov = returns.notna().rolling(63, min_periods=1).sum().iloc[-1]
                    eligible_override = [
                        int(s) for s, c in cov.items() if float(c) >= 21
                    ]
            _log_event(
                "crucible_panels",
                adv_source=adv_source,
                surface_mode=surface_mode,
                n_eligible=len(eligible_override or []),
                n_hist_rows=int((dates_full < dates_idx[0]).sum()),
                n_eval_rows=int(len(dates_idx)),
            )

    friction = friction_spec_from_cfg(cfg)
    # Size-adaptive: G1/G2 probe len(cur_secids), which may differ from campaign K
    # during repair. Locking k=campaign K caused "projector expected k=100, got 57".
    projector = make_cmdp_projector(cfg, k=None)

    u_mode = str(cfg.get("universe_cadence") or "quarterly_63d")
    u_mask = build_universe_cadence_mask(dates_idx, u_mode)
    policy_cadence = str(cfg.get("policy_cadence") or cfg.get("rebalance_cadence") or "daily")
    p_mask = build_rebalance_mask(dates_idx, policy_cadence)
    assert_universe_subset_of_policy(u_mask, p_mask)
    del rb_mask  # universe cadence is independent of policy rebalance cadence

    freeze_path = cfg.get("crucible_schedule_freeze_path") or cfg.get(
        "_crucible_schedule_freeze_path"
    )
    if freeze_path:
        from mascotrl.data.crucible import load_universe_schedule

        frozen = load_universe_schedule(freeze_path)
        slots_rows = [
            [None if s is None else int(s) for s in row]
            for row in frozen["slots_rows"]
        ]
        if len(slots_rows) != len(dates_idx):
            raise SystemExit(
                f"CRUCIBLE freeze schedule length {len(slots_rows)} != "
                f"eval dates {len(dates_idx)}"
            )
        crucible_block = {
            "surface_mode": surface_mode,
            "proxy_panels": bool(proxy_panels),
            "reselects": [],
            "fingerprint": frozen.get("selection_fingerprint")
            or frozen.get("fingerprint"),
            "schedule_fingerprint": frozen["schedule_fingerprint"],
            "schedule_freeze_path": str(freeze_path),
            "from_freeze": True,
        }
        selection_log = [
            {
                "date": str(pd.Timestamp(d).date()),
                "fingerprint": crucible_block["fingerprint"],
                "from_freeze": True,
            }
            for d in dates_idx
        ]
    else:
        slots_rows = []
        selection_log = []
        incumbent: list[int] | None = None
        last_sel: list[int] | None = None
        crucible_block = {
            "surface_mode": surface_mode,
            "proxy_panels": bool(proxy_panels),
            "reselects": [],
            "from_freeze": False,
        }

        for t, d in enumerate(dates_idx):
            if not bool(u_mask[t]) and last_sel is not None:
                slots_rows.append(list(last_sel))
                continue
            # Cold start or reselect day
            try:
                result = select_universe_crucible(
                    as_of=d,
                    pool_secids=secids[: int(spec.max_pool)],
                    returns=returns,
                    ff4_factors=ff4,
                    adv_panel=adv,
                    amihud_panel=amihud,
                    surface_panel=surface,
                    beta_panel=beta,
                    projector=projector,
                    friction_spec=friction,
                    spec=spec,
                    incumbent_secids=incumbent,
                    rng_seed=int(cfg.get("crucible_rng_seed", 0)),
                    book_notional=float(cruc_cfg.get("book_notional", 1_000_000.0)),
                    eligible_secids=eligible_override,
                    turnover_limit=(
                        float(cfg["turnover_limit"])
                        if str(cfg.get("projection_mode") or "").lower().strip()
                        == "hard"
                        and cfg.get("turnover_limit") is not None
                        else None
                    ),
                )
            except Exception as exc:
                if last_sel is not None:
                    slots_rows.append(list(last_sel))
                    selection_log.append(
                        {"date": str(d.date()), "error": str(exc), "kept_incumbent": True}
                    )
                    continue
                diag = getattr(exc, "diagnostics", None)
                if isinstance(diag, dict) and diag:
                    _log_event(
                        "crucible_gate_failure",
                        as_of=str(d.date()),
                        g1_pass=diag.get("g1_pass"),
                        g2_pass=diag.get("g2_pass"),
                        g3_pass=diag.get("g3_pass"),
                        g1_entropy_gap=diag.get("g1_entropy_gap"),
                        g1_entropy_gap_floor_effective=diag.get(
                            "g1_entropy_gap_floor_effective"
                        ),
                        g2_tc_post_projection=diag.get("g2_tc_post_projection"),
                        repair_passes_used=diag.get("repair_passes_used"),
                    )
                raise SystemExit(
                    f"CRUCIBLE selection failed at {d.date()}: {exc}. "
                    "Provide cfg['_crucible_panels'] with returns/ff4/adv/amihud/"
                    "surface/beta for a lake-backed run, or ensure the eval panel "
                    "is long enough for FF4 residualisation."
                ) from exc
            last_sel = list(result.secids)
            if len(last_sel) < int(k):
                for s in secids:
                    if s not in last_sel:
                        last_sel.append(s)
                    if len(last_sel) >= int(k):
                        break
            last_sel = last_sel[: int(k)]
            incumbent = list(last_sel)
            slots_rows.append(list(last_sel))
            selection_log.append(
                {
                    "date": str(d.date()),
                    "fingerprint": result.fingerprint,
                    "n": len(last_sel),
                }
            )
            crucible_block["reselects"].append(result.diagnostics)
            crucible_block["sleeve_primary"] = {
                str(kk): vv for kk, vv in result.sleeve_primary.items()
            }
            crucible_block["sleeve_membership"] = {
                str(sk): [int(x) for x in sv]
                for sk, sv in (result.sleeve_membership or {}).items()
            }
            crucible_block["sleeve_matrix"] = result.sleeve_matrix.tolist()
            crucible_block["fingerprint"] = result.fingerprint
            try:
                written = _write_crucible_selection_json(cfg, d, result.diagnostics)
                selection_log[-1]["selection_json"] = str(written)
            except OSError as exc:
                _log_event(
                    "crucible_selection_write_error",
                    as_of=str(d.date()),
                    error=str(exc)[:200],
                )

    if not slots_rows:
        raise SystemExit("CRUCIBLE produced empty slots_rows")

    from mascotrl.data.crucible import schedule_fingerprint, write_universe_schedule

    sched_fp = schedule_fingerprint(slots_rows)
    crucible_block["schedule_fingerprint"] = sched_fp
    try:
        sched_path = _crucible_artifact_dir(cfg) / "crucible_universe_schedule.json"
        write_universe_schedule(
            sched_path,
            slots_rows=slots_rows,
            dates=list(dates_idx),
            fingerprint=str(crucible_block.get("fingerprint") or sched_fp),
            selection_fingerprint=crucible_block.get("fingerprint"),
        )
        crucible_block["schedule_path"] = str(sched_path)
        cfg["_crucible_schedule_path"] = str(sched_path)
    except OSError as exc:
        _log_event("crucible_schedule_write_error", error=str(exc)[:200])

    col_map = {s: i for i, s in enumerate(secids)}
    slotted_panel = build_slotted_panel(
        dates=list(dates_idx),
        slots_rows=slots_rows,
        wide_returns=np.asarray(rets_e, dtype=np.float64),
        col_map=col_map,
    )
    fingerprint = sorted({s for row in slots_rows for s in row if s is not None})
    if crucible_block.get("fingerprint"):
        cfg["_crucible_universe_fingerprint"] = crucible_block["fingerprint"]
    if crucible_block.get("schedule_fingerprint"):
        cfg["_crucible_schedule_fingerprint"] = crucible_block["schedule_fingerprint"]
    turn_info = selection_turnover(slots_rows)
    # Scalar name-turnover rate for separate_turnover_keys consumers (not float(dict)).
    selection_to = float(turn_info.get("mean_added", 0.0)) / float(max(int(k), 1))
    info = {
        "arm": "dyn_crucible",
        "pool_size": len(secids),
        "turnover": turn_info,
        "n_rebalances": len(selection_log),
        "fingerprint_size": len(fingerprint),
        "selection_log": selection_log,
        "crucible": crucible_block,
        "selection_turnover": selection_to,
        # policy_turnover is filled later by the training loop; keep key distinct
        "policy_turnover": None,
    }
    valid_mask = np.ones((len(dates_idx), int(k)), dtype=bool)
    cfg["_slot_valid_mask"] = valid_mask
    cfg["_slots_rows"] = slots_rows
    cfg["_crucible_result"] = crucible_block
    cfg["_universe_reselect_mask"] = np.asarray(u_mask, dtype=bool)
    from mascotrl.eval.cpcv import stamp_reselect_purge_meta

    purge_meta = stamp_reselect_purge_meta(
        list(dates_idx),
        u_mask,
        purge_radius=int(
            (cfg.get("cpcv") or {}).get("purge_days")
            or cruc_cfg.get("cpcv_purge_radius", 21)
        ),
    )
    crucible_block["n_purged_at_reselect"] = int(purge_meta["n_purged_at_reselect"])
    crucible_block["reselect_purge"] = purge_meta
    info["n_purged_at_reselect"] = int(purge_meta["n_purged_at_reselect"])
    _stamp_dynamic_arm_pit(info, cfg=cfg, dates=list(dates_idx))
    return slotted_panel, fingerprint, info


def _build_dynamic_arm_universe(
    universe_arm: str,
    *,
    rets_e: np.ndarray,
    secids_e: list,
    dates: list,
    rb_mask: np.ndarray | None,
    k: int,
    args: argparse.Namespace,
    cfg: dict,
    rets_fit: np.ndarray,
    secids_w: list,
    lake_root,
    rets_w: np.ndarray | None = None,
    dates_w: list | None = None,
) -> tuple[np.ndarray, list, dict]:
    """Build a dynamic (per-rebalance) universe + slotted panel.

    Accepted arms: ``dyn_hrp``, ``dyn_liquidity``, ``dyn_crucible``.
    Sets ``cfg["_slot_valid_mask"]`` / ``cfg["_slots_rows"]`` so
    ``build_research_hist_env`` can mask inactive slots. Returns
    ``(slotted_panel, fingerprint, info)`` where ``fingerprint`` is every
    secid that ever occupied a slot.
    """
    del args, rets_fit
    if universe_arm == "dyn_crucible":
        return _build_crucible_universe(
            rets_e=rets_e,
            secids_e=secids_e,
            dates=dates,
            rb_mask=rb_mask,
            k=k,
            cfg=cfg,
            lake_root=lake_root,
            rets_hist=rets_w,
            secids_hist=secids_w,
            dates_hist=dates_w,
        )

    from mascotrl.data.dynamic_universe import (
        build_dynamic_universe,
        build_slotted_panel,
        select_universe_corr_cluster,
        select_universe_liquidity,
        selection_turnover,
    )

    mask = rb_mask if rb_mask is not None else np.ones(len(dates), dtype=bool)
    pool_secids = list(secids_e)
    pool_returns = rets_e
    select_kwargs: dict = {}
    info: dict = {"arm": universe_arm, "pool_size": len(pool_secids)}

    if universe_arm == "dyn_hrp":
        select_fn = select_universe_corr_cluster
    elif universe_arm == "dyn_liquidity":
        select_fn = select_universe_liquidity
    else:
        raise SystemExit(
            f"unhandled universe_arm={universe_arm!r}; accepted={ACCEPTED_UNIVERSE_ARMS}"
        )

    slots_rows, valid_mask, selection_log = build_dynamic_universe(
        dates=dates,
        rebalance_mask=mask,
        wide_returns=pool_returns,
        secids=pool_secids,
        k=int(k),
        select_fn=select_fn,
        trailing_days=252,
        select_kwargs=select_kwargs,
        eligibility_by_date=None,
    )
    col_map = {s: i for i, s in enumerate(pool_secids)}
    slotted_panel = build_slotted_panel(
        dates=dates, slots_rows=slots_rows, wide_returns=pool_returns, col_map=col_map
    )
    fingerprint = sorted({s for row in slots_rows for s in row if s is not None})
    info["turnover"] = selection_turnover(slots_rows)
    info["n_rebalances"] = len(selection_log)
    info["fingerprint_size"] = len(fingerprint)
    cfg["_slot_valid_mask"] = valid_mask
    cfg["_slots_rows"] = slots_rows
    _stamp_dynamic_arm_pit(info, cfg=cfg, dates=dates)
    return slotted_panel, fingerprint, info


def _stamp_dynamic_arm_pit(info: dict, *, cfg: dict, dates: list) -> None:
    """Stamp rolling slot-masked selection as PIT-clean."""
    from mascotrl.data.pit_guards import selection_pit_status
    from mascotrl.features.pit_universe import ROLLING_TRAILING_PIT

    cfg["universe_mode"] = ROLLING_TRAILING_PIT
    info["pit"] = selection_pit_status(
        universe_end=dates[-1] if dates else None,
        eval_start=dates[0] if dates else None,
        phase="dynamic_universe",
        universe_protocol="slot_masked",
    )


def _liquidity_k(df: pd.DataFrame, secids: list, k: int, start: str, end: str) -> list:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))]
    d = d[d["secid"].isin(secids)]
    if "volume" in d.columns and "close" in d.columns:
        d["dv"] = d["volume"].astype(float) * d["close"].astype(float)
        score = d.groupby("secid")["dv"].mean().sort_values(ascending=False)
    else:
        score = d.groupby("secid")["return"].count().sort_values(ascending=False)
    return list(score.head(k).index)


def _load_hv_table(lake_root, secids: list, start: str, end: str, *, days: int = 30) -> pd.DataFrame | None:
    """B1: OptionMetrics historical volatility, filtered to one tenor.

    ``macro/sp500_hv.parquet`` carries multiple ``days`` tenors per
    ``(secid, date)``; the surface signal wiring needs exactly one row per
    key (``vmp = log(hv) - log(atm_iv_30d)``), so this filters to the
    30-day tenor before returning.
    """
    path = Path(lake_root) / "macro" / "sp500_hv.parquet"
    if not path.is_file():
        return None
    df = pd.read_parquet(path, columns=["secid", "date", "days", "volatility"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[
        (df["secid"].isin(secids))
        & (df["date"] >= pd.Timestamp(start))
        & (df["date"] <= pd.Timestamp(end))
        & (pd.to_numeric(df["days"], errors="coerce") == int(days))
    ]
    return df.rename(columns={"volatility": "hv"})[["secid", "date", "hv"]]


def _load_option_volume_table(lake_root, secids: list, start: str, end: str) -> pd.DataFrame | None:
    """B1: total (calls + puts) option contract volume per (secid, date)."""
    path = Path(lake_root) / "macro" / "om_opvold.parquet"
    if not path.is_file():
        return None
    df = pd.read_parquet(path, columns=["secid", "date", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[
        (df["secid"].isin(secids))
        & (df["date"] >= pd.Timestamp(start))
        & (df["date"] <= pd.Timestamp(end))
    ]
    agg = df.groupby(["secid", "date"], as_index=False)["volume"].sum()
    return agg.rename(columns={"volume": "option_volume"})


def _load_borrow_table(lake_root, secids: list, start: str, end: str, *, days: int = 30) -> pd.DataFrame | None:
    """B1: OptionMetrics indicative borrow rate, filtered to one tenor."""
    path = Path(lake_root) / "macro" / "om_borrate.parquet"
    if not path.is_file():
        return None
    df = pd.read_parquet(path, columns=["secid", "date", "days", "borrowrate"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[
        (df["secid"].isin(secids))
        & (df["date"] >= pd.Timestamp(start))
        & (df["date"] <= pd.Timestamp(end))
        & (pd.to_numeric(df["days"], errors="coerce") == int(days))
    ]
    return df.rename(columns={"borrowrate": "borrow_rate"})[["secid", "date", "borrow_rate"]]


def _load_ff4(dates: list, lake) -> np.ndarray:
    """Align Ken French FF4 (Mkt-RF, SMB, HML, Mom) to ``dates``; zeros if missing."""
    path = Path(lake) / "macro" / "ff_factors.parquet"
    t = len(dates)
    out = np.zeros((t, 4), dtype=np.float64)
    if not path.is_file():
        return out
    ff = pd.read_parquet(path)
    if "date" not in ff.columns:
        return out
    ff = ff.copy()
    ff["date"] = pd.to_datetime(ff["date"])
    ff = ff.set_index("date").sort_index()
    cols = []
    for c in ("Mkt-RF", "SMB", "HML", "Mom"):
        if c in ff.columns:
            cols.append(c)
        else:
            return out
    aligned = ff[cols].reindex(pd.DatetimeIndex(dates)).fillna(0.0)
    # FF daily factors often in percent
    arr = aligned.to_numpy(dtype=np.float64)
    if np.nanmax(np.abs(arr)) > 1.0:
        arr = arr / 100.0
    return arr


def _factors_zeros(t: int, k: int) -> np.ndarray:
    return np.zeros((t, 4), dtype=np.float64)


def compute_eq_campaign_gates(
    *,
    fill_ladder: dict,
    policy_sharpe: float,
    challenger_sharpes: dict,
    series0: np.ndarray | None,
    series0_dates: list | None,
    panel_dates: list,
    lake_root,
) -> dict:
    """C7: gate1/gate2/gate3 for the eq allocation campaign, factored out of
    ``main()`` so each gate's date-alignment and fallback logic is directly
    unit-testable without running the full CPCV loop. Each gate is
    independently guarded: one gate failing (or lacking enough data) never
    blanks out the others.
    """
    from mascotrl.eval.spectrum_gates import compute_gate1, compute_gate2, compute_gate3

    gates: dict = {}

    try:
        mid = float(fill_ladder.get("mid", float("nan")))
        pct75 = float(fill_ladder.get("pct75", float("nan")))
        # Approximate break-even: fill_ladder rungs are Sharpe at spread
        # multiplier 0.5x ("mid") and 1.0x ("pct75") of the calibrated
        # om_touch spread (research_alpha_cpcv's ladder semantics), not a
        # mean-PnL cost ladder. Linearly extrapolate Sharpe-vs-multiplier to
        # its zero crossing as an honest proxy for
        # scripts/run_cpcv_campaign.py's exact gross/cost_at_full break-even
        # (which needs a full PnL-level re-run at multiple spread
        # multipliers, out of scope here).
        slope = pct75 - mid  # Sharpe change per +0.5x multiplier
        be = float("nan")
        if np.isfinite(mid) and np.isfinite(slope) and abs(slope) > 1e-9:
            be = 0.5 - mid * (0.5 / slope)
        gates["gate1"] = compute_gate1(
            {"break_even_spread_multiplier": be, "cost_source": "fill_ladder_sharpe_extrapolation"}
        )
    except Exception as e:  # noqa: BLE001
        gates["gate1_error"] = str(e)[:300]

    try:
        if series0 is not None and series0_dates and series0.size > 30:
            if len(series0_dates) == series0.size:
                fac_full = _load_ff4(panel_dates, lake_root)
                date_to_idx = {
                    str(pd.Timestamp(d).date()): i for i, d in enumerate(panel_dates)
                }
                rows = [date_to_idx.get(str(pd.Timestamp(d).date())) for d in series0_dates]
                keep = np.asarray([r is not None for r in rows], dtype=bool)
                if keep.sum() > 30:
                    idx = np.asarray([r for r in rows if r is not None], dtype=int)
                    gates["gate2"] = compute_gate2(series0[keep], fac_full[idx])
                else:
                    gates["gate2_skipped_reason"] = "fewer than 30 dates matched the FF4 panel"
            else:
                gates["gate2_skipped_reason"] = (
                    f"path pnl length {series0.size} != dates length {len(series0_dates)}"
                )
        else:
            gates["gate2_skipped_reason"] = "no policy path series available"
    except Exception as e:  # noqa: BLE001
        gates["gate2_error"] = str(e)[:300]

    try:
        gates["gate3"] = compute_gate3(policy_sharpe, challenger_sharpes)
    except Exception as e:  # noqa: BLE001
        gates["gate3_error"] = str(e)[:300]

    return gates


def _plot_campaign(results: dict, out_dir: Path) -> list[str]:
    """Ops campaign figures using figure_style (Okabe-Ito, no default matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mascotrl.reporting.book_style import FAMILY_PALETTE, family_color
    from mascotrl.reporting.figures.labels import human
    from mascotrl.reporting.figures.figure_style import (
        FIGURE_DPI,
        FIGURE_HEIGHT_DEFAULT_IN,
        FIGURE_HEIGHT_TALL_IN,
        FIGURE_WIDTH_FULL_IN,
        apply_figure_rc,
        greyscale_safe_styles,
        style_axes,
    )
    from mascotrl.reporting.figures.validate import run_figure_validators

    apply_figure_rc()
    paths: list[str] = []
    conf = results.get("confirmatory") or {}
    path_pnls = conf.get("path_pnls") or {}
    bench = conf.get("benchmark_sharpes") or {}
    olps = conf.get("olps_sharpes") or {}
    path_summary = conf.get("path_summary") or {}
    fill_ladder = conf.get("fill_ladder") or {}
    path_sharpes = list(path_summary.get("path_sharpes") or [])
    if not path_sharpes:
        for s in path_summary.get("per_seed") or []:
            if isinstance(s, (int, float)) and np.isfinite(s):
                path_sharpes.append(float(s))

    def _save(fig, stem: str) -> str:
        out = out_dir / f"{stem}.png"
        run_figure_validators(fig, stem=stem, strict=False)
        fig.savefig(
            out,
            dpi=FIGURE_DPI,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)
        paths.append(str(out))
        return str(out)

    zero_color = "#222222"

    # 1) Cumulative path fan
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_FULL_IN, FIGURE_HEIGHT_DEFAULT_IN),
        constrained_layout=True,
    )
    if path_pnls:
        styles = greyscale_safe_styles(min(len(path_pnls), 8))
        for i, (pid, series) in enumerate(sorted(path_pnls.items())):
            s = np.asarray(series, dtype=float)
            ax.plot(
                np.cumsum(np.nan_to_num(s)),
                color=family_color("policy"),
                ls=styles[i % len(styles)],
                lw=1.0 if i else 1.4,
                alpha=0.75,
                label=f"Path {pid}" if i < 8 else None,
            )
    ax.set_xlabel("OOS step (index)")
    ax.set_ylabel(human("cum_return", kind="axis"))
    if path_pnls:
        ax.legend(loc="best", frameon=False, fontsize=7)
    style_axes(ax, zero_line=True)
    _save(fig, "cum_pnl_paths")

    merged = dict(bench)
    for n, v in olps.items():
        merged[f"olps:{n}"] = v
    ceiling = conf.get("ceiling_sharpes") or {}
    for n, v in ceiling.items():
        merged[f"ceiling:{n}"] = v
    names = sorted(merged.keys(), key=lambda n: float(merged[n] or 0.0))
    pol_sr = float(path_summary.get("sharpe_mean") or float("nan"))
    if names:
        for stem in ("benchmark_sharpe_bars", "benchmark_panel_sharpe"):
            fig, ax = plt.subplots(
                figsize=(FIGURE_WIDTH_FULL_IN, FIGURE_HEIGHT_TALL_IN),
                constrained_layout=True,
            )
            sharpes = [float(merged[n]) for n in names]
            colors = [family_color(n) for n in names]
            ax.barh([human(n, kind="strategy") for n in names], sharpes, color=colors)
            if np.isfinite(pol_sr):
                ax.axvline(
                    pol_sr,
                    color=FAMILY_PALETTE["policy"],
                    ls="--",
                    lw=1.0,
                    label=f"Policy ({pol_sr:.3f})",
                )
                ax.legend(loc="lower right", frameon=False, fontsize=7)
            ax.set_xlabel(human("sharpe", kind="axis"))
            style_axes(ax, zero_line=True)
            _save(fig, stem)

    breadth = results.get("breadth") or {}
    if breadth:
        fig, ax = plt.subplots(
            figsize=(FIGURE_WIDTH_FULL_IN, FIGURE_HEIGHT_DEFAULT_IN),
            constrained_layout=True,
        )
        labels = [human(k, kind="strategy") for k in breadth.keys()]
        vals = [float(breadth[k].get("n_eff_enb", float("nan"))) for k in breadth.keys()]
        ax.bar(labels, vals, color=FAMILY_PALETTE["naive"])
        ax.set_ylabel("Effective names (count)")
        style_axes(ax)
        _save(fig, "breadth_by_universe")

    if path_sharpes:
        fig, ax = plt.subplots(
            figsize=(FIGURE_WIDTH_FULL_IN, FIGURE_HEIGHT_DEFAULT_IN),
            constrained_layout=True,
        )
        bp = ax.boxplot([path_sharpes], tick_labels=["Policy paths"], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(FAMILY_PALETTE["policy"])
            patch.set_alpha(0.55)
        ax.axhline(0.0, color=zero_color, ls="--", lw=0.8)
        ax.set_ylabel(human("sharpe", kind="axis"))
        style_axes(ax, zero_line=True)
        _save(fig, "path_sharpe_box")

    if path_pnls:
        pid0 = sorted(path_pnls.keys())[0]
        s = np.cumsum(np.nan_to_num(np.asarray(path_pnls[pid0], dtype=float)))
        peak = np.maximum.accumulate(s)
        dd = s - peak
        fig, ax = plt.subplots(
            figsize=(FIGURE_WIDTH_FULL_IN, FIGURE_HEIGHT_DEFAULT_IN),
            constrained_layout=True,
        )
        ax.fill_between(
            np.arange(dd.size),
            dd,
            0.0,
            color=FAMILY_PALETTE["ml_ceiling"],
            alpha=0.35,
        )
        ax.set_xlabel("OOS step (index)")
        ax.set_ylabel("Drawdown (fraction)")
        style_axes(ax, zero_line=True)
        _save(fig, "drawdown_path0")

        r = np.asarray(path_pnls[pid0], dtype=float)
        win = 63
        if r.size > win + 5:
            roll = []
            for i in range(win, r.size):
                w = r[i - win : i]
                mu, sd = float(np.mean(w)), float(np.std(w))
                roll.append((mu / (sd + 1e-12)) * np.sqrt(252.0) if sd > 0 else float("nan"))
            fig, ax = plt.subplots(
                figsize=(FIGURE_WIDTH_FULL_IN, FIGURE_HEIGHT_DEFAULT_IN),
                constrained_layout=True,
            )
            ax.plot(roll, color=family_color("policy"), lw=1.2)
            ax.axhline(0.0, color=zero_color, ls="--", lw=0.8)
            ax.set_xlabel("OOS step (index)")
            ax.set_ylabel(human("sharpe", kind="axis"))
            style_axes(ax, zero_line=True)
            _save(fig, "rolling_sharpe_path0")

    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH_FULL_IN, FIGURE_HEIGHT_DEFAULT_IN),
        constrained_layout=True,
    )
    if path_pnls:
        for pid, series in list(path_pnls.items())[:8]:
            r = np.asarray(series, dtype=float)
            mu = float(np.nanmean(r)) * 252.0
            vol = float(np.nanstd(r)) * np.sqrt(252.0)
            ax.scatter(
                vol,
                mu,
                color=family_color("policy"),
                label=f"Path {pid}",
                s=24,
            )
    ax.axhline(0.0, color=zero_color, ls="--", lw=0.8)
    ax.axvline(0.0, color=zero_color, ls="--", lw=0.8)
    ax.set_xlabel("Annualised volatility (fraction)")
    ax.set_ylabel("Annualised mean return (fraction)")
    if path_pnls:
        ax.legend(loc="best", frameon=False, fontsize=7)
    style_axes(ax)
    _save(fig, "risk_return_scatter")

    if fill_ladder:
        fig, ax = plt.subplots(
            figsize=(FIGURE_WIDTH_FULL_IN, FIGURE_HEIGHT_DEFAULT_IN),
            constrained_layout=True,
        )
        labs = list(fill_ladder.keys())
        vals = [float(fill_ladder[k]) for k in labs]
        ax.bar(labs, vals, color=FAMILY_PALETTE["olps"])
        ax.set_ylabel(human("sharpe", kind="axis"))
        style_axes(ax, zero_line=True)
        _save(fig, "cost_ladder")

    return paths


def _wfo_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "no_wfo", False))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "config/workflows/arm_equity.yaml"),
    )
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="Universe size K (default: workflow YAML `k:`, else 100). "
        "Bakeoff passes an explicit value; YAML remains the documented default.",
    )
    ap.add_argument("--dii-epochs", type=int, default=80)
    ap.add_argument("--seeds", default="0,1,2")
    # W4.3 / A16: CLI default None so YAML max_pool (headline 511) wins;
    # interactive callers can still pass --max-pool for a smaller pool.
    ap.add_argument("--max-pool", type=int, default=None)
    ap.add_argument("--skip-rl", action="store_true", help="benchmarks+breadth only")
    ap.add_argument(
        "--no-wfo",
        action="store_true",
        help="disable the default expanding-window equity WFO diagnostic "
        "(when run it is stamped is_cpcv=false and never capital-grade)",
    )
    ap.add_argument(
        "--universe-arm",
        choices=ACCEPTED_UNIVERSE_ARMS,
        default=None,
        help="dyn_hrp: correlation-cluster control; dyn_liquidity: coverage/vol "
        "control; dyn_crucible: CRUCIBLE behavioural-sleeve universe. "
        "dyn_hrp/dyn_liquidity re-select at every policy rebalance; "
        "dyn_crucible uses universe_cadence (default quarterly_63d). "
        "Defaults to the workflow YAML's `universe_arm:` key, or dyn_liquidity "
        "when that key is absent.",
    )
    ap.add_argument(
        "--neg-control-policy",
        action="store_true",
        help="opt-in: also re-run one seed/fold under permuted signals "
        "(expensive; default scores the signal pathway only)",
    )
    ap.add_argument(
        "--no-surface-signals",
        action="store_true",
        help="override use_surface_signals=false regardless of the config file "
        "(entry-point smoke tests that do not need the signal-gate fail-closed path)",
    )
    ap.add_argument("--train-episodes", type=int, default=None)
    ap.add_argument("--train-env-steps", type=int, default=None)
    ap.add_argument("--train-epochs", type=int, default=None)
    ap.add_argument("--min-optimizer-steps-total", type=int, default=None)
    ap.add_argument("--min-optimizer-steps", type=int, default=None)
    ap.add_argument(
        "--kelly-n-seeds",
        type=int,
        default=None,
        help="kelly_cnn ceiling-arm ensemble size (default 3; smoke tests "
        "pass a smaller value so the O(T/refit_every) expanding-window "
        "refit schedule fits in a smoke-scale wall-clock budget)",
    )
    ap.add_argument("--kelly-refit-every", type=int, default=None)
    ap.add_argument("--kelly-epochs", type=int, default=None)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Artifact root (default: logs/artifacts/eq_alloc). Bakeoff arms "
        "must pass a distinct path so they do not clobber the headline seal.",
    )
    ap.add_argument(
        "--force-overwrite",
        action="store_true",
        help="allow replacing a larger-K cpcv_path_summary.json in --out-dir",
    )
    ap.add_argument(
        "--seed-workers",
        type=_at_least_one_int,
        default=_default_seed_workers(),
        help="parallel CPCV seed fan-out (ProcessPoolExecutor spawn). "
        "Default: MASCOTRL_SEED_WORKERS env or 1. Min 1.",
    )
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    from mascotrl.data.paths import assert_lake_mounted

    assert_lake_mounted()

    global OUT, _JSONL
    if args.out_dir:
        OUT = Path(args.out_dir)
        OUT.mkdir(parents=True, exist_ok=True)
        _JSONL = OUT / "campaign.jsonl"

    _install_diagnostic_signals(OUT)
    _write_heartbeat(OUT, phase="main_start")

    from mascotrl.data.equity_panel import (
        SELECTION_START,
        SELECTION_END,
        EVAL_START,
        EVAL_END,
        load_sp500_security_returns,
    )
    from mascotrl.data.paths import LAKE_ROOT
    from mascotrl.eval.kahn_breadth import selection_breadth_metrics
    from mascotrl.eval.benchmark_panel import BENCHMARK_PANEL_NAMES
    from mascotrl.eval.stats_rigor import annualized_sharpe
    from mascotrl.eval.cpcv import CPCVConfig
    from mascotrl.eval.research_alpha_cpcv import (
        run_policy_level_negative_control,
        run_research_alpha_cpcv,
    )
    from mascotrl.eval.campaign_manifest import (
        atomic_write_json,
        load_manifest,
        save_manifest,
        is_cell_complete,
        mark_cell_complete,
        purge_orphan_fold_cells,
    )

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(OUT / "campaign_run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    _log_event(
        "campaign_start",
        config=str(args.config),
        k=args.k,
        seeds=str(args.seeds),
        out_dir=str(OUT),
    )

    cfg = _load_cfg(Path(args.config))
    cfg["_out_dir"] = str(OUT)
    from mascotrl.eval.yaml_honesty import assert_turnover_cap_honesty

    assert_turnover_cap_honesty(cfg)
    # A16: CLI --max-pool overrides YAML; otherwise YAML (511) then 400.
    if args.max_pool is None:
        args.max_pool = int(cfg.get("max_pool") or 400)
    else:
        args.max_pool = int(args.max_pool)
    cfg["max_pool"] = int(args.max_pool)
    # CLI --k overrides YAML k; always touch YAML k for tracked honesty.
    args.k = resolve_campaign_k(args, cfg)
    _assert_disk_budget(out_dir=args.out_dir, k=int(args.k))
    _log_event("k_resolved", k=int(args.k), yaml_k=cfg.get("k"))
    cfg.setdefault("headline_fill", "pct75")
    cfg.setdefault("primary_train", "historical_arm_env")
    cfg.setdefault("claim_label_stem", "stk_ret")
    cfg.setdefault("equity_bps", 5.0)
    cfg.setdefault("impact_c_eq", 0.5)
    cfg.setdefault("train_epochs", 3)
    cfg.setdefault("reward", "differential_sharpe")
    cfg.setdefault("use_equity_feature_cube", True)
    cfg.setdefault("feature_seq_len", 1)
    cfg.setdefault("rebalance_cadence", "monthly")
    cfg.setdefault("checkpoint_every_n_episodes", 1)
    # Stamp lake BEFORE any feature attach so attach_feature_net_extras cannot
    # silently no-op on lake=None (parity audit A4).
    from mascotrl.data.paths import LAKE_ROOT as _LAKE_ROOT_EARLY

    cfg.setdefault("_lake_root", str(_LAKE_ROOT_EARLY))
    cfg.setdefault("lake_root", str(_LAKE_ROOT_EARLY))
    if args.no_surface_signals:
        cfg["use_surface_signals"] = False
        cfg["surface_obs_lane"] = "off"
    cfg.setdefault("surface_obs_lane", "geometry_lite")
    if str(cfg.get("surface_obs_lane") or "") == "off":
        cfg["use_surface_signals"] = False
    if args.kelly_n_seeds is not None:
        cfg["kelly_n_seeds"] = args.kelly_n_seeds
    if args.kelly_refit_every is not None:
        cfg["kelly_refit_every"] = args.kelly_refit_every
    if args.kelly_epochs is not None:
        cfg["kelly_epochs"] = args.kelly_epochs
    # C1: fail closed on any unregistered spectrum axis value before any
    # data is touched (train_world / architecture / objective / algo).
    from mascotrl.spectrum.registry import validate_cfg

    validate_cfg(cfg)
    # C1: fail closed on any unregistered spectrum axis value.
    from mascotrl.spectrum.registry import validate_cfg as _validate_spectrum_cfg

    _validate_spectrum_cfg(cfg)
    if args.train_episodes is not None:
        cfg["train_episodes"] = int(args.train_episodes)
    if args.train_env_steps is not None:
        cfg["train_env_steps"] = int(args.train_env_steps)
    if args.train_epochs is not None:
        cfg["train_epochs"] = int(args.train_epochs)
    if args.min_optimizer_steps_total is not None:
        cfg["min_optimizer_steps_total"] = int(args.min_optimizer_steps_total)
    if args.min_optimizer_steps is not None:
        cfg["min_optimizer_steps"] = int(args.min_optimizer_steps)
    cfg["_learning_curves_dir"] = str(OUT / "curves")
    Path(cfg["_learning_curves_dir"]).mkdir(parents=True, exist_ok=True)
    cfg["claim_category"] = "eq_stk_allocation"
    cfg["claim_tier"] = "research"

    manifest = load_manifest(OUT)

    t0 = time.perf_counter()
    raw = load_sp500_security_returns(LAKE_ROOT, start="2003-01-02", end=EVAL_END)
    _log_event("load", rows=int(len(raw)), wall_s=float(time.perf_counter() - t0))

    # Selection window panel
    rets_w, secids_w, idx_w = _wide_returns(raw, start=SELECTION_START, end=SELECTION_END)
    # Cap pool for CPU: take highest-coverage names already filtered; then top by vol activity
    if rets_w.shape[1] > args.max_pool:
        activity = np.nanstd(rets_w, axis=0)
        keep = np.argsort(activity)[::-1][: args.max_pool]
        rets_w = rets_w[:, keep]
        secids_w = [secids_w[i] for i in keep]
    # Subsample selection window for densest diagnostic breadth (stride).
    stride = max(1, rets_w.shape[0] // 400)
    rets_fit = rets_w[::stride]
    _log_event(
        "selection",
        T=int(rets_w.shape[0]),
        D=int(rets_w.shape[1]),
        fit_T=int(rets_fit.shape[0]),
        stride=int(stride),
        max_pool=int(args.max_pool),
    )

    k = min(int(args.k), rets_fit.shape[1])

    results: dict = {"cfg": args.config, "k": k, "pool": int(rets_w.shape[1]), "fit_stride": stride}
    breadth: dict = {}

    # --- densest / liquidity controls (diagnostic only; never selection) ---
    dens_idx = _densest_k(rets_fit, k)
    breadth["densest"] = selection_breadth_metrics(rets_w, dens_idx)

    liq_secids = _liquidity_k(raw, secids_w, k, SELECTION_START, SELECTION_END)
    liq_idx = np.asarray([secids_w.index(s) for s in liq_secids if s in secids_w], dtype=int)
    if liq_idx.size:
        breadth["liquidity"] = selection_breadth_metrics(rets_w, liq_idx)
    results["breadth"] = breadth

    # Eval panel 2014-2024; dynamic arms select from this pool at each rebalance.
    use_avail = bool(cfg.get("use_availability_mask", True))
    rets_e, secids_e, dates_e, avail_e = _wide_returns_with_availability(
        raw,
        start=EVAL_START,
        end=EVAL_END,
        min_cov=0.70,
        ffill_limit=5,
        keep_partial_rows=use_avail,
    )
    del avail_e  # dyn arms stamp all-True slot availability below
    results["use_availability_mask"] = use_avail

    dates = list(dates_e)
    # Phase B: month-end (default) / weekly / daily rebalance mask shared by
    # policy + peers. Computed before the universe-arm branch below because
    # dyn_* arms need it to know which days trigger re-selection.
    from mascotrl.eval.cadence import build_rebalance_mask

    cadence = str(cfg.get("rebalance_cadence") or "monthly")
    cfg["rebalance_cadence"] = cadence
    rb_mask = build_rebalance_mask(dates, cadence)

    # Universe-arm resolution: CLI > YAML > dyn_liquidity.
    # Accepted: dyn_hrp, dyn_liquidity, dyn_crucible.
    universe_arm = resolve_universe_arm(cli_arm=args.universe_arm, cfg=cfg)
    if universe_arm not in ACCEPTED_UNIVERSE_ARMS:
        raise SystemExit(
            f"universe_arm={universe_arm!r} not accepted after DII/TE removal; "
            f"expected one of {ACCEPTED_UNIVERSE_ARMS}"
        )
    cfg["universe_arm"] = universe_arm
    panel, universe_secids, dynamic_universe_info = _build_dynamic_arm_universe(
        universe_arm,
        rets_e=rets_e,
        secids_e=secids_e,
        dates=dates,
        rb_mask=rb_mask,
        k=k,
        args=args,
        cfg=cfg,
        rets_fit=rets_fit,
        secids_w=secids_w,
        lake_root=LAKE_ROOT,
        rets_w=rets_w,
        dates_w=list(idx_w),
    )
    universe_secids = _set_universe_secids(results, universe_secids)
    # Dynamic arms rotate slots; stamp all-True until slot-level availability
    # is threaded through the dyn builder (still honest vs punctured dropna).
    slot_avail = np.ones(panel.shape, dtype=bool)
    results["universe_arm"] = universe_arm
    results["dynamic_universe"] = dynamic_universe_info
    # Promote CRUCIBLE block for behaviour / dissertation consumers.
    if isinstance(dynamic_universe_info.get("crucible"), dict):
        results["crucible"] = dynamic_universe_info["crucible"]
    _log_event(
        "universe_select",
        arm=universe_arm,
        n_fingerprint=int(len(universe_secids)),
        n_eff=(breadth.get("liquidity") or {}).get("n_eff_enb"),
    )
    # Align cfg arm slots to realized panel width before RL.
    realized_k = int(panel.shape[1])
    cfg["n_assets"] = realized_k
    cfg["_slot_valid_mask"] = np.asarray(slot_avail, dtype=bool)
    arm_cfg = dict(cfg.get("arm") or {})
    arm_cfg["id"] = "eq"
    arm_cfg["option_slots"] = 0
    arm_cfg["equity_slots"] = realized_k
    arm_cfg.setdefault("delta_mode", "off")
    cfg["arm"] = arm_cfg
    # Phase L extras: dollar volume for Amihud when present on the lake panel.
    extras: dict = dict(cfg.get("feature_extras") or {})
    if "volume" in raw.columns and "close" in raw.columns:
        tmp = raw.copy()
        tmp["dollar_volume"] = pd.to_numeric(tmp["volume"], errors="coerce") * pd.to_numeric(
            tmp["close"], errors="coerce"
        )
        dv = _wide_field(tmp, secids=universe_secids, dates=dates, value_col="dollar_volume")
        if dv is not None and dv.shape == panel.shape:
            extras["dollar_volume"] = dv
        elif dv is not None and cfg.get("_slots_rows") is not None:
            # Dyn arms: fingerprint ADV is (T, N>>K); slot-align onto the panel.
            from mascotrl.features.blocks.liquidity import map_wide_to_slots

            extras["dollar_volume"] = map_wide_to_slots(
                dv, secids=universe_secids, slots_rows=cfg["_slots_rows"]
            )
    cfg["feature_extras"] = extras

    # Feature-net panels (OHLC/micro/fundamentals/…) stay OFF by default.
    # Gate G0: running H0 checkpoints are K*26 (no feature-net). Enabling
    # attach_feature_net_extras silently would inflate obs dim and break
    # spectrum parity. Opt in only via use_feature_net_extras=true.
    cfg.setdefault("use_feature_net_extras", False)
    if bool(cfg.get("use_feature_net_extras")):
        try:
            from mascotrl.eval.feature_extras_loader import attach_feature_net_extras

            lake_for_feat = cfg.get("lake_root") or cfg.get("_lake_root")
            if lake_for_feat is None:
                raise RuntimeError(
                    "use_feature_net_extras=true but lake_root/_lake_root missing"
                )
            extras = attach_feature_net_extras(
                extras,
                lake=lake_for_feat,
                dates=dates,
                secids=universe_secids,
                slots_rows=cfg.get("_slots_rows"),
            )
            if "feature_groups_exclude" in cfg:
                extras["feature_groups_exclude"] = list(
                    cfg.get("feature_groups_exclude") or []
                )
            if "feature_channels_exclude" in cfg:
                extras["feature_channels_exclude"] = list(
                    cfg.get("feature_channels_exclude") or []
                )
            cfg["feature_extras"] = extras
            if not any(
                k in extras
                for k in (
                    "ohlc",
                    "microstructure",
                    "sentiment",
                    "fundamentals_pit",
                    "option_flow",
                    "jkp",
                    "macro",
                )
            ):
                raise RuntimeError(
                    "use_feature_net_extras=true but no feature-net panels attached"
                )
        except Exception as exc:
            _log_event("feature_net_extras_error", error=str(exc)[:300])
            raise

    # A5: real market-cap panel for cap_weight_bah (close * shrout); without
    # this the benchmark silently fell back to equal weight every step.
    mktcap_panel = None
    if "close" in raw.columns and "shrout" in raw.columns:
        tmp_mc = raw.copy()
        tmp_mc["mktcap"] = pd.to_numeric(tmp_mc["close"], errors="coerce") * pd.to_numeric(
            tmp_mc["shrout"], errors="coerce"
        )
        mc = _wide_field(tmp_mc, secids=universe_secids, dates=dates, value_col="mktcap")
        if mc is not None and mc.shape == panel.shape:
            mktcap_panel = mc
    if mktcap_panel is None:
        # Dyn arms rotate which secid occupies a slot over time, so a single
        # static (dates, secid)-indexed mktcap panel does not have a stable
        # column identity. cap_weight_bah degrades to equal-weight when None.
        print(
            "dyn_* universe arm: no static mktcap panel (per-slot secid "
            "identity rotates over time); cap_weight_bah benchmark falls "
            "back to equal weight for this run"
        )
    cfg["_universe_secids"] = list(universe_secids)
    # W3.1: record the CLI knobs that shape the realized universe/training
    # run into cfg so _campaign_config_fingerprint sees them (previously
    # only args.dii_epochs/args.max_pool existed, invisible to the fingerprint).
    cfg["dii_epochs"] = int(args.dii_epochs)
    cfg["max_pool"] = int(args.max_pool)
    cfg["_rebalance_mask"] = rb_mask
    from mascotrl.eval.calendar_scaling import eval_panel_meta, periods_per_year_from_dates

    ppy = float(periods_per_year_from_dates(dates))
    cfg["_periods_per_year"] = ppy
    rb_days = (
        int(np.asarray(rb_mask).sum()) if rb_mask is not None else int(panel.shape[0])
    )
    results["eval_panel"] = eval_panel_meta(
        dates,
        k=realized_k,
        intended_start=EVAL_START,
        intended_end=EVAL_END,
        rebalance_days=rb_days,
    )
    cov = float(results["eval_panel"].get("coverage_frac") or 1.0)
    if cov < 0.90:
        results["eval_panel_coverage_warning"] = True
        _log_event(
            "eval_panel_coverage_warning",
            coverage_frac=cov,
            t=int(results["eval_panel"]["t"]),
            periods_per_year=ppy,
            date_start=results["eval_panel"]["date_start"],
            date_end=results["eval_panel"]["date_end"],
        )
    print(
        f"eval panel T={panel.shape[0]} K={realized_k} feature_cube={cfg.get('use_equity_feature_cube')} "
        f"cadence={cadence} rebalance_days={rb_days} "
        f"periods_per_year={ppy:.2f} coverage={cov:.3f} "
        f"span={results['eval_panel']['date_start']}..{results['eval_panel']['date_end']}"
    )

    # B1: wire real option-implied surface signals into the observation cube.
    # B2: refuse to start if the allowlist gate has not run / admitted
    # nothing, rather than silently falling back to the momentum proxy.
    # Dual-track: surface_obs_lane selects CS allowlist vs geometry pack vs off.
    # P1: dyn_* arms use slot-aware alignment so whichever secid occupies a
    # slot contributes that name's published signal (not a static column).
    if bool(cfg.get("use_surface_signals", False)):
        from mascotrl.eval.signal_gate import (
            assert_allowlist_valid,
            assert_geometry_pack_valid,
        )
        from mascotrl.data.surface_signals import (
            align_signals_to_slots,
            materialize_surface_signals_from_lake,
        )

        lane = str(cfg.get("surface_obs_lane") or "geometry_lite")
        pack_id = None
        if lane in {"geometry_lite", "geometry"}:
            pack_path = cfg.get("obs_pack_path") or "config/obs_packs/surf_geometry_lite.yaml"
            try:
                pack = assert_geometry_pack_valid(pack_path)
            except (ValueError, FileNotFoundError) as e:
                raise SystemExit(
                    f"surface_obs_lane={lane!r} but obs pack invalid: {e}"
                ) from e
            signal_names = list(pack.get("channels") or [])
            pack_id = str(pack.get("pack_id") or "surf_geometry_lite")
            if not signal_names:
                raise SystemExit(
                    f"surface_obs_lane={lane!r} pack {pack_id!r} has empty channels"
                )
            allow_data = {"allowlist": signal_names, "pack_id": pack_id}
        elif lane in {"cs_admit", "surf_cs_admit"}:
            allowlist_path = cfg.get("signal_allowlist_path")
            try:
                allow_data = assert_allowlist_valid(allowlist_path)
            except (ValueError, FileNotFoundError) as e:
                raise SystemExit(
                    f"use_surface_signals=true but allowlist invalid/empty "
                    f"(fail-closed, run scripts/run_signal_gate.py first): {e}"
                ) from e
            signal_names = list(allow_data.get("allowlist") or [])
            pack_id = "surf_cs_admit"
        else:
            raise SystemExit(
                f"unknown surface_obs_lane={lane!r}; "
                "expected cs_admit | geometry_lite | off"
            )
        cfg["_obs_pack_id"] = pack_id
        allowlist = list(signal_names)
        t3 = time.perf_counter()
        # A month of lookback before EVAL_START so the first eval date has
        # a published (lagged) month-end value to forward-fill from.
        surf_start = str((pd.Timestamp(EVAL_START) - pd.DateOffset(months=2)).date())
        # Dyn arms: materialize over every secid that ever occupies a slot.
        slots_rows = cfg.get("_slots_rows")
        if slots_rows is None:
            raise SystemExit(
                f"universe_arm={universe_arm!r} requires cfg['_slots_rows'] "
                "from the dynamic universe builder"
            )
        signal_secids = sorted(
            {s for row in slots_rows for s in row if s is not None},
            key=lambda x: str(x),
        )
        signals_long = materialize_surface_signals_from_lake(
            LAKE_ROOT,
            secids=signal_secids,
            start=surf_start,
            end=EVAL_END,
            cache_path=OUT / "dyn_surface_signals.parquet",
            hv=_load_hv_table(LAKE_ROOT, signal_secids, surf_start, EVAL_END),
            option_volume=_load_option_volume_table(
                LAKE_ROOT, signal_secids, surf_start, EVAL_END
            ),
            equity_volume=raw.rename(columns={"volume": "equity_volume"})[
                ["secid", "date", "equity_volume"]
            ] if "volume" in raw.columns else None,
            borrow=_load_borrow_table(LAKE_ROOT, signal_secids, surf_start, EVAL_END),
            month_end_only=True,
        )
        extras["iv_surface"] = align_signals_to_slots(
            signals_long,
            dates,
            slots_rows,
            lag_days=1,
            signal_names=allowlist,
        )
        align_mode = "slots"
        cfg["feature_extras"] = extras
        results.pop("surface_signals_skipped_reason", None)
        results["surface_signals"] = {
            "allowlist": allowlist,
            "surface_obs_lane": lane,
            "obs_pack_id": pack_id,
            "n_month_end_rows": int(len(signals_long)),
            "materialize_s": time.perf_counter() - t3,
            "align_mode": align_mode,
            "n_signal_secids": int(len(signal_secids)),
        }
        # Fail-closed when an admitted surface channel is mostly all-NaN.
        from mascotrl.eval.equity_substrate import assert_surface_nan_ok

        iv = extras.get("iv_surface") or {}
        if iv:
            nan_diag = assert_surface_nan_ok(
                iv,
                channel_names=list(iv.keys()),
                admitted_channels=allowlist,
            )
            results["feature_nan_diagnostics"] = nan_diag
        _log_event(
            "surface_signals",
            allowlist=allowlist,
            surface_obs_lane=lane,
            obs_pack_id=pack_id,
            month_end_rows=int(len(signals_long)),
            align_mode=align_mode,
            wall_s=results["surface_signals"]["materialize_s"],
        )

    # Wave 3: fioracle macro on the observation cube (cfg feature_extras flag).
    from mascotrl.data.macro_loader import attach_fioracle_macro_cube, fioracle_cfg_from_feature_extras

    fio_on, _, _ = fioracle_cfg_from_feature_extras(cfg)
    cfg["_lake_root"] = str(LAKE_ROOT)
    # Macro span covers IS through OOS so CPCV folds inherit aligned columns.
    macro_start = str(cfg.get("hist_panel_start") or cfg.get("is_hist_start") or "2003-01-01")
    macro_end = str(cfg.get("oos_end") or EVAL_END)
    fio_meta = attach_fioracle_macro_cube(
        cfg,
        lake_base_dir=LAKE_ROOT,
        start_date=macro_start,
        end_date=macro_end,
        dates=dates,
        out_dir=OUT if fio_on else None,
        prefer_arctic=False,
    )
    results["fioracle_enabled"] = bool(fio_meta.get("fioracle_enabled"))
    results["macro_column_order"] = list(fio_meta.get("macro_column_order") or [])
    if fio_meta.get("regime_labels_path"):
        results["regime_labels_path"] = fio_meta["regime_labels_path"]
    if fio_on:
        extras = dict(cfg.get("feature_extras") or {})
        cfg["feature_extras"] = extras
        _log_event(
            "fioracle_macro",
            enabled=True,
            n_cols=len(results["macro_column_order"]),
            macro_shape=fio_meta.get("macro_shape"),
            regime_labels_path=results.get("regime_labels_path"),
        )

    # Benchmarks on confirmatory universe — parity harness (dual scorecard).
    _write_heartbeat(OUT, phase="pre_seed_benchmarks")
    _log_event("pre_seed_benchmarks")
    from mascotrl.arms import ArmSpec
    from mascotrl.eval.friction import friction_spec_from_cfg
    from mascotrl.eval.parity_harness import (
        assert_same_scorecard,
        estimand_hash,
        require_uniform_estimand_hashes,
        score_benchmark_panel,
        score_strategy,
    )
    from mascotrl.eval.residualization import fit_ff4_residualizer, freeze_residualizer
    from mascotrl.eval.olps import olps_claim_names, olps_weights

    t2 = time.perf_counter()
    fac_for_bench = _load_ff4(dates, LAKE_ROOT)
    if fac_for_bench.shape[1] != 4:
        fac_for_bench = _factors_zeros(panel.shape[0], 4)
    arm_eq = ArmSpec(
        id="eq", option_slots=0, equity_slots=int(panel.shape[1]), delta_mode="off"
    )
    fric = friction_spec_from_cfg(cfg)
    y_ew = np.nanmean(panel, axis=1)
    resid_state = freeze_residualizer(
        fit_ff4_residualizer(y_ew, fac_for_bench, fold_id="eq_alloc_bench"),
        "eq_alloc_bench",
    )
    rebalance_mask = cfg.get("_rebalance_mask")  # set by Phase B when present
    if rebalance_mask is not None:
        rebalance_mask = np.asarray(rebalance_mask, dtype=bool)

    bench_names = list(BENCHMARK_PANEL_NAMES)
    bench_scored = score_benchmark_panel(
        bench_names,
        panel,
        factors=fac_for_bench,
        arm=arm_eq,
        friction=fric,
        residualizer=resid_state,
        rebalance_mask=rebalance_mask,
        mktcap=mktcap_panel,
        cadence=cadence,
        universe=universe_secids,
        turnover_cap=float(cfg.get("turnover_limit") or 0.15),
        slot_valid_mask=cfg.get("_slot_valid_mask"),
    )
    results["benchmark_panel_s"] = time.perf_counter() - t2
    results["confirmatory"] = {
        "benchmark_sharpes": {
            n: float(annualized_sharpe(v["total_net"], periods=ppy)) for n, v in bench_scored.items()
        },
        "benchmark_sharpes_residual": {
            n: float(annualized_sharpe(v["residual"], periods=ppy)) for n, v in bench_scored.items()
        },
        "benchmark_turnover_ann": {
            n: float(np.nansum(v["turnover"]) * float(ppy) / max(1, panel.shape[0]))
            for n, v in bench_scored.items()
        },
        "benchmark_estimand_hashes": {
            n: v["estimand_hash"] for n, v in bench_scored.items()
        },
        "estimand_hash": estimand_hash(
            friction=fric,
            residualize=True,
            cadence=cadence,
            universe=universe_secids,
            rebalance_mask=rebalance_mask,
        ),
    }
    # Keep EW total_net series for SPA parity comparison.
    ew_total_net = bench_scored.get("equal_weight", {}).get("total_net")
    if ew_total_net is None and bench_scored:
        ew_total_net = next(iter(bench_scored.values()))["total_net"]
    results["confirmatory"]["_ew_total_net"] = (
        list(map(float, np.asarray(ew_total_net).reshape(-1)[:5000]))
        if ew_total_net is not None
        else None
    )

    olps_sharpes = {}
    olps_sharpes_residual = {}
    olps_hashes = {}
    olps_scored: dict[str, dict] = {}
    # Gate3 peers: claimable OLPS only (never stub EG aliases). Keep the
    # established 4-name panel; ``up`` is claimable but MC-heavy.
    _olps_peer_names = tuple(
        n for n in ("bah", "eg", "olmar", "pamr") if n in olps_claim_names()
    )
    for name in _olps_peer_names:
        try:

            def _olps_fn(
                returns_hist,
                *,
                t,
                w_prev,
                _name=name,
                **_kw,
            ):
                return olps_weights(_name, returns_hist, w_prev=w_prev)

            scored = score_strategy(
                _olps_fn,
                panel,
                factors=fac_for_bench,
                arm=arm_eq,
                friction=fric,
                residualizer=resid_state,
                rebalance_mask=rebalance_mask,
                cadence=cadence,
                universe=universe_secids,
            )
            olps_sharpes[name] = float(annualized_sharpe(scored["total_net"], periods=ppy))
            olps_sharpes_residual[name] = float(annualized_sharpe(scored["residual"], periods=ppy))
            olps_hashes[name] = scored["estimand_hash"]
            olps_scored[name] = scored
        except Exception as e:
            # A10: record the reason but do not pretend NaN is a legitimate
            # score; a failed OLPS peer fails the run closed at the end of
            # main() rather than silently degrading the benchmark panel.
            olps_sharpes[name] = float("nan")
            olps_sharpes_residual[name] = float("nan")
            results.setdefault("olps_errors", {})[name] = str(e)[:200]
    results["confirmatory"]["olps_sharpes"] = olps_sharpes
    results["confirmatory"]["olps_sharpes_residual"] = olps_sharpes_residual
    results["confirmatory"]["olps_estimand_hashes"] = olps_hashes

    # Phase G: non-RL ceiling arms on the same parity harness.
    from mascotrl.eval.ceiling_arms import CEILING_ARM_NAMES, ceiling_arm_weight_fn

    ceiling_sharpes = {}
    ceiling_sharpes_residual = {}
    ceiling_hashes = {}
    ceiling_scored: dict[str, dict] = {}
    # B3: prefer real allowlisted surface signals (already materialized
    # above when use_surface_signals=true and the gate admitted names)
    # over the synthetic momentum proxy; only degrade to the proxy when no
    # gated signal is available, and say so explicitly in the artifact.
    iv_surface_extras = dict(cfg.get("feature_extras") or {}).get("iv_surface")
    if iv_surface_extras:
        proxy_signals = dict(iv_surface_extras)
        results["ceiling_arm_signals"] = "surface_allowlist"
    else:
        mom = np.zeros_like(panel)
        mom[21:] = panel[21:] - panel[:-21]
        proxy_signals = {"mom_21": mom}
        results["ceiling_arm_signals"] = "mom_21_proxy"
    # kelly_images is keyed one-per-secid with a static column identity;
    # dyn_* arms rotate slot occupants, so skip rather than feed a mismatched
    # fingerprint-width image tensor.
    kelly_images = None
    results["kelly_images_skipped_reason"] = "dynamic_universe_no_static_secid_identity"
    if bool(cfg.get("use_surface_image_encoder", False)):
        raise SystemExit(
            "use_surface_image_encoder=true is incompatible with dynamic "
            "universe arms (no static kelly_images grid)"
        )
    for cname in CEILING_ARM_NAMES:
        if cname == "kelly_cnn":
            results.setdefault("ceiling_skipped", {})[cname] = "dynamic_universe_no_kelly_images"
            continue
        try:
            fn = ceiling_arm_weight_fn(
                cname,
                signals=proxy_signals,
                long_only=True,
                kelly_images=kelly_images,
                returns=panel,
                kelly_n_seeds=int(cfg.get("kelly_n_seeds", 3)),
                kelly_refit_every=int(cfg.get("kelly_refit_every", 21)),
                kelly_epochs=int(cfg.get("kelly_epochs", 10)),
            )
            scored = score_strategy(
                fn,
                panel,
                factors=fac_for_bench,
                arm=arm_eq,
                friction=fric,
                residualizer=resid_state,
                rebalance_mask=rebalance_mask,
                cadence=cadence,
                universe=universe_secids,
            )
            ceiling_sharpes[cname] = float(annualized_sharpe(scored["total_net"], periods=ppy))
            ceiling_sharpes_residual[cname] = float(annualized_sharpe(scored["residual"], periods=ppy))
            ceiling_hashes[cname] = scored["estimand_hash"]
            ceiling_scored[cname] = scored
        except Exception as e:
            # A10: record the reason; fail closed at end of main() instead
            # of quietly shipping a NaN ceiling arm.
            ceiling_sharpes[cname] = float("nan")
            results.setdefault("ceiling_errors", {})[cname] = str(e)[:200]
    results["confirmatory"]["ceiling_sharpes"] = ceiling_sharpes
    results["confirmatory"]["ceiling_sharpes_residual"] = ceiling_sharpes_residual
    results["confirmatory"]["ceiling_estimand_hashes"] = ceiling_hashes

    # Phase I negative controls: score the same signal pathway the policy
    # reads (zscore_composite on proxy_signals, via the shared parity
    # harness) under three corruptions and demand a real fail-closed
    # verdict. Headline estimand is beta-free: long-short demeaned weights
    # scored on residual PnL, compared to an identically built uncorrupted
    # arm via degradation ratio. Long-only total_net numbers are retained
    # as continuity fields only (market beta alone can breach abs floors).
    # Re-running the RL agent itself under each corruption is opt-in
    # (--neg-control-policy) because it multiplies campaign cost.
    from mascotrl.eval.negative_controls import (
        date_shift_signals,
        permute_signals_across_names,
        run_negative_controls,
        shuffled_return_rows,
    )

    shuffled_panel = shuffled_return_rows(panel, seed=0)
    ew_shuf = float(annualized_sharpe(np.nanmean(shuffled_panel, axis=1), periods=ppy))
    results["confirmatory"]["negative_controls_prelim"] = {
        "shuffled_panel_ew_sharpe": ew_shuf,
    }

    def _signal_control_sharpe(
        signals_ctrl: dict,
        returns_ctrl: np.ndarray,
        *,
        long_only: bool,
        scorecard: str,
    ) -> float:
        fn_ctrl = ceiling_arm_weight_fn(
            "zscore_composite", signals=signals_ctrl, long_only=bool(long_only)
        )
        scored_ctrl = score_strategy(
            fn_ctrl,
            returns_ctrl,
            factors=fac_for_bench,
            arm=arm_eq,
            friction=fric,
            residualizer=resid_state,
            rebalance_mask=rebalance_mask,
            cadence=cadence,
            universe=universe_secids,
        )
        key = "residual" if scorecard == "residual" else "total_net"
        return float(annualized_sharpe(scored_ctrl[key], periods=ppy))

    try:
        n_ctrl_reps = 5
        # Beta-free headline rung (degradation verdict).
        clean_residual = _signal_control_sharpe(
            proxy_signals, panel, long_only=False, scorecard="residual"
        )
        sh_shuffled = float(
            np.mean(
                [
                    _signal_control_sharpe(
                        proxy_signals,
                        shuffled_return_rows(panel, seed=s),
                        long_only=False,
                        scorecard="residual",
                    )
                    for s in range(n_ctrl_reps)
                ]
            )
        )
        sh_permuted = float(
            np.mean(
                [
                    _signal_control_sharpe(
                        permute_signals_across_names(proxy_signals, seed=s),
                        panel,
                        long_only=False,
                        scorecard="residual",
                    )
                    for s in range(n_ctrl_reps)
                ]
            )
        )
        sh_date_shifted = _signal_control_sharpe(
            date_shift_signals(proxy_signals, shift=21),
            panel,
            long_only=False,
            scorecard="residual",
        )
        neg_control_verdict = run_negative_controls(
            control_sharpe_on_shuffled=sh_shuffled,
            control_sharpe_on_permuted_signals=sh_permuted,
            control_sharpe_on_date_shifted=sh_date_shifted,
            clean_sharpe=clean_residual,
        )
        # Continuity: long-only total_net under the same corruptions.
        lo_clean = _signal_control_sharpe(
            proxy_signals, panel, long_only=True, scorecard="total_net"
        )
        lo_shuffled = float(
            np.mean(
                [
                    _signal_control_sharpe(
                        proxy_signals,
                        shuffled_return_rows(panel, seed=s),
                        long_only=True,
                        scorecard="total_net",
                    )
                    for s in range(n_ctrl_reps)
                ]
            )
        )
        lo_permuted = float(
            np.mean(
                [
                    _signal_control_sharpe(
                        permute_signals_across_names(proxy_signals, seed=s),
                        panel,
                        long_only=True,
                        scorecard="total_net",
                    )
                    for s in range(n_ctrl_reps)
                ]
            )
        )
        lo_date_shifted = _signal_control_sharpe(
            date_shift_signals(proxy_signals, shift=21),
            panel,
            long_only=True,
            scorecard="total_net",
        )
        results["confirmatory"]["negative_controls"] = {
            "target": "signal_pathway_zscore_composite",
            "estimand": "long_short_residual",
            **neg_control_verdict,
            "long_only_total_net": {
                "clean": lo_clean,
                "shuffled_labels": lo_shuffled,
                "permuted_signals": lo_permuted,
                "date_shifted_signals": lo_date_shifted,
            },
        }
    except Exception as e:
        results.setdefault("negative_controls_errors", {})["exception"] = str(e)[:300]

    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip() != ""]
    # Orphan fold completions (crash after fold cache, before seed JSON) cause
    # resume to skip every fold with optimizer_steps=0. Purge those first.
    removed = purge_orphan_fold_cells(
        manifest, out_dir=OUT, arm="eq_dii", seeds=seeds
    )
    if removed:
        save_manifest(OUT, manifest)
        print(
            f"purged {len(removed)} orphan fold resume cell(s) "
            f"(no cpcv_seed_*.json for those seeds)"
        )
    cpcv = CPCVConfig(
        n_splits=int(cfg.get("cpcv_n_splits", 6)),
        n_test_groups=int(cfg.get("cpcv_n_test_groups", 2)),
        purge_days=int(cfg.get("cpcv_purge_days", 21)),
        embargo_days=int(cfg.get("cpcv_embargo_days", 21)),
    )
    # The manifest is resumable-by-design (a multi-hour production run should
    # survive a crash without re-paying already-completed seeds), but that is
    # only safe if the cached seed artifact was produced under the *same*
    # training config. Without this fingerprint, changing --k / seeds /
    # train_env_steps / the surface-signal allowlist between invocations
    # silently reuses a stale cpcv_seed_N.json computed under a different
    # config, which is an estimand-integrity defect, not a resume feature.
    run_config_hash = _campaign_config_fingerprint(cfg, realized_k=realized_k)
    results["run_config_hash"] = run_config_hash
    results["confirmatory"]["run_config_hash"] = run_config_hash

    policy_frame_kw: dict[str, Any] | None = None
    if not args.skip_rl:
        fac = _load_ff4(dates, LAKE_ROOT)
        if fac.shape[1] != 4:
            fac = _factors_zeros(panel.shape[0], 4)
        seed_arts: list[dict] = []
        path_pnls_all: dict[str, Any] = {}
        path_dates_all: dict[str, list] = {}
        arts_by_seed: dict[int, dict] = {}
        pending_seeds: list[int] = []
        # CRUCIBLE: freeze schedule path so workers never reselect universe.
        sched = cfg.get("_crucible_schedule_path") or cfg.get(
            "crucible_schedule_freeze_path"
        )
        if sched:
            cfg["crucible_schedule_freeze_path"] = str(sched)

        for seed in seeds:
            if is_cell_complete(manifest, fold_id=-1, seed=seed, arm="eq_dii"):
                entry = (manifest.get("completed") or {}).get(f"-1|{seed}|eq_dii") or {}
                cached_hash = (entry.get("extra") or {}).get("run_config_hash")
                if cached_hash != run_config_hash:
                    print(
                        f"seed={seed} manifest cell stale (run_config_hash "
                        f"{cached_hash!r} != {run_config_hash!r}); retraining"
                    )
                else:
                    art_path = OUT / f"cpcv_seed_{seed}.json"
                    prior_sh = entry.get("extra", {}).get("sharpe_mean")
                    if prior_sh in (None, "nan") and not art_path.is_file():
                        pass
                    elif art_path.is_file():
                        art = json.loads(art_path.read_text())
                        arts_by_seed[int(seed)] = art
                        for key in entry.get("extra", {}).get("accessed_keys") or []:
                            cfg.get(key)
                        print(f"skip completed seed={seed} (loaded {art_path.name})")
                        continue
                    else:
                        print(f"skip completed seed={seed}")
                        continue
            completed = dict(manifest.get("completed") or {})
            completed.pop(f"-1|{seed}|eq_dii", None)
            manifest["completed"] = completed
            pending_seeds.append(int(seed))

        if pending_seeds:
            save_manifest(OUT, manifest)

        seed_workers = max(1, int(getattr(args, "seed_workers", 1) or 1))
        use_pool = seed_workers > 1 and len(seeds) > 1 and len(pending_seeds) > 0

        if use_pool:
            pack_dir = OUT / "_seed_pack"
            threads_per = max(
                1, int(os.environ.get("MASCOTRL_THREADS_PER_WORKER", "4") or "4")
            )
            payload_base = _write_seed_pack(
                pack_dir,
                panel=panel,
                factors=fac,
                dates=dates,
                cfg=cfg,
                cpcv=cpcv,
                run_config_hash=run_config_hash,
                realized_k=realized_k,
            )
            _log_event(
                "seed_pool_start",
                n_pending=len(pending_seeds),
                seed_workers=seed_workers,
                threads_per_worker=threads_per,
                run_config_hash=run_config_hash,
            )
            _write_heartbeat(
                OUT,
                phase="seed_pool_start",
                n_pending=len(pending_seeds),
                seed_workers=seed_workers,
            )
            payloads = [
                {
                    **payload_base,
                    "seed": int(seed),
                    "out_dir": str(OUT),
                    "repo_root": str(ROOT),
                    "threads_per": threads_per,
                    "panel_source": "equity_sp500",
                    "resume": True,
                }
                for seed in pending_seeds
            ]
            ctx = multiprocessing.get_context("spawn")
            max_workers = min(seed_workers, len(pending_seeds))
            # Spawn pickles by module path; when this file is executed as
            # __main__, re-bind to the importable scripts.* symbol.
            worker_fn = _seed_worker_main
            if __name__ == "__main__":
                import scripts.run_eq_alloc_campaign as _camp_mod

                worker_fn = _camp_mod._seed_worker_main
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers, mp_context=ctx
            ) as pool:
                futures = [pool.submit(worker_fn, p) for p in payloads]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
            manifest = load_manifest(OUT)
            for seed in pending_seeds:
                art_path = OUT / f"cpcv_seed_{seed}.json"
                if not art_path.is_file():
                    raise SystemExit(
                        f"seed worker finished but missing artifact: {art_path}"
                    )
                arts_by_seed[int(seed)] = json.loads(
                    art_path.read_text(encoding="utf-8")
                )
                entry = (manifest.get("completed") or {}).get(
                    f"-1|{seed}|eq_dii"
                ) or {}
                for key in entry.get("extra", {}).get("accessed_keys") or []:
                    cfg.get(key)
            _log_event(
                "seed_pool_done",
                n_pending=len(pending_seeds),
                run_config_hash=run_config_hash,
            )
        else:
            import threading

            for seed in pending_seeds:
                _log_event(
                    "seed_start", seed=int(seed), run_config_hash=run_config_hash
                )
                _write_heartbeat(
                    OUT,
                    phase="seed_start",
                    seed=int(seed),
                    run_config_hash=run_config_hash,
                )
                stop_hb = threading.Event()
                hb_thread = threading.Thread(
                    target=_heartbeat_loop,
                    args=(OUT, stop_hb),
                    kwargs={
                        "phase": "seed_train",
                        "interval_s": 300.0,
                        "seed": int(seed),
                        "run_config_hash": run_config_hash,
                    },
                    name=f"heartbeat-seed{seed}",
                    daemon=True,
                )
                hb_thread.start()
                try:
                    art = _run_one_seed_cpcv(
                        int(seed),
                        dates,
                        panel,
                        fac,
                        cfg,
                        cpcv,
                        OUT,
                        run_config_hash,
                        panel_source="equity_sp500",
                        resume=True,
                        repo_root=ROOT,
                    )
                finally:
                    stop_hb.set()
                    hb_thread.join(timeout=2.0)
                manifest = load_manifest(OUT)
                arts_by_seed[int(seed)] = art
                _log_event(
                    "seed_done",
                    seed=int(seed),
                    wall_s=art.get("wall_s"),
                    sharpe_mean=(art.get("path_summary") or {}).get("sharpe_mean"),
                    estimand_hash=art.get("estimand_hash"),
                    run_config_hash=run_config_hash,
                )
                _write_heartbeat(
                    OUT,
                    phase="seed_done",
                    seed=int(seed),
                    sharpe_mean=(art.get("path_summary") or {}).get("sharpe_mean"),
                )

        # Aggregate in CLI seed order (resume + newly trained).
        for seed in seeds:
            art = arts_by_seed.get(int(seed))
            if art is None:
                continue
            seed_arts.append(art)
            _collect_seed_art_paths(int(seed), art, path_pnls_all, path_dates_all)

        if seed_arts:
            sharpes = [
                float((a.get("path_summary") or {}).get("sharpe_mean") or float("nan"))
                for a in seed_arts
            ]
            if bool(args.neg_control_policy):
                policy_seed = int(seeds[0]) if seeds else 0
                results["confirmatory"]["negative_controls"][
                    "policy_level"
                ] = run_policy_level_negative_control(
                    dates,
                    panel,
                    fac,
                    cfg,
                    cpcv=cpcv,
                    seed=policy_seed,
                    clean_sharpe=float(sharpes[0]),
                    signals=proxy_signals,
                )
            # Prefer path sharpes from first completed seed for boxplots.
            path_sharpes = list(
                ((seed_arts[0].get("path_summary") or {}).get("path_sharpes")) or []
            )
            results["confirmatory"]["path_summary"] = {
                "sharpe_mean": float(np.nanmean(sharpes)),
                "sharpe_std": float(np.nanstd(sharpes)),
                "n_seeds": len(sharpes),
                "per_seed": sharpes,
                "path_sharpes": path_sharpes,
            }
            # PREREG arm-selection fields: collapse / turnover binding / L1 vs EW.
            results["confirmatory"]["decision_fields"] = _stamp_selection_vs_sizing(
                _aggregate_decision_fields(seed_arts, sharpes),
                confirmatory=results["confirmatory"],
                path_pnls_all=path_pnls_all,
            )
            results["confirmatory"]["fill_ladder"] = dict(seed_arts[0].get("fill_ladder") or {})
            results["confirmatory"]["path_pnls"] = {
                k: list(map(float, np.asarray(v).reshape(-1)[:5000]))
                for k, v in list(path_pnls_all.items())[:8]
            }
            # Fail-closed capital (gates only flip after confirmatory spec §22d)

            # W7 / Wave 2: policy-behavior archetype harness (interpretation only).
            # Pass CRUCIBLE sleeves + fioracle regimes/macros when available.
            try:
                from mascotrl.reporting.policy_behavior import (
                    build_policy_behavior,
                    extract_crucible_behaviour_inputs,
                    load_behaviour_macro_context,
                    pack_policy_behavior_campaign_record,
                    plot_archetype_figures,
                    write_policy_behavior,
                )

                path0_w = None
                path0_to = None
                path0_dates: list = []
                for a in seed_arts:
                    p0 = (a.get("paths") or {}).get("0") or {}
                    if isinstance(p0, dict) and p0.get("weights") is not None:
                        path0_w = np.asarray(p0["weights"], dtype=float)
                        path0_to = p0.get("turnover")
                        path0_dates = list(p0.get("dates") or [])
                        break
                diag0 = (seed_arts[0].get("policy_diagnostics") or {}) if seed_arts else {}
                cruc_bh = extract_crucible_behaviour_inputs(results=results, cfg=cfg)
                macro_dates = path0_dates if path0_dates else list(dates)
                fioracle_cfg = dict(
                    (cfg.get("feature_extras") or {}).get("fioracle_macro") or {}
                )
                lake_subdir = str(
                    fioracle_cfg.get("lake_subdir") or "macro/fioracle"
                )
                macro_ctx = load_behaviour_macro_context(
                    macro_dates,
                    lake_root=LAKE_ROOT,
                    lake_subdir=lake_subdir,
                )
                # Align panel returns to path-0 dates when lengths match.
                asset_returns = None
                if path0_w is not None:
                    if path0_dates and len(path0_dates) == int(path0_w.shape[0]):
                        date_to_i = {
                            str(pd.Timestamp(d).date()): i
                            for i, d in enumerate(dates)
                        }
                        rows = []
                        for d in path0_dates:
                            key = str(pd.Timestamp(d).date())
                            if key in date_to_i:
                                rows.append(panel[date_to_i[key]])
                        if len(rows) == int(path0_w.shape[0]):
                            asset_returns = np.asarray(rows, dtype=np.float64)
                    elif int(path0_w.shape[0]) == int(panel.shape[0]):
                        asset_returns = np.asarray(panel, dtype=np.float64)
                extras_bh: dict[str, Any] = {
                    "seed_sharpes": sharpes,
                    "policy_diagnostics": diag0,
                    "entropy_series": diag0.get("entropy_series"),
                    "turnover_series": path0_to,
                    "cmdp_slack_series": seed_arts[0].get("cmdp_slack_series"),
                    "macro_context_status": dict(macro_ctx.get("status") or {}),
                }
                if cruc_bh.get("sleeve_membership") is not None:
                    extras_bh["sleeve_membership"] = cruc_bh["sleeve_membership"]
                if cruc_bh.get("sleeve_primary") is not None:
                    extras_bh["sleeve_primary"] = cruc_bh["sleeve_primary"]
                behavior = build_policy_behavior(
                    algo=str(cfg.get("algo") or "ppo"),
                    architecture=str(cfg.get("architecture") or cfg.get("temporal_backend") or ""),
                    objective=str(cfg.get("objective") or cfg.get("reward") or ""),
                    train_world=str(cfg.get("train_world") or ""),
                    policy_mode=str(cfg.get("policy_mode") or "balanced"),
                    universe_fingerprint=str(
                        cruc_bh.get("universe_fingerprint")
                        or cfg.get("_crucible_universe_fingerprint")
                        or ""
                    ),
                    weights=path0_w,
                    turnovers=path0_to,
                    entropies=diag0.get("entropy_series"),
                    sensitivities=seed_arts[0].get("policy_sensitivities"),
                    asset_returns=asset_returns,
                    sleeve_matrix=cruc_bh.get("sleeve_matrix"),
                    regimes=macro_ctx.get("regimes"),
                    vix_z=macro_ctx.get("vix_z"),
                    hy_oas_z=macro_ctx.get("hy_oas_z"),
                    term_spread=macro_ctx.get("term_spread"),
                    epu_z=macro_ctx.get("epu_z"),
                    gpri_z=macro_ctx.get("gpri_z"),
                    turnover_cap=(
                        float(cfg["turnover_limit"])
                        if cfg.get("turnover_limit") is not None
                        else None
                    ),
                    cell_cfg=cfg,
                    extras=extras_bh,
                )
                behavior_path = write_policy_behavior(
                    OUT / "policy_behavior.json", behavior
                )
                fig_paths = plot_archetype_figures(behavior, OUT / "report" / "archetypes")
                results["policy_behavior"] = pack_policy_behavior_campaign_record(
                    behavior,
                    path=behavior_path,
                    figures=fig_paths,
                    macro_status=macro_ctx.get("status"),
                )
            except Exception as e:
                results["policy_behavior_error"] = str(e)[:300]

            # W6: optional expanding-window WFO reported alongside CPCV.
            if _wfo_enabled(args):
                from mascotrl.eval.equity_nested_wfo import run_equity_nested_wfo

                wfo_seed = int(seeds[0]) if seeds else 0
                results["nested_wfo_eq"] = run_equity_nested_wfo(
                    dates, panel, fac, cfg, seed=wfo_seed, n_folds=5
                )
                _log_event(
                    "wfo_done",
                    sharpe_mean=(results["nested_wfo_eq"] or {}).get("sharpe_mean"),
                    is_cpcv=False,
                )

            # Lightweight DSR / SPA / Romano-Wolf table (policy as challenger; fail-closed).
            try:
                from mascotrl.eval.stats_rigor import deflated_sharpe_ratio, hansen_spa_test
                from mascotrl.eval.stats_inference import romano_wolf_stepdown, cscv_pbo_from_paths
                from mascotrl.eval.olps import filter_olps_stubs_from_peers

                pol = float(np.nanmean(sharpes))
                bench_map = dict(results["confirmatory"].get("benchmark_sharpes") or {})
                challenger_raw = {
                    **bench_map,
                    **{f"olps:{k}": v for k, v in (results["confirmatory"].get("olps_sharpes") or {}).items()},
                    **{
                        f"ceiling:{k}": v
                        for k, v in (results["confirmatory"].get("ceiling_sharpes") or {}).items()
                    },
                }
                # P5: EG-fallback stubs are not distinct peers for beat counts.
                challenger = filter_olps_stubs_from_peers(challenger_raw)
                series0 = None
                series0_key = None
                if path_pnls_all:
                    series0_key = next(iter(path_pnls_all))
                    series0 = np.asarray(path_pnls_all[series0_key], dtype=float)
                stats_tbl: dict = {
                    "policy_sharpe_mean": pol,
                    "benchmarks_beaten_by_mean_sharpe": [
                        n for n, v in challenger.items() if np.isfinite(v) and pol > float(v)
                    ],
                    "n_benchmarks": len(challenger),
                    "spa_polarity": "policy_as_challenger",
                }
                if series0 is not None and series0.size > 30:
                    dsr_n_trials, dsr_meta = _estimate_campaign_dsr_trials(
                        ROOT / "logs" / "trial_ledger.json", cfg=dict(cfg)
                    )
                    dsr = deflated_sharpe_ratio(
                        series0, n_trials=dsr_n_trials, periods_per_year=ppy
                    )
                    stats_tbl["deflated_sharpe"] = dsr
                    stats_tbl["dsr_n_trials"] = int(dsr_n_trials)
                    stats_tbl["dsr_n_trials_source"] = dsr_meta.get("source")
                    stats_tbl["dsr_n_trials_auditable"] = bool(
                        dsr_meta.get("auditable", False)
                    )
                    stats_tbl["dsr_n_trials_breakdown"] = dsr_meta
                    # Parity: compare policy residual/total_net to EW total_net from harness.
                    ew_series = results["confirmatory"].get("_ew_total_net")
                    if ew_series is None:
                        ew = np.asarray(
                            bench_scored.get("equal_weight", {}).get("total_net"),
                            dtype=float,
                        )
                    else:
                        ew = np.asarray(ew_series, dtype=float)
                    n = min(series0.size, ew.size)
                    # A2: EW is scored total_net; the headline policy series
                    # (`art["paths"]`) is also total_net (was residual before
                    # the fix). Fail closed if that ever drifts.
                    policy_scorecard = str(seed_arts[0].get("scorecard") or "")
                    assert_same_scorecard(policy_scorecard, "total_net")
                    spa = hansen_spa_test(
                        ew[:n],
                        {"policy": series0[:n]},
                        n_boot=199,
                        block_mean=21,
                        seed=0,
                    )
                    stats_tbl["hansen_spa_vs_ew"] = spa
                    try:
                        from mascotrl.eval.ledoit_wolf_sharpe import sharpe_difference_test

                        stats_tbl["sharpe_diff_vs_equal_weight"] = sharpe_difference_test(
                            series0[:n],
                            ew[:n],
                            n_boot=199,
                            block_mean=21,
                            seed=0,
                            periods=ppy,
                        )
                    except Exception as _sde:  # noqa: BLE001
                        stats_tbl["sharpe_diff_vs_equal_weight_error"] = str(_sde)[:200]
                    rw = romano_wolf_stepdown(
                        ew[:n],
                        {"policy": series0[:n]},
                        n_boot=199,
                        block_mean=21,
                        seed=0,
                    )
                    stats_tbl["romano_wolf_vs_ew"] = {
                        "rejected": rw.get("rejected"),
                        "results": rw.get("results"),
                        "reason": rw.get("reason"),
                    }
                    path_mats = [
                        np.asarray(v, dtype=float).reshape(-1)
                        for v in list(path_pnls_all.values())[:15]
                    ]
                    if len(path_mats) >= 2:
                        stats_tbl["cscv_pbo"] = cscv_pbo_from_paths(path_mats, seed=0)
                # Fail-closed: every scored peer AND the policy must share the
                # estimand hash (A6: the policy used to be silently excluded
                # from this gate).
                hash_entries = {
                    "policy": {"estimand_hash": seed_arts[0].get("estimand_hash")},
                    **{
                        f"bench:{n}": {"estimand_hash": h}
                        for n, h in (
                            results["confirmatory"].get("benchmark_estimand_hashes") or {}
                        ).items()
                    },
                    **{
                        f"olps:{n}": {"estimand_hash": h}
                        for n, h in (
                            results["confirmatory"].get("olps_estimand_hashes") or {}
                        ).items()
                        if h
                    },
                    **{
                        f"ceiling:{n}": {"estimand_hash": h}
                        for n, h in (
                            results["confirmatory"].get("ceiling_estimand_hashes") or {}
                        ).items()
                        if h
                    },
                }
                common_h = require_uniform_estimand_hashes(hash_entries)
                stats_tbl["estimand_hash"] = common_h
                results["confirmatory"]["stats_table"] = stats_tbl
                atomic_write_json(OUT / "stats_table.json", stats_tbl)
            except Exception as e:
                # A10: record the reason; fail closed at end of main() since
                # a broken stats table (SPA/hash-uniformity) means the run
                # produced no valid evidence.
                results["confirmatory"]["stats_table_error"] = str(e)[:400]

            # C7: gate1 (cost break-even), gate2 (positive FF4 alpha), gate3
            # (same-fold vs peer panel), one shared module
            # (src/eval/spectrum_gates.py).
            gates = compute_eq_campaign_gates(
                fill_ladder=results["confirmatory"].get("fill_ladder") or {},
                policy_sharpe=pol,
                challenger_sharpes=challenger,
                series0=series0,
                series0_dates=path_dates_all.get(series0_key) if series0_key else None,
                panel_dates=dates,
                lake_root=LAKE_ROOT,
            )
            results["confirmatory"]["gates"] = gates
            arms_dir = ROOT / "logs" / "artifacts" / "arms" / "eq"
            arms_dir.mkdir(parents=True, exist_ok=True)
            (arms_dir / "gate3_same_fold.json").write_text(
                json.dumps({"gate3": gates.get("gate3") or {}}, indent=2, default=str, sort_keys=True)
            )

            # D2: the frozen policy's real path-0 holdings, keyed the same
            # way _reconstruct_path0_aux_series build them in
            # src/eval/research_alpha_cpcv.py, so strategy_frame() below can
            # build the policy's parquet the same way it builds every peer's.
            path0 = (seed_arts[0].get("paths") or {}).get("0") or {}
            if path0.get("dates") and path0.get("weights"):
                policy_frame_kw = {
                    "dates": path0["dates"],
                    "weights": np.asarray(path0["weights"], dtype=np.float64),
                    "turnover": path0.get("turnover") or [],
                    "cost": path0.get("cost") or [],
                    "gross": path0.get("gross") or [],
                    "total_net": path0.get("pnl"),
                }

    results["breadth"] = breadth
    results["pilot_only"] = False
    results["wall_total_s"] = time.perf_counter() - t0

    # D2: persist every strategy's real OOS weights/turnover/cost/gross to
    # parquet (one file per strategy) plus a combined PortfolioAccountingLedger,
    # so the reporting book (D3/D4) reads holdings off disk instead of
    # re-deriving them from raw CPCV/parity-harness artifacts.
    from mascotrl.reporting.strategy_persistence import (
        build_accounting_ledger,
        persist_strategy_frames,
        strategy_frame,
    )

    strategy_frames: dict[str, pd.DataFrame] = {}
    for name, scored in bench_scored.items():
        strategy_frames[name] = strategy_frame(
            dates=[dates[i] for i in scored["t_index"]],
            secids=universe_secids,
            weights=scored["weights"],
            turnover=scored["turnover"],
            cost=scored["cost"],
            gross=scored["gross"],
            total_net=scored["total_net"],
            residual=scored["residual"],
        )
    for name, scored in olps_scored.items():
        strategy_frames[f"olps:{name}"] = strategy_frame(
            dates=[dates[i] for i in scored["t_index"]],
            secids=universe_secids,
            weights=scored["weights"],
            turnover=scored["turnover"],
            cost=scored["cost"],
            gross=scored["gross"],
            total_net=scored["total_net"],
            residual=scored["residual"],
        )
    for name, scored in ceiling_scored.items():
        strategy_frames[f"ceiling:{name}"] = strategy_frame(
            dates=[dates[i] for i in scored["t_index"]],
            secids=universe_secids,
            weights=scored["weights"],
            turnover=scored["turnover"],
            cost=scored["cost"],
            gross=scored["gross"],
            total_net=scored["total_net"],
            residual=scored["residual"],
        )
    if policy_frame_kw is not None:
        strategy_frames["policy"] = strategy_frame(secids=universe_secids, **policy_frame_kw)

    strategy_frames_dir = ROOT / "logs" / "artifacts" / "arms" / "eq" / "strategy_frames"
    written = persist_strategy_frames(strategy_frames, strategy_frames_dir)
    results["strategy_frames"] = written
    if strategy_frames:
        ledger = build_accounting_ledger(secids=universe_secids, frames=strategy_frames)
        ledger_path = ledger.export_parquet(strategy_frames_dir / "portfolio_ledger.parquet")
        results["portfolio_ledger_parquet"] = str(ledger_path) if ledger_path else None
        # W5: single robust holdings book in report/ (Excel + CSV twin).
        try:
            from mascotrl.reporting.strategy_persistence import write_holdings_book

            report_dir = OUT / "report"
            report_dir.mkdir(parents=True, exist_ok=True)
            ticker_by_secid: dict[Any, str] = {}
            if {"secid", "ticker"} <= set(raw.columns):
                ticker_rows = raw[["secid", "ticker"]].dropna().drop_duplicates(
                    subset=["secid"], keep="last"
                )
                ticker_by_secid = {
                    row.secid: str(row.ticker)
                    for row in ticker_rows.itertuples(index=False)
                    if str(row.ticker).strip()
                }
            holdings_written = write_holdings_book(
                frames=strategy_frames,
                secids=universe_secids,
                out_xlsx=report_dir / "holdings_book.xlsx",
                out_csv=report_dir / "holdings_book.csv",
                ticker_by_secid=ticker_by_secid,
            )
            results["holdings_book"] = holdings_written
        except Exception as e:
            results["holdings_book_error"] = str(e)[:300]

    plot_paths = _plot_campaign(results, OUT)
    results["plots"] = plot_paths

    # D3/D4: render the reporting book from the frames just persisted above,
    # so the campaign itself produces book.pdf/BOOK.md/index.json rather
    # than only being exercised by test_eq_alloc_book.py's synthetic inputs.
    # ff4_factors is intentionally omitted here: fac_for_bench is aligned to
    # the full panel, not to each strategy's t_index-subset date axis, and
    # the FF4 attribution sub-figure already degrades gracefully to "skip"
    # without it (see src/reporting/eq_alloc_book.py _section5_attribution).
    signal_gate_result = None
    allowlist_path = Path(cfg.get("signal_allowlist_path") or "config/signal_allowlist.json")
    if allowlist_path.exists():
        try:
            signal_gate_result = json.loads(allowlist_path.read_text())
        except (OSError, json.JSONDecodeError):
            signal_gate_result = None
    try:
        from mascotrl.reporting.eq_alloc_book import render_eq_alloc_book

        book_dir = OUT / "report" / "book"
        # Load learning curves if present so S7 is not silently skipped.
        learning_curves: dict[str, list] = {}
        curves_root = OUT / "curves"
        if curves_root.is_dir():
            for p in sorted(curves_root.glob("seed*_fold*.json")):
                try:
                    blob = json.loads(p.read_text())
                    series = blob if isinstance(blob, list) else blob.get("curve") or []
                    learning_curves[p.stem] = [
                        float(r.get("mean_reward", float("nan")))
                        for r in series
                        if isinstance(r, dict)
                    ]
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    continue
        book_index = render_eq_alloc_book(
            strategy_frames=strategy_frames,
            out_dir=book_dir,
            results=results,
            focus_strategy="policy",
            cfg=cfg,
            signal_gate_result=signal_gate_result,
            signal_panels=proxy_signals,
            returns_panel=panel,
            returns_panel_secids=universe_secids,
            arms_root=ROOT / "logs" / "artifacts" / "arms",
            known_limitations=list(cfg.get("known_limitations") or []),
            learning_curves=learning_curves or None,
        )
        results["book"] = {
            "out_dir": str(book_dir),
            "n_figures": book_index.get("n_figures"),
            "n_written": book_index.get("n_written"),
            "n_skipped": book_index.get("n_skipped"),
        }
    except Exception as e:
        # A10: record the reason; fail closed at end of main() instead of
        # silently shipping a campaign that claims book coverage it lacks.
        results["book_error"] = str(e)[:300]

    # A11: verify every top-level YAML key was actually read at runtime
    # (not just documented as "should be read" in a hand-maintained list).
    # Only meaningful for a full run: --skip-rl deliberately never touches
    # the training-only keys.
    from mascotrl.eval.yaml_honesty import assert_yaml_honesty_tracked, load_workflow_keys

    yaml_honesty_error: str | None = None
    if not args.skip_rl:
        try:
            honesty_report = assert_yaml_honesty_tracked(
                cfg, load_workflow_keys(args.config), path=args.config
            )
            results["yaml_honesty"] = honesty_report
        except AssertionError as e:
            yaml_honesty_error = str(e)
            results["yaml_honesty_error"] = yaml_honesty_error

    out_json = OUT / "cpcv_path_summary.json"
    _assert_safe_to_write_summary(
        OUT, int(results["k"]), force_overwrite=args.force_overwrite
    )
    atomic_write_json(out_json, results)
    _log_event("finalize", summary=str(out_json), n_plots=len(plot_paths or []))

    # C7: mirror the eq arm's summary under logs/artifacts/arms/eq/ (the
    # figure loaders in src/reporting/figures/loaders.py read per-arm
    # artifacts from there, not from OUT=logs/artifacts/eq_alloc/), and
    # write a top-level spectrum_summary.json roll-up.
    arms_eq_dir = ROOT / "logs" / "artifacts" / "arms" / "eq"
    arms_eq_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(arms_eq_dir / "cpcv_path_summary.json", results)
    gates_summary = results.get("confirmatory", {}).get("gates") or {}
    spectrum_summary = {
        "arm": "eq",
        "policy_sharpe_mean": (results.get("confirmatory", {}).get("path_summary") or {}).get(
            "sharpe_mean"
        ),
        "gate1_pass": (gates_summary.get("gate1") or {}).get("pass"),
        "gate2_pass": (gates_summary.get("gate2") or {}).get("pass"),
        "gate3_pass": (gates_summary.get("gate3") or {}).get("pass"),
        "wall_total_s": results.get("wall_total_s"),
    }
    (ROOT / "logs" / "artifacts" / "spectrum_summary.json").write_text(
        json.dumps(spectrum_summary, indent=2, default=str, sort_keys=True)
    )
    print(f"wrote {arms_eq_dir / 'cpcv_path_summary.json'}")

    # A10: fail closed after artifacts are persisted for post-mortem. A
    # broken OLPS peer, ceiling arm, or stats table means the run produced
    # incomplete evidence and must not exit 0.
    hard_errors: dict = {}
    if results.get("olps_errors"):
        hard_errors["olps_errors"] = results["olps_errors"]
    if results.get("ceiling_errors"):
        hard_errors["ceiling_errors"] = results["ceiling_errors"]
    stats_err = results.get("confirmatory", {}).get("stats_table_error")
    if stats_err:
        hard_errors["stats_table_error"] = stats_err
    if yaml_honesty_error:
        hard_errors["yaml_honesty_error"] = yaml_honesty_error
    if results.get("negative_controls_errors"):
        hard_errors["negative_controls_errors"] = results["negative_controls_errors"]
    if results.get("book_error"):
        hard_errors["book_error"] = results["book_error"]
    if hard_errors:
        raise SystemExit(
            "eq_alloc campaign produced incomplete evidence (fail-closed, "
            f"see {out_json}): {json.dumps(hard_errors, default=str)[:1000]}"
        )


if __name__ == "__main__":
    main()
