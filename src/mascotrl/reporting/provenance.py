"""
Run provenance: seed/config manifest and artifact schema assertions.

A referee (and a replication attempt) needs to know exactly what produced a
number: the config, the seeds, the code version, the library versions, and the
protocol switches that gate the claim. This module freezes that into one
artifact per run and validates that the report carries the fields the paper's
tables depend on, so a missing field fails loudly at write time instead of
silently becoming a blank cell.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mascotrl._root import REPO_ROOT
from mascotrl.logging_utils import get_logger

log = get_logger("mascotrl.reporting.provenance")

# Report fields the paper's headline tables read. Missing any of these means a
# table cell would be blank or, worse, silently defaulted.
REQUIRED_REPORT_FIELDS: tuple[str, ...] = (
    "eval_protocol",
    "train_distribution",
    "n_assets",
)

# Fields required specifically to support a published alpha claim.
REQUIRED_CLAIM_FIELDS: tuple[str, ...] = (
    "deflated_sharpe_oos",
    "hac_inference_oos",
    "factor_alpha",
    "cost_ladder",
    "n_trials_breakdown",
)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(REPO_ROOT),
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


def _package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "torch", "duckdb", "scipy", "cvxpy", "pyarrow")
    out: dict[str, str] = {}
    for n in names:
        try:
            mod = __import__(n)
            out[n] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[n] = "not_installed"
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable SHA-256 over the config, so two runs can be proven identical."""
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class RunManifest:
    """Everything needed to re-run and to audit what produced the numbers."""

    run_label: str
    cfg: dict[str, Any] = field(default_factory=dict)

    def build(self) -> dict[str, Any]:
        cfg = dict(self.cfg or {})
        seeds = {
            "seed": cfg.get("seed"),
            "eval_seeds": cfg.get("eval_seeds"),
            "torch_deterministic_requested": bool(
                cfg.get("torch_deterministic", False)
            ),
        }
        protocol_switches = {
            k: cfg.get(k)
            for k in (
                "nested_wfo_retrain",
                "train_distribution",
                "capital_gates_require_stability",
                "capital_gates_require_retrain_wfo",
                "capital_gates_require_factor_alpha",
                "capital_gates_require_after_cost",
                "capital_gates_require_pack_gates",
                "capital_gates_require_sharpe_vs_best_baseline",
                "capital_gates_require_om_touch_claim",
                "claim_label_stem",
                "claim_return_definition",
                "min_break_even_spread_multiplier",
                "hedge_frequency",
                "hedge_leg_spread_bps",
                "hedge_impact_enabled",
                "publication_n_trials",
            )
            if k in cfg
        }
        return {
            "run_label": self.run_label,
            "config_sha256": config_hash(cfg),
            "seeds": seeds,
            "protocol_switches": protocol_switches,
            "code": {
                "git_commit": _git("rev-parse", "HEAD"),
                "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": _package_versions(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "use_gpu": cfg.get("use_gpu"),
            },
            "reproduction": {
                "command": (
                    f"python scripts/train_happo.py --config <config> "
                    f"--run-label {self.run_label}"
                ),
                "note": (
                    "git_dirty=true means the working tree had uncommitted "
                    "changes; the run is then not reproducible from the commit "
                    "alone and must not back a published table."
                ),
            },
        }

    def write(self, run_dir: str | Path) -> Path:
        path = Path(run_dir) / "RUN_MANIFEST.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build()
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        if payload["code"]["git_dirty"]:
            log.warning(
                "RUN_MANIFEST: working tree dirty — run is not reproducible "
                "from git commit alone"
            )
        log.info("Run manifest → %s", path)
        return path


def validate_report_schema(
    report: dict[str, Any],
    *,
    require_claim_fields: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Check the report carries the fields the paper's tables read.

    ``require_claim_fields`` additionally demands the evidence needed to publish
    an alpha claim. With ``strict`` the function raises; otherwise it returns the
    result so the caller can record it and continue (useful for smoke runs that
    intentionally skip publication stages).
    """
    missing = [f for f in REQUIRED_REPORT_FIELDS if report.get(f) is None]
    missing_claim: list[str] = []
    if require_claim_fields:
        missing_claim = [f for f in REQUIRED_CLAIM_FIELDS if not report.get(f)]
    ok = not missing and not missing_claim
    out = {
        "ok": bool(ok),
        "missing_required": missing,
        "missing_claim_evidence": missing_claim,
        "checked_required": list(REQUIRED_REPORT_FIELDS),
        "checked_claim": list(REQUIRED_CLAIM_FIELDS) if require_claim_fields else [],
    }
    if not ok and strict:
        raise RuntimeError(
            "report schema incomplete: "
            f"missing={missing} missing_claim_evidence={missing_claim}"
        )
    if not ok:
        log.warning(
            "report schema incomplete: missing=%s claim=%s", missing, missing_claim
        )
    return out
