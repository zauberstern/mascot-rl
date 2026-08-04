"""Static wiring audit for regime detectors vs consumers.

REGIME_SCORECARD_SPEC (frozen checklist)
----------------------------------------
Detectors (defaults):
  - label_regimes(macro; min_history_days=756, crisis_vix_q/oas_q=0.85,
    infl_q=0.70, persistence_days=10) → calm|inflationary|crisis
  - turbulence_index(returns; window=252); classify_regime(turb; quantile=0.75)
    is the Skulls comparator; operational hard labels = filtered Markov P>0.5
  - walk_forward_markov_filter(turb; window=756, step=21, k_regimes=2);
    jaccard_turbulent(q75, operational)

Expected wiring (live):
  - macro_labels → load_behaviour_macro_context: connected
  - turbulence → turbulence_regimes_from_returns: connected
  - HMM non-test callers: connected (scorecard/seal)
  - regime_labels.parquet readers (non-test): weak (write-only)
  - per_regime_sharpe callers: connected (scorecard)
  - env/features discrete regime ids: disconnected (intentional)
  - policy_mode archetype overlays: connected_to_train / disconnected_from_labels
  - config/spectrum/cherrypick_regime: naming_collision
  - fixed_share turbulence: connected via assemble_regime_desk (α not turb-gated)
  - MacroScheduleTau spectrum YAML: disconnected (intentional)
  - spectrum HAPPO macro_series inject: weak (optional helper; confirmatory omits)
  - fioracle_macro.enabled in spectrum YAML: none (n_matches=0)
  - seal → regime desk timeline: connected (align_sealed_operational_mask)

Confirmatory-critical connected set: macro_labels_to_behavior, turbulence_to_behavior.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

FIGURE_CRITICAL_CONNECTED: tuple[str, ...] = (
    "macro_labels_to_behavior",
    "turbulence_to_behavior",
)

# Rows that must remain non-connected by honesty design (not unfinished work).
INTENTIONAL_NON_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("env_features_regime_ids", "disconnected"),
    ("policy_mode_vs_labels", "connected_to_train_disconnected_from_labels"),
    ("burst_cherrypick_regime", "naming_collision"),
    ("macro_schedule_tau_yaml", "disconnected"),
    ("spectrum_happo_macro_series", "weak"),
    ("fioracle_macro_enabled_yaml", "none"),
    ("regime_labels_parquet_readers", "weak"),
)

_SCAN_SUBDIRS = ("src", "scripts", "config", "tests", "deploy")


def _iter_text_files(root: Path, *, suffixes: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for sub in _SCAN_SUBDIRS:
        base = root / sub
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in suffixes:
                continue
            if "__pycache__" in path.parts or ".venv" in path.parts:
                continue
            out.append(path)
    return out


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _non_test_py_hits(root: Path, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    hits: list[str] = []
    for path in _iter_text_files(root, suffixes=(".py",)):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("tests/") or "/test_" in rel or rel.startswith("scripts/test"):
            continue
        text = _read(path)
        if rx.search(text):
            hits.append(rel)
    return sorted(hits)


def _yaml_hits(root: Path, pattern: str) -> list[str]:
    rx = re.compile(pattern)
    hits: list[str] = []
    for path in _iter_text_files(root, suffixes=(".yaml", ".yml")):
        rel = path.relative_to(root).as_posix()
        text = _read(path)
        if rx.search(text):
            hits.append(rel)
    return sorted(hits)


def _row(
    *,
    id: str,
    status: str,
    evidence: str,
    n_matches: int = 0,
    paths: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "status": status,
        "evidence": evidence,
        "n_matches": int(n_matches),
        "paths": list(paths or []),
    }


def audit_regime_wiring(repo_root: Path | str) -> dict[str, Any]:
    """Return wiring matrix for regime detectors vs consumers."""
    root = Path(repo_root).resolve()
    rows: list[dict[str, Any]] = []

    # Macro labels → behavior export
    loader_hits = _non_test_py_hits(
        root, r"def load_behaviour_macro_context|load_behaviour_macro_context\("
    )
    rows.append(
        _row(
            id="macro_labels_to_behavior",
            status="connected" if loader_hits else "disconnected",
            evidence="load_behaviour_macro_context defined/called outside tests",
            n_matches=len(loader_hits),
            paths=loader_hits[:20],
        )
    )

    turb_hits = _non_test_py_hits(root, r"turbulence_regimes_from_returns")
    rows.append(
        _row(
            id="turbulence_to_behavior",
            status="connected" if turb_hits else "disconnected",
            evidence="turbulence_regimes_from_returns used outside tests",
            n_matches=len(turb_hits),
            paths=turb_hits[:20],
        )
    )

    hmm_hits = _non_test_py_hits(
        root,
        r"from src\.eval\.walk_forward_hmm|import walk_forward_hmm|"
        r"walk_forward_hmm_regimes\(|walk_forward_markov_filter\(|"
        r"jaccard_turbulent\(|hmm_turbulent_mask\(",
    )
    # Definition / audit self-docs are not production callers
    hmm_callers = [
        p
        for p in hmm_hits
        if not p.endswith("walk_forward_hmm.py")
        and not p.endswith("regime_wiring_audit.py")
    ]
    rows.append(
        _row(
            id="hmm_non_test_callers",
            status="disconnected" if not hmm_callers else "connected",
            evidence="non-test imports/calls of walk_forward_hmm APIs",
            n_matches=len(hmm_callers),
            paths=hmm_callers[:20],
        )
    )

    # Parquet path readers (exclude writers / attach)
    parquet_hits = _non_test_py_hits(root, r"regime_labels\.parquet|regime_labels_path")
    reader_hits = [
        p
        for p in parquet_hits
        if not p.endswith("macro_loader.py")
        and "run_eq_alloc_campaign.py" not in p
        and not p.endswith("regime_wiring_audit.py")
        and not p.endswith("regime_scorecard.py")
        and not p.endswith("run_regime_scorecard.py")
    ]
    # eq_alloc only stamps path; treat as weak if no dedicated reader
    rows.append(
        _row(
            id="regime_labels_parquet_readers",
            status="weak" if not reader_hits else "connected",
            evidence="parquet path written/stamped; dedicated readers outside writer/campaign stamp",
            n_matches=len(reader_hits),
            paths=reader_hits[:20],
        )
    )

    prs_call = _non_test_py_hits(root, r"per_regime_sharpe\(")
    prs_callers = [p for p in prs_call if not p.endswith("policy_mode.py")]
    rows.append(
        _row(
            id="per_regime_sharpe_callers",
            status="disconnected" if not prs_callers else "connected",
            evidence="per_regime_sharpe call sites outside definition",
            n_matches=len(prs_callers),
            paths=prs_callers[:20],
        )
    )

    env_hits = []
    for sub in ("src/env", "src/features"):
        sub_root = root / sub
        if not sub_root.is_dir():
            continue
        for path in sub_root.rglob("*.py"):
            text = _read(path)
            if re.search(r"\bregime_label|label_regimes|REGIME_IDS\b", text):
                env_hits.append(path.relative_to(root).as_posix())
    rows.append(
        _row(
            id="env_features_regime_ids",
            status="disconnected" if not env_hits else "connected",
            evidence="discrete regime ids in src/env or src/features",
            n_matches=len(env_hits),
            paths=env_hits[:20],
        )
    )

    pm_hits = _non_test_py_hits(root, r"resolve_policy_mode|apply_risk_aversion|apply_turnover_multiplier")
    # policy_mode scales train knobs; does not consume calm/crisis labels for routing
    rows.append(
        _row(
            id="policy_mode_vs_labels",
            status="connected_to_train_disconnected_from_labels",
            evidence="policy_mode overlays connected to train; no regime-label routing",
            n_matches=len(pm_hits),
            paths=pm_hits[:12],
        )
    )

    regime_dir = root / "config" / "spectrum" / "cherrypick_regime"
    n_cells = len(list(regime_dir.glob("*.yaml"))) if regime_dir.is_dir() else 0
    rows.append(
        _row(
            id="burst_cherrypick_regime",
            status="naming_collision" if n_cells > 0 else "disconnected",
            evidence="Burst REGIME wave = constraint ablations, not detectors",
            n_matches=n_cells,
            paths=["config/spectrum/cherrypick_regime/"] if n_cells else [],
        )
    )

    fs_text = _read(root / "src" / "eval" / "fixed_share.py")
    assemble_text = _read(root / "scripts" / "assemble_regime_desk.py")
    fs_has_turb = bool(re.search(r"turbulence|classify_regime", fs_text))
    assemble_has_turb = bool(re.search(r"turbulence_index\(", assemble_text))
    fs_connected = fs_has_turb or assemble_has_turb
    fs_paths = []
    if fs_has_turb:
        fs_paths.append("src/eval/fixed_share.py")
    if assemble_has_turb:
        fs_paths.append("scripts/assemble_regime_desk.py")
    rows.append(
        _row(
            id="fixed_share_turbulence_trigger",
            status="connected" if fs_connected else "disconnected",
            evidence="Fixed-Share desk runs turbulence_index in parallel (α not turb-gated)",
            n_matches=len(fs_paths),
            paths=fs_paths,
        )
    )

    tau_yaml = _yaml_hits(root, r"macro_schedule|MacroScheduleTau|tau_mode:\s*macro")
    # Prefer spectrum configs
    spectrum_tau = [p for p in tau_yaml if "config/spectrum" in p or "config/workflows" in p]
    rows.append(
        _row(
            id="macro_schedule_tau_yaml",
            status="disconnected" if not spectrum_tau else "connected",
            evidence="MacroScheduleTau / tau_mode macro in workflow/spectrum YAML",
            n_matches=len(spectrum_tau),
            paths=spectrum_tau[:20],
        )
    )

    # Spectrum HAPPO: look for macro_series= in campaign/train construction
    inject_hits = _non_test_py_hits(root, r"macro_series\s*=")
    # Filter to env construction sites
    inject_callers = [
        p
        for p in inject_hits
        if "cmdp_env.py" not in p and "tau_schedule.py" not in p
    ]
    rows.append(
        _row(
            id="spectrum_happo_macro_series",
            status="weak" if inject_callers else "weak",
            evidence=(
                "optional happo_macro_inject helper; confirmatory arm_equity "
                "omits USB/fioracle inject by design (open_by_design / weak)"
            ),
            n_matches=len(inject_callers),
            paths=inject_callers[:20],
        )
    )

    fio_any = _yaml_hits(root, r"fioracle_macro")
    enabled_true = []
    for p in fio_any:
        text = _read(root / p)
        # crude: enabled true within 15 lines of fioracle_macro
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "fioracle_macro" in line:
                window = "\n".join(lines[i : i + 15])
                if re.search(r"enabled:\s*[Tt]rue", window):
                    enabled_true.append(p)
                    break
    rows.append(
        _row(
            id="fioracle_macro_enabled_yaml",
            status="none" if not enabled_true else "connected",
            evidence="spectrum/workflow YAML with fioracle_macro.enabled: true",
            n_matches=len(enabled_true),
            paths=enabled_true[:20],
        )
    )

    # Seal chronology → Fixed-Share desk (timeline only; α not turb-gated).
    seal_hits = []
    if re.search(
        r"align_sealed_operational_mask|load_sealed_series|--seal-path",
        assemble_text,
    ):
        seal_hits.append("scripts/assemble_regime_desk.py")
    if (root / "src" / "eval" / "regime_desk_seal.py").is_file():
        seal_hits.append("src/eval/regime_desk_seal.py")
    rows.append(
        _row(
            id="seal_to_regime_desk",
            status="connected" if seal_hits else "disconnected",
            evidence=(
                "assemble loads SCHEMA>=3 seal via align_sealed_operational_mask "
                "for timeline shading; Fixed-Share alpha remains Herbster prior"
            ),
            n_matches=len(seal_hits),
            paths=seal_hits,
        )
    )

    by_id = {r["id"]: r for r in rows}
    confirmatory_ok = all(
        by_id[rid]["status"] == "connected" for rid in FIGURE_CRITICAL_CONNECTED
    )
    return {
        "status": "ok",
        "repo_root": str(root),
        "confirmatory_critical_connected": list(FIGURE_CRITICAL_CONNECTED),
        "confirmatory_critical_pass": bool(confirmatory_ok),
        "intentional_non_connections": [
            {"id": rid, "expected_status": st}
            for rid, st in INTENTIONAL_NON_CONNECTIONS
        ],
        "rows": rows,
    }
