"""Quantify productive use of fioracle + USB macro panels for regime eval.

Does not admit features onto the confirmatory arm_equity cube. Reports idle /
dead / underused series and ranked eval-path gaps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mascotrl.data.fioracle_macro import DEFAULT_SERIES, FIORACLE_FEATURE_COLUMNS
from mascotrl.data.paths import CANONICAL_LAKE, MASCOTRL_ROOT

# Pre-registered dual-source candidates (not the full H.15 dump).
USB_DUAL_SOURCE_CANDIDATES: tuple[str, ...] = (
    "t10y2y",
    "bamlh0a0hym2",
    "t10yie",
    "dff",
)

USB_HAPPO_VIX: tuple[str, ...] = ("vix", "vxn", "vxd")
USB_HAPPO_RATES: tuple[str, ...] = ("sofr", "effr", "dtb3")

# Fioracle derived cols with ≥1 live consumer today (behavior / labels).
_CONSUMED_FEATURES: frozenset[str] = frozenset(
    {
        "vix_level",
        "vix_z_252",
        "hy_oas_level",
        "hy_oas_z_252",
        "term_spread_level",
        "inflation_yoy_level",
        "epu_z_252",
        "gpri_z_252",
    }
)

_REGIME_LABEL_INPUTS: frozenset[str] = frozenset(
    {"vix_level", "hy_oas_level", "inflation_yoy_level"}
)

_BEHAVIOR_REGRESSORS: frozenset[str] = frozenset(
    {
        "vix_z_252",
        "hy_oas_z_252",
        "term_spread_level",
        "epu_z_252",
        "gpri_z_252",
    }
)

_ENGINEERED_IDLE: frozenset[str] = frozenset(
    {"unemployment_yoy_chg", "vix_chg_21", "hy_oas_chg_21"}
)

PRODUCTIVE_GAPS: tuple[dict[str, Any], ...] = (
    {
        "id": "turbulence_macro_comovement",
        "rank": 1,
        "status": "closed",
        "summary": "CLOSED: VIX/HY OAS/term in y_t with windowed z-score (scale-safe)",
        "target_files": [
            "src/mascotrl/eval/turbulence.py",
            "src/mascotrl/eval/regime_dual_source.py",
        ],
    },
    {
        "id": "b3_widen_epu_gpri",
        "rank": 2,
        "status": "closed",
        "summary": "CLOSED: optional EPU/GPRI lag-1 in macro_tilt_sensitivity",
        "target_files": [
            "src/mascotrl/reporting/behavior_metrics.py",
        ],
    },
    {
        "id": "hmm_macro_jaccard",
        "rank": 3,
        "status": "closed_superseded",
        "summary": "CLOSED/superseded: KPT filtered-Markov on turbulence (not a second return clusterer)",
        "target_files": [
            "src/mascotrl/eval/walk_forward_hmm.py",
            "src/mascotrl/eval/regime_scorecard.py",
        ],
    },
    {
        "id": "h15_dual_source",
        "rank": 4,
        "status": "closed",
        "summary": "CLOSED: H.15 lag-1 fallback for term/OAS via regime_dual_source",
        "target_files": [
            "src/mascotrl/eval/regime_dual_source.py",
        ],
    },
    {
        "id": "per_regime_sharpe_producer",
        "rank": 5,
        "status": "closed",
        "summary": "CLOSED: causal_per_regime_sharpe beside calendar_stress_windows in scorecard",
        "target_files": [
            "src/mascotrl/eval/regime_scorecard.py",
            "src/mascotrl/spectrum/policy_mode.py",
        ],
    },
    {
        "id": "happo_macro_series_inject",
        "rank": 6,
        "status": "open_by_design",
        "summary": (
            "helper exists (happo_macro_inject); confirmatory arm_equity keeps "
            "happo_usb_macro / fioracle_macro.enabled false by honesty lock"
        ),
        "target_files": [
            "src/mascotrl/spectrum/happo_macro_inject.py",
            "scripts/run_spectrum_campaign.py",
        ],
    },
)


def _asset(
    *,
    id: str,
    status: str,
    family: str,
    evidence: str = "",
    present: bool | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "status": status,
        "family": family,
        "evidence": evidence,
        "present": present,
    }


def _fioracle_dir(repo_root: Path) -> Path:
    return repo_root / "lake" / "macro" / "fioracle"


def _usb_macro_dir(usb_lake_root: Path) -> Path:
    return usb_lake_root / "macro"


def build_macro_lake_leverage(
    *,
    repo_root: Path | str | None = None,
    usb_lake_root: Path | str | None = None,
) -> dict[str, Any]:
    """Inventory fioracle + USB designed macro vs productive consumers."""
    root = Path(repo_root or MASCOTRL_ROOT).resolve()
    usb_root = Path(usb_lake_root) if usb_lake_root is not None else Path(CANONICAL_LAKE)
    fio_dir = _fioracle_dir(root)
    assets: list[dict[str, Any]] = []
    denominator_ids: list[str] = []

    # --- Fioracle raw ---
    present_raw = 0
    for sid in DEFAULT_SERIES:
        path = fio_dir / f"{sid}.parquet"
        ok = path.is_file()
        present_raw += int(ok)
        if sid == "yield_2y":
            assets.append(
                _asset(
                    id="yield_2y",
                    status="loaded_dead" if ok else "unavailable",
                    family="fioracle_raw",
                    evidence="in DEFAULT_SERIES but not FIORACLE_FEATURE_COLUMNS",
                    present=ok,
                )
            )
            denominator_ids.append("yield_2y")
        else:
            denominator_ids.append(f"raw_{sid}")
            assets.append(
                _asset(
                    id=f"raw_{sid}",
                    status="optional_eval" if ok else "unavailable",
                    family="fioracle_raw",
                    evidence=str(path) if ok else "missing parquet",
                    present=ok,
                )
            )

    raw_frac = present_raw / max(len(DEFAULT_SERIES), 1)
    assets.append(
        _asset(
            id="fioracle_raw",
            status="ok" if raw_frac == 1.0 else ("partial" if raw_frac > 0 else "unavailable"),
            family="summary",
            evidence=f"{present_raw}/{len(DEFAULT_SERIES)} series present",
            present=raw_frac == 1.0,
        )
    )
    # Attach present_frac for tests
    assets[-1]["present_frac"] = float(raw_frac)

    # --- Fioracle derived features ---
    consumed = 0
    for col in FIORACLE_FEATURE_COLUMNS:
        denominator_ids.append(col)
        if col in _REGIME_LABEL_INPUTS or col in _BEHAVIOR_REGRESSORS:
            status = "optional_eval"
            consumed += 1
        elif col in _ENGINEERED_IDLE:
            status = "engineered_idle"
        else:
            status = "engineered_idle"
        assets.append(
            _asset(
                id=col,
                status=status,
                family="fioracle_feature",
                evidence="behavior/labels" if status == "optional_eval" else "no live consumer",
                present=fio_dir.is_dir(),
            )
        )

    feat_consumer_frac = consumed / max(len(FIORACLE_FEATURE_COLUMNS), 1)

    # --- USB designed HAPPO slots ---
    usb_macro = _usb_macro_dir(usb_root)
    cboe = usb_macro / "cboe_vix.parquet"
    ir = usb_macro / "interest_rate.parquet"
    usb_ok = usb_macro.is_dir()

    if not usb_ok:
        assets.append(
            _asset(
                id="usb_cboe_vix_designed",
                status="unavailable",
                family="usb",
                evidence=f"USB macro missing: {usb_macro}",
                present=False,
            )
        )
        assets.append(
            _asset(
                id="usb_interest_rate_designed",
                status="unavailable",
                family="usb",
                evidence=f"USB macro missing: {usb_macro}",
                present=False,
            )
        )
        usb_happo_frac = float("nan")
        dual_present = 0
    else:
        cboe_ok = cboe.is_file()
        ir_ok = ir.is_file()
        assets.append(
            _asset(
                id="usb_cboe_vix_designed",
                status="usb_underused" if cboe_ok else "unavailable",
                family="usb",
                evidence="designed HAPPO vix/vxn/vxd; spectrum inject often weak/dead",
                present=cboe_ok,
            )
        )
        assets.append(
            _asset(
                id="usb_interest_rate_designed",
                status="usb_underused" if ir_ok else "unavailable",
                family="usb",
                evidence="designed HAPPO sofr/effr/dtb3; spectrum inject often weak/dead",
                present=ir_ok,
            )
        )
        denominator_ids.extend(
            [f"usb_{c}" for c in USB_HAPPO_VIX] + [f"usb_{c}" for c in USB_HAPPO_RATES]
        )
        # Designed slots exist on disk → countable; injection still underused
        designed_n = len(USB_HAPPO_VIX) + len(USB_HAPPO_RATES)
        usb_happo_frac = (designed_n if (cboe_ok and ir_ok) else 0) / designed_n

        # Dual-source candidates: column presence only
        dual_present = 0
        if ir_ok:
            try:
                import pyarrow.parquet as pq

                cols = set(pq.read_schema(ir).names)
            except Exception:
                cols = set()
            for cand in USB_DUAL_SOURCE_CANDIDATES:
                denominator_ids.append(f"dual_{cand}")
                ok = cand in cols
                dual_present += int(ok)
                # t10y2y / bamlh0a0hym2 consumed by regime_dual_source.
                used = cand in ("t10y2y", "bamlh0a0hym2")
                if ok and used:
                    dual_status = "optional_eval"
                elif ok:
                    dual_status = "engineered_idle"
                else:
                    dual_status = "unavailable"
                assets.append(
                    _asset(
                        id=f"dual_{cand}",
                        status=dual_status,
                        family="usb_dual_source",
                        evidence=(
                            "H.15 dual-source via regime_dual_source"
                            if used
                            else "pre-registered H.15 dual-source candidate"
                        ),
                        present=ok,
                    )
                )

    # Quarantined (listed for honesty, excluded from denominator)
    for qid, reason in (
        ("compustat", "disclosure / not feature-admitted"),
        ("ibes", "disclosure / not feature-admitted"),
        ("lseg_p3", "P3_REFUSED overlay"),
        ("jkp", "MISS stub 0 rows"),
    ):
        assets.append(
            _asset(
                id=qid,
                status="quarantined",
                family="quarantine",
                evidence=reason,
                present=None,
            )
        )

    metrics = {
        "fioracle_raw_present_frac": float(raw_frac),
        "fioracle_feature_consumer_frac": float(feat_consumer_frac),
        "usb_happo_designed_frac": usb_happo_frac,
        "behavior_regressors_used": len(_BEHAVIOR_REGRESSORS),
        "behavior_regressors_available": len(_BEHAVIOR_REGRESSORS)
        + len({"unemployment_yoy_chg"}),
        "regime_label_inputs_used": len(_REGIME_LABEL_INPUTS),
        "regime_label_inputs_available": len(FIORACLE_FEATURE_COLUMNS),
        "usb_dual_source_present": int(dual_present),
        "usb_dual_source_candidates": len(USB_DUAL_SOURCE_CANDIDATES),
    }

    status = "ok"
    if raw_frac < 1.0 or not usb_ok:
        status = "partial" if raw_frac > 0 else "unavailable"

    return {
        "status": status,
        "repo_root": str(root),
        "usb_lake_root": str(usb_root),
        "fioracle_dir": str(fio_dir),
        "denominator_ids": list(denominator_ids),
        "metrics": metrics,
        "assets": assets,
        "productive_gaps": [dict(g) for g in PRODUCTIVE_GAPS],
        "non_goals": [
            "admit full fioracle onto confirmatory arm_equity",
            "bulk H.15 dump as obs",
            "GAN curriculum",
            "turbulence-gated alpha retuning",
        ],
    }
