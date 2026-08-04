#!/usr/bin/env python3
"""Run regime detection scorecard + wiring audit + macro lake leverage.

Usage:
    python scripts/run_regime_scorecard.py
    python scripts/run_regime_scorecard.py --out logs/artifacts/regime_scorecard
    python scripts/run_regime_scorecard.py --synthetic-fallback
    python scripts/run_regime_scorecard.py --seal my_run
    python scripts/run_regime_scorecard.py --from-seal my_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "logs" / "artifacts" / "regime_scorecard"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _load_macro_from_fioracle(
    *,
    lake_root: Path,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    try:
        from src.data.fioracle_macro import (
            build_fioracle_feature_frame,
            load_fioracle_macro,
        )

        levels = load_fioracle_macro(
            lake_root=lake_root,
            start_date=start,
            end_date=end,
        )
        feats = build_fioracle_feature_frame(levels)
        bdays = pd.bdate_range(start, end)
        return feats.reindex(bdays).ffill()
    except Exception as exc:
        print(f"[regime_scorecard] fioracle load failed: {exc}", file=sys.stderr)
        return None


def _synthetic_returns(n: int, k: int = 8, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, size=(n, k))
    a, b = n // 3, n // 3 + max(40, n // 20)
    r[a:b] = rng.normal(0.0, 0.04, size=(b - a, k))
    return r


def _try_usb_equity_returns(
    usb_root: Path,
    *,
    n: int | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> np.ndarray | None:
    """Build a (T, k) return panel from USB sp500 prices if present."""
    path = usb_root / "macro" / "sp500_prices.parquet"
    if not path.is_file():
        # Campaign / style-desk style artifacts occasionally live elsewhere.
        alt = usb_root / "equity" / "sp500_prices.parquet"
        path = alt if alt.is_file() else path
    if not path.is_file():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    # CRSP-style USB dump uses PERMNO/RET/date; tolerate mixed case.
    lower = {str(c).lower(): c for c in df.columns}

    def _pick(*names: str) -> str | None:
        for name in names:
            if name in df.columns:
                return name
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    date_col = _pick("date", "Date", "datadate")
    ret_col = _pick("RET", "ret", "return", "RETX")
    id_col = _pick("PERMNO", "permno", "secid", "TICKER", "ticker", "symbol")
    if date_col is None or id_col is None:
        return None
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    if ret_col is None:
        px_col = _pick("adj_close", "close", "PRC", "prc", "price", "OPENPRC")
        if px_col is None:
            return None
        work = work[[date_col, id_col, px_col]].dropna(subset=[date_col, id_col])
        work[px_col] = pd.to_numeric(work[px_col], errors="coerce").abs()
        work = work.dropna(subset=[px_col])
        work = work.sort_values([id_col, date_col])
        work["ret"] = work.groupby(id_col, sort=False)[px_col].pct_change()
        ret_col = "ret"
    else:
        work = work[[date_col, id_col, ret_col]].dropna(subset=[date_col, id_col])
        work[ret_col] = pd.to_numeric(work[ret_col], errors="coerce")
        work = work.dropna(subset=[ret_col])
    # Prefer names with the densest history in the requested calendar window.
    if dates is not None:
        d0, d1 = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
        in_cal = work[(work[date_col] >= d0) & (work[date_col] <= d1)]
        top = in_cal[id_col].value_counts().head(10).index
        sub = in_cal[in_cal[id_col].isin(top)]
    else:
        top = work[id_col].value_counts().head(10).index
        sub = work[work[id_col].isin(top)]
    wide = sub.pivot_table(
        index=date_col, columns=id_col, values=ret_col, aggfunc="last"
    )
    wide = wide.sort_index()
    if dates is not None:
        wide = wide.reindex(pd.DatetimeIndex(dates))
    elif n is not None:
        wide = wide.tail(int(n))
    # Drop all-NaN columns; require a usable cross-section.
    wide = wide.dropna(axis=1, how="all")
    if wide.shape[1] < 2:
        return None
    arr = wide.to_numpy(dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 100:
        return None
    # Keep NaNs; turbulence_index drops incomplete names (never invent zeros).
    return arr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lake-root", type=Path, default=ROOT / "lake")
    ap.add_argument(
        "--usb-lake-root",
        type=Path,
        default=None,
        help="USB production lake (default: CANONICAL_LAKE from paths)",
    )
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--require-lake", action="store_true")
    ap.add_argument("--skip-wiring", action="store_true")
    ap.add_argument("--skip-leverage", action="store_true")
    ap.add_argument("--behavior-path", type=Path, default=None)
    ap.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Explicit opt-in: invent returns when USB panel missing "
        "(agreement then diagnostic-only). Default: agreement=unavailable.",
    )
    ap.add_argument(
        "--desk-returns",
        type=Path,
        default=None,
        help="Style-desk / campaign panel_returns JSON (campaign y_t priority).",
    )
    ap.add_argument(
        "--panel-legacy-permno10",
        action="store_true",
        help="Diagnostic only: densest-PERMNO panel (not confirmatory).",
    )
    ap.add_argument("--hmm-step", type=int, default=21)
    ap.add_argument("--hmm-n-iter", type=int, default=50)
    ap.add_argument("--hmm-window", type=int, default=252 * 3)
    ap.add_argument("--turbulence-window", type=int, default=252)
    ap.add_argument(
        "--markov-robustness",
        action="store_true",
        help="Growing-window + KPT monthly Markov robustness (slow).",
    )
    ap.add_argument(
        "--no-markov-robustness",
        action="store_true",
        help="Disable robustness even when --seal is set.",
    )
    ap.add_argument(
        "--seal",
        default=None,
        help="After a live run, seal labels + per-window HMM under sealed/NAME",
    )
    ap.add_argument(
        "--from-seal",
        default=None,
        help="Replay scorecard agreement from sealed/NAME (no HMM refit)",
    )
    args = ap.parse_args(argv)

    from src.data.paths import CANONICAL_LAKE
    from src.eval.macro_lake_leverage import build_macro_lake_leverage
    from src.eval.regime_scorecard import build_regime_scorecard
    from src.eval.regime_wiring_audit import audit_regime_wiring

    usb = args.usb_lake_root or Path(CANONICAL_LAKE)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_seal and args.seal:
        print("[regime_scorecard] refuse both --seal and --from-seal", file=sys.stderr)
        return 2

    if args.from_seal:
        from src.eval.regime_seal import scorecard_from_seal, sealed_dir

        seal_path = sealed_dir(out_dir, args.from_seal)
        if not seal_path.is_dir():
            print(f"[regime_scorecard] missing seal {seal_path}", file=sys.stderr)
            return 2
        # Macro / wiring still refresh; agreement comes from seal.
        macro = _load_macro_from_fioracle(
            lake_root=args.lake_root, start=args.start, end=args.end
        )
        base = build_regime_scorecard(
            macro=macro,
            asset_returns=None,
            repo_root=ROOT,
            usb_lake_root=usb,
            include_wiring=not args.skip_wiring,
            include_leverage=not args.skip_leverage,
            behavior_path=args.behavior_path,
        )
        scorecard = scorecard_from_seal(seal_path, base_scorecard=base)
        print(
            "[regime_scorecard] --from-seal: skipped HMM/turbulence refit",
            file=sys.stderr,
        )
    else:
        macro = _load_macro_from_fioracle(
            lake_root=args.lake_root,
            start=args.start,
            end=args.end,
        )
        if macro is None and args.require_lake:
            print(
                "[regime_scorecard] fioracle required but unavailable",
                file=sys.stderr,
            )
            return 2

        asset_returns = None
        returns_source = "none"
        panel_meta: dict = {}
        gics = None
        kpt = None
        if macro is not None:
            from src.eval.regime_return_panel import (
                load_kpt_gics_sector_returns,
                load_kpt_sector_returns,
                load_style_desk_returns,
            )

            dates_ix = pd.DatetimeIndex(macro.index)
            desk = load_style_desk_returns(args.desk_returns)
            if desk is not None:
                asset_returns, panel_meta = desk
                # Align length to macro calendar when possible
                if asset_returns.shape[0] != len(macro):
                    print(
                        "[regime_scorecard] desk-returns T mismatch macro; "
                        "using desk T as-is",
                        file=sys.stderr,
                    )
                returns_source = "desk"
            else:
                gics = load_kpt_gics_sector_returns(usb, dates_ix)
                kpt = load_kpt_sector_returns(usb, dates_ix)
                if gics is not None:
                    asset_returns, panel_meta = gics
                    returns_source = "kpt10_gics"
                elif kpt is not None:
                    asset_returns, panel_meta = kpt
                    returns_source = "kpt10"
                    print(
                        "[regime_scorecard] GICS sector join below coverage "
                        "gate; SIC10 approximation",
                        file=sys.stderr,
                    )
                elif args.panel_legacy_permno10:
                    asset_returns = _try_usb_equity_returns(
                        usb, n=len(macro), dates=dates_ix
                    )
                    if asset_returns is not None:
                        returns_source = "legacy_permno10"
                if asset_returns is None and args.synthetic_fallback:
                    asset_returns = _synthetic_returns(len(macro))
                    returns_source = "synthetic"
                    print(
                        "[regime_scorecard] USB/KPT panel unavailable; "
                        "using --synthetic-fallback (agreement diagnostic only)",
                        file=sys.stderr,
                    )
                elif asset_returns is None:
                    print(
                        "[regime_scorecard] KPT/desk returns unavailable; "
                        "agreement=unavailable (pass --synthetic-fallback to invent)",
                        file=sys.stderr,
                    )
            if panel_meta:
                print(
                    f"[regime_scorecard] panel source={returns_source} "
                    f"meta={panel_meta.get('source')} "
                    f"shape={None if asset_returns is None else asset_returns.shape}",
                    file=sys.stderr,
                )

        if macro is None and args.synthetic_fallback:
            dates = pd.bdate_range(args.start, periods=900)
            rng = np.random.default_rng(0)
            macro = pd.DataFrame(
                {
                    "vix_level": 15.0 + rng.normal(0, 1.0, len(dates)),
                    "hy_oas_level": 4.0 + rng.normal(0, 0.2, len(dates)),
                    "inflation_yoy_level": 2.0 + rng.normal(0, 0.1, len(dates)),
                    "term_spread_level": 1.0 + rng.normal(0, 0.1, len(dates)),
                },
                index=dates,
            )
            macro.iloc[500:560, 0] = 40.0
            asset_returns = _synthetic_returns(len(macro))
            returns_source = "synthetic"
            print(
                "[regime_scorecard] fioracle unavailable; "
                "--synthetic-fallback macro+returns",
                file=sys.stderr,
            )
            if args.require_lake:
                return 2

        need_models = args.seal is not None
        # --seal implies robustness unless --no-markov-robustness.
        want_robust = bool(args.markov_robustness) or (
            args.seal is not None and not args.no_markov_robustness
        )
        scorecard = build_regime_scorecard(
            macro=macro,
            asset_returns=asset_returns,
            repo_root=ROOT,
            usb_lake_root=usb,
            include_wiring=not args.skip_wiring,
            include_leverage=not args.skip_leverage,
            behavior_path=args.behavior_path,
            hmm_window=int(args.hmm_window),
            hmm_step=int(args.hmm_step),
            hmm_n_iter=int(args.hmm_n_iter),
            turbulence_window=int(args.turbulence_window),
            return_series=need_models,
            return_models=need_models,
            include_markov_robustness=want_robust,
        )
        scorecard["returns_source"] = returns_source

        # Report-only: Jaccard of SIC vs GICS operational Markov masks.
        if returns_source == "kpt10_gics" and kpt is not None and gics is not None:
            from src.eval.regime_scorecard import jaccard_sic_vs_gics_operational

            try:
                j_sg = jaccard_sic_vs_gics_operational(
                    kpt[0],
                    gics[0],
                    turbulence_window=int(args.turbulence_window),
                    hmm_window=int(args.hmm_window),
                    hmm_step=int(args.hmm_step),
                )
                agr = scorecard.setdefault("agreement", {})
                agr["jaccard_sic_vs_gics_operational"] = float(j_sg)
            except Exception as exc:
                agr = scorecard.setdefault("agreement", {})
                agr["jaccard_sic_vs_gics_operational"] = float("nan")
                agr["jaccard_sic_vs_gics_note"] = f"unavailable: {exc}"

        if args.seal is not None:
            from src.eval.regime_seal import seal_regime_run

            if asset_returns is None:
                print(
                    "[regime_scorecard] refuse --seal without asset_returns",
                    file=sys.stderr,
                )
                return 2
            series = scorecard.pop("_series", None)
            models = scorecard.pop("_models", None)
            if series is None:
                print(
                    "[regime_scorecard] refuse --seal: missing series payload",
                    file=sys.stderr,
                )
                return 2
            try:
                dest = seal_regime_run(
                    name=args.seal,
                    out_root=out_dir,
                    scorecard=scorecard,
                    series=series,
                    models=models,
                    asset_returns=asset_returns,
                    hyperparams={
                        "turbulence_window": int(args.turbulence_window),
                        "hmm_window": int(args.hmm_window),
                        "hmm_n_iter": int(args.hmm_n_iter),
                        "returns_source": returns_source,
                    },
                    repo_root=ROOT,
                    hmm_step=int(args.hmm_step),
                )
            except (ValueError, FileExistsError) as exc:
                print(f"[regime_scorecard] seal failed: {exc}", file=sys.stderr)
                return 2
            print(f"[regime_scorecard] sealed -> {dest}", file=sys.stderr)
        else:
            scorecard.pop("_series", None)
            scorecard.pop("_models", None)

    wiring = scorecard.get("wiring") or (
        None if args.skip_wiring else audit_regime_wiring(ROOT)
    )
    leverage = None
    if not args.skip_leverage:
        leverage = build_macro_lake_leverage(repo_root=ROOT, usb_lake_root=usb)
        scorecard["leverage"] = {
            "status": leverage.get("status"),
            "metrics": leverage.get("metrics"),
            "productive_gaps": leverage.get("productive_gaps"),
            "n_assets": len(leverage.get("assets", [])),
        }

    for key in ("turbulent_mask", "hmm_mask"):
        if isinstance(scorecard.get("agreement"), dict):
            scorecard["agreement"].pop(key, None)
    scorecard.pop("_series", None)
    scorecard.pop("_models", None)

    paths = {
        "regime_scorecard": out_dir / "regime_scorecard.json",
        "regime_wiring_audit": out_dir / "regime_wiring_audit.json",
        "macro_lake_leverage": out_dir / "macro_lake_leverage.json",
    }
    paths["regime_scorecard"].write_text(
        json.dumps(scorecard, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    if wiring is not None:
        paths["regime_wiring_audit"].write_text(
            json.dumps(wiring, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
    if leverage is not None:
        paths["macro_lake_leverage"].write_text(
            json.dumps(leverage, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )

    print("=== Regime scorecard ===")
    print(f"status: {scorecard.get('status')}")
    agr = scorecard.get("agreement") or {}
    print(
        f"Jaccard turb↔filtered-HMM: {agr.get('jaccard_turbulence_hmm')} "
        f"grade={agr.get('jaccard_grade')}"
    )
    print(
        f"duration mean_turbulent_run_days={agr.get('mean_turbulent_run_days')} "
        f"switch_rate={agr.get('turbulence_switch_rate')}"
    )
    print(f"hygiene: {(scorecard.get('hygiene') or {}).get('status')}")
    print(f"returns_source: {scorecard.get('returns_source') or agr.get('reason')}")
    occ = scorecard.get("occupancy") or {}
    print(f"occupancy crisis_frac: {(occ.get('fractions') or {}).get('crisis')}")
    if wiring is not None:
        print(
            f"wiring confirmatory_critical_pass: {wiring.get('confirmatory_critical_pass')} "
            f"rows={len(wiring.get('rows') or [])}"
        )
        weak = [
            r["id"]
            for r in wiring.get("rows") or []
            if r.get("status")
            in ("disconnected", "weak", "naming_collision", "none")
        ]
        print(f"wiring gaps/notes: {', '.join(weak[:8])}")
    if leverage is not None:
        m = leverage.get("metrics") or {}
        print(
            "leverage fioracle_feature_consumer_frac="
            f"{m.get('fioracle_feature_consumer_frac')} "
            f"raw_present={m.get('fioracle_raw_present_frac')}"
        )
        gaps = leverage.get("productive_gaps") or []
        print("top productive gaps:")
        for g in gaps[:3]:
            print(f"  {g.get('rank')}. {g.get('id')}: {g.get('summary')}")
    print("wrote:")
    for k, p in paths.items():
        if p.is_file():
            print(f"  {k}: {p}")

    if args.require_lake and scorecard.get("status") == "unavailable":
        return 2
    if wiring is not None and not wiring.get("confirmatory_critical_pass"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
