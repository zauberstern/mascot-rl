"""Alpha v2 conjunctive G0–G6 gate ladder (Block E Steps 24, 28).

Pure predicates over precomputed metrics in a bundle. Campaign code computes
Sharpes / HAC / DSR / PBO / SPA elsewhere; this module only decides PASS/FAIL.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


MIN_ALPHA_ANNUAL = 0.0
MIN_ECONOMIC_SPA_RIVALS = 3


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _b(x: Any) -> bool:
    return bool(x)


def _gate_g0(g: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    checks = {
        "explicit_arm": _b(g.get("explicit_arm")),
        "pit": _b(g.get("pit")),
        "friction_parity": _b(g.get("friction_parity")),
        "no_silent_delist": _b(g.get("no_silent_delist")),
        "no_rbergomi_ancestry": _b(g.get("no_rbergomi_ancestry")),
        "eq_allowlist": _b(g.get("eq_allowlist")),
    }
    ok = all(checks.values())
    return ok, {"checks": checks}


def _gate_g1(g: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    sr = _f(g.get("residual_sr"))
    months_pos = int(g.get("months_positive") or 0)
    n_months = int(g.get("n_months") or 12)
    ok = sr > 0.0 and months_pos >= 7 and n_months >= 12
    return ok, {
        "residual_sr": sr,
        "months_positive": months_pos,
        "n_months": n_months,
        "require": "residual_sr>0 and months_positive>=7/12",
    }


def _gate_g2(g: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    med = _f(g.get("median_residual_sr"))
    p10 = _f(g.get("p10_residual_sr"))
    n_seeds = int(g.get("n_seeds") or 0)
    ok = n_seeds >= 10 and med >= 0.25 and p10 > 0.0
    return ok, {
        "median_residual_sr": med,
        "p10_residual_sr": p10,
        "n_seeds": n_seeds,
        "require": "n_seeds>=10; median_residual_sr>=0.25; p10>0",
    }


def _gate_g3(g: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    # Research (ultra8x-refine): 30 untouched acceptance seeds.
    p05 = _f(g.get("p05_sr"))
    med = _f(g.get("median_sr"))
    ens = _f(g.get("ensemble_residual_return"))
    ci_lo = _f(g.get("bootstrap_ci_low_median"))
    min_a = _f(g.get("min_alpha_annual"), MIN_ALPHA_ANNUAL)
    max_share = _f(g.get("max_seed_profit_share"))
    n_pos = int(g.get("n_positive_seeds") or 0)
    n_seeds = int(g.get("n_seeds") or 0)
    ok = (
        n_seeds >= 30
        and p05 > 0.0
        and med >= 0.50
        and ens > 0.0
        and ci_lo > min_a
        and max_share <= 0.50
        and n_pos >= 15
    )
    return ok, {
        "p05_sr": p05,
        "median_sr": med,
        "ensemble_residual_return": ens,
        "bootstrap_ci_low_median": ci_lo,
        "min_alpha_annual": min_a,
        "max_seed_profit_share": max_share,
        "n_positive_seeds": n_pos,
        "n_seeds": n_seeds,
        "require": "n_seeds>=30; p05>0; median>=0.50; ens>0; CI>min_alpha; "
        "max_seed_share<=0.5; n_pos>=15/30",
    }


def _gate_g4(g: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    # Research CPCV(6,2): 15 combos, 5 paths; >=12/15 positive.
    paths = [float(x) for x in (g.get("path_srs") or [])]
    n_pos = int(g.get("n_positive_combos") or 0)
    n_combos = int(g.get("n_combos") or 15)
    med = float("nan")
    if paths:
        s = sorted(paths)
        mid = len(s) // 2
        med = float(s[mid]) if len(s) % 2 else float((s[mid - 1] + s[mid]) / 2.0)
    all_pos = len(paths) >= 5 and all(sr > 0.0 for sr in paths)
    ok = all_pos and med >= 0.50 and n_pos >= 12 and n_combos >= 15
    return ok, {
        "path_srs": paths,
        "median_path_sr": med,
        "n_positive_combos": n_pos,
        "n_combos": n_combos,
        "require": "all path SR>0; median>=0.50; >=12/15 combos positive",
    }


def _gate_g5(g: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    hac_t = _f(g.get("hac_t"))
    dsr = _f(g.get("dsr"))
    pbo = _f(g.get("pbo"))
    spa_p = _f(g.get("spa_p"))
    white_rc_p = _f(g.get("white_rc_p"))
    white_rc_ok = (white_rc_p != white_rc_p) or white_rc_p <= 0.05
    ok = hac_t >= 3.0 and dsr >= 0.95 and pbo <= 0.10 and spa_p <= 0.05 and white_rc_ok
    return ok, {
        "hac_t": hac_t,
        "dsr": dsr,
        "pbo": pbo,
        "spa_p": spa_p,
        "white_rc_p": white_rc_p,
        "require": "NW HAC t>=3.0 (3 monthly lags); DSR>=0.95; PBO<=0.10; "
        "SPA p<=0.05; White RC p<=0.05 when reported",
    }


def _gate_g6(g: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    # Research primary stress is 1.5x; 2x remains an optional stricter field.
    sr15 = _f(g.get("residual_sr_1_5x"), _f(g.get("residual_sr_2x")))
    a15 = _f(g.get("alpha_1_5x"), _f(g.get("alpha_2x")))
    n_pos = int(
        g.get("n_paths_positive_1_5x")
        if g.get("n_paths_positive_1_5x") is not None
        else (g.get("n_paths_positive_2x") or 0)
    )
    n_paths = int(g.get("n_paths") or 5)
    cap = _f(g.get("capacity_alpha_10m"))
    pub1 = _b(g.get("published_1x"))
    pub15 = _b(g.get("published_1_5x", True))
    pub3 = _b(g.get("published_3x"))
    ok = (
        sr15 >= 0.25
        and a15 > 0.0
        and n_pos >= 4
        and n_paths >= 5
        and cap > 0.0
        and pub1
        and pub15
        and pub3
    )
    return ok, {
        "residual_sr_1_5x": sr15,
        "alpha_1_5x": a15,
        "n_paths_positive_1_5x": n_pos,
        "n_paths": n_paths,
        "capacity_alpha_10m": cap,
        "published_1x": pub1,
        "published_1_5x": pub15,
        "published_3x": pub3,
        "require": "at 1.5x: residual SR>=0.25, alpha>0, >=4/5 paths>0; "
        "capacity alpha>0 at $10M; publish 1x/1.5x/3x",
    }


_GATE_FNS = (
    ("G0", "g0", _gate_g0),
    ("G1", "g1", _gate_g1),
    ("G2", "g2", _gate_g2),
    ("G3", "g3", _gate_g3),
    ("G4", "g4", _gate_g4),
    ("G5", "g5", _gate_g5),
    ("G6", "g6", _gate_g6),
)


def run_gate_ladder(bundle: dict) -> dict:
    """Evaluate locked G0–G6 predicates on a precomputed metrics bundle.

    Returns a report with per-gate PASS/FAIL. Overall ``pass`` is True iff every
    gate passes (conjunctive). Golden FAIL whenever any gate is false.
    """
    gates: dict[str, Any] = {}
    fails: list[str] = []
    for name, key, fn in _GATE_FNS:
        body = bundle.get(key) or {}
        if not isinstance(body, Mapping):
            body = {}
        ok, detail = fn(body)
        gates[name] = {"pass": bool(ok), **detail}
        if not ok:
            fails.append(name)
    overall = len(fails) == 0
    return {
        "pass": overall,
        "any_fail": not overall,
        "fails": fails,
        "gates": gates,
        "protocol": "residual_equity_g0_g6",
    }


def _spa_rivals_thin(report: Mapping[str, Any]) -> bool:
    spa = report.get("hansen_spa") or report.get("spa") or {}
    if not isinstance(spa, Mapping):
        return True
    if spa.get("reason") == "spa_rivals_insufficient":
        return True
    if spa.get("ok") is False and "rival" in str(spa.get("reason") or "").lower():
        return True
    n_econ = spa.get("n_economic_rivals")
    if n_econ is None:
        rivals = spa.get("rival_names") or spa.get("rivals") or []
        n_econ = len(rivals) if isinstance(rivals, Sequence) else 0
    return int(n_econ) < MIN_ECONOMIC_SPA_RIVALS


def _ledger_incomplete(report: Mapping[str, Any]) -> bool:
    ledger = report.get("trial_ledger")
    if ledger is None:
        return True
    if isinstance(ledger, Sequence) and not isinstance(ledger, (str, bytes)):
        return len(ledger) == 0
    if not isinstance(ledger, Mapping):
        return True
    if ledger.get("complete") is False:
        return True
    trials = ledger.get("trials") or ledger.get("rows") or []
    return len(trials) == 0


def refuse_alpha_stamp(report: Mapping[str, Any]) -> None:
    """Raise if an evidence pack must not stamp an alpha claim.

    Refuses when arm is mix / alpha_claim false, G0 is false, SPA economic
    rivals are thin, or the trial ledger is incomplete. Does not soft-pass.
    """
    arm = report.get("arm") or report.get("arm_id")
    alpha_claim = report.get("alpha_claim")
    if arm == "mix" or alpha_claim is False:
        raise ValueError(
            "refuse alpha stamp: mix/spectrum diagnostic cannot claim residual alpha"
        )
    ladder = report.get("gate_ladder")
    if not isinstance(ladder, Mapping):
        ladder = run_gate_ladder(dict(report.get("bundle") or report))
    g0 = (ladder.get("gates") or {}).get("G0") or {}
    if not _b(g0.get("pass")):
        raise ValueError("refuse alpha stamp: G0 false")
    if _spa_rivals_thin(report):
        raise ValueError("refuse alpha stamp: SPA rivals thin")
    if _ledger_incomplete(report):
        raise ValueError("refuse alpha stamp: ledger incomplete")
