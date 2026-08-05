#!/usr/bin/env python3
"""Generate RC6_HEADS dose-response cherrypick YAMLs (entmax / Tsallis / tilt ladder).

Writes under config/spectrum/cherrypick/rc6_heads/. Inherits RC6_OVERRIDES and
``_cell`` from generate_rc6_panel.py so locks stay bit-identical to RC6.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from scripts.generate_rc6_panel import RC6_OVERRIDES, _cell
from mascotrl.spectrum.cell_schema import validate_cell_cfg
from mascotrl.spectrum.registry import validate_cfg

OUT = ROOT / "config" / "spectrum" / "cherrypick" / "rc6_heads"

# H1: head dose-response (softmax / entmax_15 / sparse_tilt) x mean_std_cao
H1 = [
    ("ppo", "softmax", "mean_std_cao", ""),
    ("ppo", "entmax_15", "mean_std_cao", ""),
    ("ppo", "sparse_tilt", "mean_std_cao", ""),
]

# H2: entropy-bonus isolation (Tsallis variant; sparse_tilt twin already in RC6)
H2 = [
    ("ppo", "sparse_tilt_tsallis", "mean_std_cao", ""),
]

# H3: tilt-gain ladder on sparse_tilt (5.0 already in RC6 / H1)
H3 = [
    ("ppo", "sparse_tilt", "mean_std_cao", "_tg2p5"),
    ("ppo", "sparse_tilt", "mean_std_cao", "_tg10"),
]

# H4: objective cross-check on entmax_15
H4 = [
    ("ppo", "entmax_15", "cvar_ru", ""),
    ("ppo", "entmax_15", "differential_sharpe", ""),
    ("ppo", "entmax_15", "mtm_pnl", ""),
]


def _tilt_for_suffix(suffix: str) -> float | None:
    if suffix == "_tg2p5":
        return 2.5
    if suffix == "_tg10":
        return 10.0
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    errors: list[str] = []
    # Touch RC6_OVERRIDES so import stays live for auditors.
    assert float(RC6_OVERRIDES["weight_head_tilt_gain"]) == 5.0

    for algo, head, obj, suffix in H1 + H2 + H3 + H4:
        cfg = _cell("eq", algo, head, obj, suffix=suffix)
        tg = _tilt_for_suffix(suffix)
        if tg is not None:
            cfg["weight_head_tilt_gain"] = float(tg)
        stem = cfg["spectrum_cell_id"]
        try:
            validate_cell_cfg(cfg)
            validate_cfg(cfg)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{stem}: {exc}")
            continue
        path = OUT / f"{stem}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        written += 1

    print(f"wrote {written} RC6_HEADS cells to {OUT}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(" ", e)
        raise SystemExit(1)
    if written != 9:
        raise SystemExit(f"expected 9 cells, wrote {written}")


if __name__ == "__main__":
    main()
