#!/usr/bin/env python3
"""Clone full-budget HAPPO narrative YAMLs (cvar / entropic / meanvar)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from mascotrl.spectrum.cell_schema import validate_cell_cfg
from mascotrl.spectrum.registry import validate_cfg

TEMPLATE = (
    ROOT
    / "config"
    / "spectrum"
    / "cherrypick"
    / "rc6_narrative"
    / "eq_K100_multi_happo_mlp_mean_std_cao.yaml"
)
OUT = ROOT / "config" / "spectrum" / "cherrypick" / "rc6_happo_full"
SPECS = (
    ("eq_K100_multi_happo_mlp_cvar_ru", "cvar_ru", "multi_agent_cvar"),
    ("eq_K100_multi_happo_mlp_entropic_oce", "entropic_oce", "multi_agent_entropic"),
    ("eq_K100_multi_happo_mlp_meanvar_kolm", "meanvar_kolm", "multi_agent_meanvar"),
)


def main() -> int:
    raw = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for stem, objective, role in SPECS:
        cfg = copy.deepcopy(raw)
        cfg["spectrum_cell_id"] = stem
        cfg["objective"] = objective
        cfg["narrative_role"] = role
        cfg.pop("happo_dispatch_only", None)
        try:
            validate_cell_cfg(cfg)
            validate_cfg(cfg)
        except Exception as exc:
            errors.append(f"{stem}: {exc}")
            continue
        path = OUT / f"{stem}.yaml"
        path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"wrote {len(SPECS)} yamls under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
