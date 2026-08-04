#!/usr/bin/env python3
"""Generate config/spectrum/cherrypick_val/ for the VAL validation wave."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "config" / "spectrum" / "cherrypick_val"
FINAL_MANIFEST = ROOT / "config" / "spectrum" / "cherrypick_final" / "manifest.json"

HYBRID_SRC = (
    ROOT
    / "config"
    / "spectrum"
    / "cherrypick"
    / "_deferred_train_world"
    / "eq_K100_single_ppo_mlp_softmax_mean_std_cao_tw-hybrid_pretrain_finetune.yaml"
)
RB_SB3_SRC = (
    ROOT
    / "config"
    / "spectrum"
    / "cherrypick"
    / "narrative"
    / "eq_K100_single_ppo_mlp_softmax_mean_std_cao_rb-sb3.yaml"
)
MAMBA_K100_SRC = (
    ROOT
    / "config"
    / "spectrum"
    / "cherrypick"
    / "_dropped_mamba"
    / "eq_K100_single_ppo_mamba_softmax_mean_std_cao.yaml"
)


def _load_final() -> dict:
    return json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))


def _dropped_stems(data: dict) -> set[str]:
    out: set[str] = set()
    for item in data.get("dropped_cells") or []:
        if isinstance(item, dict):
            stem = str(item.get("stem") or "").strip()
        else:
            stem = str(item).strip()
        if stem:
            out.add(stem)
    return out


def _served_pick_stems(data: dict) -> list[str]:
    cells = [str(c).strip() for c in (data.get("cells") or []) if str(c).strip()]
    dropped = _dropped_stems(data)
    return sorted(set(cells) - dropped)


def _served_k200_stems(data: dict) -> list[str]:
    k200 = data.get("k200") or {}
    cells = [str(c).strip() for c in (k200.get("cells") or []) if str(c).strip()]
    dropped = _dropped_stems(data)
    return sorted(set(cells) - dropped)


def _copy_yaml(src: Path, dst: Path, *, restamp: dict[str, object] | None = None) -> None:
    text = src.read_text(encoding="utf-8")
    if restamp:
        for key, val in restamp.items():
            if isinstance(val, list):
                rep = "[" + ", ".join(str(x) for x in val) + "]"
            elif isinstance(val, bool):
                rep = "true" if val else "false"
            else:
                rep = str(val)
            pattern = rf"^{re.escape(key)}:.*$"
            repl = f"{key}: {rep}"
            new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
            if n == 0:
                text = text.rstrip() + f"\n{key}: {rep}\n"
            else:
                text = new_text
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def _mamba_probe(k: int) -> tuple[str, Path]:
    stem = f"eq_K{k}_single_ppo_mamba_softmax_mean_std_cao"
    dst = OUT_DIR / f"{stem}.yaml"
    text = MAMBA_K100_SRC.read_text(encoding="utf-8")
    text = text.replace("K100", f"K{k}")
    text = text.replace(
        "eq_K100_single_ppo_mamba_softmax_mean_std_cao",
        stem,
    )
    text = re.sub(r"^n_assets: \d+$", f"n_assets: {k}", text, count=1, flags=re.MULTILINE)
    text = re.sub(
        r"^# Cherrypick Tier 1 sweep B cell:.*$",
        f"# VAL mamba memory probe K={k} (code-path canary; not panel-comparable)",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    # Probes are not panel-comparable; cut peak RSS via high n_minibatches.
    # Mamba requires use_equity_feature_cube=true; free-tier 6912 MiB OOMs → himem.
    minibatches = 64 if k >= 25 else 32
    if "n_minibatches:" in text:
        text = re.sub(r"^n_minibatches: \d+$", f"n_minibatches: {minibatches}", text, flags=re.MULTILINE)
    else:
        text = re.sub(
            r"^(train_env_steps: \d+)$",
            rf"\1\nn_minibatches: {minibatches}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    if "requires_himem:" not in text:
        text = text.rstrip() + "\nrequires_himem: true\n"
    dst.write_text(text, encoding="utf-8")
    return stem, dst


def generate(*, dry_run: bool = False) -> dict:
    data = _load_final()
    entries: list[dict] = []

    for stem in _served_pick_stems(data):
        src = ROOT / "config" / "spectrum" / "cherrypick" / f"{stem}.yaml"
        if not src.is_file():
            raise FileNotFoundError(f"missing PICK source: {src}")
        dst = OUT_DIR / f"{stem}.yaml"
        entries.append(
            {
                "stem": stem,
                "source": str(src.relative_to(ROOT)),
                "role": "pick_served",
                "acceptance": "validated_final",
            }
        )
        if not dry_run:
            _copy_yaml(src, dst)

    for stem in _served_k200_stems(data):
        src = ROOT / "config" / "spectrum" / "cherrypick" / "k200" / f"{stem}.yaml"
        if not src.is_file():
            raise FileNotFoundError(f"missing K200 source: {src}")
        dst = OUT_DIR / f"{stem}.yaml"
        entries.append(
            {
                "stem": stem,
                "source": str(src.relative_to(ROOT)),
                "role": "k200_served",
                "acceptance": "validated_final",
            }
        )
        if not dry_run:
            _copy_yaml(src, dst)

    hybrid_stem = "eq_K100_single_ppo_mlp_softmax_mean_std_cao_tw-hybrid_pretrain_finetune"
    entries.append(
        {
            "stem": hybrid_stem,
            "source": str(HYBRID_SRC.relative_to(ROOT)),
            "role": "hybrid_negative_control",
            "acceptance": "validated_final_or_single_attempt_fail_closed_error",
        }
    )
    if not dry_run:
        _copy_yaml(HYBRID_SRC, OUT_DIR / f"{hybrid_stem}.yaml")

    rb_stem = "eq_K100_single_ppo_mlp_softmax_mean_std_cao_rb-sb3"
    entries.append(
        {
            "stem": rb_stem,
            "source": str(RB_SB3_SRC.relative_to(ROOT)),
            "role": "sb3_backend_parity",
            "acceptance": "validated_final",
            "restamp": "screening tier from narrative",
        }
    )
    if not dry_run:
        _copy_yaml(
            RB_SB3_SRC,
            OUT_DIR / f"{rb_stem}.yaml",
            restamp={
                "protocol_tier": "screening",
                "claim_tier": "research",
                "seeds": [0],
                "train_env_steps": 25000,
                "cpcv_n_splits": 6,
                "cpcv_n_test_groups": 2,
                "grid_kind": "cherrypick_val",
            },
        )

    for k in (10, 25):
        stem, path = _mamba_probe(k)
        entries.append(
            {
                "stem": stem,
                "source": str(MAMBA_K100_SRC.relative_to(ROOT)),
                "role": "mamba_memory_probe",
                "acceptance": "validated_final",
                "note": f"K={k} mamba probe; requires_himem; n_minibatches={64 if k >= 25 else 32}",
            }
        )
        if dry_run:
            pass  # stem computed; file not written in dry_run for mamba helper

    manifest = {
        "wave": "VAL",
        "n_cells": len(entries),
        "scope": "eq_only_validation",
        "purpose": "Widest free-tier correctness net before paid Tier A/B",
        "cells": [e["stem"] for e in entries],
        "entries": entries,
    }
    if not dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    manifest = generate(dry_run=bool(args.dry_run))
    print(json.dumps({"n_cells": manifest["n_cells"], "wave": "VAL"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
