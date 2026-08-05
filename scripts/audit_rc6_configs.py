#!/usr/bin/env python3
"""RC6 config audit: fail-stop checklist before fleet submit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

RC6 = ROOT / "config" / "spectrum" / "cherrypick" / "rc6"
CANARY = ROOT / "config" / "spectrum" / "cherrypick" / "rc6_canary"

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def _audit_one(p: Path, *, require_cube_false_non_hist: bool = True) -> None:
    cfg = yaml.safe_load(p.read_text())
    stem = p.stem
    check(int(cfg.get("train_env_steps") or 0) >= 300000, f"{stem}: train_env_steps")
    check(int(cfg.get("train_updates_per_fold") or 0) >= 3, f"{stem}: updates")
    check(int(cfg.get("train_epochs") or 0) >= 8, f"{stem}: epochs")
    # weekly_rebal is an intentional REGIME ablation under RC6 locks.
    if stem.endswith("_weekly_rebal"):
        check(str(cfg.get("rebalance_cadence")) == "weekly", f"{stem}: cadence")
    else:
        check(str(cfg.get("rebalance_cadence")) == "daily", f"{stem}: cadence")
    world = str(cfg.get("train_world") or "historical")
    if world == "historical":
        check(
            bool(cfg.get("use_equity_feature_cube")),
            f"{stem}: feature cube required for historical",
        )
        check(
            int(cfg.get("ppo_hidden") or 0) >= 256
            or str(cfg.get("algo")) in ("dqn", "happo"),
            f"{stem}: ppo_hidden",
        )
    elif require_cube_false_non_hist:
        # RASP lock: non-historical worlds must not load the equity feature cube.
        check(
            bool(cfg.get("use_equity_feature_cube")) is False,
            f"{stem}: use_equity_feature_cube must be false for non-historical",
        )
    if str(cfg.get("algo")) in ("ppo", "cppo"):
        check(float(cfg.get("clip_eps") or 0) >= 0.3, f"{stem}: clip_eps")
        check(str(cfg.get("rl_backend")) == "custom", f"{stem}: rl_backend")
        check(float(cfg.get("entropy_coef") or 0) >= 0.01, f"{stem}: entropy")
    if str(cfg.get("weight_head") or "") == "sparse_tilt":
        check(
            float(cfg.get("weight_head_tilt_gain") or 0) >= 5.0,
            f"{stem}: sparse_tilt tilt_gain",
        )
    if not stem.endswith("_hardtau"):
        check(str(cfg.get("projection_mode")) == "soft", f"{stem}: soft proj")
    else:
        check(str(cfg.get("projection_mode")) == "hard", f"{stem}: hardtau")

    from mascotrl.spectrum.cell_schema import validate_cell_cfg
    from mascotrl.spectrum.registry import validate_cfg

    try:
        validate_cell_cfg(cfg)
        validate_cfg(cfg)
    except Exception as exc:  # noqa: BLE001
        FAILS.append(f"{stem}: validate {exc}")


def main() -> None:
    paths = sorted(RC6.glob("*.yaml"))
    # Eq-only RC6 fleet (mix/opt deferred); includes DESKORG HAPPO.
    check(len(paths) >= 79, f"expected >=79 RC6 eq cells, got {len(paths)}")
    canary = sorted(CANARY.glob("*.yaml"))
    check(len(canary) == 10, f"expected 10 canary cells, got {len(canary)}")

    for p in paths:
        _audit_one(p)

    for p in canary:
        _audit_one(p)
        cfg = yaml.safe_load(p.read_text())
        stem = p.stem
        # Canary field audit: must share RC6 locks (daily cadence + projection).
        check(str(cfg.get("rebalance_cadence")) == "daily", f"canary {stem}: cadence")
        check(
            str(cfg.get("projection_mode")) in ("soft", "hard"),
            f"canary {stem}: projection_mode",
        )

    sparse = [p for p in paths if "sparse_tilt" in p.stem]
    softmax_ctrl = [
        p for p in paths if "_softmax_" in p.stem and "sparse_tilt" not in p.stem
    ]
    # Dual-variant panel: sparse_tilt is the RC6 pivot; softmax remains as control arm.
    check(len(sparse) >= 40, f"sparse_tilt cells={len(sparse)}")
    check(len(softmax_ctrl) >= 10, f"softmax controls={len(softmax_ctrl)}")

    if FAILS:
        print(f"AUDIT FAIL ({len(FAILS)}):")
        for f in FAILS[:50]:
            print(" ", f)
        if len(FAILS) > 50:
            print(f"  ... +{len(FAILS)-50} more")
        sys.exit(1)
    print(f"AUDIT PASS: {len(paths)} RC6 cells, {len(canary)} canary")


if __name__ == "__main__":
    main()
