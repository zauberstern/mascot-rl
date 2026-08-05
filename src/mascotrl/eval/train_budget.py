"""Training budget helpers: optimizer-step floor and learning-curve IO."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def assert_optimizer_step_floor(optimizer_steps: int, *, min_steps: int) -> None:
    """Fail closed when a fold trains with too few optimizer steps."""
    steps = int(optimizer_steps)
    floor = int(min_steps)
    if floor > 0 and steps < floor:
        raise RuntimeError(
            f"optimizer_steps={steps} below configured floor min_optimizer_steps={floor}"
        )


def write_learning_curve(
    curve: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Persist per-episode learning curve JSON for campaign telemetry."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [dict(row) for row in curve]
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return out
