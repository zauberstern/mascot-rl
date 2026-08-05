"""Append-only trial ledger for Alpha v2 (baseline + seed + fold, incl. failures).

Past rows are immutable: callers receive MappingProxyType views; there is no
in-place replace / update API for historical indices.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

SCHEMA = "mascotrl.trial_ledger.v2"
REQUIRED_KEYS = ("baseline", "seed", "fold", "status")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _freeze_row(row: Mapping[str, Any]) -> MappingProxyType:
    return MappingProxyType(dict(row))


class TrialLedger:
    """Append-only ledger backed by JSONL (``.jsonl``) or a JSON list (``.json``)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []
        if self.path.is_file():
            self._rows = [dict(r) for r in load_ledger(self.path)]

    def append(
        self,
        *,
        baseline: str,
        seed: int,
        fold: int,
        status: str,
        error: str | None = None,
        sharpe: float | None = None,
        metrics: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> Mapping[str, Any]:
        row: dict[str, Any] = {
            "baseline": str(baseline),
            "seed": int(seed),
            "fold": int(fold),
            "status": str(status),
            "timestamp": _utc_now(),
            "schema": SCHEMA,
        }
        if error is not None:
            row["error"] = str(error)
        if sharpe is not None:
            row["sharpe"] = float(sharpe)
        if metrics:
            row["metrics"] = dict(metrics)
        for k, v in extra.items():
            if k in row and k in REQUIRED_KEYS:
                raise ValueError(f"cannot override required key {k!r}")
            row[k] = v
        self._rows.append(row)
        self._persist_append(row)
        return _freeze_row(row)

    def rows(self) -> Sequence[Mapping[str, Any]]:
        """Immutable views of all rows (mutations raise TypeError)."""
        return tuple(_freeze_row(r) for r in self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def replace(self, index: int, **_kwargs: Any) -> None:
        """Past rows are immutable; replace is refused."""
        raise ValueError(
            f"trial ledger is append-only; refuse replace of index {index}"
        )

    def _persist_append(self, row: dict[str, Any]) -> None:
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            return
        # JSON list document (rewrite full file on append; rows never edited).
        blob = {
            "schema": SCHEMA,
            "trials": list(self._rows),
            "n_trials": len(self._rows),
            "updated_at": row["timestamp"],
        }
        self.path.write_text(json.dumps(blob, indent=2, default=str) + "\n")


def append_trial(
    path: Path | str,
    *,
    baseline: str,
    seed: int,
    fold: int,
    status: str,
    error: str | None = None,
    sharpe: float | None = None,
    metrics: Mapping[str, Any] | None = None,
    **extra: Any,
) -> Mapping[str, Any]:
    """Append one trial row to ``path`` (creates ledger if missing)."""
    return TrialLedger(path).append(
        baseline=baseline,
        seed=seed,
        fold=fold,
        status=status,
        error=error,
        sharpe=sharpe,
        metrics=metrics,
        **extra,
    )


def load_ledger(path: Path | str) -> list[Mapping[str, Any]]:
    """Load frozen rows from JSONL or JSON-list ledger."""
    p = Path(path)
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(_freeze_row(json.loads(line)))
        return rows
    blob = json.loads(text)
    if isinstance(blob, list):
        return [_freeze_row(r) for r in blob]
    trials = blob.get("trials") or []
    return [_freeze_row(r) for r in trials]
