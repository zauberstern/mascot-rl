"""A-9: checkpoint payload None must stamp empty and fail when requested."""
from __future__ import annotations

from pathlib import Path

import pytest


class _NoCheckpointAgent:
    name = "noop"


def test_save_checkpoint_stamps_empty_when_no_payload(tmp_path: Path) -> None:
    from mascotrl.eval.research_alpha_train import _save_checkpoint

    cfg: dict = {"_checkpoint_dir": str(tmp_path / "ckpt")}
    _save_checkpoint(_NoCheckpointAgent(), cfg, seed=0, episode=1, optimizer_steps=0)
    assert cfg.get("checkpoint_payload_empty") is True
    assert not list((tmp_path / "ckpt").glob("*.pt"))


def test_save_checkpoint_raises_when_explicit_every_and_no_payload(tmp_path: Path) -> None:
    from mascotrl.eval.research_alpha_train import _save_checkpoint

    cfg = {
        "_checkpoint_dir": str(tmp_path / "ckpt"),
        "checkpoint_every_n_episodes": 5,
    }
    with pytest.raises(RuntimeError, match="checkpoint_payload_empty|checkpoint_every"):
        _save_checkpoint(_NoCheckpointAgent(), cfg, seed=0, episode=5, optimizer_steps=1)
