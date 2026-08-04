"""Intra-fold checkpoints resume only matching campaign cells."""
from __future__ import annotations

from pathlib import Path

import torch

from src.eval.research_alpha_train import _discover_latest_checkpoint


def _write_checkpoint(
    path: Path, *, seed: int, fold_id: int, run_config_hash: str, episode: int
) -> None:
    torch.save(
        {
            "seed": seed,
            "fold_id": fold_id,
            "run_config_hash": run_config_hash,
            "episode": episode,
            "policy": {},
            "optimizer": None,
        },
        path,
    )


def test_discover_latest_checkpoint_matches_seed_fold_and_hash(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path / "fold2_seed9_ep00001.pt",
        seed=9,
        fold_id=2,
        run_config_hash="same",
        episode=1,
    )
    latest = tmp_path / "fold2_seed9_ep00003.pt"
    _write_checkpoint(
        latest, seed=9, fold_id=2, run_config_hash="same", episode=3
    )
    _write_checkpoint(
        tmp_path / "fold2_seed9_ep00009.pt",
        seed=9,
        fold_id=2,
        run_config_hash="other",
        episode=9,
    )
    _write_checkpoint(
        tmp_path / "fold3_seed9_ep00010.pt",
        seed=9,
        fold_id=3,
        run_config_hash="same",
        episode=10,
    )

    assert _discover_latest_checkpoint(tmp_path, 9, 2, "same") == latest
    assert _discover_latest_checkpoint(tmp_path, 9, 2, "missing") is None
