"""Spectrum resume wiring: out_dir/resume, HAPPO ckpt load, S3 pull/push helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch


def test_spectrum_run_config_hash_stable() -> None:
    from scripts.run_spectrum_campaign import _spectrum_run_config_hash

    cfg_a = {"algo": "ppo", "n_assets": 8, "objective": "sharpe"}
    cfg_b = {"objective": "sharpe", "n_assets": 8, "algo": "ppo"}
    cfg_c = {"algo": "ppo", "n_assets": 8, "objective": "cvar"}
    assert _spectrum_run_config_hash(cfg_a) == _spectrum_run_config_hash(cfg_b)
    assert _spectrum_run_config_hash(cfg_a) != _spectrum_run_config_hash(cfg_c)
    assert len(_spectrum_run_config_hash(cfg_a)) == 16


def test_prepare_spectrum_resume_dirs_sets_checkpoint_and_cpcv(tmp_path: Path) -> None:
    from scripts.run_spectrum_campaign import _prepare_spectrum_resume_dirs

    cfg: dict = {"algo": "ppo", "n_assets": 4}
    cpcv_dir, ckpt_dir, run_hash = _prepare_spectrum_resume_dirs(cfg, tmp_path)
    assert cpcv_dir == tmp_path / "cpcv"
    assert ckpt_dir == tmp_path / "ckpt"
    assert cpcv_dir.is_dir() and ckpt_dir.is_dir()
    assert cfg["_checkpoint_dir"] == str(ckpt_dir)
    assert cfg["_run_config_hash"] == run_hash
    assert len(run_hash) == 16


def test_run_research_arm_passes_out_dir_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Spectrum research path must forward out_dir/resume into CPCV."""
    from scripts import run_spectrum_campaign as camp

    captured: dict = {}

    def _fake_cpcv(*_a, **kwargs):
        captured.update(kwargs)
        return {
            "path_summary": {"mean_sharpe": 0.0, "n_paths": 1},
            "panel_source": "toy",
            "claim_tier": "research",
        }

    monkeypatch.setattr(camp, "_try_om_research_panel", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "mascotrl.eval.research_alpha_cpcv.run_research_alpha_cpcv",
        _fake_cpcv,
    )
    # Import path used inside _run_research_arm
    import mascotrl.eval.research_alpha_cpcv as rac

    monkeypatch.setattr(rac, "run_research_alpha_cpcv", _fake_cpcv)
    monkeypatch.setattr(rac, "dry_run_research_alpha_cpcv", lambda *_a, **_k: {"dry_run": True})

    cfg = {
        "algo": "ppo",
        "n_assets": 4,
        "headline_fill": "pct75",
        "claim_tier": "research",
        "spectrum_budget_tier": "dispatch_only",
    }
    # Force a tiny budget via resolve_spectrum_budget mock
    monkeypatch.setattr(
        camp,
        "resolve_spectrum_budget",
        lambda _c: {
            "claim_tier": "research",
            "n_episodes": 1,
            "horizon": 8,
            "seeds": [0],
            "cpcv_n_splits": 2,
            "cpcv_n_test_groups": 1,
            "dispatch_only": False,
        },
    )
    monkeypatch.setattr(camp, "validate_cfg", lambda c: dict(c))
    monkeypatch.setattr(
        camp,
        "_toy_research_panel",
        lambda **_k: (
            list(pd.bdate_range("2020-01-01", periods=32)),
            __import__("numpy").zeros((32, 4)),
            __import__("numpy").zeros((32, 3)),
        ),
    )
    monkeypatch.setattr(
        camp,
        "_aggregate_spectrum_seed_arts",
        lambda arts: arts[0],
    )

    art, err = camp._run_research_arm(
        cfg,
        "eq",
        allow_toy_panel=True,
        no_dry_run=True,
        cell_out_dir=tmp_path,
    )
    assert err is None
    assert art is not None
    assert captured.get("out_dir") is not None
    assert Path(captured["out_dir"]) == tmp_path / "cpcv"
    assert captured.get("resume") is True
    assert (tmp_path / "ckpt").is_dir()
    assert (tmp_path / "cpcv").is_dir()


def test_discover_latest_happo_checkpoint(tmp_path: Path) -> None:
    from mascotrl.eval.research_happo_cpcv import _discover_latest_happo_checkpoint

    def _write(path: Path, *, seed: int, fold_id: int, run_hash: str, episode: int) -> None:
        torch.save(
            {
                "policy": {},
                "seed": seed,
                "fold_id": fold_id,
                "run_config_hash": run_hash,
                "episode": episode,
                "optimizer_steps": episode,
            },
            path,
        )

    _write(tmp_path / "fold1_seed0_ep00001.pt", seed=0, fold_id=1, run_hash="h", episode=1)
    latest = tmp_path / "fold1_seed0_ep00005.pt"
    _write(latest, seed=0, fold_id=1, run_hash="h", episode=5)
    _write(tmp_path / "fold1_seed0_ep00009.pt", seed=0, fold_id=1, run_hash="other", episode=9)

    assert _discover_latest_happo_checkpoint(tmp_path, seed=0, fold_id=1, run_config_hash="h") == latest
    assert _discover_latest_happo_checkpoint(tmp_path, seed=0, fold_id=1, run_config_hash="missing") is None


def test_maybe_resume_happo_checkpoint_loads_policy(tmp_path: Path) -> None:
    from mascotrl.eval.research_happo_cpcv import (
        _maybe_resume_happo_checkpoint,
        _save_happo_checkpoint,
    )

    class _Pol(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

    src = _Pol()
    with torch.no_grad():
        src.w.fill_(3.14)
    cfg = {
        "_checkpoint_dir": str(tmp_path),
        "_fold_id": 0,
        "_run_config_hash": "rh",
    }
    _save_happo_checkpoint(src, cfg, seed=1, episode=2, optimizer_steps=2)
    ckpts = list(tmp_path.glob("*.pt"))
    assert ckpts
    dst = _Pol()
    blob = _maybe_resume_happo_checkpoint(
        dst,
        {"_resume_checkpoint": str(ckpts[0]), "_run_config_hash": "rh"},
    )
    assert blob is not None
    assert blob["episode"] == 2
    assert torch.allclose(dst.w, src.w)


def test_run_happo_cpcv_forwards_out_dir_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mascotrl.eval import research_happo_cpcv as hap

    captured: dict = {}

    def _fake_run_cpcv(*_a, **kwargs):
        captured.update(kwargs)
        return {"path_summary": {"mean_sharpe": 0.0, "n_paths": 1}}

    monkeypatch.setattr(hap, "run_cpcv", _fake_run_cpcv)
    # HAPPO now defaults to purgedcv; stub that path too.
    import mascotrl.eval.cpcv_lib as cpcv_lib

    monkeypatch.setattr(cpcv_lib, "run_cpcv_lib", _fake_run_cpcv, raising=False)
    monkeypatch.setattr(
        hap,
        "_try_om_research_panel",
        lambda *_a, **_k: None,
        raising=False,
    )

    import scripts.run_spectrum_campaign as camp

    monkeypatch.setattr(camp, "_try_om_research_panel", lambda *_a, **_k: None)
    monkeypatch.setattr(
        camp,
        "_toy_research_panel",
        lambda **_k: (
            list(pd.bdate_range("2020-01-01", periods=40)),
            __import__("numpy").zeros((40, 4)),
            __import__("numpy").zeros((40, 3)),
        ),
    )
    import mascotrl.eval.equity_substrate as es

    monkeypatch.setattr(
        es,
        "load_lake_dyn_hrp_panel",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("lake_disabled_in_unit_test")),
    )

    art, err = hap.run_happo_cpcv(
        {"algo": "happo", "n_assets": 4, "seed": 0, "_happo_toy_fast": True},
        "eq",
        budget={
            "claim_tier": "research",
            "cpcv_n_splits": 2,
            "cpcv_n_test_groups": 1,
            "seeds": [0],
        },
        allow_toy_panel=True,
        no_dry_run=True,
        out_dir=tmp_path / "cpcv",
        resume=True,
    )
    assert err is None
    assert art is not None
    assert captured.get("resume") is True
    assert Path(captured["out_dir"]) == tmp_path / "cpcv"


def test_incomplete_cells_resubmits_checkpoint_only(tmp_path: Path) -> None:
    """Final artifact missing => incomplete even if resume/ ckpt prefix exists."""
    from scripts.aws_submit_wave import _incomplete_cells

    client = MagicMock()
    client.account_id.return_value = "111"
    # Only resume state, no final cell JSON
    client.list_keys.return_value = [
        "PICK/resume/cell_a/cpcv/campaign_manifest.json",
        "PICK/resume/cell_a/ckpt/fold0_seed0_ep00001.pt",
        "PICK/cell_b.json",
        "PICK/cell_b.json.sha256",
    ]
    with patch("scripts.aws_submit_wave.artifact_bucket", return_value="arts"):
        incomplete = _incomplete_cells(
            client,
            "PICK",
            ["config/spectrum/cherrypick/cell_a.yaml", "config/spectrum/cherrypick/cell_b.yaml"],
        )
    assert incomplete == ["config/spectrum/cherrypick/cell_a.yaml"]


def test_incomplete_cells_ignores_archive_prefix() -> None:
    """Archived finals must not suppress a resubmit of the same stem."""
    from scripts.aws_submit_wave import _incomplete_cells

    client = MagicMock()
    client.account_id.return_value = "111"
    client.list_keys.return_value = [
        "PICK_SMOKE/_archive_20260824T162126Z/eq_K50_multi_happo_mlp_cvar_ru.json",
        "PICK_SMOKE/_archive_20260824T162126Z/eq_K50_multi_happo_mlp_cvar_ru.json.sha256",
    ]
    with patch("scripts.aws_submit_wave.artifact_bucket", return_value="arts"):
        incomplete = _incomplete_cells(
            client,
            "PICK_SMOKE",
            ["config/spectrum/cherrypick_smoke/eq_K50_multi_happo_mlp_cvar_ru.yaml"],
        )
    assert incomplete == [
        "config/spectrum/cherrypick_smoke/eq_K50_multi_happo_mlp_cvar_ru.yaml"
    ]


def test_cell_runner_pull_push_resume_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deploy.aws_burst.docker import cell_runner

    bucket = "arts"
    stem = "cell_x"
    oprefix = "PICK/"
    local = tmp_path / "out"
    local.mkdir()
    (local / "cpcv").mkdir()
    (local / "ckpt").mkdir()
    (local / "cpcv" / "campaign_manifest.json").write_text('{"version":1,"completed":{}}\n')
    ckpt = local / "ckpt" / "fold0_seed0_ep00001.pt"
    torch.save({"episode": 1}, ckpt)

    uploaded: list[str] = []
    downloaded: list[tuple[str, str]] = []

    class _Body:
        def read(self) -> bytes:
            return b'{"version":1,"completed":{}}'

    s3 = MagicMock()

    def _list_objects_v2(**kwargs):
        prefix = kwargs.get("Prefix", "")
        if "resume/cell_x/" in prefix:
            return {
                "Contents": [
                    {"Key": f"{oprefix}resume/{stem}/cpcv/campaign_manifest.json"},
                    {"Key": f"{oprefix}resume/{stem}/ckpt/fold0_seed0_ep00001.pt"},
                ]
            }
        return {"Contents": []}

    def _download_file(b, key, dest):
        downloaded.append((b, key))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        if key.endswith(".pt"):
            torch.save({"episode": 1}, dest)
        else:
            Path(dest).write_text('{"version":1,"completed":{}}\n')

    def _put_object(**kwargs):
        uploaded.append(kwargs["Key"])

    s3.get_paginator.return_value.paginate.side_effect = lambda **kw: [_list_objects_v2(**kw)]
    s3.download_file.side_effect = _download_file
    s3.put_object.side_effect = _put_object
    monkeypatch.setattr(cell_runner, "_s3_client", lambda: s3)

    pull_dest = tmp_path / "pulled"
    n = cell_runner.pull_resume_state(bucket, oprefix, stem, pull_dest)
    assert n >= 2
    assert (pull_dest / "cpcv" / "campaign_manifest.json").is_file()
    assert any(k.endswith(".pt") for _, k in downloaded)

    cell_runner.push_resume_state(bucket, oprefix, stem, local)
    assert any("resume/cell_x/cpcv/campaign_manifest.json" in k for k in uploaded)
    assert any("resume/cell_x/ckpt/" in k and k.endswith(".pt") for k in uploaded)


def test_fold_manifest_resume_skips_completed(tmp_path: Path) -> None:
    """Tiny CPCV: completed fold in manifest is not re-run on resume=True."""
    from mascotrl.eval.campaign_manifest import mark_cell_complete, save_manifest
    from mascotrl.eval.cpcv import CPCVConfig, run_cpcv

    dates = list(pd.bdate_range("2015-01-01", periods=40))
    cfg = CPCVConfig(n_splits=2, n_test_groups=1, purge_days=0, embargo_days=0)
    seed, arm = 0, "eq"
    man = mark_cell_complete(
        {"version": 1, "completed": {}},
        fold_id=0,
        seed=seed,
        arm=arm,
        pnl={"2015-01-02": 0.1},
    )
    save_manifest(tmp_path, man)
    calls: list[int] = []

    def fold_runner(fold):
        calls.append(int(fold.fold_id))
        return {str(dates[0].date()): 0.0}

    out = run_cpcv(
        dates,
        fold_runner,
        cfg,
        resume=True,
        out_dir=tmp_path,
        seed=seed,
        arm=arm,
    )
    assert 0 not in calls
    assert out["resume"]["enabled"] is True
