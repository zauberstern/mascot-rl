"""B-1: HAPPO checkpoint v2 roundtrip + v1 backward compatibility."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch


class _ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(3))

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if not hasattr(self, "_opt"):
            self._opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        return self._opt


class _ToyTrainer:
    def __init__(self, policy: _ToyPolicy) -> None:
        self.policy = policy
        self.actor_opts = [policy.optimizer]
        self.critic_opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)


def _perturb(policy: _ToyPolicy, trainer: _ToyTrainer) -> None:
    with torch.no_grad():
        policy.w.fill_(1.23)
        trainer.critic_opt.param_groups[0]["params"][0].fill_(4.56)
    loss = policy.w.sum() + trainer.critic_opt.param_groups[0]["params"][0].sum()
    loss.backward()
    policy.optimizer.step()
    trainer.critic_opt.step()


def _policy_tensor(policy: _ToyPolicy) -> torch.Tensor:
    return policy.w.detach().clone()


def _optimizer_tensors(trainer: _ToyTrainer) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for opt in trainer.actor_opts:
        for group in opt.state_dict()["state"].values():
            for v in group.values():
                if isinstance(v, torch.Tensor):
                    out.append(v.detach().clone())
    for group in trainer.critic_opt.state_dict()["state"].values():
        for v in group.values():
            if isinstance(v, torch.Tensor):
                out.append(v.detach().clone())
    return out


def test_happo_checkpoint_v2_roundtrip_policy_and_optimizer(tmp_path: Path) -> None:
    from src.eval.research_happo_cpcv import (
        _maybe_resume_happo_checkpoint,
        _save_happo_checkpoint,
    )

    src_policy = _ToyPolicy()
    src_trainer = _ToyTrainer(src_policy)
    _perturb(src_policy, src_trainer)
    before_policy = _policy_tensor(src_policy)
    before_opt = _optimizer_tensors(src_trainer)

    cfg = {
        "_checkpoint_dir": str(tmp_path),
        "_fold_id": 2,
        "_run_config_hash": "happo-v2",
    }
    _save_happo_checkpoint(
        src_policy,
        cfg,
        seed=3,
        episode=7,
        optimizer_steps=11,
        trainer=src_trainer,
    )
    ckpt = next(tmp_path.glob("*.pt"))
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert blob["format"] == 2
    assert blob["optimizer"] is not None

    dst_policy = _ToyPolicy()
    dst_trainer = _ToyTrainer(dst_policy)
    resume_cfg = {
        "_resume_checkpoint": str(ckpt),
        "_run_config_hash": "happo-v2",
    }
    out = _maybe_resume_happo_checkpoint(
        dst_policy, resume_cfg, trainer=dst_trainer
    )
    assert out is not None
    assert out.get("resumed_with_fresh_optimizer") is not True
    assert resume_cfg.get("resumed_with_fresh_optimizer") is not True
    assert torch.allclose(_policy_tensor(dst_policy), before_policy)
    for got, exp in zip(_optimizer_tensors(dst_trainer), before_opt):
        assert torch.allclose(got, exp)


def test_happo_checkpoint_v1_formatless_stamps_fresh_optimizer(tmp_path: Path) -> None:
    from src.eval.research_happo_cpcv import _maybe_resume_happo_checkpoint

    src = _ToyPolicy()
    with torch.no_grad():
        src.w.fill_(2.71)
    v1_payload = {
        "policy": src.state_dict(),
        "seed": 1,
        "fold_id": 0,
        "run_config_hash": "legacy",
        "episode": 4,
        "optimizer_steps": 9,
    }
    ckpt = tmp_path / "fold0_seed1_ep00004.pt"
    torch.save(v1_payload, ckpt)

    dst = _ToyPolicy()
    dst_trainer = _ToyTrainer(dst)
    resume_cfg = {
        "_resume_checkpoint": str(ckpt),
        "_run_config_hash": "legacy",
    }
    out = _maybe_resume_happo_checkpoint(dst, resume_cfg, trainer=dst_trainer)
    assert out is not None
    assert out["resumed_with_fresh_optimizer"] is True
    assert resume_cfg["resumed_with_fresh_optimizer"] is True
    assert dst_trainer.resumed_with_fresh_optimizer is True
    assert torch.allclose(dst.w, src.w)


def test_happo_checkpoint_v1_bare_state_dict_stamps_fresh_optimizer(tmp_path: Path) -> None:
    from src.eval.research_happo_cpcv import _maybe_resume_happo_checkpoint

    src = _ToyPolicy()
    with torch.no_grad():
        src.w.fill_(-0.5)
    ckpt = tmp_path / "bare.pt"
    torch.save(src.state_dict(), ckpt)

    dst = _ToyPolicy()
    resume_cfg = {"_resume_checkpoint": str(ckpt)}
    out = _maybe_resume_happo_checkpoint(dst, resume_cfg)
    assert out is not None
    assert out["resumed_with_fresh_optimizer"] is True
    assert resume_cfg["resumed_with_fresh_optimizer"] is True
    assert torch.allclose(dst.w, src.w)
