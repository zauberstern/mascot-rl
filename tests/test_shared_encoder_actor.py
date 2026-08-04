"""Phase N1: shared temporal encoder + shared actor (param count vs ModuleList)."""
from __future__ import annotations

import torch

from src.features.extractor import AlphaFeatureExtractor
from src.plugins.resolve import _ALLOWED_ACTOR, resolve_plugins
from src.policy.happo import HAPPOEngine


def test_share_temporal_encoder_shapes_and_fewer_params():
    k, d, t = 10, 8, 6
    per = AlphaFeatureExtractor(
        num_assets=k,
        d_model=d,
        d_state=4,
        temporal_backend="mlp",
        use_dhgnn=False,
        share_temporal_encoder=False,
    )
    shared = AlphaFeatureExtractor(
        num_assets=k,
        d_model=d,
        d_state=4,
        temporal_backend="mlp",
        use_dhgnn=False,
        share_temporal_encoder=True,
    )
    x = torch.randn(2, k, t, d)
    iv = torch.randn(2, k, d)
    out_per = per(x, iv)
    out_sh = shared(x, iv)
    assert out_per.shape == (2, k, d)
    assert out_sh.shape == (2, k, d)
    n_per = sum(p.numel() for p in per.parameters())
    n_sh = sum(p.numel() for p in shared.parameters())
    assert n_sh < n_per
    assert shared.share_temporal_encoder is True
    assert len(shared.temporal_blocks) == 1
    assert len(per.temporal_blocks) == k


def test_actor_backend_shared_shapes_and_fewer_params():
    assert "shared" in _ALLOWED_ACTOR or "shared_mappo" in _ALLOWED_ACTOR
    k, d, m = 10, 8, 4
    mod = HAPPOEngine(
        k, d, m, turnover_limit=0.15, use_projection=False, actor_backend="modulelist"
    )
    shared = HAPPOEngine(
        k, d, m, turnover_limit=0.15, use_projection=False, actor_backend="shared"
    )
    e = torch.randn(3, k, d)
    macro = torch.randn(3, m)
    w_prev = torch.zeros(3, k)
    deltas = torch.ones(3, k) * 0.1
    w_mod, v_mod = mod(e, macro, w_prev, deltas)
    w_sh, v_sh = shared(e, macro, w_prev, deltas)
    assert w_mod.shape == w_sh.shape == (3, k)
    assert v_mod.shape == v_sh.shape == (3,)
    n_mod = sum(p.numel() for p in mod.actors.parameters())
    n_sh = sum(p.numel() for p in shared.actors.parameters())
    assert n_sh < n_mod
    assert shared.actor_backend in ("shared", "shared_mappo")
    assert len(shared.actors) == 1
    assert len(mod.actors) == k


def test_eq_alloc_opt_in_shared_actor_resolves():
    """STATUS_QUO stays modulelist; eq_alloc YAML opts into actor_backend=shared."""
    sq = resolve_plugins({})
    assert sq["actor_backend"] == "modulelist"
    eq = resolve_plugins({"plugins": {"actor_backend": "shared"}})
    assert eq["actor_backend"] == "shared"
    # Alias used by spectrum / capital hygiene stamps.
    alias = resolve_plugins({"plugins": {"actor_backend": "shared_mappo"}})
    assert alias["actor_backend"] == "shared_mappo"
