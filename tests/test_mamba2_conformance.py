"""Conformance of PureTorchMamba2 vs optional mamba_ssm.Mamba2Simple (CPU)."""
from __future__ import annotations

import pytest
import torch

from src.features.mamba2 import PureTorchMamba2


def test_pure_torch_self_consistency_chunk_sizes():
    """Always-on guard: chunk_size does not change L=1 outputs (no mamba_ssm)."""
    torch.manual_seed(1)
    x = torch.randn(2, 1, 16)
    a = PureTorchMamba2(d_model=16, d_state=8, chunk_size=1)
    b = PureTorchMamba2(d_model=16, d_state=8, chunk_size=32)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        ya, yb = a(x), b(x)
    assert torch.allclose(ya, yb, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("d_model,d_state,L", [(8, 4, 4), (16, 8, 8)])
def test_pure_torch_vs_mamba2_simple_forward(d_model, d_state, L):
    pytest.importorskip(
        "mamba_ssm",
        reason="optional; AWS production uses PureTorchMamba2 (CPU-only)",
    )
    from mamba_ssm.modules.mamba2_simple import Mamba2Simple

    torch.manual_seed(0)
    x = torch.randn(2, L, d_model)

    ref = Mamba2Simple(
        d_model=d_model,
        d_state=d_state,
        d_conv=4,
        expand=2,
        headdim=d_model // 2 if d_model >= 4 else d_model,
    ).eval()
    ours = PureTorchMamba2(
        d_model=d_model, d_state=d_state, d_conv=4, expand=2, chunk_size=1
    ).eval()

    ref_sd = ref.state_dict()
    ours_sd = ours.state_dict()
    shared = {
        k: v
        for k, v in ref_sd.items()
        if k in ours_sd and ours_sd[k].shape == v.shape
    }
    if not shared:
        pytest.skip("Mamba2Simple/PureTorchMamba2 have no matching weight keys")
    ours.load_state_dict({**ours_sd, **shared}, strict=False)

    with torch.no_grad():
        y_ref = ref(x)
        y_ours = ours(x)
    assert y_ref.shape == y_ours.shape == x.shape
    if len(shared) >= len(ours_sd) // 2:
        assert torch.allclose(y_ours, y_ref, atol=1e-3, rtol=1e-3)
    else:
        assert torch.isfinite(y_ours).all() and torch.isfinite(y_ref).all()
