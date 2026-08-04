import numpy as np
import pytest

pytest.importorskip("cpp_rbergomi")


def test_generate_surfaces_smoke():
    import cpp_rbergomi
    from src.simulator import get_surface_tensor

    cfg = {
        "n_paths": 2,
        "n_assets": 2,
        "n_steps": 4,
        "n_strikes": 5,
        "n_maturities": 2,
        "hurst_exponent": 0.1,
    }
    t = get_surface_tensor(cfg)
    assert t.shape == (2, 2, 4, 5, 2)
    assert t.dtype == torch_float32()
    assert np.isfinite(t.numpy()).all()


def torch_float32():
    import torch

    return torch.float32
