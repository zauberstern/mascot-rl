"""WP-P7: mask honesty + EarnMore mask tokens."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mascotrl.features.mask_tokens import (
    MaskTokenEncoder,
    apply_mask_token,
    refuse_logit_only_masking,
)
from mascotrl.policy.rasp_locks import assert_mask_honesty


def test_all_true_mask_refused_with_availability() -> None:
    with pytest.raises(ValueError, match="mask_all_true_with_availability"):
        assert_mask_honesty(np.ones((5, 3), dtype=bool), availability_exists=True)


def test_mask_token_changes_invalid_slots() -> None:
    torch.manual_seed(0)
    enc = MaskTokenEncoder(n_channels=4)
    feats = torch.randn(2, 5, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0]])
    out = enc(feats, mask)
    assert torch.allclose(out[0, 0], feats[0, 0])
    assert not torch.allclose(out[0, 2], feats[0, 2])
    assert torch.allclose(out[0, 2], enc.mask_token)


def test_logit_only_masking_refused() -> None:
    with pytest.raises(ValueError, match="logit_only_masking_refused"):
        refuse_logit_only_masking(representation_masked=False)
    refuse_logit_only_masking(representation_masked=True)


def test_apply_mask_token_functional() -> None:
    feats = torch.ones(3, 2)
    mask = torch.tensor([1.0, 0.0, 1.0])
    token = torch.tensor([-1.0, -2.0])
    out = apply_mask_token(feats, mask, token)
    assert torch.allclose(out[0], torch.ones(2))
    assert torch.allclose(out[1], token)
