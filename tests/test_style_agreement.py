"""Holdings vs RBSA mapped-vector cosine and disagreement flag."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import FLOAT_TOL


def test_style_agreement_perfect_alignment() -> None:
    from mascotrl.reporting.style_agreement import style_agreement

    exposures = {
        "exposure_size": 1.0,
        "exposure_value": 1.0,
        "exposure_momentum": 0.5,
    }
    rbsa = {
        "factor_names": ["mkt", "smb", "hml", "mom"],
        "rbsa_loadings": [0.1, -0.5, 0.8, 0.4],
    }
    hold = np.array([-1.0, 1.0, 0.5])
    rvec = np.array([-0.5, 0.8, 0.4])
    expected = float(
        np.dot(hold, rvec) / (np.linalg.norm(hold) * np.linalg.norm(rvec))
    )
    out = style_agreement(exposures, rbsa)
    assert out["style_agreement_cosine"] == pytest.approx(expected, **FLOAT_TOL)
    assert out["style_disagreement_flag"] is False


def test_style_agreement_opposite_flag() -> None:
    from mascotrl.reporting.style_agreement import style_agreement

    exposures = {
        "exposure_size": -1.0,
        "exposure_value": -1.0,
        "exposure_momentum": -1.0,
    }
    # holdings mapped = -(-1), -1, -1 = (1, -1, -1)
    rbsa = {
        "factor_names": ["mkt", "smb", "hml", "mom"],
        "rbsa_loadings": [0.0, -1.0, 1.0, 1.0],
    }
    out = style_agreement(exposures, rbsa)
    assert out["style_agreement_cosine"] < 0.2
    assert out["style_disagreement_flag"] is True


def test_style_agreement_missing_rbsa_nan() -> None:
    from mascotrl.reporting.style_agreement import style_agreement

    out = style_agreement(
        {"exposure_size": 1.0, "exposure_value": 0.0, "exposure_momentum": 0.0},
        {"rbsa_loadings": [], "factor_names": []},
    )
    assert math.isnan(out["style_agreement_cosine"])
    assert out["style_disagreement_flag"] is False
    assert out["reason"] == "rbsa_unavailable"


def test_umd_alias_for_mom() -> None:
    from mascotrl.reporting.style_agreement import style_agreement

    exposures = {
        "exposure_size": 0.0,
        "exposure_value": 0.0,
        "exposure_momentum": 1.0,
    }
    rbsa = {
        "factor_names": ["mkt", "smb", "hml", "umd"],
        "rbsa_loadings": [0.0, 0.0, 0.0, 1.0],
    }
    out = style_agreement(exposures, rbsa)
    assert out["style_agreement_cosine"] == pytest.approx(1.0, **FLOAT_TOL)
    assert out["rbsa_style_vec"][2] == pytest.approx(1.0, **FLOAT_TOL)
