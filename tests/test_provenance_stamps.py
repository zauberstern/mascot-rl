"""Provenance stamps and logger presence for library backends."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

from src.eval.arch_bootstrap import log as arch_log
from src.eval.cpcv_backend import resolve_use_purgedcv
from src.eval.residualization import ResidualizerState, fit_ipca3_residualizer
from src.logging_utils import get_logger
from src.policy.harl_adapter import log as harl_log
from src.policy.omnisafe_adapter import log as omnisafe_log
from src.policy.sb3_adapter import log as sb3_log, resolve_rl_backend


def test_adapter_loggers_named():
    assert sb3_log.name
    assert omnisafe_log.name
    assert harl_log.name
    assert arch_log.name
    assert "sb3" in sb3_log.name
    assert "omnisafe" in omnisafe_log.name


def test_get_logger_nonempty():
    log = get_logger("mascotrl.eval.research_alpha_cpcv")
    assert log.name == "mascotrl.eval.research_alpha_cpcv"


def test_residualizer_backend_used_stamp():
    import numpy as np

    panel = np.random.randn(40, 5)
    state = fit_ipca3_residualizer(panel, backend="custom")
    assert isinstance(state, ResidualizerState)
    assert state.backend_used == "custom"


def test_resolve_defaults():
    assert resolve_rl_backend({}) == "sb3"
    assert resolve_use_purgedcv({}) is True
    assert resolve_rl_backend({"rl_backend": "custom"}) == "custom"
    assert resolve_use_purgedcv({"use_purgedcv": False}) is False
