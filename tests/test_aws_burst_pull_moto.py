"""Pull script validation helpers + completeness."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

from scripts.validate_remote_cell import validate_remote_cell


def test_rejects_non_remote_host() -> None:
    out = validate_remote_cell({"compute_host": "local", "instance_type": "x"})
    assert out["ok"] is False
