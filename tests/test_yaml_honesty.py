"""YAML honesty helpers + arm_equity spine parse locks."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.plumbing
import yaml

from mascotrl.eval.yaml_honesty import (
    KNOWN_UNUSED_EQ_ALLOC,
    RESEARCH_READ_KEYS,
    TrackingDict,
    assert_yaml_honesty,
    assert_yaml_honesty_tracked,
)


ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "config" / "workflows" / "arm_equity.yaml"


def test_arm_equity_yaml_parses_and_has_required_keys():
    cfg = yaml.safe_load(YAML.read_text(encoding="utf-8")) or {}
    assert cfg.get("arm", {}).get("id") == "eq"
    assert cfg.get("claim_label_stem") == "stk_ret"
    assert cfg.get("cost_in_decision") is True
    assert int(cfg.get("cpcv_n_splits") or 0) == 8
    assert int(cfg.get("cpcv_n_test_groups") or 0) == 3


def test_orphan_key_fails_honesty(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    # Minimal spine-shaped YAML that the honesty gate accepts once all keys are known-used,
    # then add a decorative orphan.
    base = {
        "turnover_limit": 0.15,
        "lr": 3e-4,
        "projection_mode": "hard",
    }
    # Seed with a known-unused key so the gate has a baseline allowlist path.
    for k in list(KNOWN_UNUSED_EQ_ALLOC)[:1]:
        base[k] = True
    p.write_text(yaml.safe_dump(base))
    # Force an orphan by adding a nonsense key that is neither read nor known-unused.
    cfg = yaml.safe_load(p.read_text()) or {}
    cfg["totally_decorative_unused_key"] = 123
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError, match="unread keys"):
        assert_yaml_honesty(p)


def test_known_unused_and_read_disjoint():
    overlap = KNOWN_UNUSED_EQ_ALLOC & RESEARCH_READ_KEYS
    assert not overlap, f"key listed as both read and unused: {overlap}"


def test_tracked_honesty_fails_when_a_wired_key_is_never_touched(tmp_path: Path):
    cfg = TrackingDict({"turnover_limit": 0.15, "lr": 3e-4})
    cfg.get("lr")
    with pytest.raises(AssertionError, match="turnover_limit"):
        assert_yaml_honesty_tracked(cfg, {"turnover_limit", "lr"})
