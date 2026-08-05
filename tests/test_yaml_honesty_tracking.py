"""A11: TrackingDict records real config access; the gate checks reality,
not a hand-maintained "should read" list.
"""
from __future__ import annotations

import pytest

from mascotrl.eval.yaml_honesty import TrackingDict, assert_yaml_honesty_tracked, track_copy


def test_tracking_dict_records_getitem_and_get() -> None:
    d = TrackingDict({"a": 1, "b": 2, "c": 3})
    assert d.accessed_keys == frozenset()
    _ = d["a"]
    _ = d.get("b")
    assert d.accessed_keys == {"a", "b"}
    assert "c" not in d.accessed_keys


def test_tracking_dict_contains_counts_as_access() -> None:
    d = TrackingDict({"a": 1})
    assert "a" in d
    assert "a" in d.accessed_keys


def test_track_copy_preserves_shared_accessed_set() -> None:
    d = TrackingDict({"a": 1, "b": 2})
    copy1 = track_copy(d)
    _ = copy1.get("a")
    # The original TrackingDict sees the access made through the copy.
    assert "a" in d.accessed_keys

    copy2 = track_copy(copy1)
    _ = copy2.get("b")
    assert "b" in d.accessed_keys


def test_track_copy_plain_dict_is_a_plain_copy() -> None:
    plain = {"a": 1}
    out = track_copy(plain)
    assert isinstance(out, dict)
    assert not isinstance(out, TrackingDict)
    assert out == plain


def test_assert_yaml_honesty_tracked_passes_when_all_read_or_unused() -> None:
    d = TrackingDict({"a": 1, "b": 2})
    _ = d.get("a")
    report = assert_yaml_honesty_tracked(d, {"a", "b"}, known_unused={"b"})
    assert report["orphans"] == []


def test_assert_yaml_honesty_tracked_fails_on_never_read_key() -> None:
    d = TrackingDict({"a": 1, "b": 2})
    _ = d.get("a")
    with pytest.raises(AssertionError, match="never read at runtime"):
        assert_yaml_honesty_tracked(d, {"a", "b"}, known_unused=set())


def test_assert_yaml_honesty_tracked_reflects_nested_copy_reads() -> None:
    """A key read only through a dict(cfg)-style copy still counts, as
    long as the copy is made via ``track_copy`` (the campaign's actual
    call sites were updated to use it instead of a bare ``dict(cfg)``)."""
    d = TrackingDict({"turnover_limit": 0.15, "unused_key": 1})
    cfg_local = track_copy(d)
    _ = cfg_local.get("turnover_limit")
    report = assert_yaml_honesty_tracked(
        d, {"turnover_limit", "unused_key"}, known_unused={"unused_key"}
    )
    assert report["orphans"] == []


def test_friction_spec_from_cfg_preserves_tracking() -> None:
    """D1: bare dict(cfg) inside friction_spec_from_cfg must not drop
    TrackingDict; reading plugins must mark it accessed."""
    from mascotrl.eval.friction import friction_spec_from_cfg

    cfg = TrackingDict({"plugins": {"om_touch": {"enabled": True}}, "arm": {}})
    friction_spec_from_cfg(cfg)
    assert "plugins" in cfg._accessed


def test_assert_yaml_honesty_tracked_plugins_via_friction() -> None:
    """D1: config with plugins + friction_spec_from_cfg must pass tracked honesty."""
    from mascotrl.eval.friction import friction_spec_from_cfg

    cfg = TrackingDict(
        {
            "plugins": {"om_touch": {"enabled": True}},
            "arm": {"friction_spec_id": "v2_quote_touch"},
            "equity_bps": 5.0,
        }
    )
    friction_spec_from_cfg(cfg)
    report = assert_yaml_honesty_tracked(
        cfg,
        {"plugins", "arm", "equity_bps"},
        known_unused=set(),
    )
    assert report["orphans"] == []


def test_plugins_in_research_read_keys() -> None:
    from mascotrl.eval.yaml_honesty import RESEARCH_READ_KEYS

    assert "plugins" in RESEARCH_READ_KEYS
