"""Campaign surface_obs_lane resolution (dual-track gate)."""
from __future__ import annotations

from pathlib import Path

import pytest

from mascotrl.eval.signal_gate import assert_geometry_pack_valid, load_obs_pack


ROOT = Path(__file__).resolve().parents[1]


def test_obs_packs_on_disk():
    lite = assert_geometry_pack_valid(ROOT / "config/obs_packs/surf_geometry_lite.yaml")
    assert lite["channels"] == ["mfiv_30", "iv_term_slope", "iv_skew_30d"]
    cs = assert_geometry_pack_valid(ROOT / "config/obs_packs/surf_cs_admit.yaml")
    assert cs["resolve_from"] == "signal_allowlist"
    off = load_obs_pack(ROOT / "config/obs_packs/surf_off.yaml")
    assert off["channels"] == []
    with pytest.raises(ValueError, match="empty channels"):
        assert_geometry_pack_valid(ROOT / "config/obs_packs/surf_off.yaml")


def test_campaign_surface_obs_lane_default_is_geometry_lite():
    """Spine default matches docs: geometry_lite (cs_admit fail-closes on empty allowlist)."""

    from scripts import run_eq_alloc_campaign as camp

    src = Path(camp.__file__).read_text(encoding="utf-8")
    assert 'setdefault("surface_obs_lane", "geometry_lite")' in src
    assert 'setdefault("surface_obs_lane", "cs_admit")' not in src


def test_fingerprint_includes_surface_obs_lane():
    from scripts.run_eq_alloc_campaign import _campaign_config_fingerprint

    cfg_a = {
        "use_surface_signals": True,
        "surface_obs_lane": "cs_admit",
        "feature_extras": {"iv_surface": {"mfis_30": None}},
    }
    cfg_b = {
        "use_surface_signals": True,
        "surface_obs_lane": "geometry_lite",
        "_obs_pack_id": "surf_geometry_lite",
        "feature_extras": {
            "iv_surface": {
                "mfiv_30": None,
                "iv_term_slope": None,
                "iv_skew_30d": None,
            }
        },
    }
    fa = _campaign_config_fingerprint(cfg_a, realized_k=10)
    fb = _campaign_config_fingerprint(cfg_b, realized_k=10)
    assert fa != fb
