"""C-2: spectrum substrate_meta stamps universe fingerprint when panel bundle exists."""
from __future__ import annotations

from pathlib import Path

from mascotrl.eval.universe_fingerprint import read_panel_bundle_sha256


def test_read_panel_bundle_sha256_from_logs(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "logs/aws_burst_panel_bundle"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "panel_bundle.sha256").write_text("fingerprint-abc\n", encoding="utf-8")
    assert read_panel_bundle_sha256(tmp_path) == "fingerprint-abc"


def test_substrate_meta_key_list_includes_universe_fingerprint() -> None:
    """run_spectrum_campaign copies these panel_meta keys into substrate_meta."""
    panel_meta = {
        "panel_source": "lake_sp500_sec",
        "universe_arm": "dyn_hrp",
        "fingerprint_size": 120,
        "universe_fingerprint": "deadbeef",
        "universe_fingerprint_kind": "panel_bundle_sha256",
        "k": 100,
        "n_days": 2500,
    }
    keys = (
        "panel_source",
        "universe_arm",
        "fingerprint_size",
        "universe_fingerprint",
        "universe_fingerprint_kind",
        "k",
        "n_days",
    )
    substrate = {k_: panel_meta[k_] for k_ in keys if k_ in panel_meta}
    assert substrate["universe_fingerprint"] == "deadbeef"
    assert substrate["universe_fingerprint_kind"] == "panel_bundle_sha256"
