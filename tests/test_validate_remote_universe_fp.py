"""WP-R2: validate_remote_cell accepts universe fingerprint without crucible."""
from __future__ import annotations

from scripts.validate_remote_cell import validate_remote_cell


def _base(**kw):
    art = {
        "compute_host": "remote",
        "instance_type": "c6i.large",
        "container_digest": "sha256:abc",
        "requirements_lock_sha256": "deadbeef",
    }
    art.update(kw)
    return art


def test_universe_fingerprint_ok() -> None:
    art = _base(
        universe_fingerprint="aaa",
        universe_fingerprint_kind="dyn_hrp",
    )
    res = validate_remote_cell(art, expected_universe_fingerprint="aaa")
    assert res["ok"] is True


def test_missing_both_fingerprints_fails() -> None:
    res = validate_remote_cell(_base())
    assert res["ok"] is False


def test_crucible_mismatch_fails() -> None:
    art = _base(crucible_fingerprint="x")
    res = validate_remote_cell(art, expected_fingerprint="y")
    assert res["ok"] is False
