"""Regression coverage for the protected headline CPCV summary."""

from __future__ import annotations

import json

import pytest


def test_smaller_k_cannot_overwrite_headline_summary(tmp_path) -> None:
    from scripts.run_eq_alloc_campaign import _assert_safe_to_write_summary

    (tmp_path / "cpcv_path_summary.json").write_text(
        json.dumps({"k": 100}), encoding="utf-8"
    )

    with pytest.raises(SystemExit):
        _assert_safe_to_write_summary(tmp_path, k=8, force_overwrite=False)

    _assert_safe_to_write_summary(tmp_path, k=8, force_overwrite=True)
    _assert_safe_to_write_summary(tmp_path, k=100, force_overwrite=False)
    _assert_safe_to_write_summary(tmp_path, k=200, force_overwrite=False)
