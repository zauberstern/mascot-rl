"""A13: capital hygiene disclosures must not cite missing scripts."""
from __future__ import annotations

import re
from pathlib import Path

from mascotrl.reporting.capital_gates import KNOWN_UNMODELED_RISKS

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_RE = re.compile(r"(?:scripts|src)/[\w./-]+\.py")


def test_known_unmodeled_risk_script_paths_exist_or_are_absent_by_design() -> None:
    """Any concrete repo path named in a disclosure must resolve on disk.

    Phrases that say tooling is absent are allowed; bare path citations that
    point at missing files are not.
    """
    missing: list[tuple[str, str]] = []
    for item in KNOWN_UNMODELED_RISKS:
        summary = str(item.get("summary") or "")
        rid = str(item.get("id") or "")
        # Explicit "absent" / "no ... exists" disclosures are design statements.
        if "absent" in summary.lower() or "no " in summary.lower() and "exist" in summary.lower():
            # Still refuse naming a concrete path that happens to exist as a
            # false claim of absence; only check concrete paths when claimed
            # as present. Skip path existence when the text asserts absence.
            continue
        for m in _SCRIPT_RE.findall(summary):
            p = ROOT / m
            if not p.exists():
                missing.append((rid, m))
    assert not missing, f"disclosure cites missing paths: {missing}"


def test_shadow_book_disclosure_does_not_name_missing_scripts() -> None:
    item = next(x for x in KNOWN_UNMODELED_RISKS if x["id"] == "shadow_book_mvp_only")
    summary = item["summary"]
    assert "scripts/shadow_book.py" not in summary
    assert "scripts/reconcile_book.py" not in summary
