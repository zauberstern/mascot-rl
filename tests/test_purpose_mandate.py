"""Purpose mandate locks: allocation-only; no hedge_mdp product pages."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENWIKI = ROOT / "openwiki"

PURPOSE_NEEDLE = "portfolio allocation performance"


def _read(name: str) -> str:
    return (OPENWIKI / name).read_text(encoding="utf-8")


def test_purpose_in_manifest_quickstart_instructions() -> None:
    for name in (
        "00_MASCOTRL_MANIFEST.md",
        "quickstart.md",
        "INSTRUCTIONS.md",
    ):
        text = _read(name)
        assert PURPOSE_NEEDLE in text, f"missing purpose in {name}"


def test_information_geometry_retired_after_crucible() -> None:
    """DII/TE openwiki page is deleted; CRUCIBLE / Kahn breadth replace it."""
    assert not (OPENWIKI / "07_INFORMATION_GEOMETRY.md").exists()
    quick = _read("quickstart.md")
    assert "07_INFORMATION_GEOMETRY" not in quick
    assert (ROOT / "src" / "data" / "crucible.py").is_file()
    assert (ROOT / "src" / "eval" / "kahn_breadth.py").is_file()
    assert not (ROOT / "src" / "infotheory").exists()


def test_openwiki_has_no_active_hedge_mdp_product() -> None:
    """Historical notes may mention sealed nulls; active claim paths must not."""
    banned_active = (
        "arm: hedge_mdp",
        "CLAIM_CATEGORY_DEEP_HEDGE",
        "parallel hedge-MDP arm",
        "Capital-alpha evidence lead` is the parallel",
    )
    for path in OPENWIKI.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        # Allow archival / historical wording in 07 or sealed-null context
        if "sealed null" in text.lower() or "out of scope" in text.lower():
            # still forbid live YAML arm declaration
            assert "arm: hedge_mdp" not in text, path.name
            continue
        for ban in banned_active:
            assert ban not in text, f"{path.name} contains banned active string {ban!r}"
