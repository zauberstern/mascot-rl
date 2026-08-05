"""Universe fingerprint helpers (panel bundle sha256 continuity)."""
from __future__ import annotations

from mascotrl._root import REPO_ROOT

ROOT = REPO_ROOT

EQ_BURST_WAVES = frozenset({"PICK", "PICK2", "K200", "PICK_SMOKE", "PICK_CANARY", "VAL"})


def read_panel_bundle_sha256(repo_root: Path | str | None = None) -> str | None:
    """Return ``logs/aws_burst_panel_bundle/panel_bundle.sha256`` when present."""
    root = Path(repo_root) if repo_root is not None else ROOT
    sha_path = root / "logs/aws_burst_panel_bundle/panel_bundle.sha256"
    if not sha_path.is_file():
        return None
    text = sha_path.read_text(encoding="utf-8").strip()
    return text or None
