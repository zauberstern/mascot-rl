"""Regression: fold_runner must not shadow pathlib.Path with a local import."""
from __future__ import annotations

import ast
from pathlib import Path


def test_research_alpha_cpcv_fold_runner_does_not_shadow_path() -> None:
    """Local ``from pathlib import Path`` inside fold_runner made Path unbound
    when ``_learning_curves_dir`` was unset but ``_checkpoint_dir`` was set —
    every CPCV fold then failed with UnboundLocalError after training.
    """
    src = Path("src/eval/research_alpha_cpcv.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Find nested function fold_runner and assert it has no local Path import.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fold_runner":
            for child in ast.walk(node):
                if isinstance(child, ast.ImportFrom) and child.module == "pathlib":
                    names = [a.name for a in child.names]
                    assert "Path" not in names, (
                        "fold_runner must not locally import Path; "
                        "it shadows the module-level import and breaks "
                        "prune_fold_checkpoints when curves_dir is unset"
                    )
            break
    else:
        # fold_runner is nested; search Assign/FunctionDef recursively by source
        assert "def fold_runner" in src
        # Fallback: ensure the offending pattern is gone from the file region.
        assert "if curves_dir:\n            from pathlib import Path" not in src
