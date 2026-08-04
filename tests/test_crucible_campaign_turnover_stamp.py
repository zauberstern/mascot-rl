"""CRUCIBLE campaign must stamp selection_turnover without float(dict)."""
from __future__ import annotations

import ast
from pathlib import Path


def test_crucible_info_does_not_float_selection_turnover_dict():
    src = Path("scripts/run_eq_alloc_campaign.py").read_text(encoding="utf-8")
    # Guard the known TypeError: float(selection_turnover(slots_rows))
    assert "float(selection_turnover(slots_rows))" not in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "float":
            if not node.args:
                continue
            arg = node.args[0]
            if (
                isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Name)
                and arg.func.id == "selection_turnover"
            ):
                raise AssertionError(
                    "float(selection_turnover(...)) will TypeError; "
                    "stamp the dict or a scalar field like mean_added"
                )
