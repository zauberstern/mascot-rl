"""Strict YAML loader for spectrum cell configs (preserves scr_mix: off as string)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class StrictBoolLoader(yaml.SafeLoader):
    """SafeLoader without YAML 1.1 on/off/yes/no bool coercion.

    Fullgrid cells store ``scr_mix: off`` as the string enum value. PyYAML's
  default SafeLoader treats bare ``off``/``on`` as booleans.
    """


StrictBoolLoader.yaml_implicit_resolvers = {
    first_char: [
        r for r in resolvers
        if not (r[0] == "tag:yaml.org,2002:bool" and first_char in "yYnNoO")
    ]
    for first_char, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def load_cell_yaml(path: Path | str) -> dict[str, Any]:
    """Load a spectrum cell YAML with strict bool handling."""
    p = Path(path)
    return yaml.load(p.read_text(encoding="utf-8"), Loader=StrictBoolLoader) or {}


def load_cell_yaml_text(text: str) -> dict[str, Any]:
    """Load cell config from YAML text (strict bool handling)."""
    return yaml.load(text, Loader=StrictBoolLoader) or {}
