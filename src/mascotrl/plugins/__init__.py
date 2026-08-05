"""Config-gated plugin factories. Omit ``plugins:`` → status-quo defaults."""
from __future__ import annotations

from mascotrl.plugins.resolve import (
    STATUS_QUO_PLUGINS,
    dump_resolved_plugins,
    is_status_quo,
    resolve_plugins,
)

__all__ = [
    "STATUS_QUO_PLUGINS",
    "resolve_plugins",
    "dump_resolved_plugins",
    "is_status_quo",
]


def __getattr__(name: str):
    # Lazy re-exports to avoid circular imports with HAPPOEngine.
    if name in {
        "build_projection",
        "build_feature_extractor",
        "build_happo_engine",
        "build_tau_schedule",
        "build_funding",
        "env_drag_kwargs",
        "oos_friction_kwargs",
    }:
        from mascotrl.plugins import registry as _reg

        return getattr(_reg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
