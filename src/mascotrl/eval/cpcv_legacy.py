"""Frozen alias for the hand-rolled CPCV module (pre-purgedcv path).

Internal / rollback only. Production default is purgedcv via
:func:`src.eval.cpcv_backend.resolve_use_purgedcv` when the package imports.
Set ``use_purgedcv: false`` to force this legacy path.
"""
from src.eval.cpcv import *  # noqa: F403
