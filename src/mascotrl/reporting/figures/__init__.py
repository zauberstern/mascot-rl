"""Spectrum and figure suites."""

from __future__ import annotations

__all__ = ["render_spectrum_figures"]


def __getattr__(name: str):
    if name == "render_spectrum_figures":
        from mascotrl.reporting.figures.core_suite import render_spectrum_figures

        return render_spectrum_figures
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
