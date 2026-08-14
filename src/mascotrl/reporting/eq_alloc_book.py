"""D3/D4: the equity allocation reporting book.

Renders the ten sections specified in Workstream D (provenance, headline,
risk, holdings, cost and capacity, attribution, signals, learning,
statistical rigor, spectrum, limitations) from the artifacts the eq
allocation campaign (``scripts/run_eq_alloc_campaign.py``) already produces:
the ``results`` dict it writes to ``cpcv_path_summary.json`` plus the
per-strategy parquet frames from ``src/reporting/strategy_persistence.py``.

Every figure degrades to a ``"skipped"`` manifest entry with a ``note``
rather than fabricating data when its inputs are absent (the same contract
``src/reporting/figures/core_suite.py`` already uses); the one hard failure
is missing weights entirely, since without at least one strategy's holdings
there is no book to render.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from mascotrl.reporting.book_style import PdfBook, build_manifest, section_divider, stamp_footer

SECTION_TITLES: tuple[str, ...] = (
    "Provenance",
    "Headline",
    "Risk",
    "Holdings",
    "Cost and Capacity",
    "Attribution",
    "Signals",
    "Learning",
    "Statistical Rigor",
    "Spectrum",
    "Limitations",
)


from mascotrl.reporting.eq_alloc_book_primitives import _finish, _focus_frame, _weight_cols
from mascotrl.reporting.eq_alloc_book_sections_early import (
    _section0_provenance,
    _section1_headline,
    _section2_risk,
    _section3_holdings,
)
from mascotrl.reporting.eq_alloc_book_sections_mid import (
    _section4_cost_capacity,
    _section5_attribution,
)
from mascotrl.reporting.eq_alloc_book_sections_late import (
    _section10_limitations,
    _section6_signals,
    _section7_learning,
    _section8_stat_rigor,
    _section9_spectrum,
)

def render_eq_alloc_book(
    *,
    strategy_frames: Mapping[str, pd.DataFrame],
    out_dir: str | Path,
    results: Mapping[str, Any] | None = None,
    focus_strategy: str = "policy",
    cfg: Mapping[str, Any] | None = None,
    scorecard: str = "total_net",
    signal_gate_result: Mapping[str, Any] | None = None,
    signal_panels: Mapping[str, np.ndarray] | None = None,
    iv_surface_grid: np.ndarray | None = None,
    ff4_factors: np.ndarray | None = None,
    industry_group: Mapping[str, str] | None = None,
    returns_panel: np.ndarray | None = None,
    returns_panel_secids: Sequence[str] | None = None,
    learning_curves: Mapping[str, Sequence[float]] | None = None,
    known_limitations: Sequence[str] | None = None,
    arms_root: str | Path | None = None,
    artifacts_flat: str | Path | None = None,
) -> dict[str, Any]:
    """Render the full ten-section book; fail closed if no weights exist."""
    non_empty = {k: v for k, v in strategy_frames.items() if v is not None and not v.empty}
    has_weights = any(_weight_cols(v) for v in non_empty.values())
    if not non_empty or not has_weights:
        raise ValueError(
            "render_eq_alloc_book: no strategy carries any weight columns "
            "(w_<secid>); refusing to render a book with no holdings evidence"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = dict(results or {})
    conf = results.get("confirmatory") or {}

    focus_name, focus_df = _focus_frame(non_empty, focus_strategy)
    date_start = date_end = None
    if focus_df is not None and "date" in focus_df.columns and len(focus_df):
        dts = pd.to_datetime(focus_df["date"])
        date_start, date_end = str(dts.min().date()), str(dts.max().date())

    manifest = build_manifest(
        cfg=cfg,
        estimand_hash=conf.get("estimand_hash"),
        scorecard=scorecard,
        date_start=date_start,
        date_end=date_end,
    )

    run_meta = {
        "focus_strategy": focus_name,
        "n_strategies": len(non_empty),
        "k": results.get("k"),
        "pool": results.get("pool"),
        "date_start": date_start,
        "date_end": date_end,
        "wall_total_s": results.get("wall_total_s"),
    }

    arms_root_p = Path(arms_root) if arms_root is not None else None
    artifacts_flat_p = Path(artifacts_flat) if artifacts_flat is not None else None

    entries: list[dict[str, Any]] = []
    with PdfBook(out_dir / "book.pdf") as book:
        for idx, title in enumerate(SECTION_TITLES):
            fig = section_divider(f"Section {idx}: {title}")
            _finish(fig, stem=out_dir / f"divider_S{idx}", manifest=manifest, book=book)

            if idx == 0:
                entries += _section0_provenance(out_dir=out_dir, manifest=manifest, book=book, run_meta=run_meta, results=results)
            elif idx == 1:
                entries += _section1_headline(out_dir=out_dir, manifest=manifest, book=book, results=results, strategy_frames=non_empty, focus=focus_strategy)
            elif idx == 2:
                entries += _section2_risk(out_dir=out_dir, manifest=manifest, book=book, strategy_frames=non_empty, focus=focus_strategy)
            elif idx == 3:
                entries += _section3_holdings(out_dir=out_dir, manifest=manifest, book=book, strategy_frames=non_empty, focus=focus_strategy)
            elif idx == 4:
                entries += _section4_cost_capacity(out_dir=out_dir, manifest=manifest, book=book, results=results, strategy_frames=non_empty, focus=focus_strategy)
            elif idx == 5:
                entries += _section5_attribution(
                    out_dir=out_dir, manifest=manifest, book=book, strategy_frames=non_empty, focus=focus_strategy,
                    ff4_factors=ff4_factors, industry_group=industry_group, returns_panel=returns_panel,
                    secids=returns_panel_secids,
                )
            elif idx == 6:
                entries += _section6_signals(
                    out_dir=out_dir, manifest=manifest, book=book, results=results,
                    signal_gate_result=signal_gate_result, signal_panels=signal_panels,
                    iv_surface_grid=iv_surface_grid,
                )
            elif idx == 7:
                entries += _section7_learning(out_dir=out_dir, manifest=manifest, book=book, results=results, learning_curves=learning_curves)
            elif idx == 8:
                entries += _section8_stat_rigor(out_dir=out_dir, manifest=manifest, book=book, results=results)
            elif idx == 9:
                entries += _section9_spectrum(out_dir=out_dir, manifest=manifest, book=book, results=results, arms_root=arms_root_p, artifacts_flat=artifacts_flat_p)
            elif idx == 10:
                entries += _section10_limitations(out_dir=out_dir, manifest=manifest, book=book, results=results, known_limitations=known_limitations)

        n_pages = book.n_pages

    n_written = sum(1 for e in entries if e.get("status") == "written")
    n_skipped = sum(1 for e in entries if e.get("status") == "skipped")
    # Write BOOK.md before hashing so the sha256 map covers every on-disk
    # artifact (index.json itself is excluded as the digest container).
    md_lines = [
        "# Equity Allocation Reporting Book",
        "",
        f"Date range: {date_start} .. {date_end}",
        f"Focus strategy: {focus_name}",
        f"Figures written: {n_written} / {len(entries)} (skipped: {n_skipped})",
        "",
    ]
    for idx, title in enumerate(SECTION_TITLES):
        md_lines.append(f"## Section {idx}: {title}")
        md_lines.append("")
        for e in entries:
            if str(e.get("id", "")).startswith(f"S{idx}."):
                status = e.get("status")
                note = f" -- {e['note']}" if e.get("note") else ""
                md_lines.append(f"- `{e['id']}` {e['title']} [{status}]{note}")
        md_lines.append("")
    (out_dir / "BOOK.md").write_text("\n".join(md_lines))

    # W5: loud list of skipped figures so a silent empty section is obvious.
    skipped_lines = [
        "# Skipped figures",
        "",
        "Each entry names the figure id and the reason (missing input key).",
        "",
    ]
    for e in entries:
        if e.get("status") == "skipped":
            skipped_lines.append(
                f"- `{e.get('id')}` {e.get('title')}: {e.get('note') or 'missing input'}"
            )
    if n_skipped == 0:
        skipped_lines.append("_None skipped._")
    (out_dir / "SKIPPED.md").write_text("\n".join(skipped_lines) + "\n")

    index: dict[str, Any] = {
        "manifest": manifest,
        "n_pages": n_pages,
        "n_figures": len(entries),
        "n_written": n_written,
        "n_skipped": n_skipped,
        "figures": entries,
        "sha256": {},
    }
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "index.json":
            index["sha256"][str(p.relative_to(out_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, default=str, sort_keys=True))

    return index
