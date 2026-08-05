"""Wave definitions and cell discovery for AWS Batch burst submits."""
from __future__ import annotations

import glob as _glob
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WaveSpec:
    name: str
    glob: str | None
    config_dir: str
    n_cap: int | None
    out_subdir: str
    local_equivalent: str | None = None
    script: str | None = None
    script_args: tuple[str, ...] = ()


WAVES: dict[str, WaveSpec] = {
    "H0": WaveSpec(
        name="H0",
        glob=None,
        config_dir="",
        n_cap=None,
        out_subdir="headline",
        script="scripts/validate_headline_packs.py",
        script_args=("--run-surf-off",),
    ),
    "CAL": WaveSpec(
        name="CAL",
        glob="config/spectrum/fullgrid/*_K100_single_ppo_mlp_*.yaml",
        config_dir="config/spectrum/fullgrid",
        n_cap=12,
        out_subdir="fullgrid",
    ),
    "E100": WaveSpec(
        name="E100",
        glob="config/spectrum/fullgrid/*_K100_*.yaml",
        config_dir="config/spectrum/fullgrid",
        n_cap=None,
        out_subdir="fullgrid",
    ),
    "E200": WaveSpec(
        name="E200",
        glob="config/spectrum/fullgrid/*_K200_*.yaml",
        config_dir="config/spectrum/fullgrid",
        n_cap=None,
        out_subdir="fullgrid",
    ),
    "T3": WaveSpec(
        name="T3",
        glob="config/spectrum/fullgrid/*_K100_single_ppo_*.yaml",
        config_dir="config/spectrum/fullgrid",
        n_cap=None,
        out_subdir="fullgrid",
    ),
    "EKmax": WaveSpec(
        name="EKmax",
        glob="config/spectrum/fullgrid/*_K*_*.yaml",
        config_dir="config/spectrum/fullgrid",
        n_cap=None,
        out_subdir="fullgrid",
    ),
    "PICK": WaveSpec(
        name="PICK",
        glob="config/spectrum/cherrypick/*.yaml",
        config_dir="config/spectrum/cherrypick",
        n_cap=None,
        out_subdir="cherrypick",
    ),
    "PICK2": WaveSpec(
        name="PICK2",
        glob="config/spectrum/cherrypick/narrative/*.yaml",
        config_dir="config/spectrum/cherrypick/narrative",
        n_cap=None,
        out_subdir="cherrypick/narrative",
    ),
    "K200": WaveSpec(
        name="K200",
        glob="config/spectrum/cherrypick/k200/*.yaml",
        config_dir="config/spectrum/cherrypick/k200",
        n_cap=None,
        out_subdir="cherrypick/k200",
    ),
    "VAL": WaveSpec(
        name="VAL",
        glob="config/spectrum/cherrypick_val/*.yaml",
        config_dir="config/spectrum/cherrypick_val",
        n_cap=None,
        out_subdir="cherrypick_val",
    ),
    "PICK_SMOKE": WaveSpec(
        name="PICK_SMOKE",
        glob="config/spectrum/cherrypick_smoke/*.yaml",
        config_dir="config/spectrum/cherrypick_smoke",
        n_cap=1,
        out_subdir="cherrypick_smoke",
    ),
    "PICK_CANARY": WaveSpec(
        name="PICK_CANARY",
        glob="config/spectrum/cherrypick/*.yaml",
        config_dir="config/spectrum/cherrypick",
        n_cap=10,
        out_subdir="cherrypick",
    ),
    "FEATNET": WaveSpec(
        name="FEATNET",
        glob="config/spectrum/cherrypick_featnet/*.yaml",
        config_dir="config/spectrum/cherrypick_featnet",
        n_cap=None,
        out_subdir="cherrypick_featnet",
    ),
    "HYBRID": WaveSpec(
        name="HYBRID",
        glob="config/spectrum/cherrypick_hybrid/*.yaml",
        config_dir="config/spectrum/cherrypick_hybrid",
        n_cap=None,
        out_subdir="cherrypick_hybrid",
    ),
    "DESKORG": WaveSpec(
        name="DESKORG",
        glob="config/spectrum/cherrypick_deskorg/*.yaml",
        config_dir="config/spectrum/cherrypick_deskorg",
        n_cap=None,
        out_subdir="cherrypick_deskorg",
    ),
    "REGIME": WaveSpec(
        name="REGIME",
        glob="config/spectrum/cherrypick_regime/*.yaml",
        config_dir="config/spectrum/cherrypick_regime",
        n_cap=None,
        out_subdir="cherrypick_regime",
    ),
    "RC6": WaveSpec(
        name="RC6",
        glob="config/spectrum/cherrypick/rc6/*.yaml",
        config_dir="config/spectrum/cherrypick/rc6",
        n_cap=None,
        out_subdir="cherrypick/rc6",
    ),
    "RC6_CANARY": WaveSpec(
        name="RC6_CANARY",
        glob="config/spectrum/cherrypick/rc6_canary/*.yaml",
        config_dir="config/spectrum/cherrypick/rc6_canary",
        n_cap=10,
        out_subdir="cherrypick/rc6_canary",
    ),
    "RC6_NARRATIVE": WaveSpec(
        name="RC6_NARRATIVE",
        glob="config/spectrum/cherrypick/rc6_narrative/*.yaml",
        config_dir="config/spectrum/cherrypick/rc6_narrative",
        n_cap=None,
        out_subdir="cherrypick/rc6_narrative",
    ),
    "RC6_K200": WaveSpec(
        name="RC6_K200",
        glob="config/spectrum/cherrypick/rc6_k200/*.yaml",
        config_dir="config/spectrum/cherrypick/rc6_k200",
        n_cap=None,
        out_subdir="cherrypick/rc6_k200",
    ),
    "RC6_HEADS": WaveSpec(
        name="RC6_HEADS",
        glob="config/spectrum/cherrypick/rc6_heads/*.yaml",
        config_dir="config/spectrum/cherrypick/rc6_heads",
        n_cap=None,
        out_subdir="cherrypick/rc6_heads",
    ),
    "RC6_HAPPO": WaveSpec(
        name="RC6_HAPPO",
        glob="config/spectrum/cherrypick/rc6_happo_full/*.yaml",
        config_dir="config/spectrum/cherrypick/rc6_happo_full",
        n_cap=None,
        out_subdir="cherrypick/rc6_happo_full",
    ),
    "RC6_NARRATIVE_REVIVE": WaveSpec(
        name="RC6_NARRATIVE_REVIVE",
        glob="config/spectrum/cherrypick/rc6_narrative_revive/*.yaml",
        config_dir="config/spectrum/cherrypick/rc6_narrative_revive",
        n_cap=None,
        out_subdir="cherrypick/rc6_narrative",
    ),
}


def _local_equivalent_for(wave: str) -> str:
    if wave in {
        "PICK",
        "PICK2",
        "PICK_SMOKE",
        "PICK_CANARY",
        "K200",
        "VAL",
        "FEATNET",
        "HYBRID",
        "DESKORG",
        "REGIME",
        "RC6",
        "RC6_CANARY",
        "RC6_NARRATIVE",
        "RC6_K200",
        "RC6_HEADS",
        "RC6_HAPPO",
    }:
        pick_dir = {
            "PICK2": "config/spectrum/cherrypick/narrative",
            "PICK_SMOKE": "config/spectrum/cherrypick_smoke",
            "PICK_CANARY": "config/spectrum/cherrypick",
            "PICK": "config/spectrum/cherrypick",
            "K200": "config/spectrum/cherrypick/k200",
            "VAL": "config/spectrum/cherrypick_val",
            "FEATNET": "config/spectrum/cherrypick_featnet",
            "HYBRID": "config/spectrum/cherrypick_hybrid",
            "DESKORG": "config/spectrum/cherrypick_deskorg",
            "REGIME": "config/spectrum/cherrypick_regime",
            "RC6": "config/spectrum/cherrypick/rc6",
            "RC6_CANARY": "config/spectrum/cherrypick/rc6_canary",
            "RC6_NARRATIVE": "config/spectrum/cherrypick/rc6_narrative",
            "RC6_K200": "config/spectrum/cherrypick/rc6_k200",
            "RC6_HEADS": "config/spectrum/cherrypick/rc6_heads",
            "RC6_HAPPO": "config/spectrum/cherrypick/rc6_happo_full",
        }[wave]
        return (
            f".venv/bin/python scripts/run_spectrum_campaign.py "
            f"--config-dir {pick_dir} --no-dry-run"
        )
    if wave.startswith("E") or wave in {"CAL", "T3"}:
        return (
            ".venv/bin/python scripts/run_spectrum_campaign.py "
            "--config-dir config/spectrum/fullgrid --no-dry-run"
        )
    spec = WAVES[wave]
    if spec.script:
        args = " ".join(spec.script_args)
        return f".venv/bin/python {spec.script} {args}".rstrip()
    return ".venv/bin/python scripts/run_spectrum_campaign.py"


def _cherrypick_final_served_stems(root: Path, wave: str) -> set[str] | None:
    """Served stems from cherrypick_final/manifest.json for production waves.

    On-disk globs under config/spectrum/cherrypick{,/narrative,/k200} still
    contain deferred mix/opt and dropped mamba YAMLs. Batch discovery must
    submit only the eq-only stems the final manifest sealed, else deferred
    cells burn capacity and fail with cell_yaml_missing inside the image.
    Returns None when the manifest is absent or the wave has no served list
    (caller keeps the raw glob).
    """
    if wave not in {"PICK", "PICK2", "K200", "PICK_CANARY"}:
        return None
    path = root / "config" / "spectrum" / "cherrypick_final" / "manifest.json"
    if not path.is_file():
        return None
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if wave == "PICK" or wave == "PICK_CANARY":
        cells = data.get("cells") or []
    elif wave == "PICK2":
        cells = (data.get("pick2") or {}).get("cells") or []
    else:
        cells = (data.get("k200") or {}).get("cells") or []
    if not isinstance(cells, list) or not cells:
        return None
    dropped: set[str] = set()
    for item in data.get("dropped_cells") or []:
        if isinstance(item, dict):
            stem = str(item.get("stem") or "").strip()
        else:
            stem = str(item).strip()
        if stem:
            dropped.add(stem)
    return {str(c).strip() for c in cells if str(c).strip()} - dropped


def discover_wave_cells(root: Path, wave: str) -> list[str]:
    """Return sorted relative paths to cell YAMLs for a wave."""
    if wave not in WAVES:
        raise ValueError(f"unknown wave {wave!r}; allowed={list(WAVES)}")
    spec = WAVES[wave]
    if not spec.glob:
        return []
    cells = sorted(_glob.glob(str(root / spec.glob)))
    served = _cherrypick_final_served_stems(root, wave)
    if served is not None:
        filtered = [c for c in cells if Path(c).stem in served]
        # Preserve manifest order when possible; fall back to path sort.
        by_stem = {Path(c).stem: c for c in filtered}
        ordered = [by_stem[s] for s in sorted(served) if s in by_stem]
        cells = ordered
    if spec.n_cap is not None:
        cells = cells[: int(spec.n_cap)]
    return [str(Path(c).relative_to(root)) for c in cells]


def shard_cells(
    cells: list[str],
    n_shards: int,
    *,
    weights: list[float] | None = None,
) -> list[list[str]]:
    """Shard cells across n_shards.

    Default: equal round-robin. When ``weights`` is provided (e.g. MaxvCpus per
    profile), assign each cell to the shard with the lowest load/weight ratio so
    larger accounts receive proportionally more cells.
    """
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    if not cells:
        return shards

    if weights is None:
        for i, cell in enumerate(cells):
            shards[i % n_shards].append(cell)
        return shards

    if len(weights) != n_shards:
        raise ValueError(
            f"weights length {len(weights)} must equal n_shards={n_shards}"
        )
    w = [float(x) for x in weights]
    if any(x <= 0.0 for x in w):
        raise ValueError("weights must be strictly positive")

    loads = [0.0] * n_shards
    for cell in cells:
        # Prefer the under-filled shard relative to capacity (deterministic).
        i = min(range(n_shards), key=lambda j: (loads[j] / w[j], j))
        shards[i].append(cell)
        loads[i] += 1.0
    return shards
