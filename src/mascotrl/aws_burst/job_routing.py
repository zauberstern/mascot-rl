"""Route burst cells to standard vs high-memory Batch job definitions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

JOB_DEFINITION_STANDARD = "volsurf-burst-cell"
JOB_DEFINITION_HIMEM = "volsurf-burst-cell-himem"
JOB_DEFINITION_HIMEM56 = "volsurf-burst-cell-himem56"
MAMBA_ARCHITECTURES = frozenset({"mamba", "mamba2"})
GRU_ARCHITECTURES = frozenset({"gru", "lstm"})
# Transformer was missing from auto-himem and OOMed on 12 GiB under RC4 100k steps.
TRANSFORMER_ARCHITECTURES = frozenset({"transformer"})
TEMPORAL_HIMEM_ARCHITECTURES = (
    MAMBA_ARCHITECTURES | GRU_ARCHITECTURES | TRANSFORMER_ARCHITECTURES
)
HIMEM_N_ASSETS_THRESHOLD = 100
STANDARD_MEMORY_MIB = 12288
# Proven fleet himem JD memory (burst-1/2/3 account_shape himem_job_memory_mib).
FLEET_HIMEM_MEMORY_MIB = 28672
# Proven one-off GRU recovery JD (logs/aws_burst/pick_gru56_submit.json).
HIMEM56_MEMORY_MIB = 57344
# Unproven r7i.4xlarge recovery for cells that already SIGKILL'd at 57344.
HIMEM112_MEMORY_MIB = 114688
# RC4 screening budget; large on-policy buffers dominate peak RSS.
RC4_TRAIN_ENV_STEPS = 100_000


@dataclass(frozen=True)
class CellJobPartition:
    """One Batch array submit: shared job definition + optional memory override."""

    job_definition: str
    cells: list[str]
    memory_mib: int | None = None


def estimate_peak_memory(cfg: Mapping[str, Any]) -> int:
    """Estimate peak memory in MiB from cell config (pre-submit OOM guard)."""
    k = int(cfg.get("n_assets") or cfg.get("K") or 100)
    arch = str(cfg.get("architecture") or "mlp").lower()
    base = 4096
    channels = 26
    if bool(cfg.get("use_feature_net_extras")):
        # Feature-net extras inflate the cube well past G0 Kx26.
        channels = 48
    # Feature cube rough upper bound: K * channels * days * 8 bytes
    obs_mem = k * channels * 252 * 8 / 1024 / 1024
    model_mem = {
        "mlp": 512,
        "gru": 2048,
        "lstm": 2048,
        "transformer": 3072,
        "mamba": 4096,
        "mamba2": 4096,
    }.get(arch, 512)
    # Rollout buffer dominates under RC4 (100k daily steps, obs_dim ~ K*C).
    steps = int(cfg.get("train_env_steps") or 0)
    if steps <= 0:
        steps = RC4_TRAIN_ENV_STEPS
    obs_dim = k * channels
    # obs + action + (rew, value, logp, done) float32
    buffer_mem = steps * (obs_dim * 4 + k * 4 + 16) / 1024 / 1024
    safety = 1.5
    return int((base + obs_mem + model_mem + buffer_mem) * safety)


def _load_cell_cfg(root: Path, cell_rel: str) -> Mapping[str, Any] | None:
    from src.spectrum.yaml_loader import load_cell_yaml

    path = Path(cell_rel)
    if not path.is_file():
        path = root / cell_rel
    if not path.is_file():
        return None
    return load_cell_yaml(path)


def cell_requires_himem(root: Path, cell_rel: str) -> bool:
    """True when a cell needs the himem job def (mamba/gru K>=100 or estimate)."""
    cfg = _load_cell_cfg(root, cell_rel)
    if cfg is None:
        return False
    if bool(cfg.get("requires_himem")):
        return True
    if cfg.get("himem_job_memory_mib") is not None:
        return True
    # Feature-net extras + RC4 100k rollout buffers OOMed on 12 GiB in fleet.
    if bool(cfg.get("use_feature_net_extras")):
        return True
    if estimate_peak_memory(cfg) > STANDARD_MEMORY_MIB:
        return True
    arch = str(cfg.get("architecture") or "").lower()
    n_assets = int(cfg.get("n_assets") or 0)
    if arch in TEMPORAL_HIMEM_ARCHITECTURES and n_assets >= HIMEM_N_ASSETS_THRESHOLD:
        return True
    return False


def cell_himem_memory_mib(root: Path, cell_rel: str) -> int | None:
    """Proven himem MiB recorded on the cell YAML, else None (use profile JD default)."""
    cfg = _load_cell_cfg(root, cell_rel)
    if cfg is None:
        return None
    raw = cfg.get("himem_job_memory_mib")
    if raw is None:
        return None
    mem = int(raw)
    if mem <= 0:
        raise ValueError(f"himem_job_memory_mib_invalid: cell={cell_rel} mem={mem}")
    return mem


def himem_job_definition_for_memory(memory_mib: int) -> str:
    """Map a proven himem MiB request to the Batch job-definition name.

    114688 MiB (himem112) reuses the himem56 JD with a container memory override
    so Batch can place on r7i.4xlarge (~128 GiB). No separate CFN JD.
    """
    mem = int(memory_mib)
    if mem <= FLEET_HIMEM_MEMORY_MIB:
        return JOB_DEFINITION_HIMEM
    if mem <= HIMEM112_MEMORY_MIB:
        return JOB_DEFINITION_HIMEM56
    raise ValueError(
        f"himem_memory_unsupported: mem={mem}MiB exceeds himem112={HIMEM112_MEMORY_MIB}MiB"
    )


def partition_cells_by_jobdef(root: Path, cells: list[str]) -> dict[str, list[str]]:
    """Split cells into standard vs himem job-definition buckets."""
    standard: list[str] = []
    himem: list[str] = []
    for cell_rel in cells:
        if cell_requires_himem(root, cell_rel):
            himem.append(cell_rel)
        else:
            standard.append(cell_rel)
    out: dict[str, list[str]] = {}
    if standard:
        out[JOB_DEFINITION_STANDARD] = standard
    if himem:
        out[JOB_DEFINITION_HIMEM] = himem
    return out


def partition_cells_for_submit(root: Path, cells: list[str]) -> list[CellJobPartition]:
    """Split cells for submit: standard vs himem, further by recorded memory tier."""
    standard: list[str] = []
    himem_by_mem: dict[int | None, list[str]] = {}
    for cell_rel in cells:
        if not cell_requires_himem(root, cell_rel):
            standard.append(cell_rel)
            continue
        mem = cell_himem_memory_mib(root, cell_rel)
        himem_by_mem.setdefault(mem, []).append(cell_rel)

    parts: list[CellJobPartition] = []
    if standard:
        parts.append(
            CellJobPartition(
                job_definition=JOB_DEFINITION_STANDARD,
                cells=standard,
                memory_mib=None,
            )
        )
    # Stable order: None (JD default) first, then ascending recorded MiB.
    for mem in sorted(himem_by_mem.keys(), key=lambda m: (m is not None, m or 0)):
        job_def = (
            JOB_DEFINITION_HIMEM
            if mem is None
            else himem_job_definition_for_memory(mem)
        )
        parts.append(
            CellJobPartition(
                job_definition=job_def,
                cells=himem_by_mem[mem],
                memory_mib=mem,
            )
        )
    return parts
