"""Job-definition routing for mamba / himem cells."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.plumbing
from tests.conftest import FLOAT_TOL

from mascotrl.aws_burst.job_routing import (
    JOB_DEFINITION_HIMEM,
    JOB_DEFINITION_HIMEM56,
    JOB_DEFINITION_STANDARD,
    cell_himem_memory_mib,
    cell_requires_himem,
    himem_job_definition_for_memory,
    partition_cells_by_jobdef,
    partition_cells_for_submit,
)

ROOT = Path(__file__).resolve().parents[1]


def test_mamba_k100_requires_himem() -> None:
    cell = "config/spectrum/cherrypick/eq_K100_single_ppo_mamba_softmax_mean_std_cao.yaml"
    assert cell_requires_himem(ROOT, cell) is True


def test_mamba_k25_probe_requires_himem_via_flag() -> None:
    cell = "config/spectrum/cherrypick_val/eq_K25_single_ppo_mamba_softmax_mean_std_cao.yaml"
    assert cell_requires_himem(ROOT, cell) is True


def test_partition_splits_mamba_k100_from_mlp() -> None:
    mamba = "config/spectrum/cherrypick/eq_K100_single_ppo_mamba_softmax_mean_std_cao.yaml"
    mlp = "config/spectrum/cherrypick/eq_K100_single_ppo_mlp_softmax_mean_std_cao.yaml"
    parts = partition_cells_by_jobdef(ROOT, [mlp, mamba])
    assert parts[JOB_DEFINITION_STANDARD] == [mlp]
    assert parts[JOB_DEFINITION_HIMEM] == [mamba]


def test_paid_example_shape_values() -> None:
    import json

    example = ROOT / "deploy/aws_burst/config/account_shape.paid.example.json"
    data = json.loads(example.read_text(encoding="utf-8"))
    shape = data["profiles"]["volsurf-burst-1"]
    assert shape["max_vcpus"] == 512
    assert shape["job_memory_mib"] == 12288
    assert shape["himem_job_memory_mib"] == 16384
    assert shape["credit_usd"] == pytest.approx(196.0, **FLOAT_TOL)


def test_cppo_records_proven_himem_memory() -> None:
    cell = (
        "config/spectrum/cherrypick/narrative/"
        "eq_K100_single_cppo_mlp_softmax_mean_std_cao.yaml"
    )
    assert cell_requires_himem(ROOT, cell) is True
    # Narrative CPPO is proven at himem56 under RC4 100k-step budgets.
    assert cell_himem_memory_mib(ROOT, cell) == 57344


def test_gru_records_proven_himem56_memory() -> None:
    cell = "config/spectrum/cherrypick/eq_K100_single_ppo_gru_softmax_mean_std_cao.yaml"
    assert cell_requires_himem(ROOT, cell) is True
    assert cell_himem_memory_mib(ROOT, cell) == 57344
    assert himem_job_definition_for_memory(57344) == JOB_DEFINITION_HIMEM56


def test_partition_for_submit_splits_himem_memory_tiers() -> None:
    mlp = "config/spectrum/cherrypick/eq_K100_single_ppo_mlp_softmax_mean_std_cao.yaml"
    featnet = (
        "config/spectrum/cherrypick_featnet/"
        "eq_K100_single_ppo_mlp_softmax_mean_std_cao_featnet.yaml"
    )
    gru = "config/spectrum/cherrypick/eq_K100_single_ppo_gru_softmax_mean_std_cao.yaml"
    parts = partition_cells_for_submit(ROOT, [mlp, featnet, gru])
    by_key = {(p.job_definition, p.memory_mib): p.cells for p in parts}
    assert by_key[(JOB_DEFINITION_STANDARD, None)] == [mlp]
    assert by_key[(JOB_DEFINITION_HIMEM, 28672)] == [featnet]
    assert by_key[(JOB_DEFINITION_HIMEM56, 57344)] == [gru]


def test_transformer_k100_requires_himem() -> None:
    cell = "config/spectrum/cherrypick/eq_K100_single_ppo_transformer_softmax_mean_std_cao.yaml"
    assert cell_requires_himem(ROOT, cell) is True
    assert cell_himem_memory_mib(ROOT, cell) == 57344


def test_featnet_mlp_requires_himem() -> None:
    cell = (
        "config/spectrum/cherrypick_featnet/"
        "eq_K100_single_ppo_mlp_softmax_mean_std_cao_featnet.yaml"
    )
    assert cell_requires_himem(ROOT, cell) is True
    assert cell_himem_memory_mib(ROOT, cell) == 28672


def test_feature_net_extras_flag_routes_himem(tmp_path: Path) -> None:
    """use_feature_net_extras alone must force himem (RC4 FEATNET OOM)."""
    from mascotrl.aws_burst.job_routing import estimate_peak_memory

    cfg = {
        "n_assets": 100,
        "architecture": "mlp",
        "train_env_steps": 100000,
        "use_feature_net_extras": True,
    }
    assert estimate_peak_memory(cfg) > 0
    cell = tmp_path / "featnet_probe.yaml"
    cell.write_text(
        "n_assets: 100\narchitecture: mlp\n"
        "train_env_steps: 100000\nuse_feature_net_extras: true\n"
    )
    assert cell_requires_himem(tmp_path, str(cell)) is True

