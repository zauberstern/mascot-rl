"""Wave discovery and sharding tests."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.plumbing

from src.aws_burst.waves import WAVES, discover_wave_cells, shard_cells


ROOT = Path(__file__).resolve().parents[1]


def test_pick_wave_cell_count() -> None:
    cells = discover_wave_cells(ROOT, "PICK")
    # 54 minus free-tier quarantine: tw-hybrid_pretrain_finetune (NaN fail-closed).
    assert len(cells) == 53
    assert all(Path(c).name.startswith("eq_") for c in cells)
    assert not any("mamba" in Path(c).name for c in cells)
    assert not any("hybrid_pretrain_finetune" in Path(c).name for c in cells)


def test_pick2_wave_cell_count() -> None:
    cells = discover_wave_cells(ROOT, "PICK2")
    assert len(cells) == 6
    assert all(Path(c).name.startswith("eq_") for c in cells)


def test_k200_wave_cell_count() -> None:
    cells = discover_wave_cells(ROOT, "K200")
    # 36 minus free-tier quarantine: K200 gru/lstm/transformer (OOM on ~8 GiB).
    assert len(cells) == 33
    assert all(Path(c).name.startswith("eq_") for c in cells)
    assert not any(
        any(tag in Path(c).name for tag in ("_gru_", "_lstm_", "_transformer_"))
        for c in cells
    )


def test_pick_smoke_cap() -> None:
    cells = discover_wave_cells(ROOT, "PICK_SMOKE")
    assert len(cells) == 1


def test_pick_canary_cap() -> None:
    cells = discover_wave_cells(ROOT, "PICK_CANARY")
    assert len(cells) == 10
    pick = discover_wave_cells(ROOT, "PICK")
    assert cells == pick[:10]


def test_shard_round_robin() -> None:
    cells = ["a", "b", "c", "d", "e"]
    shards = shard_cells(cells, 3)
    assert shards == [["a", "d"], ["b", "e"], ["c"]]


def test_shard_weighted_by_capacity() -> None:
    cells = [f"c{i}" for i in range(10)]
    # 256:256:32:32 -> roughly 44%/44%/6%/6%
    shards = shard_cells(cells, 4, weights=[256.0, 256.0, 32.0, 32.0])
    assert sum(len(s) for s in shards) == 10
    assert len(shards[0]) >= len(shards[2])
    assert len(shards[1]) >= len(shards[3])
    assert len(shards[0]) + len(shards[1]) >= 8


def test_shard_weighted_rejects_bad_weights() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        shard_cells(["a"], 2, weights=[1.0, 0.0])
    with pytest.raises(ValueError, match="weights length"):
        shard_cells(["a"], 2, weights=[1.0])


def test_unknown_wave_raises() -> None:
    with pytest.raises(ValueError, match="unknown wave"):
        discover_wave_cells(ROOT, "NOPE")


def test_h0_is_local_only_submit_refused(tmp_path: Path) -> None:
    import scripts.aws_submit_wave as submit_mod

    cfg = tmp_path / "deploy" / "aws_burst" / "config"
    cfg.mkdir(parents=True)
    for p in ("volsurf-burst-1", "volsurf-burst-2", "volsurf-burst-3"):
        (cfg / f"budget_armed_{p}.json").write_text(
            '{"verified": true, "action_id": "a", "armed": true}\n',
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="local-only"):
        submit_mod.build_plan(tmp_path, "H0")


def test_all_waves_defined() -> None:
    assert "PICK" in WAVES
    assert "PICK_SMOKE" in WAVES
    assert "PICK_CANARY" in WAVES
    assert "FEATNET" in WAVES
    assert "HYBRID" in WAVES
    assert "DESKORG" in WAVES
    assert "REGIME" in WAVES


def test_discover_regime_wave() -> None:
    cells = discover_wave_cells(ROOT, "REGIME")
    assert len(cells) == 3
    stems = {Path(c).stem for c in cells}
    assert "eq_K100_single_ppo_mlp_softmax_mean_std_cao_relaxed_turnover" in stems
    assert "eq_K100_single_ppo_mlp_softmax_mean_std_cao_weekly_rebal" in stems
    assert "eq_K100_single_ppo_mlp_softmax_mean_std_cao_hybrid_heston_relaxed" in stems
    assert WAVES["REGIME"].out_subdir == "cherrypick_regime"
    assert all("cherrypick_regime" in c for c in cells)


def test_discover_featnet_and_hybrid_waves() -> None:
    feat = discover_wave_cells(ROOT, "FEATNET")
    assert len(feat) >= 1
    assert all("cherrypick_featnet" in c for c in feat)
    assert all(Path(c).stem.endswith("_featnet") for c in feat)
    hybrid = discover_wave_cells(ROOT, "HYBRID")
    assert len(hybrid) == 3
    stems = {Path(c).stem for c in hybrid}
    assert any("hybrid_heston" in s for s in stems)
    assert any("tw-heston" in s for s in stems)
    assert any("tw-historical" in s for s in stems)


def test_discover_deskorg_wave() -> None:
    cells = discover_wave_cells(ROOT, "DESKORG")
    assert len(cells) == 1
    assert Path(cells[0]).stem == "eq_K100_multi_happo_mlp_mean_std_cao_deskorg"
    assert WAVES["DESKORG"].out_subdir == "cherrypick_deskorg"


def test_rc6_happo_wave_cell_count() -> None:
    cells = discover_wave_cells(ROOT, "RC6_HAPPO")
    assert len(cells) == 3
    stems = {Path(c).stem for c in cells}
    assert stems == {
        "eq_K100_multi_happo_mlp_cvar_ru",
        "eq_K100_multi_happo_mlp_entropic_oce",
        "eq_K100_multi_happo_mlp_meanvar_kolm",
    }
    for p in cells:
        cfg = Path(p).read_text(encoding="utf-8")
        assert "happo_dispatch_only" not in cfg
        assert "claim_tier: narrative" in cfg
        assert "himem_job_memory_mib: 57344" in cfg
