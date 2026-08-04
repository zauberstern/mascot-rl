"""Phase 5 cherry-pick panel: manifest counts, ID legality, tier stamps."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import scripts.generate_cherrypick_panel as gen
from src.spectrum.registry import validate_cfg

ROOT = Path(__file__).resolve().parents[1]
FULLGRID_DIR = ROOT / "config" / "spectrum" / "fullgrid"
# Production cherrypick dirs (read-only for most tests; generator writes to tmp).
PROD_CHERRYPICK_DIR = ROOT / "config" / "spectrum" / "cherrypick"
PROD_NARRATIVE_DIR = PROD_CHERRYPICK_DIR / "narrative"

# Post-RL-audit panel: cppo + sdr_composite clones expand A/D/I and narrative.
TIER1_N = 147
NARRATIVE_N = 24
K200_N = 111  # A-F only at K=200
AF_N = 111

# PICK2 production lock: six eq narrative cells (see cherrypick_final/manifest.json).
PICK2_LOCKED_STEMS = (
    "eq_K100_multi_happo_mlp_mean_std_cao",
    "eq_K100_single_cppo_mlp_softmax_mean_std_cao",
    "eq_K100_single_ddpg_mlp_softmax_mtm_pnl",
    "eq_K100_single_ppo_mlp_softmax_differential_sharpe",
    "eq_K100_single_ppo_mlp_softmax_mean_std_cao",
    "eq_K100_single_ppo_mlp_softmax_mean_std_cao_uni-crucible",
)


@pytest.fixture(scope="module")
def panel_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate the panel once into an isolated tmp tree (never clobber production)."""
    root = tmp_path_factory.mktemp("cherrypick_panel")
    out_dir = root / "cherrypick"
    narrative_dir = out_dir / "narrative"
    k200_dir = out_dir / "k200"
    monkey = pytest.MonkeyPatch()
    monkey.setattr(gen, "OUT_DIR", out_dir)
    monkey.setattr(gen, "NARRATIVE_DIR", narrative_dir)
    monkey.setattr(gen, "K200_DIR", k200_dir)
    try:
        gen.main(["--tier3"])
        yield {
            "cherrypick": out_dir,
            "narrative": narrative_dir,
            "k200": k200_dir,
        }
    finally:
        monkey.undo()


@pytest.fixture(scope="module", autouse=True)
def _generated_panel(panel_dirs: dict[str, Path]) -> None:
    """Keep module-scoped panel generation for dependent tests."""
    yield


def _manifest(panel_dirs: dict[str, Path]) -> dict:
    return json.loads((panel_dirs["cherrypick"] / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_tier_counts(panel_dirs: dict[str, Path]) -> None:
    m = _manifest(panel_dirs)
    assert m["tiers"]["screening"]["n_cells"] == TIER1_N
    assert m["tiers"]["narrative"]["n_cells"] == NARRATIVE_N
    assert m["tiers"]["k200"]["n_cells"] == K200_N
    assert m["id_mismatches_resolved"][0]["sweep"] == "C"
    assert "dirichlet_mean" in m["id_mismatches_resolved"][0]["resolution"]
    assert m["unexpected_refusals"] == []


def test_sweep_subtotals(panel_dirs: dict[str, Path]) -> None:
    m = _manifest(panel_dirs)
    sweeps = m["tiers"]["screening"]["sweeps_a_to_f"]
    assert sweeps["A"]["total"] == 27
    assert sweeps["B"]["total"] == 12
    assert sweeps["C"]["total"] == 9
    assert sweeps["D"]["total"] == 27
    assert sweeps["E"]["total"] == 27
    assert sweeps["F"]["total"] == 9
    af_total = sum(sweeps[label]["total"] for label in "ABCDEF")
    assert af_total == AF_N
    assert m["tiers"]["screening"]["sweep_g_train_world"]["total"] == 18
    assert m["tiers"]["screening"]["sweep_h_policy_mode"]["total"] == 9
    assert m["tiers"]["screening"]["sweep_i_crucible_foil"]["total"] == 9
    assert af_total + 18 + 9 + 9 == TIER1_N


def test_all_fullgrid_backed_a_to_f_ids_exist() -> None:
    clone_set = set(gen.SWEEP_C_CLONE) | set(gen.SWEEP_A_CLONE) | set(gen.SWEEP_D_CLONE)
    for label in "ABCDEF":
        for algo, body, head, objective in getattr(gen, f"SWEEP_{label}"):
            if (algo, body, head, objective) in clone_set:
                continue
            for arm in gen.ARMS:
                cell_id = gen._cell_id(
                    arm=arm, k=100, algo=algo, body=body, head=head, objective=objective
                )
                assert (FULLGRID_DIR / f"{cell_id}.yaml").is_file(), cell_id


def test_dirichlet_mean_clone_exists_and_validates(panel_dirs: dict[str, Path]) -> None:
    from src.spectrum.yaml_loader import load_cell_yaml

    for arm in gen.ARMS:
        cell_id = gen._cell_id(
            arm=arm,
            k=100,
            algo="ppo",
            body="mlp",
            head="dirichlet_mean",
            objective="mean_std_cao",
        )
        path = panel_dirs["cherrypick"] / f"{cell_id}.yaml"
        assert path.is_file(), cell_id
        assert not (FULLGRID_DIR / f"{cell_id}.yaml").is_file()
        cfg = load_cell_yaml(path)
        assert cfg["weight_head"] == "dirichlet_mean"
        assert cfg["head_axis_id"] == "dirichlet_mean"
        assert cfg["action_law"] == "dirichlet_mean"
        assert "body" not in cfg
        assert "head" not in cfg
        validate_cfg(cfg)


def test_cppo_and_sdr_composite_clones(panel_dirs: dict[str, Path]) -> None:
    from src.spectrum.yaml_loader import load_cell_yaml

    for arm in gen.ARMS:
        cppo = gen._cell_id(
            arm=arm, k=100, algo="cppo", body="mlp", head="softmax", objective="mean_std_cao"
        )
        sdr = gen._cell_id(
            arm=arm, k=100, algo="ppo", body="mlp", head="softmax", objective="sdr_composite"
        )
        for cell_id, key, val in (
            (cppo, "algo", "cppo"),
            (sdr, "objective", "sdr_composite"),
        ):
            path = panel_dirs["cherrypick"] / f"{cell_id}.yaml"
            assert path.is_file(), cell_id
            cfg = load_cell_yaml(path)
            assert cfg[key] == val
            validate_cfg(cfg)


def test_sweep_a_to_f_fullgrid_copies_match_except_grid_kind(panel_dirs: dict[str, Path]) -> None:
    from src.spectrum.yaml_loader import load_cell_yaml

    clone_set = set(gen.SWEEP_C_CLONE) | set(gen.SWEEP_A_CLONE) | set(gen.SWEEP_D_CLONE)
    for algo, body, head, objective in gen.SWEEP_A:
        if (algo, body, head, objective) in clone_set:
            continue
        cell_id = gen._cell_id(
            arm="eq", k=100, algo=algo, body=body, head=head, objective=objective
        )
        src_cfg = load_cell_yaml(FULLGRID_DIR / f"{cell_id}.yaml")
        dst_cfg = load_cell_yaml(panel_dirs["cherrypick"] / f"{cell_id}.yaml")
        src_cfg.pop("grid_kind", None)
        dst_cfg.pop("grid_kind", None)
        assert src_cfg == dst_cfg


def test_no_yaml_lands_in_fullgrid(panel_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    before = set(FULLGRID_DIR.glob("*.yaml"))
    monkeypatch.setattr(gen, "OUT_DIR", panel_dirs["cherrypick"])
    monkeypatch.setattr(gen, "NARRATIVE_DIR", panel_dirs["narrative"])
    monkeypatch.setattr(gen, "K200_DIR", panel_dirs["k200"])
    gen.main(["--tier3"])
    after = set(FULLGRID_DIR.glob("*.yaml"))
    assert before == after


def test_all_emitted_yamls_pass_validate_cfg(panel_dirs: dict[str, Path]) -> None:
    dirs = [panel_dirs["cherrypick"], panel_dirs["narrative"], panel_dirs["k200"]]
    n_checked = 0
    for d in dirs:
        for path in d.glob("*.yaml"):
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            validate_cfg(cfg)
            n_checked += 1
    assert n_checked == TIER1_N + NARRATIVE_N + K200_N


def test_narrative_tier_stamps(panel_dirs: dict[str, Path]) -> None:
    for path in panel_dirs["narrative"].glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["protocol_tier"] == "narrative"
        assert cfg["seeds"] == list(range(10))
        assert cfg["cpcv_n_splits"] == 8
        assert cfg["cpcv_n_test_groups"] == 3
        assert cfg["cpcv_purge_days"] == 21
        assert cfg["cpcv_embargo_days"] == 21
        assert cfg["claim_tier"] == "narrative"


def test_narrative_eight_configs_per_arm(panel_dirs: dict[str, Path]) -> None:
    for arm in gen.ARMS:
        ids = [f"{arm}_K100_{suffix}" for suffix, _role, _ov in gen.NARRATIVE_SPECS]
        assert len(ids) == 8
        for cell_id in ids:
            assert (panel_dirs["narrative"] / f"{cell_id}.yaml").is_file()


def test_narrative_sb3_backend_stamp(panel_dirs: dict[str, Path]) -> None:
    path = panel_dirs["narrative"] / "eq_K100_single_ppo_mlp_softmax_mean_std_cao_rb-sb3.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["rl_backend"] == "sb3"


def test_pick2_production_narrative_locked_to_six() -> None:
    """Production PICK2 dir must stay at the six-cell redesign (generator writes tmp)."""
    present = sorted(p.stem for p in PROD_NARRATIVE_DIR.glob("*.yaml"))
    assert present == sorted(PICK2_LOCKED_STEMS)


def test_sweep_i_crucible_foil_eq_only_and_universe_arm(panel_dirs: dict[str, Path]) -> None:
    for algo, body, head, objective in gen.SWEEP_A:
        base_id = gen._cell_id(
            arm="eq", k=100, algo=algo, body=body, head=head, objective=objective
        )
        cell_id = f"{base_id}_uni-crucible"
        path = panel_dirs["cherrypick"] / f"{cell_id}.yaml"
        assert path.is_file()
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["universe_arm"] == "dyn_crucible"
    for arm in ("opt", "mix"):
        assert not list(panel_dirs["cherrypick"].glob(f"{arm}_*_uni-crucible.yaml"))


def test_k200_tier3_only_has_sweep_a_to_f(panel_dirs: dict[str, Path]) -> None:
    m = _manifest(panel_dirs)
    k200 = m["tiers"]["k200"]
    assert set(k200["sweeps_a_to_f"].keys()) == set("ABCDEF")
    assert k200["n_cells"] == K200_N
    for path in panel_dirs["k200"].glob("*.yaml"):
        assert "_K200_" in path.stem
