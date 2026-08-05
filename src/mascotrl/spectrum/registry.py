"""Evidence-gated spectrum registry: single source of truth for relaxed constraints.

Each axis lists selectable options with literature citations. Code validation,
campaign YAML, and generated openwiki pages all read this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


AXES = ("train_world", "architecture", "objective", "algo", "policy_mode")

# Portfolio product surface (ArmSpec ids). Not a training axis; mascotrl
# allocates all three under the same spectrum evidence gates.
PORTFOLIO_ARM_IDS = ("opt", "eq", "mix")

# Algos that can consume requires_episode_returns / episode_weight objectives.
_EPISODE_RETURN_ALGOS = frozenset({"ppo", "happo", "cppo"})

# Continuous-weight heads. dqn / happo collapse to a single foil / native law.
# Single source of truth for fullgrid generation and validate_cfg head legality.
_CONTINUOUS_HEADS = (
    "softmax",
    "tanh_l1",
    "dirichlet_tilt",
    "sparse_tilt",
    "sparse_tilt_tsallis",
    "entmax_15",
)
ALGO_HEADS: dict[str, tuple[str, ...]] = {
    "ppo": _CONTINUOUS_HEADS,
    "cppo": _CONTINUOUS_HEADS,
    "sac": (
        "softmax",
        "tanh_l1",
        "dirichlet_entropy",
        "sparse_tilt",
        "sparse_tilt_tsallis",
        "entmax_15",
    ),
    "td3": (
        "softmax",
        "tanh_l1",
        "dirichlet_mean",
        "sparse_tilt",
        "sparse_tilt_tsallis",
        "entmax_15",
    ),
    "ddpg": (
        "softmax",
        "tanh_l1",
        "dirichlet_mean",
        "sparse_tilt",
        "sparse_tilt_tsallis",
        "entmax_15",
    ),
    "mcpg": _CONTINUOUS_HEADS,
    "rrl": _CONTINUOUS_HEADS,  # B-RRL fixed: pre-head logp path
    "dqn": ("discrete",),
    "happo": ("delta_w",),
}
# Back-compat alias used by generators / docs.
_ALGO_HEADS = ALGO_HEADS

# Cherrypick Sweep C exception: ppo + dirichlet_mean is legal outside fullgrid.
_CHERRYPICK_HEAD_EXCEPTIONS: dict[str, frozenset[str]] = {
    "ppo": frozenset({"dirichlet_mean"}),
}


def _axis_id_to_weight_head(head_id: str) -> str:
    if head_id == "discrete":
        return "softmax"  # unused by DQN levels path; stamped for honesty
    if head_id == "delta_w":
        return "raw"
    return head_id


def allowed_head_axis_ids(algo: str) -> frozenset[str]:
    """Legal head_axis_id values for algo (includes cherrypick exceptions)."""
    base = ALGO_HEADS.get(str(algo).lower().strip(), ())
    extra = _CHERRYPICK_HEAD_EXCEPTIONS.get(str(algo).lower().strip(), frozenset())
    return frozenset(base) | extra


def allowed_weight_heads(algo: str) -> frozenset[str]:
    """Legal weight_head values for algo (axis ids mapped + exceptions)."""
    axes = allowed_head_axis_ids(algo)
    return frozenset(_axis_id_to_weight_head(h) for h in axes) | axes


@dataclass(frozen=True)
class Citation:
    paper_slug: str
    claim_id: str


@dataclass(frozen=True)
class SpectrumOption:
    id: str
    default: bool = False
    citation: Citation = field(default_factory=lambda: Citation("", ""))
    status: str = "unproven"  # proven | plausible | unproven
    requires: tuple[str, ...] = ()
    note: str = ""
    # D.2: refuse at validate_cfg when paired with an inadmissible algo.
    requires_episode_returns: bool = False
    # Wave 5: DQN keeps discrete flat-Q heads; non-mlp bodies are refused.
    requires_discrete: bool = False


def _c(slug: str, claim: str) -> Citation:
    return Citation(paper_slug=slug, claim_id=claim)


PORTFOLIO_ARMS: tuple[SpectrumOption, ...] = (
    SpectrumOption(
        id="opt",
        default=True,
        citation=_c("cao_2021_deep_hedging_rl", "C1"),
        status="proven",
        note=(
            "Delta-hedged option allocator (DH-panel / HAPPO+CMDP). "
            "Headline stem dh_ret_lagdelta. Unique field contribution."
        ),
    ),
    SpectrumOption(
        id="eq",
        default=False,
        citation=_c("jiang_2017_drl_portfolio_management", "C1"),
        status="plausible",
        note=(
            "Equity-only allocation (stk_ret). Parked/research; does not "
            "unlock capital claims by itself."
        ),
    ),
    SpectrumOption(
        id="mix",
        default=False,
        citation=_c("li_2019_ensemble_strategy", "C2"),
        status="plausible",
        note=(
            "Joint option+equity book (overlay projection). Reopens K>50; "
            "alpha_claim refused on mix until honesty gates pass."
        ),
    ),
)


# Claim-metric orientation for transfer_gap = o*(train-eval). Cost-like metrics
# use lower_better so a positive gap still means "train flattered the policy".
METRIC_ORIENTATION: dict[str, str] = {
    "sharpe_mean": "higher_better",
    "sharpe": "higher_better",
    "mean_pl": "higher_better",
    "cao_y": "lower_better",
    "mean_cost": "lower_better",
    "shortfall_rate": "lower_better",
    "smse": "lower_better",
    "rsqp": "lower_better",
    "cvar": "lower_better",
}


def metric_orientation(claim_metric: str) -> str:
    """Return higher_better | lower_better for a claim metric id."""
    key = str(claim_metric or "").strip().lower()
    return METRIC_ORIENTATION.get(key, "higher_better")


_REGISTRY: dict[str, tuple[SpectrumOption, ...]] = {
    "train_world": (
        SpectrumOption(
            id="historical",
            default=True,
            citation=_c("mikkila_2023_drl_option_trading", "C4"),
            status="proven",
            note="Empiric path training beats Heston sim-to-real on SPX.",
        ),
        SpectrumOption(
            id="rbergomi",
            default=False,
            citation=_c("buehler_2019_deep_hedging", "C8"),
            status="plausible",
            note="Synth generator; keep as ablation, not exclusive law.",
        ),
        SpectrumOption(
            id="gbm",
            default=False,
            citation=_c("cao_2021_deep_hedging_rl", "C1"),
            status="proven",
        ),
        SpectrumOption(
            id="heston",
            default=False,
            citation=_c("mikkila_2023_drl_option_trading", "C3"),
            status="proven",
        ),
        SpectrumOption(
            id="garch",
            default=False,
            citation=_c("neagu_2025_drl_algorithms_option_hedging", "C1"),
            status="proven",
        ),
        SpectrumOption(
            id="sabr",
            default=False,
            citation=_c("cao_2021_deep_hedging_rl", "M2"),
            status="proven",
            note="Cao M2 SABR beta=1 parity sim world.",
        ),
        SpectrumOption(
            id="hybrid_pretrain_finetune",
            default=False,
            citation=_c("mikkila_2023_drl_option_trading", "C7"),
            status="plausible",
            note="Sim useful when tape scarce; measure transfer_gap.",
        ),
    ),
    "architecture": (
        SpectrumOption(
            id="mlp",
            default=True,
            citation=_c("du_2020_drl_option_hedging", "C6"),
            status="proven",
            note="Reference temporal backend; must beat this to promote exotic.",
        ),
        SpectrumOption(
            id="gru",
            default=False,
            citation=_c("cartea_2021_drl_algo_trading", "M1"),
            status="proven",
        ),
        SpectrumOption(
            id="lstm",
            default=False,
            citation=_c("jiang_2017_drl_portfolio_management", "C1"),
            status="proven",
        ),
        SpectrumOption(
            id="transformer",
            default=False,
            citation=_c("sirignano_2019_universal_features_price", "C5"),
            status="plausible",
            note="LOB direction evidence; not hedge alpha.",
        ),
        SpectrumOption(
            id="mamba",
            default=False,
            citation=_c("buehler_2019_deep_hedging", "C5"),
            status="unproven",
            note="Repo default historically; no corpus hedge win.",
        ),
    ),
    "objective": (
        SpectrumOption(
            id="mean_std_cao",
            default=True,
            citation=_c("cao_2021_deep_hedging_rl", "C1"),
            status="proven",
            note="Y = E[C] + c*std; +16-42% vs delta under costs.",
            requires_episode_returns=True,
        ),
        SpectrumOption(
            id="mtm_pnl",
            default=False,
            citation=_c("cao_2021_deep_hedging_rl", "C2"),
            status="plausible",
            note="Accounting MTM path; paper-protocol cell.",
        ),
        SpectrumOption(
            id="meanvar_kolm",
            default=False,
            citation=_c("kolm_2019_dynamic_replication_hedging", "C3"),
            status="proven",
            requires=("kappa",),
            requires_episode_returns=True,
        ),
        SpectrumOption(
            id="cvar_ru",
            default=False,
            citation=_c("buehler_2019_deep_hedging", "C4"),
            status="proven",
            requires=("alpha",),
            requires_episode_returns=True,
        ),
        SpectrumOption(
            id="entropic_oce",
            default=False,
            citation=_c("buehler_2019_deep_hedging", "C3"),
            status="proven",
            requires=("lam",),
            requires_episode_returns=True,
        ),
        SpectrumOption(
            id="smse",
            default=False,
            citation=_c("francois_2025_deep_hedging_iv_surface_feedback", "C4"),
            status="proven",
            requires_episode_returns=True,
        ),
        SpectrumOption(
            id="rsqp",
            default=False,
            citation=_c("neagu_2025_drl_algorithms_option_hedging", "C1"),
            status="proven",
            requires_episode_returns=True,
        ),
        SpectrumOption(
            id="differential_sharpe",
            default=False,
            citation=_c("moody_2001_direct_reinforcement", "C3"),
            status="proven",
        ),
        SpectrumOption(
            id="mikkila_asym",
            default=False,
            citation=_c("mikkila_2023_drl_option_trading", "C2"),
            status="proven",
            note="Dense r_t = dPi - xi*|dPi|; xi in {1,2,3}.",
        ),
        SpectrumOption(
            id="sdr_composite",
            default=False,
            citation=_c("srivastava_2025_risk_aware_rl_reward", "C1"),
            status="plausible",
            note="Composite anti-hack: ann return - downside + diff + Treynor.",
        ),
    ),
    "algo": (
        SpectrumOption(
            id="ppo",
            default=True,
            citation=_c("du_2020_drl_option_hedging", "C6"),
            status="proven",
        ),
        SpectrumOption(
            id="sac",
            default=False,
            citation=_c("huang_2025_deep_hedging_market_frictions_drl", "C1"),
            status="proven",
        ),
        SpectrumOption(
            id="td3",
            default=False,
            citation=_c("mikkila_2023_drl_option_trading", "C4"),
            status="proven",
        ),
        SpectrumOption(
            id="ddpg",
            default=False,
            citation=_c("cao_2021_deep_hedging_rl", "M2"),
            status="plausible",
        ),
        SpectrumOption(
            id="dqn",
            default=False,
            citation=_c("zhang_2020_drl_for_trading", "C2"),
            status="proven",
            requires_discrete=True,
            note=(
                "Discrete per-asset Q heads (flat MLP only). "
                "Non-mlp architecture is an honest refusal, not a silent MLP fallback."
            ),
        ),
        SpectrumOption(
            id="mcpg",
            default=False,
            citation=_c("neagu_2025_drl_algorithms_option_hedging", "C1"),
            status="proven",
        ),
        SpectrumOption(
            id="rrl",
            default=False,
            citation=_c("moody_2001_direct_reinforcement", "C3"),
            status="proven",
        ),
        SpectrumOption(
            id="happo",
            default=False,
            citation=_c("du_2020_drl_option_hedging", "C6"),
            status="plausible",
            note="Multi-agent HAPPO allocator spine (repo field contribution).",
        ),
        SpectrumOption(
            id="cppo",
            default=False,
            citation=_c("ying_2022_cvar_constrained_safe_rl", "C10"),
            status="plausible",
            requires_episode_returns=True,
            note="CVaR-constrained PPO (trajectory hard constraint, not cvar_ru).",
        ),
    ),
    "policy_mode": (
        SpectrumOption(
            id="shared",
            default=True,
            citation=_c("jiang_2017_drl_portfolio_management", "C1"),
            status="proven",
            note="Base turnover + risk aversion; Amihud 95th pct.",
        ),
        SpectrumOption(
            id="archetype_carry",
            default=False,
            citation=_c("jiang_2017_drl_portfolio_management", "C1"),
            status="plausible",
            note="0.5x turnover; risk aversion halved.",
        ),
        SpectrumOption(
            id="archetype_inflation",
            default=False,
            citation=_c("jiang_2017_drl_portfolio_management", "C1"),
            status="plausible",
            note="Base turnover; term-spread-conditioned risk aversion.",
        ),
        SpectrumOption(
            id="archetype_crisis",
            default=False,
            citation=_c("jiang_2017_drl_portfolio_management", "C1"),
            status="plausible",
            note="2.0x turnover; risk aversion doubled; Amihud 90th pct.",
        ),
    ),
}


def allowed_ids(axis: str) -> tuple[str, ...]:
    if axis not in _REGISTRY:
        raise ValueError(f"unknown axis={axis!r}; expected one of {list(AXES)}")
    return tuple(o.id for o in _REGISTRY[axis])


def get_option(axis: str, option_id: str) -> SpectrumOption:
    if axis not in _REGISTRY:
        raise ValueError(f"unknown axis={axis!r}; expected one of {list(AXES)}")
    for opt in _REGISTRY[axis]:
        if opt.id == option_id:
            return opt
    raise ValueError(
        f"unknown {axis} option={option_id!r}; allowed={list(allowed_ids(axis))}"
    )


def default_id(axis: str) -> str:
    for opt in _REGISTRY[axis]:
        if opt.default:
            return opt.id
    raise RuntimeError(f"no default for axis={axis!r}")


def validate_choice(axis: str, option_id: str) -> str:
    """Return option_id if valid; raise ValueError otherwise."""
    if axis not in _REGISTRY:
        raise ValueError(f"unknown axis={axis!r}; expected one of {list(AXES)}")
    ids = allowed_ids(axis)
    if option_id not in ids:
        raise ValueError(f"unknown {axis} option={option_id!r}; allowed={list(ids)}")
    return option_id


def all_options(axis: str) -> tuple[SpectrumOption, ...]:
    if axis not in _REGISTRY:
        raise ValueError(f"unknown axis={axis!r}")
    return _REGISTRY[axis]


def validate_cfg(cfg: Mapping[str, object]) -> dict[str, str]:
    """Resolve spectrum keys from cfg with defaults; raise on unknown values."""
    out: dict[str, str] = {}
    key_map = {
        "train_world": ("train_world", "train_distribution"),
        "architecture": ("architecture", "temporal_backend"),
        "objective": ("objective", "reward"),
        "algo": ("algo", "policy_algo"),
        "policy_mode": ("policy_mode", "mandate", "archetype_mode"),
    }
    for axis, keys in key_map.items():
        raw = None
        for k in keys:
            if k in cfg and cfg[k] is not None and str(cfg[k]).strip():
                # Skip booleans / non-id values (e.g. objective_primary: true).
                val = cfg[k]
                if isinstance(val, bool):
                    continue
                raw = str(val).strip()
                if raw.lower() in ("true", "false"):
                    continue
                break
        if raw is None:
            out[axis] = default_id(axis)
        else:
            # Map legacy train_distribution aliases.
            if axis == "train_world" and raw in {"rbergomi_dupire", "optionmetrics"}:
                raw = "rbergomi" if raw == "rbergomi_dupire" else "historical"
            if axis == "architecture" and raw == "mamba2":
                raw = "mamba"
            if axis == "objective":
                aliases = {
                    "cvar": "cvar_ru",
                    "entropic": "entropic_oce",
                    "residual_pnl": "mtm_pnl",
                }
                raw = aliases.get(raw, raw)
            if axis == "algo" and raw in ("single_agent", "policy"):
                raw = default_id(axis)
            out[axis] = validate_choice(axis, raw)
    # D.2: refuse episode-weight objectives under algos without that path.
    obj_opt = get_option("objective", out["objective"])
    if obj_opt.requires_episode_returns and out["algo"] not in _EPISODE_RETURN_ALGOS:
        raise ValueError(
            f"objective={out['objective']!r} requires_episode_returns; "
            f"algo={out['algo']!r} is not admissible "
            f"(allowed={sorted(_EPISODE_RETURN_ALGOS)})"
        )
    # Reverse: algos that require episode returns must not pair with
    # step-reward-only objectives (e.g. cppo + mtm_pnl).
    algo_opt = get_option("algo", out["algo"])
    if getattr(algo_opt, "requires_episode_returns", False) and not obj_opt.requires_episode_returns:
        raise ValueError(
            f"algo={out['algo']!r} requires_episode_returns; "
            f"objective={out['objective']!r} does not provide them"
        )
    # Wave 5: DQN (requires_discrete) stays fail-closed under temporal bodies.
    if algo_opt.requires_discrete and out["architecture"] not in ("mlp",):
        raise ValueError(
            f"algo={out['algo']!r} requires_discrete; "
            f"architecture={out['architecture']!r} is not admissible "
            "(allowed=['mlp']; discrete Q heads are flat-MLP only)"
        )
    # RRL double-DSR: refuse at validate time (not only train time).
    from src.eval.yaml_honesty import refuse_rrl_double_dsr

    refuse_rrl_double_dsr({**dict(cfg), **out})
    # Weight-head legality (fullgrid SoT + cherrypick ppo/dirichlet_mean exception).
    _assert_algo_head_legal(cfg, out["algo"])
    # HAPPO screening must stamp dispatch-only honesty (or explicit full budget).
    _assert_happo_screening_stamp(cfg, out["algo"])
    # RASP shared locks (Part A.5). Existing OFAT soft cells without
    # turnover_limit still pass; new dirichlet/scr/hard-tau cells are gated.
    from src.policy.rasp_locks import assert_rasp_locks

    merged = dict(cfg)
    merged.update(out)
    assert_rasp_locks(merged)
    return out


def _assert_algo_head_legal(cfg: Mapping[str, object], algo: str) -> None:
    """Raise if weight_head / head_axis_id is illegal for algo."""
    if algo not in ALGO_HEADS:
        return
    allowed_axis = allowed_head_axis_ids(algo)
    allowed_wh = allowed_weight_heads(algo)
    head_axis = cfg.get("head_axis_id")
    if head_axis is not None and str(head_axis).strip():
        hid = str(head_axis).strip()
        if hid not in allowed_axis:
            raise ValueError(
                f"head_axis_id={hid!r} illegal for algo={algo!r}; "
                f"allowed={sorted(allowed_axis)}"
            )
    weight_head = cfg.get("weight_head")
    if weight_head is not None and str(weight_head).strip():
        wh = str(weight_head).strip()
        if wh not in allowed_wh:
            raise ValueError(
                f"weight_head={wh!r} illegal for algo={algo!r}; "
                f"allowed={sorted(allowed_wh)}"
            )


def _assert_happo_screening_stamp(cfg: Mapping[str, object], algo: str) -> None:
    """Refuse HAPPO screening/research cells that omit the dispatch-only stamp.

    Generated screening HAPPO YAMLs stamp ``happo_dispatch_only: true``.
    Narrative / ``happo_full_budget`` cells opt out. Explicit claim_tier
    ``dispatch_only`` is already honest without the stamp.
    """
    if algo != "happo":
        return
    if bool(cfg.get("happo_full_budget")):
        return
    if bool(cfg.get("happo_dispatch_only") or cfg.get("dispatch_only")):
        return
    protocol_tier = str(cfg.get("protocol_tier") or "").strip().lower()
    claim_tier = str(cfg.get("claim_tier") or "").strip().lower()
    if protocol_tier == "narrative" or claim_tier in ("narrative", "dispatch_only"):
        return
    # Screening protocol or research/screening claim without stamp.
    needs_stamp = protocol_tier == "screening" or claim_tier in (
        "research",
        "screening",
    )
    if needs_stamp:
        raise ValueError(
            "happo_screening_requires_dispatch_stamp: set happo_dispatch_only=true "
            "(or happo_full_budget=true for intentional full-scale HAPPO)"
        )


def render_markdown_table(axis: str) -> str:
    """Markdown table for generated docs."""
    lines = [
        f"| id | default | status | citation | note |",
        f"|----|---------|--------|----------|------|",
    ]
    for opt in all_options(axis):
        cite = f"{opt.citation.paper_slug} {opt.citation.claim_id}"
        note = (opt.note or "").replace("|", "/")
        lines.append(
            f"| `{opt.id}` | {'yes' if opt.default else ''} | {opt.status} | {cite} | {note} |"
        )
    return "\n".join(lines)


def render_portfolio_arms_table() -> str:
    """Markdown table for opt / eq / mix product surface."""
    lines = [
        "| id | default | status | citation | note |",
        "|----|---------|--------|----------|------|",
    ]
    for opt in PORTFOLIO_ARMS:
        cite = f"{opt.citation.paper_slug} {opt.citation.claim_id}"
        note = (opt.note or "").replace("|", "/")
        lines.append(
            f"| `{opt.id}` | {'yes' if opt.default else ''} | {opt.status} | {cite} | {note} |"
        )
    return "\n".join(lines)


def validate_portfolio_arm(arm_id: str) -> str:
    if arm_id not in PORTFOLIO_ARM_IDS:
        raise ValueError(
            f"unknown portfolio_arm={arm_id!r}; allowed={list(PORTFOLIO_ARM_IDS)}"
        )
    return arm_id


def iter_options() -> Iterable[tuple[str, SpectrumOption]]:
    for axis in AXES:
        for opt in _REGISTRY[axis]:
            yield axis, opt
    for opt in PORTFOLIO_ARMS:
        yield "portfolio_arm", opt
