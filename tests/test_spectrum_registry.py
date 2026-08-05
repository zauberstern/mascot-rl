"""TDD: spectrum registry - four axes, citations, one default each."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.plumbing

from mascotrl.spectrum.registry import (
    AXES,
    SpectrumOption,
    allowed_ids,
    default_id,
    get_option,
    validate_choice,
)


def test_four_axes_present() -> None:
    assert set(AXES) == {
        "train_world",
        "architecture",
        "objective",
        "algo",
        "policy_mode",
    }


def test_exactly_one_default_per_axis() -> None:
    for axis in AXES:
        opts = [o for o in allowed_ids(axis)]
        defaults = [oid for oid in opts if get_option(axis, oid).default]
        assert len(defaults) == 1, f"{axis} defaults={defaults}"


def test_every_option_has_citation() -> None:
    for axis in AXES:
        for oid in allowed_ids(axis):
            opt = get_option(axis, oid)
            assert isinstance(opt, SpectrumOption)
            assert opt.citation.paper_slug, f"{axis}/{oid} missing paper"
            assert opt.citation.claim_id, f"{axis}/{oid} missing claim_id"
            assert opt.status in {"proven", "plausible", "unproven"}


def test_validate_choice_accepts_known() -> None:
    assert validate_choice("train_world", "historical") == "historical"
    assert validate_choice("architecture", "mlp") == "mlp"


def test_validate_choice_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown"):
        validate_choice("train_world", "not_a_world")
    with pytest.raises(ValueError, match="unknown axis"):
        validate_choice("not_an_axis", "mlp")


def test_reference_defaults_are_simple_proven() -> None:
    # Literature-backed reference cell: real data, MLP, mean_std_cao, ppo.
    assert default_id("train_world") == "historical"
    assert default_id("architecture") == "mlp"
    assert default_id("objective") == "mean_std_cao"
    assert default_id("algo") == "ppo"


def test_portfolio_arms_opt_eq_mix() -> None:
    from mascotrl.spectrum.registry import PORTFOLIO_ARM_IDS, PORTFOLIO_ARMS, validate_portfolio_arm

    assert PORTFOLIO_ARM_IDS == ("opt", "eq", "mix")
    assert {o.id for o in PORTFOLIO_ARMS} == {"opt", "eq", "mix"}
    assert sum(1 for o in PORTFOLIO_ARMS if o.default) == 1
    assert validate_portfolio_arm("opt") == "opt"
    with pytest.raises(ValueError, match="portfolio_arm"):
        validate_portfolio_arm("fx")
