"""Gap-freeze: research alpha trial estimand + claim_tier locked strings."""
from __future__ import annotations
from pathlib import Path
import yaml
from mascotrl.data.oos_panel import LABEL_STEM
from mascotrl.reporting.claim_stamps import CLAIM_RETURN_DEFINITION, stamp_research_positive_alpha
ROOT = Path(__file__).resolve().parents[1]

def test_estimand_strings_frozen() -> None:
    assert LABEL_STEM == 'dh_ret_lagdelta'
    assert CLAIM_RETURN_DEFINITION == 'delta_hedged_call_lagdelta_scaled_by_delta_S_minus_C'

def test_stamp_defaults_claim_tier_research() -> None:
    out = stamp_research_positive_alpha({'train_objective_equals_claim_metric': True, 'friction_applied': True, 'headline_fill': 'pct75', 'fill_ladder': {'mid': 0.0, 'pct75': 0.1, 'worst': -0.1}, 'path_summary': {'sharpe_mean': 0.5}, 'random_baseline_sharpe': 0.0, 'panel_source': 'optionmetrics'})
    assert out['claim_tier'] == 'research'
    assert out['research_positive_alpha'] is True
