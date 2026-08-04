"""TDD: equity stk_ret research claim YAML + stamp locks."""
from __future__ import annotations
from pathlib import Path
import yaml
from src.arms.spec import EQUITY_LABEL_STEM
from src.reporting.claim_language import CLAIM_CATEGORY_RANK1
from src.reporting.claim_stamps import CLAIM_CATEGORY_EQ_STK, stamp_research_positive_alpha
ROOT = Path(__file__).resolve().parents[1]
EQ_YAML = ROOT / 'config' / 'workflows' / 'research_alpha_eq_stk.yaml'

def test_stamp_eq_stem_requires_matching_category() -> None:
    base = {'train_objective_equals_claim_metric': True, 'friction_applied': True, 'headline_fill': 'pct75', 'fill_ladder': {'mid': 0.1, 'pct75': 0.2, 'worst': -0.1}, 'path_summary': {'sharpe_mean': 0.5}, 'random_baseline_sharpe': 0.0, 'panel_source': 'optionmetrics', 'claim_label_stem': 'stk_ret', 'claim_category': CLAIM_CATEGORY_EQ_STK, 'sign_lag_baseline_sharpe': -0.1, 'long_baseline_sharpe': 0.0}
    ok = stamp_research_positive_alpha(base)
    assert ok['research_positive_alpha'] is True
    bad = stamp_research_positive_alpha({**base, 'claim_category': CLAIM_CATEGORY_RANK1})
    assert bad['research_positive_alpha'] is False
    assert any(('claim_category' in f for f in bad.get('research_positive_failures') or []))

def test_stamp_eq_refuses_dh_stem() -> None:
    out = stamp_research_positive_alpha({'train_objective_equals_claim_metric': True, 'friction_applied': True, 'headline_fill': 'pct75', 'fill_ladder': {'mid': 0.1, 'pct75': 0.2, 'worst': -0.1}, 'path_summary': {'sharpe_mean': 0.5}, 'random_baseline_sharpe': 0.0, 'panel_source': 'optionmetrics', 'claim_label_stem': 'dh_ret_lagdelta', 'claim_category': CLAIM_CATEGORY_EQ_STK, 'sign_lag_baseline_sharpe': -0.1, 'long_baseline_sharpe': 0.0})
    assert out['research_positive_alpha'] is False
