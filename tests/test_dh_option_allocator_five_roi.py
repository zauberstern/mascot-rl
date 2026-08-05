"""delta-hedged option allocator five-ROI stamps: estimand hire, friction XOR, transfer, pack, arm lock."""
from __future__ import annotations
from mascotrl.data.oos_panel import LABEL_STEM
from mascotrl.reporting.capital_gates import assert_protocol_provenance, default_estimand_residuals
from mascotrl.reporting.claim_language import CLAIM_CATEGORY_RANK1
from mascotrl.reporting.claim_stamps import assert_rank1_arm_lock, stamp_rank1_estimand_and_transfer
from tests.conftest import capital_gate_pass_extras

def _lake() -> dict:
    return {'single_write_immutable_lake': True, 'lake_checksum': 'test', 'estimand_residuals': default_estimand_residuals({})}

def _passing_factor() -> dict:
    return {'alpha': {'alpha_significant_05': True, 'alpha_t_hac': 3.1}}

def _passing_ladder() -> dict:
    return {'break_even_spread_multiplier': 1.4}

def _passing_baselines() -> dict:
    return {'baselines': {'summary': {'short_vol_carry': {'mean_pnl': 0.01, 'sharpe': 0.5}}}, 'best_baseline': 'short_vol_carry', 'edge_vs_best_baseline': 0.1}

def _base_capital_report(**extra) -> dict:
    report = {'eval_protocol': 'pit_optionmetrics_atm_is_oos', 'claim_category': CLAIM_CATEGORY_RANK1, 'campaign': 'rank1_allocator', 'historical_oos': {'alpha_found_historical': True, 'sharpe_beats_best_baseline': True, 'label_stem': LABEL_STEM, 'friction_applied': True, 'summary': {'happo': {'sharpe': 3.0, 'mean_pnl': 0.1}}}, 'capital_gates_require_stability': False, 'capital_gates_require_retrain_wfo': False, 'capital_gates_require_pack_gates': False, 'factor_alpha': _passing_factor(), 'cost_ladder': _passing_ladder(), 'finetune_friction_applied': True, 'transfer_protocol': 'rbergomi_dupire_pretrain_then_optionmetrics_finetune', 'transfer_ok': True, **_passing_baselines(), **_lake(), **capital_gate_pass_extras()}
    report.update(extra)
    return report

def test_stamp_rank1_estimand_fields():
    report = {'historical_oos': {'label_stem': LABEL_STEM, 'return_definition': 'delta_hedged_call_lagdelta_scaled_by_delta_S_minus_C', 'friction_applied': True, 'friction_model': 'om_touch'}, 'om_touch_enabled': True, 'execution_spread_bps': 0.0, 'nested_wfo': {'mode': 'retrain_per_fold', 'finetune_friction_applied': True, 'n_folds': 3}}
    out = stamp_rank1_estimand_and_transfer(report)
    assert out['claim_label_stem'] == LABEL_STEM
    assert out['claim_return_definition'].startswith('delta_hedged_call')
    assert out['train_reward'] == 'clean_mtm_synth'
    assert out['eval_friction'] == 'om_touch'
    assert out['train_objective_equals_claim_metric'] is False
    assert out['transfer_protocol'] == 'rbergomi_dupire_pretrain_then_optionmetrics_finetune'
    assert out['finetune_friction_applied'] is True
    assert out['transfer_ok'] is True
    assert out['campaign'] == 'rank1_allocator'

def test_capital_blocks_wrong_label_stem():
    report = _base_capital_report()
    report['historical_oos']['label_stem'] = 'fwd_ret'
    report = stamp_rank1_estimand_and_transfer(report)
    out = assert_protocol_provenance(report)
    assert any(('claim_label_stem' in f for f in out['protocol_gate']['gate_failures']))

def test_capital_blocks_friction_off_when_om_touch_required():
    report = _base_capital_report(om_touch_enabled=True)
    report['historical_oos']['friction_applied'] = False
    report = stamp_rank1_estimand_and_transfer(report)
    out = assert_protocol_provenance(report)
    assert any(('friction_applied' in f for f in out['protocol_gate']['gate_failures']))

def test_capital_blocks_finetune_friction_off_when_retrain_required():
    report = _base_capital_report(capital_gates_require_retrain_wfo=True, capital_gates_require_stability=True, eval_protocol='pit_optionmetrics_nested_wfo_retrain', nested_wfo={'mode': 'retrain_per_fold', 'positive_fold_rate': 0.8, 'finetune_friction_applied': False, 'n_folds': 5}, multiseed_oos={'sharpe_p05': 0.1}, adversarial_iv_stress={'sharpe_degradation': 0.1, 'fragile': False}, finetune_friction_applied=False, transfer_ok=False)
    report = stamp_rank1_estimand_and_transfer(report)
    out = assert_protocol_provenance(report)
    fails = out['protocol_gate']['gate_failures']
    assert any(('finetune_friction' in f or 'transfer_ok' in f for f in fails))

def test_arm_lock_refuses_beef_art_in_pack():
    report = {'claim_category': CLAIM_CATEGORY_RANK1, 'publication_evidence_pack': {'arm': 'hedge_mdp', 'source': 'cpcv_hedge_gate1_om_beef_tier_s.json'}}
    out = assert_rank1_arm_lock(report)
    assert out['claim_category'] == CLAIM_CATEGORY_RANK1
    assert 'deep_hedge_mdp' not in str(out.get('claim_category'))
    assert 'beef_art_in_rank1_pack' in (out.get('arm_lock_failures') or [])

def test_protocol_hygiene_ok_with_full_rank1_stamps():
    report = stamp_rank1_estimand_and_transfer(_base_capital_report(om_touch_enabled=True))
    out = assert_protocol_provenance(report)
    assert out['claim_label_stem'] == LABEL_STEM
    assert out['protocol_gate']['protocol_hygiene_ok'] is True

def test_spa_families_exclude_nonsense_peers():
    from mascotrl.eval.alpha_gates import NONSENSE_PEERS
    from mascotrl.eval.publication import SPA_FAMILIES
    assert not set(SPA_FAMILIES) & set(NONSENSE_PEERS)
