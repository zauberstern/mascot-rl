"""Re-exports CRUCIBLE stage helpers (split across screening and gates)."""
from mascotrl.data.crucible_gates import (
    _g3_pass_from_ladder,
    apply_reselect_churn_cap,
    effective_g1_entropy_gap_floor,
    entropy_gap_upper_bound_l1,
    feasible_action_diversity_probe,
    pack_slots_by_community,
    repair_swaps,
    ridge_residual_signal,
    separate_turnover_keys,
    structure_participation_gate,
    transfer_coefficient_probe,
)
from mascotrl.data.crucible_screening import (
    amihud_screen,
    assign_sleeves,
    attrition_funnel_report,
    build_sleeve_matrix,
    lottery_risk_budget_trim,
    option_eligibility,
    residual_communities,
    residualize_pool,
    sleeve_scores,
    stratify_by_adv,
)
