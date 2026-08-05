import torch

from mascotrl.features.dhgnn import SpatialDHGNN
from mascotrl.features.mamba2 import PureTorchMamba2
from mascotrl.policy.convex_projection import ConvexProjectionLayer


def test_mamba_shapes():
    m = PureTorchMamba2(d_model=16, d_state=4, d_conv=4)
    x = torch.randn(2, 10, 16)
    y = m(x)
    assert y.shape == (2, 10, 16)


def test_hippo_legs_init():
    from mascotrl.features.mamba2 import hippo_legs_diagonal

    legs = hippo_legs_diagonal(8)
    assert torch.allclose(legs, -torch.arange(1, 9, dtype=torch.float32))
    m = PureTorchMamba2(d_model=8, d_state=8)
    A = -torch.exp(m.A_log.detach())
    assert torch.allclose(A, legs, atol=1e-5)


def test_teamtr_kl_bounds():
    from mascotrl.policy.happo import HAPPOEngine
    from mascotrl.policy.trainer import HAPPOTrainer

    eng = HAPPOEngine(num_assets=4, enriched_dim=8, macro_dim=4, turnover_limit=0.15)
    tr = HAPPOTrainer(
        eng,
        teamtr_kl0=0.01,
        teamtr_kl_floor=0.002,
        teamtr_enabled=True,
        use_compile=False,
    )
    assert abs(tr.teamtr_kl_bound(0) - 0.01) < 1e-12
    assert abs(tr.teamtr_kl_bound(3) - 0.005) < 1e-12
    assert abs(tr.teamtr_kl_bound(49) - 0.002) < 1e-12


def test_teamtr_shuffle_flag():
    from mascotrl.policy.happo import HAPPOEngine
    from mascotrl.policy.trainer import HAPPOTrainer

    eng = HAPPOEngine(num_assets=4, enriched_dim=8, macro_dim=4, turnover_limit=0.15)
    tr = HAPPOTrainer(eng, teamtr_shuffle_order=False, use_compile=False)
    assert tr.teamtr_shuffle_order is False


def test_iv_hypergraph_greedy_prefers_clique():
    """Unit-level check: densest pairwise clique beats high-OI loner."""
    import numpy as np

    # Synthetic IV panel: names 0..4 form a tight clique; 5 is liquid but uncorrelated.
    rng = np.random.default_rng(0)
    T = 80
    factor = rng.normal(size=T)
    noise = rng.normal(size=(T, 6)) * 0.05
    iv = np.column_stack([factor + noise[:, i] for i in range(5)] + [rng.normal(size=T)])
    corr = np.corrcoef(iv.T)
    adj = (corr >= 0.5).astype(float)
    np.fill_diagonal(adj, 0.0)
    index = iv[:, :5].mean(axis=1)
    idx_corr = np.array([np.corrcoef(iv[:, i], index)[0, 1] for i in range(6)])
    survivors = list(range(6))
    chosen = [int(np.argmax(idx_corr))]
    rem = [i for i in survivors if i != chosen[0]]
    while len(chosen) < 3 and rem:
        nxt = max(rem, key=lambda i: (float(adj[i, chosen].sum()), float(idx_corr[i])))
        chosen.append(nxt)
        rem.remove(nxt)
    assert 5 not in chosen  # uncorrelated loner excluded
    assert set(chosen).issubset({0, 1, 2, 3, 4})


def test_copula_tail_selection_prefers_joint_shocks():
    """Tail λ_U selection keeps crisis co-movers; excludes baseline-only loner."""
    import numpy as np

    rng = np.random.default_rng(1)
    T = 200
    # Calm factor + rare joint tail shocks on names 0..4.
    calm = rng.normal(size=(T, 6)) * 0.3
    shock = np.zeros(T)
    shock_idx = rng.choice(T, size=20, replace=False)
    shock[shock_idx] = 5.0
    iv = calm.copy()
    for i in range(5):
        iv[:, i] += shock
    # Name 5: high linear corr via calm factor only, never joint shock.
    iv[:, 5] = calm[:, 0] * 2.0

    ranks = np.argsort(np.argsort(iv, axis=0), axis=0).astype(np.float64) + 1.0
    U = ranks / (T + 1.0)
    thr = 0.90
    exceed = U > thr
    joint = np.logical_and(exceed[:, :, None], exceed[:, None, :]).sum(axis=0, dtype=np.float64)
    marg = np.maximum(exceed.sum(axis=0, dtype=np.float64), 1.0)
    lam = 0.5 * (joint / marg[:, None] + (joint / marg[:, None]).T)
    np.fill_diagonal(lam, 0.0)
    index = iv[:, :5].mean(axis=1)
    idx_ranks = np.argsort(np.argsort(index)).astype(np.float64) + 1.0
    U_idx = idx_ranks / (T + 1.0)
    idx_exceed = U_idx > thr
    idx_score = np.logical_and(exceed, idx_exceed[:, None]).sum(axis=0) / max(
        float(idx_exceed.sum()), 1.0
    )
    chosen = [int(np.argmax(idx_score))]
    rem = [i for i in range(6) if i != chosen[0]]
    while len(chosen) < 3 and rem:
        nxt = max(rem, key=lambda i: (float(lam[i, chosen].sum()), float(idx_score[i])))
        chosen.append(nxt)
        rem.remove(nxt)
    assert 5 not in chosen
    assert set(chosen).issubset({0, 1, 2, 3, 4})


def test_proj_penalty_pulls_mean_toward_exec_delta():
    """Projection-gap regularizer reduces ||μ_Δw − Δw_exec|| over a few steps."""
    from mascotrl.policy.happo import HAPPOEngine
    from mascotrl.policy.trainer import HAPPOTrainer, TrainBatch

    torch.manual_seed(0)
    K = 4
    T = 16
    eng = HAPPOEngine(num_assets=K, enriched_dim=8, macro_dim=4, turnover_limit=0.15)
    tr = HAPPOTrainer(
        eng,
        proj_penalty_coef=25.0,
        entropy_coef=0.0,
        teamtr_enabled=False,
        use_compile=False,
        teamtr_shuffle_order=False,
    )
    enriched = torch.randn(T, K, 8)
    macro = torch.randn(T, 4)
    w_prev = torch.zeros(T, K)
    deltas = torch.randn(T, K)
    # Huge raw Δw proposals vs small executed increments.
    raw = torch.randn(T, K) * 5.0
    actions = w_prev + torch.randn(T, K) * 0.02
    with torch.no_grad():
        old_lp = eng.evaluate_raw_log_probs(enriched, raw)
    batch = TrainBatch(
        enriched=enriched,
        macro=macro,
        w_prev=w_prev,
        deltas=deltas,
        actions=actions,
        log_probs=old_lp,
        values=torch.zeros(T),
        rewards=torch.zeros(T),
        dones=torch.zeros(T),
        raw_actions=raw,
    )
    with torch.no_grad():
        mean0 = eng._actor_means(enriched)
        gap0 = float((mean0 - (actions - w_prev)).pow(2).mean())
    for _ in range(8):
        tr.update(batch, epochs=1)
    with torch.no_grad():
        mean1 = eng._actor_means(enriched)
        gap1 = float((mean1 - (actions - w_prev)).pow(2).mean())
    assert gap1 < gap0 * 0.85


def test_convex_projection_keeps_inverse_vol_lambda():
    """Do NOT replace λ=λ₀/(σ+ε) with logistic — high σ must loosen slack."""
    layer = ConvexProjectionLayer(num_assets=4, penalty_weight=1e4, lambda_eps=1e-2)
    low = layer._lambda_from_vol(torch.tensor([0.1]))
    high = layer._lambda_from_vol(torch.tensor([4.0]))
    assert float(high) < float(low)


def test_dhgnn_incidence():
    g = SpatialDHGNN(d_model=8, num_assets=4, hist_len=32)
    # Seed history so copula path engages.
    for t in range(16):
        iv = torch.randn(4).abs() + 0.1
        g._push_iv_history(iv)
        g.iv_hist_count.fill_(t + 1)
    z = torch.randn(2, 4, 8)
    iv = torch.randn(2, 4).abs() + 0.1
    out = g(z, iv)
    assert out.shape == (2, 4, 8)
    H = g._build_dynamic_incidence_matrix(iv)
    assert H.shape[0] == 2 and H.shape[1] == 4 and H.shape[2] == 4
    # Each hyper-edge includes the anchor (+ soft peers).
    assert (H.sum(dim=1) >= 0.9).all()


def test_dhgnn_rejects_pearson_path():
    """Incidence must come from tail-dependence buffers, not a Pearson attr."""
    g = SpatialDHGNN(d_model=8, num_assets=4)
    assert not hasattr(g, "running_corr")
    assert hasattr(g, "running_tail")
    assert hasattr(g, "_empirical_copula_tail_dependence")


def test_convex_projection_delta_neutral():
    layer = ConvexProjectionLayer(num_assets=4, turnover_limit=0.5)
    w_raw = torch.tensor([[0.4, -0.2, 0.3, -0.1]], dtype=torch.float32)
    w_prev = torch.zeros(1, 4)
    deltas = torch.tensor([[1.0, 0.5, -0.5, 0.2]], dtype=torch.float32)
    w = layer(w_raw, w_prev, deltas, vol_scale=0.2)
    d_hat = deltas / deltas.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    soft_dot = (w * d_hat).sum().item()
    assert abs(soft_dot) < 1e-2
    turn = (w - w_prev).abs().sum().item()
    assert turn <= 0.5 + 1e-3


def test_convex_projection_enforces_name_box():
    """Hard |w_i| ≤ max_name must bind when the actor proposes a spike."""
    layer = ConvexProjectionLayer(
        num_assets=4, turnover_limit=2.0, max_name_abs_weight=5.0
    )
    w_raw = torch.tensor([[20.0, -20.0, 0.0, 0.0]], dtype=torch.float32)
    w_prev = torch.zeros(1, 4)
    deltas = torch.tensor([[0.01, -0.01, 0.0, 0.0]], dtype=torch.float32)
    w = layer(w_raw, w_prev, deltas, vol_scale=0.2)
    assert float(w.abs().max()) <= 5.0 + 1e-3


def test_convex_projection_inverse_vol_lambda():
    """λ = λ₀/(σ+ε): higher σ → lower slack penalty."""
    layer = ConvexProjectionLayer(num_assets=4, penalty_weight=1e4, lambda_eps=1e-2)
    low = layer._lambda_from_vol(torch.tensor([0.1]))
    high = layer._lambda_from_vol(torch.tensor([4.0]))
    assert float(high) < float(low)


def test_convex_projection_vol_scaled_slacks():
    layer = ConvexProjectionLayer(num_assets=4, turnover_limit=0.15, penalty_weight=1e4)
    w_raw = torch.randn(2, 4)
    w_prev = torch.zeros(2, 4)
    deltas = torch.tensor(
        [[12.0, -8.0, 5.0, -3.0], [20.0, 1.0, -15.0, 4.0]], dtype=torch.float32
    )
    for sigma in (0.1, 1.0, 4.0):
        w = layer(w_raw, w_prev, deltas, vol_scale=sigma)
        assert torch.isfinite(w).all()
        turn = (w - w_prev).abs().sum(dim=-1)
        assert float(turn.max()) < 50.0
