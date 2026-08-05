"""HAPPO-style sequential factorized PPO (Kuba Alg 3) + optional TeamTR / risk / CMDP."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from mascotrl.policy.happo import HAPPOEngine
from mascotrl.policy.objective_factory import (
    episode_weights,
    objective_gradient_path_for,
)


@dataclass
class TrainBatch:
    enriched: torch.Tensor
    macro: torch.Tensor
    w_prev: torch.Tensor
    deltas: torch.Tensor
    actions: torch.Tensor  # executed (projected) weights
    log_probs: torch.Tensor  # (T, K) or (T,) legacy — per-asset preferred
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor  # legacy: terminated OR truncated (compat)
    raw_actions: torch.Tensor | None = None  # (T, K) pre-projection samples
    # WP1: Pardo / Gymnasium — GAE masks on terminated only; truncations bootstrap.
    terminateds: torch.Tensor | None = None
    truncateds: torch.Tensor | None = None
    # Genuine V(s_T) after last transition (None → 0.0)
    last_value: float | None = None
    # WP6: per-step constraint costs C(s,a)
    costs: torch.Tensor | None = None
    cost_values: torch.Tensor | None = None
    last_cost_value: float | None = None


def _gae_loop(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(λ). Recursion mask uses terminated only (truncation bootstraps)."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    next_value = float(last_value)
    for t in reversed(range(T)):
        mask = 1.0 - terminateds[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    return advantages, advantages + values


_gae_compiled = None


def _get_gae_compiled():
    global _gae_compiled
    if _gae_compiled is False:
        return None
    if _gae_compiled is not None:
        return _gae_compiled
    try:
        _gae_compiled = torch.compile(_gae_loop, mode="reduce-overhead", fullgraph=False)
        return _gae_compiled
    except Exception:
        _gae_compiled = False
        return None


def _k3_kl(logratio: torch.Tensor) -> torch.Tensor:
    """Schulman k3 estimator: E[(r-1) - log r], unbiased, lower variance than k1."""
    ratio = torch.exp(logratio)
    return ((ratio - 1.0) - logratio).mean()


class HAPPOTrainer:
    """
    HAPPO-style sequential factorized PPO (Kuba Algorithm 3).

    Ordering (Alg 3):
      L7  compute advantages once from pre-update critic
      L8  draw random agent permutation (once per update)
      L9  M = A-hat
      L10-13 sequential PPO-clip actor updates with M compounding
      L14 critic update AFTER the actor loop

    Multi-epoch PPO ratios always against fixed rollout log-probs (π_old).

    TeamTR occupancy mitigation is engineering (not Kuba) and defaults OFF.
    """

    def __init__(
        self,
        engine: HAPPOEngine,
        lr: float = 3e-4,
        actor_lr: float | None = None,
        critic_lr: float | None = None,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.95,
        use_compile: bool = True,
        teamtr_kl0: float = 0.01,
        teamtr_enabled: bool = False,
        teamtr_kl_floor: float = 0.002,
        teamtr_shuffle_order: bool = False,
        agent_permutation: str = "random",
        proj_penalty_coef: float = 2.5,
        ppo_minibatch_size: int | None = None,
        adv_norm_scope: str = "minibatch",
        target_kl: float | None = None,
        max_grad_norm: float = 10.0,
        adam_eps: float = 1e-5,
        anneal_lr: bool = False,
        total_updates: int = 1,
        risk_objective: nn.Module | None = None,
        cmdp_enabled: bool = False,
        cmdp_limit_d: float = 0.0,
        cmdp_kp: float = 0.0,
        cmdp_ki: float = 1e-3,
        cmdp_kd: float = 0.0,
        cmdp_scale_invariant: bool = True,
    ):
        self.engine = engine
        self._hypernet = str(getattr(engine, "actor_backend", "modulelist")) == "hypernet"
        self._shared_actor = str(getattr(engine, "actor_backend", "modulelist")) in (
            "shared",
            "shared_mappo",
        )
        a_lr = float(actor_lr if actor_lr is not None else lr)
        c_lr = float(critic_lr if critic_lr is not None else lr)
        self.actor_lr0 = a_lr
        self.critic_lr0 = c_lr
        self.adam_eps = float(adam_eps)
        self.max_grad_norm = float(max_grad_norm)
        self.anneal_lr = bool(anneal_lr)
        self.total_updates = max(int(total_updates), 1)
        self._update_idx = 0

        if self._hypernet:
            if engine.hypernet_actors is None:
                raise ValueError("actor_backend=hypernet but hypernet_actors is None")
            shared = torch.optim.Adam(
                list(engine.hypernet_actors.parameters())
                + list(engine._log_std.parameters()),
                lr=a_lr,
                eps=self.adam_eps,
            )
            self.actor_opts = [shared for _ in range(engine.num_assets)]
        elif self._shared_actor:
            if len(engine.actors) != 1:
                raise ValueError(
                    f"shared actor expects len(actors)==1, got {len(engine.actors)}"
                )
            shared = torch.optim.Adam(
                list(engine.actors[0].parameters())
                + list(engine._log_std.parameters()),
                lr=a_lr,
                eps=self.adam_eps,
            )
            self.actor_opts = [shared for _ in range(engine.num_assets)]
        else:
            if len(engine.actors) != engine.num_assets:
                raise ValueError(
                    f"modulelist actors length {len(engine.actors)} != "
                    f"num_assets {engine.num_assets}"
                )
            self.actor_opts = [
                torch.optim.Adam(
                    list(actor.parameters()) + [engine._log_std[k]],
                    lr=a_lr,
                    eps=self.adam_eps,
                )
                for k, actor in enumerate(engine.actors)
            ]
        critic_params = list(engine.critic.parameters())
        if getattr(engine, "cost_critic", None) is not None:
            critic_params = critic_params + list(engine.cost_critic.parameters())
        self.critic_opt = torch.optim.Adam(
            critic_params, lr=c_lr, eps=self.adam_eps
        )
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.teamtr_kl0 = float(teamtr_kl0)
        self.teamtr_enabled = bool(teamtr_enabled)
        self.teamtr_kl_floor = float(teamtr_kl_floor)
        self.teamtr_shuffle_order = bool(teamtr_shuffle_order)
        # Kuba Alg 3 L8: random permutation per update (default on).
        self.agent_permutation = str(agent_permutation or "random").lower()
        self.proj_penalty_coef = float(proj_penalty_coef)
        self.ppo_minibatch_size = (
            int(ppo_minibatch_size) if ppo_minibatch_size is not None else None
        )
        self.adv_norm_scope = str(adv_norm_scope or "minibatch").lower()
        self.target_kl = float(target_kl) if target_kl is not None else None
        self.risk_objective = risk_objective
        self.cmdp_enabled = bool(cmdp_enabled)
        self.cmdp_limit_d = float(cmdp_limit_d)
        self.cmdp_kp = float(cmdp_kp)
        self.cmdp_ki = float(cmdp_ki)
        self.cmdp_kd = float(cmdp_kd)
        self.cmdp_scale_invariant = bool(cmdp_scale_invariant)
        self._lambda_c = 0.0
        self._lambda_I = 0.0
        self._J_c_prev = 0.0
        self._lambda_trace: list[float] = []
        self._j_c_violations = 0
        self._j_c_iters = 0
        self._eval_lp = engine.evaluate_raw_log_probs
        if use_compile:
            try:
                self._eval_lp = torch.compile(
                    engine.evaluate_raw_log_probs,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            except Exception:
                self._eval_lp = engine.evaluate_raw_log_probs

    def teamtr_kl_bound(self, queue_pos: int) -> float:
        return max(
            self.teamtr_kl0 / float(queue_pos + 1) ** 0.5,
            self.teamtr_kl_floor,
        )

    def _apply_lr_anneal(self) -> None:
        if not self.anneal_lr:
            return
        frac = max(0.0, 1.0 - self._update_idx / float(self.total_updates))
        for opt in self.actor_opts:
            for g in opt.param_groups:
                g["lr"] = frac * self.actor_lr0
        for g in self.critic_opt.param_groups:
            g["lr"] = frac * self.critic_lr0

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        terminateds: torch.Tensor,
        last_value: float = 0.0,
    ):
        fn = _get_gae_compiled() or _gae_loop
        try:
            return fn(
                rewards, values, terminateds, self.gamma, self.lam, float(last_value)
            )
        except Exception:
            return _gae_loop(
                rewards, values, terminateds, self.gamma, self.lam, float(last_value)
            )

    def _resolve_terminateds(self, batch: TrainBatch) -> torch.Tensor:
        if batch.terminateds is not None:
            return batch.terminateds
        # Legacy: dones conflated termination+truncation; treat as terminated.
        return batch.dones

    def _slice_batch(self, batch: TrainBatch, idx: torch.Tensor) -> TrainBatch:
        def _maybe(t: torch.Tensor | None):
            return None if t is None else t[idx]

        return TrainBatch(
            enriched=batch.enriched[idx],
            macro=batch.macro[idx],
            w_prev=batch.w_prev[idx],
            deltas=batch.deltas[idx],
            actions=batch.actions[idx],
            log_probs=batch.log_probs[idx],
            values=batch.values[idx],
            rewards=batch.rewards[idx],
            dones=batch.dones[idx],
            raw_actions=_maybe(batch.raw_actions),
            terminateds=_maybe(batch.terminateds),
            truncateds=_maybe(batch.truncateds),
            last_value=None,  # only meaningful for full traj
            costs=_maybe(batch.costs),
            cost_values=_maybe(batch.cost_values),
            last_cost_value=None,
        )

    def _normalize_adv(self, advantages: torch.Tensor) -> torch.Tensor:
        if advantages.numel() > 1:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages - advantages.mean()

    def _pid_dual_update(self, j_c: float) -> float:
        """Stooke Algorithm 2 PID Lagrangian dual step."""
        delta = j_c - self.cmdp_limit_d
        partial = max(j_c - self._J_c_prev, 0.0)
        self._lambda_I = max(self._lambda_I + delta, 0.0)
        lam = (
            self.cmdp_kp * delta
            + self.cmdp_ki * self._lambda_I
            + self.cmdp_kd * partial
        )
        self._lambda_c = max(lam, 0.0)
        self._J_c_prev = j_c
        self._lambda_trace.append(self._lambda_c)
        self._j_c_iters += 1
        if j_c > self.cmdp_limit_d:
            self._j_c_violations += 1
        return self._lambda_c

    def _epoch_on_batch(
        self,
        batch: TrainBatch,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        rollout_lp: torch.Tensor,
        raw: torch.Tensor,
        delta_exec: torch.Tensor,
        stats: dict,
        agent_order: list[int],
        cost_advantages: torch.Tensor | None = None,
        cost_returns: torch.Tensor | None = None,
    ) -> float:
        """Returns max approx_kl seen this epoch pass (for target_kl)."""
        K = self.engine.num_assets
        adv = advantages
        if self.adv_norm_scope == "minibatch":
            adv = self._normalize_adv(advantages)

        M_factor = torch.ones_like(adv)
        last_policy = 0.0
        max_kl = 0.0

        # Primary spectrum objectives: score-function episode weights on actor.
        obj_w = None
        obj_path = "critic_only"
        if self.risk_objective is not None:
            raw_mode = str(
                getattr(
                    self.risk_objective,
                    "raw_mode",
                    getattr(self.risk_objective, "mode", "none"),
                )
            )
            primary = bool(
                getattr(self.risk_objective, "objective_primary", False)
            )
            obj_path = objective_gradient_path_for(raw_mode, primary)
            if obj_path == "episode_weight":
                zeta = None
                z_param = getattr(self.risk_objective, "zeta", None)
                if z_param is not None:
                    zeta = z_param.detach()
                obj_w = episode_weights(
                    raw_mode,
                    returns.detach(),
                    cao_c=float(getattr(self.risk_objective, "cao_c", 1.5)),
                    kappa=float(getattr(self.risk_objective, "kappa", 1.0)),
                    alpha=float(getattr(self.risk_objective, "alpha", 0.95)),
                    lam=float(getattr(self.risk_objective, "lam", 1.0)),
                    zeta=zeta,
                )
        stats["objective_gradient_path"] = obj_path

        for queue_pos, k in enumerate(agent_order):
            kl_bound = self.teamtr_kl_bound(queue_pos)
            new_lp = self._eval_lp(batch.enriched, raw)
            logratio = new_lp[:, k] - rollout_lp[:, k]
            with torch.no_grad():
                approx_kl = abs(float(_k3_kl(logratio.detach())))
            max_kl = max(max_kl, approx_kl)
            stats["_kl_acc"] = stats.get("_kl_acc", 0.0) + approx_kl

            ratio = torch.exp(logratio)
            with torch.no_grad():
                stats["_n_clip"] = stats.get("_n_clip", 0) + int(
                    ((ratio - 1.0).abs() > self.clip_eps).sum()
                )
                stats["_n_ratio"] = stats.get("_n_ratio", 0) + int(ratio.numel())
                if stats.get("_first_ratio") is None:
                    stats["_first_ratio"] = float(ratio.mean())

            if self.teamtr_enabled and approx_kl > kl_bound:
                stats["_teamtr_skips"] = stats.get("_teamtr_skips", 0) + 1
                with torch.no_grad():
                    updated_lp = self._eval_lp(batch.enriched, raw)
                    M_factor = M_factor * torch.exp(
                        updated_lp[:, k] - rollout_lp[:, k]
                    )
                continue

            # Clip rho alone, then multiply by M (Kuba Eq 11).
            adv_k = adv * M_factor
            surr1 = ratio * adv_k
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_k
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            if (
                self.cmdp_enabled
                and cost_advantages is not None
                and self._lambda_c > 0.0
            ):
                c_adv = cost_advantages
                if self.adv_norm_scope == "minibatch" and c_adv.numel() > 1:
                    c_adv = self._normalize_adv(c_adv)
                c_adv_k = c_adv * M_factor
                c_surr1 = ratio * c_adv_k
                c_surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * c_adv_k
                )
                cost_loss = torch.min(c_surr1, c_surr2).mean()
                scale = (1.0 + self._lambda_c) if self.cmdp_scale_invariant else 1.0
                policy_loss = (policy_loss - self._lambda_c * cost_loss) / scale

            if self._hypernet:
                mean_k = self.engine._actor_means(batch.enriched)[:, k]
                actor_params = list(self.engine.hypernet_actors.parameters())
                clip_params = actor_params + list(self.engine._log_std.parameters())
            elif self._shared_actor:
                mean_k = self.engine._actor_means(batch.enriched)[:, k]
                actor_params = list(self.engine.actors[0].parameters())
                clip_params = actor_params + list(self.engine._log_std.parameters())
            else:
                mean_k = self.engine.actors[k](batch.enriched[:, k, :]).squeeze(-1)
                actor_params = list(self.engine.actors[k].parameters())
                clip_params = actor_params + [self.engine._log_std[k]]
            proj_gap_k = mean_k - delta_exec[:, k]
            proj_penalty = (proj_gap_k * proj_gap_k).mean()
            entropy = self.engine.gaussian_entropy(self.engine.log_std[k])
            stats["_ent_acc"] = stats.get("_ent_acc", 0.0) + float(entropy.detach())
            stats["_proj_pen_acc"] = stats.get("_proj_pen_acc", 0.0) + float(
                proj_penalty.detach()
            )
            loss = (
                policy_loss
                + self.proj_penalty_coef * proj_penalty
                - self.entropy_coef * entropy
            )
            if obj_w is not None:
                # L_hat = mean(w * log π); w detached (G detached), grad via log_prob.
                loss = loss + (obj_w.detach() * new_lp[:, k]).mean()

            self.actor_opts[k].zero_grad(set_to_none=True)
            loss.backward()
            p_gn = float(
                nn.utils.clip_grad_norm_(clip_params, self.max_grad_norm).item()
            )
            stats["policy_grad_norm"] = max(
                float(stats.get("policy_grad_norm", 0.0)), p_gn
            )
            self.actor_opts[k].step()
            stats["_actor_updates"] = stats.get("_actor_updates", 0) + 1
            stats["_actor_steps_order"] = stats.get("_actor_steps_order", []) + [
                ("actor", k)
            ]

            with torch.no_grad():
                updated_lp = self._eval_lp(batch.enriched, raw)
                M_factor = M_factor * torch.exp(updated_lp[:, k] - rollout_lp[:, k])
            last_policy = float(policy_loss.detach())

        # Kuba Alg 3 L14: critic AFTER sequential actors.
        value = self.engine._value(batch.enriched, batch.macro)
        value_loss = self.value_coef * F.mse_loss(value, returns)
        if (
            self.cmdp_enabled
            and cost_returns is not None
            and getattr(self.engine, "cost_critic", None) is not None
        ):
            c_val = self.engine._cost_value(batch.enriched, batch.macro)
            value_loss = value_loss + self.value_coef * F.mse_loss(c_val, cost_returns)

        if self.risk_objective is not None:
            # Static risk on batch returns (episode-proxy: use returns as G).
            risk_loss = self.risk_objective.loss(returns.detach())
            value_loss = value_loss + risk_loss
            stats["risk_loss"] = float(risk_loss.detach()) if torch.is_tensor(risk_loss) else float(risk_loss)

        self.critic_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        critic_params = list(self.engine.critic.parameters())
        if getattr(self.engine, "cost_critic", None) is not None:
            critic_params = critic_params + list(self.engine.cost_critic.parameters())
        v_gn = float(
            nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm).item()
        )
        stats["value_grad_norm"] = v_gn
        self.critic_opt.step()
        stats["_actor_steps_order"] = stats.get("_actor_steps_order", []) + [
            ("critic", -1)
        ]
        if self.risk_objective is not None and hasattr(
            self.risk_objective, "step_zeta"
        ):
            self.risk_objective.step_zeta()
        stats["value_loss"] = float(value_loss.detach())

        with torch.no_grad():
            w_det, _ = self.engine(
                batch.enriched, batch.macro, batch.w_prev, batch.deltas
            )
            stats["action_l1"] = float(raw.abs().sum(dim=-1).mean())
            stats["proj_gap"] = float((raw - delta_exec).pow(2).mean().sqrt())
            stats["exec_weight_l1"] = float(w_det.abs().sum(dim=-1).mean())
            stats["exec_turnover"] = float(
                (w_det - batch.w_prev).abs().sum(dim=-1).mean()
            )
        stats["policy_loss"] = last_policy
        return max_kl

    def update(self, batch: TrainBatch, epochs: int = 4) -> dict:
        if batch.raw_actions is None:
            raise ValueError(
                "raw_actions required for PPO log-probs "
                "(pre-projection Δw; executed actions are not the policy RV)"
            )
        raw = batch.raw_actions
        self._apply_lr_anneal()
        self._update_idx += 1

        terminateds = self._resolve_terminateds(batch)
        last_value = 0.0
        if batch.last_value is not None:
            last_value = float(batch.last_value)
        elif terminateds.numel() > 0 and float(terminateds[-1]) < 0.5:
            # Legacy fallback (wrong state) — prefer callers to pass last_value.
            last_value = float(batch.values[-1].detach())

        advantages, returns = self.compute_gae(
            batch.rewards,
            batch.values.detach(),
            terminateds,
            last_value=last_value,
        )
        if self.adv_norm_scope == "batch":
            advantages = self._normalize_adv(advantages)

        cost_advantages = None
        cost_returns = None
        if self.cmdp_enabled and batch.costs is not None:
            c_vals = (
                batch.cost_values
                if batch.cost_values is not None
                else torch.zeros_like(batch.rewards)
            )
            c_last = (
                float(batch.last_cost_value)
                if batch.last_cost_value is not None
                else 0.0
            )
            cost_advantages, cost_returns = self.compute_gae(
                batch.costs, c_vals.detach(), terminateds, last_value=c_last
            )
            j_c = float(batch.costs.sum())
            self._pid_dual_update(j_c)

        rollout_lp = batch.log_probs.detach()
        if rollout_lp.dim() == 1:
            rollout_lp = rollout_lp.unsqueeze(-1).expand(
                -1, self.engine.num_assets
            ).clone()
        else:
            rollout_lp = rollout_lp.clone()

        stats: dict = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "action_l1": 0.0,
            "proj_gap": 0.0,
            "proj_penalty": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "teamtr_skips": 0.0,
            "adv_mean": float(advantages.mean()),
            "adv_std": float(advantages.std()) if advantages.numel() > 1 else 0.0,
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()) if returns.numel() > 1 else 0.0,
            "_kl_acc": 0.0,
            "_n_clip": 0,
            "_n_ratio": 0,
            "_ent_acc": 0.0,
            "_proj_pen_acc": 0.0,
            "_teamtr_skips": 0,
            "_actor_updates": 0,
            "_first_ratio": None,
            "_actor_steps_order": [],
        }
        K = self.engine.num_assets

        # Kuba Alg 3 L8: one permutation per update (iteration k).
        if self.agent_permutation == "random" and K > 1:
            agent_order = torch.randperm(K).tolist()
        elif self.teamtr_shuffle_order and K > 1:
            agent_order = torch.randperm(K).tolist()
        else:
            agent_order = list(range(K))
        stats["agent_order"] = list(agent_order)

        with torch.no_grad():
            delta_exec = (batch.actions - batch.w_prev).detach()

        T = int(batch.rewards.shape[0])
        mb = self.ppo_minibatch_size
        use_mb = mb is not None and T >= 2 * int(mb)

        epochs_run = 0
        for _ in range(epochs):
            epochs_run += 1
            if not use_mb:
                max_kl = self._epoch_on_batch(
                    batch,
                    advantages,
                    returns,
                    rollout_lp,
                    raw,
                    delta_exec,
                    stats,
                    agent_order,
                    cost_advantages=cost_advantages,
                    cost_returns=cost_returns,
                )
            else:
                mb_size = int(mb)
                perm = torch.randperm(T)
                max_kl = 0.0
                for start in range(0, T, mb_size):
                    idx = perm[start : start + mb_size]
                    if idx.numel() < 1:
                        continue
                    mb_batch = self._slice_batch(batch, idx)
                    c_adv_mb = (
                        None
                        if cost_advantages is None
                        else cost_advantages[idx]
                    )
                    c_ret_mb = None if cost_returns is None else cost_returns[idx]
                    kl = self._epoch_on_batch(
                        mb_batch,
                        advantages[idx],
                        returns[idx],
                        rollout_lp[idx],
                        raw[idx],
                        delta_exec[idx],
                        stats,
                        agent_order,
                        cost_advantages=c_adv_mb,
                        cost_returns=c_ret_mb,
                    )
                    max_kl = max(max_kl, kl)
            if self.target_kl is not None and max_kl > self.target_kl:
                stats["early_stop_kl"] = 1.0
                break

        denom = max(epochs_run * K, 1)
        actor_updates = max(int(stats.get("_actor_updates", 0)), 1)
        stats["entropy"] = float(stats["_ent_acc"]) / actor_updates
        stats["proj_penalty"] = float(stats["_proj_pen_acc"]) / actor_updates
        stats["proj_penalty_coef"] = float(self.proj_penalty_coef)
        stats["approx_kl"] = float(stats["_kl_acc"]) / denom
        stats["clip_frac"] = float(stats["_n_clip"]) / max(
            int(stats["_n_ratio"]), 1
        )
        stats["teamtr_skips"] = float(stats["_teamtr_skips"])
        stats["teamtr_kl_floor"] = float(self.teamtr_kl_floor)
        stats["teamtr_shuffle_order"] = float(self.teamtr_shuffle_order)
        stats["teamtr_enabled"] = float(self.teamtr_enabled)
        stats["value_coef"] = float(self.value_coef)
        stats["ppo_minibatch_size"] = (
            float(self.ppo_minibatch_size)
            if self.ppo_minibatch_size is not None
            else 0.0
        )
        stats["adv_norm_scope"] = self.adv_norm_scope
        stats["max_grad_norm"] = float(self.max_grad_norm)
        stats["adam_eps"] = float(self.adam_eps)
        stats["epochs_run"] = float(epochs_run)
        stats["first_ratio"] = float(stats["_first_ratio"] or 0.0)
        stats["critic_after_actors"] = 1.0
        if self.cmdp_enabled:
            stats["cmdp_lambda"] = float(self._lambda_c)
            stats["cmdp_limit_d"] = float(self.cmdp_limit_d)
            stats["cmdp_j_c_violation_frac"] = float(
                self._j_c_violations / max(self._j_c_iters, 1)
            )
        with torch.no_grad():
            ls = self.engine.log_std
            stats["log_std_mean"] = float(ls.mean())
            stats["log_std_min"] = float(ls.min())
            stats["log_std_max"] = float(ls.max())
        for k in list(stats.keys()):
            if k.startswith("_"):
                del stats[k]
        return stats

    def cmdp_report(self) -> dict:
        return {
            "enabled": self.cmdp_enabled,
            "limit_d": self.cmdp_limit_d,
            "lambda": self._lambda_c,
            "lambda_trace": list(self._lambda_trace),
            "j_c_violation_frac": float(
                self._j_c_violations / max(self._j_c_iters, 1)
            ),
            "n_iters": self._j_c_iters,
        }
