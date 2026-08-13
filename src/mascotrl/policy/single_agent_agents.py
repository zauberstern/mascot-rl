"""Single-agent RL algorithm implementations (PPO, SAC, TD3, DDPG, MCPG, RRL, DQN)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mascotrl.policy.single_agent_common import (
    ActorCritic as _ActorCritic,
    AssetTemporalActorCritic as _AssetTemporalActorCritic,
    BaseAgent as _BaseAgent,
    QNet as _QNet,
    RunningMeanStd,
    WEIGHT_HEADS as _WEIGHT_HEADS,
    actor_body as _actor_body,
    apply_weight_head as _apply_weight_head,
    compute_gae,
    explained_variance as _explained_variance,
    mlp as _mlp,
)

class PPOAgent(_BaseAgent):
    """Correct PPO: stored old log-probs, GAE, entropy, minibatch, weight head.

    ``weight_head``:
      - ``softmax``: sample Gaussian in logit space, emit simplex weights (long-only).
      - ``tanh_l1``: sample Gaussian, tanh, L1-normalize (long-short).
      - ``raw``: legacy unbounded Gaussian (smoke / bakeoff only).
      - ``dirichlet_tilt``: sample Dirichlet; score Dir.log_prob; multiplicative tilt.
    """

    name = "ppo"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        hidden: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        weight_head: str = "softmax",
        normalize_obs: bool = True,
        normalize_adv: bool = True,
        architecture: str = "mlp",
        num_assets: int | None = None,
        d_model: int | None = None,
        seq_len: int = 1,
        d_state: int = 16,
        share_temporal_encoder: bool = True,
        use_surface_image_encoder: bool = False,
        image_channels: int = 0,
        surface_image_embed_dim: int = 16,
        actor_final_gain: float = 0.1,
        weight_head_temperature: float = 1.0,
        weight_head_tilt_gain: float = 1.0,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.clip_eps = float(clip_eps)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        head = str(weight_head or "softmax").lower()
        if head not in _WEIGHT_HEADS:
            raise ValueError(f"unknown weight_head={weight_head!r}")
        self.weight_head = head
        self.weight_head_temperature = float(weight_head_temperature)
        self.weight_head_tilt_gain = float(weight_head_tilt_gain)
        self.normalize_adv = bool(normalize_adv)
        self.architecture = str(architecture or "mlp").lower()
        self._dirichlet = head.startswith("dirichlet")
        # C2: architecture axis. "mlp" keeps the original flat joint-asset
        # body exactly (no behaviour change for the calibrated default);
        # gru/lstm/transformer/mamba route through AlphaFeatureExtractor's
        # per-asset temporal backends on the asset-major feature cube.
        if self.architecture == "mlp":
            self.net: nn.Module = _ActorCritic(
                obs_dim, action_dim, hidden, actor_final_gain=actor_final_gain
            )
        else:
            if num_assets is None or d_model is None:
                raise ValueError(
                    f"architecture={architecture!r} requires num_assets and "
                    "d_model (needs use_equity_feature_cube=true so the "
                    "asset-major observation layout is known)"
                )
            self.net = _AssetTemporalActorCritic(
                num_assets=int(num_assets),
                d_model=int(d_model),
                action_dim=action_dim,
                seq_len=int(seq_len),
                d_state=int(d_state),
                temporal_backend=self.architecture,
                share_temporal_encoder=bool(share_temporal_encoder),
                use_surface_image_encoder=bool(use_surface_image_encoder),
                image_channels=int(image_channels),
                surface_image_embed_dim=int(surface_image_embed_dim),
            )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.obs_rms = RunningMeanStd(obs_dim) if normalize_obs else None
        self._optimizer_steps = 0

    def checkpoint_state(self) -> dict:
        state: dict = {
            "net": self.net.state_dict(),
            "optimizer_steps": int(self._optimizer_steps),
        }
        if self.obs_rms is not None:
            state["obs_rms"] = {
                "mean": self.obs_rms.mean.detach().clone(),
                "var": self.obs_rms.var.detach().clone(),
                "count": float(self.obs_rms.count),
                "frozen": bool(getattr(self.obs_rms, "frozen", False)),
            }
        return state

    def load_checkpoint_state(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
        if "optimizer_steps" in state:
            self._optimizer_steps = int(state["optimizer_steps"])
        rms = state.get("obs_rms")
        if rms is not None and self.obs_rms is not None:
            self.obs_rms.mean.copy_(rms["mean"])
            self.obs_rms.var.copy_(rms["var"])
            self.obs_rms.count = float(rms["count"])
            self.obs_rms.frozen = bool(rms.get("frozen", False))

    def _prep_obs(self, obs: torch.Tensor, *, update_rms: bool = False) -> torch.Tensor:
        x = obs
        if self.obs_rms is not None:
            if update_rms:
                self.obs_rms.update(x)
            x = self.obs_rms.normalize(x)
        return x

    def freeze_obs_norm(self) -> None:
        if self.obs_rms is not None:
            self.obs_rms.frozen = True

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net.mean(obs)
        # Clamp both sides: unclamped-above log_std let a handful of samples
        # blow up the Gaussian scale during early training and destabilize
        # the ratio; unclamped-below collapses entropy to zero (A8).
        log_std = torch.clamp(self.net.log_std, -5.0, 1.0)
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def _dirichlet_alpha(self, obs: torch.Tensor) -> torch.Tensor:
        from mascotrl.policy.dirichlet_tilt import concentrations_from_logits

        return concentrations_from_logits(self.net.mean(obs))

    def raw_to_weights(
        self, raw: torch.Tensor, *, w_base: torch.Tensor | None = None
    ) -> torch.Tensor:
        base = w_base
        if base is None and getattr(self, "_last_w_base", None) is not None:
            base = self._last_w_base
        return _apply_weight_head(
            raw,
            self.weight_head,
            temperature=self.weight_head_temperature,
            tilt_gain=self.weight_head_tilt_gain,
            w_base=base,
        )

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        x = self._prep_obs(obs, update_rms=False)
        if self._dirichlet:
            from mascotrl.policy.dirichlet_tilt import dirichlet_sample

            alpha = self._dirichlet_alpha(x)
            u, _, _ = dirichlet_sample(alpha, deterministic=deterministic)
            return self.raw_to_weights(u)
        dist = self._dist(x)
        raw = dist.mean if deterministic else dist.rsample()
        return self.raw_to_weights(raw)

    def act_and_logp(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (weights, log_prob of pre-head stochastic object)."""
        raw, logp = self.act_and_logp_raw(obs, deterministic=deterministic)
        return self.raw_to_weights(raw), logp

    def act_and_logp_raw(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pre-head sample, log_prob). Train on the pre-head sample.

        For Dirichlet heads the sample is the simplex draw ``u``; for Gaussian
        heads it is the unbounded logit sample.
        """
        x = self._prep_obs(obs, update_rms=True)
        if self._dirichlet:
            from mascotrl.policy.dirichlet_tilt import dirichlet_sample

            alpha = self._dirichlet_alpha(x)
            u, logp, _ = dirichlet_sample(alpha, deterministic=deterministic)
            return u, logp
        dist = self._dist(x)
        raw = dist.mean if deterministic else dist.rsample()
        logp = dist.log_prob(raw).sum(dim=-1)
        return raw, logp

    def _constraint_penalty(self, returns_mb: torch.Tensor) -> torch.Tensor:
        """Optional CMDP/CPPO Lagrangian term; default zero (vanilla PPO)."""
        return returns_mb.new_zeros(())

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        old_logprobs: torch.Tensor | None = None,
        n_epochs: int = 1,
        n_minibatches: int = 1,
        sample_weight: torch.Tensor | None = None,
        policy_step_mask: torch.Tensor | None = None,
        scr_mix: str = "off",
        scr_beta: float = 0.5,
        scr_y_cf: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """PPO update with GAE, real importance ratio, entropy, minibatching.

        ``actions`` are **pre-head** Gaussian samples (logits), not simplex weights
        (or Dirichlet simplex samples when ``weight_head`` starts with dirichlet).
        When ``old_logprobs`` is omitted (legacy smoke), they are computed under
        the *current* policy once before updates (ratio starts at 1 for epoch 0).

        ``sample_weight`` (C3): optional per-timestep multiplier on the GAE
        advantage, the research-PPO analogue of HAPPO's score-function
        episode weights (``src.policy.objective_factory.episode_weights``)
        for the ``objective: episode_weight`` axis (mean_std_cao,
        meanvar_kolm, cvar_ru, entropic_oce, smse, rsqp). ``None`` reproduces
        the pre-existing vanilla-advantage behaviour exactly.

        ``policy_step_mask`` (RC5): optional bool/float mask; non-rebalance
        steps get zero advantage so the PPO surrogate ignores daily no-ops
        under monthly cadence while the value head still fits all steps.

        ``scr_mix`` / ``scr_beta`` / ``scr_y_cf``: SCR critic target mix
        (plan A.3.8). Actor ``rewards`` are never modified.
        """
        # A8: the running observation normalizer is updated once per sample
        # during rollout (act_and_logp_raw); re-updating here on the full
        # batch double-counted every sample's contribution to the running
        # mean/var, skewing normalization for long training runs.
        x = self._prep_obs(obs, update_rms=False)
        with torch.no_grad():
            values = self.net.value(x)
            next_x = self._prep_obs(next_obs, update_rms=False)
            next_values = self.net.value(next_x)
            from mascotrl.eval.scr_critic import build_scr_returns

            advantages, returns, scr_meta = build_scr_returns(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                scr_mix=str(scr_mix or "off"),
                scr_beta=float(scr_beta),
                y_cf=scr_y_cf,
            )
            # RC5: episode_weights before z-norm so CAO scale survives.
            # Zero-weight / non-rebalance steps stay at 0 after norm (do not
            # leak -mean/std into the surrogate).
            active = torch.ones_like(advantages, dtype=torch.bool)
            if sample_weight is not None:
                sw = sample_weight.detach().reshape(-1).to(advantages.dtype)
                if sw.shape[0] != advantages.shape[0]:
                    raise ValueError(
                        f"sample_weight length {sw.shape[0]} != batch {advantages.shape[0]}"
                    )
                advantages = advantages * sw
                active = active & (sw.abs() > 0)
            if policy_step_mask is not None:
                m = policy_step_mask.detach().reshape(-1).to(dtype=torch.bool)
                if m.shape[0] != advantages.shape[0]:
                    raise ValueError(
                        f"policy_step_mask length {m.shape[0]} != batch {advantages.shape[0]}"
                    )
                active = active & m
            advantages = torch.where(active, advantages, torch.zeros_like(advantages))
            # RC6 diagnostics: advantage variance + logit cross-sectional std.
            adv_var = float(torch.var(advantages).item()) if advantages.numel() else 0.0
            logit_xsec_std = 0.0
            if actions is not None and actions.numel() > 0:
                logit_xsec_std = float(
                    torch.std(actions, dim=-1).mean().item()
                )
            if self.normalize_adv:
                if bool(active.any().item()):
                    act = advantages[active]
                    mu = act.mean()
                    sd = act.std(unbiased=False) + 1e-8
                    advantages = torch.where(
                        active, (advantages - mu) / sd, torch.zeros_like(advantages)
                    )
                # else: leave all-zero advantages untouched
            if old_logprobs is None:
                if self._dirichlet:
                    from mascotrl.policy.dirichlet_tilt import dirichlet_log_prob

                    old_logprobs = dirichlet_log_prob(self._dirichlet_alpha(x), actions)
                else:
                    dist0 = self._dist(x)
                    old_logprobs = dist0.log_prob(actions).sum(dim=-1)
            else:
                old_logprobs = old_logprobs.detach()

        n = int(obs.shape[0])
        mb = max(1, int(n_minibatches))
        batch_size = max(1, n // mb)
        last: dict[str, float] = {}
        ratio_means: list[float] = []
        kl_vals: list[float] = []
        entropies: list[float] = []
        clip_fracs: list[float] = []
        steps = 0

        for _ in range(max(1, int(n_epochs))):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                if idx.numel() == 0:
                    continue
                xb = self._prep_obs(obs[idx], update_rms=False)
                if self._dirichlet:
                    from mascotrl.policy.dirichlet_tilt import (
                        dirichlet_entropy,
                        dirichlet_log_prob,
                    )

                    alpha_b = self._dirichlet_alpha(xb)
                    logp = dirichlet_log_prob(alpha_b, actions[idx])
                    entropy = dirichlet_entropy(alpha_b).mean()
                else:
                    dist = self._dist(xb)
                    logp = dist.log_prob(actions[idx]).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    # RC6_HEADS: Tsallis-2 over executed weights isolates the
                    # entropy-bonus effect; projection stays sparsemax.
                    if self.weight_head == "sparse_tilt_tsallis":
                        from mascotrl.policy.entmax import tsallis_entropy

                        w_exec = self.raw_to_weights(actions[idx])
                        entropy = tsallis_entropy(w_exec, alpha=2.0).mean()
                ratio = torch.exp(logp - old_logprobs[idx])
                adv_b = advantages[idx]
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                value_pred = self.net.value(xb)
                critic_loss = F.mse_loss(value_pred, returns[idx])
                constraint_pen = self._constraint_penalty(returns[idx])
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                    + constraint_pen
                )
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.max_grad_norm
                    )
                )
                self.opt.step()
                steps += 1
                self._optimizer_steps += 1
                with torch.no_grad():
                    approx_kl = float(((old_logprobs[idx] - logp).mean()).clamp_min(0.0))
                    clip_frac = float(
                        ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                    )
                    ratio_means.append(float(ratio.mean()))
                    kl_vals.append(approx_kl)
                    entropies.append(float(entropy.detach()))
                    clip_fracs.append(clip_frac)
                last = {
                    "loss": float(loss.detach()),
                    "actor_loss": float(actor_loss.detach()),
                    "policy_loss": float(actor_loss.detach()),
                    "critic_loss": float(critic_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "approx_kl": float(np.mean(kl_vals)) if kl_vals else 0.0,
                    "mean_ratio": float(np.mean(ratio_means)) if ratio_means else 1.0,
                    "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
                    "grad_norm": grad_norm,
                    "policy_grad_norm": grad_norm,
                    "optimizer_steps": float(steps),
                }
        if last:
            # A8: explained_variance must reflect the *updated* critic, not
            # the pre-update snapshot taken before any gradient step.
            with torch.no_grad():
                values_post = self.net.value(x)
            last["explained_variance"] = _explained_variance(
                returns.detach(), values_post.detach()
            )
            last["scr_mix"] = str(scr_meta.get("scr_mix") or "off")
            last["advantage_variance"] = adv_var
            last["logit_xsec_std"] = logit_xsec_std
            last["scr_beta"] = float(scr_meta.get("scr_beta") or 0.0)
        return last

class SACAgent(_BaseAgent):
    """Soft Actor-Critic with the same ``weight_head`` contract as PPO.

    Actor samples a Gaussian in unbounded space; ``act`` and critic targets
    apply ``_apply_weight_head`` so env interaction and Q-learning share
    admissible portfolio geometry (P4).
    """

    name = "sac"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        hidden: int = 64,
        weight_head: str = "softmax",
        weight_head_tilt_gain: float = 1.0,
        architecture: str = "mlp",
        num_assets: int | None = None,
        d_model: int | None = None,
        seq_len: int = 1,
        d_state: int = 16,
        share_temporal_encoder: bool = True,
        use_surface_image_encoder: bool = False,
        image_channels: int = 0,
        surface_image_embed_dim: int = 16,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        head = str(weight_head or "softmax").lower()
        if head not in _WEIGHT_HEADS:
            raise ValueError(f"unknown weight_head={weight_head!r}")
        self.weight_head = head
        self.weight_head_tilt_gain = float(weight_head_tilt_gain)
        self.action_law = head if head.startswith("dirichlet") else "gaussian_weight_head"
        self._dirichlet = head.startswith("dirichlet")
        self.architecture = str(architecture or "mlp").lower()
        self.actor = _actor_body(
            architecture=self.architecture,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=hidden,
            num_assets=num_assets,
            d_model=d_model,
            seq_len=seq_len,
            d_state=d_state,
            share_temporal_encoder=share_temporal_encoder,
            use_surface_image_encoder=use_surface_image_encoder,
            image_channels=image_channels,
            surface_image_embed_dim=surface_image_embed_dim,
            with_critic=False,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        self.q1 = _QNet(obs_dim, action_dim, hidden)
        self.q2 = _QNet(obs_dim, action_dim, hidden)
        self.q1_t = _QNet(obs_dim, action_dim, hidden)
        self.q2_t = _QNet(obs_dim, action_dim, hidden)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        # Dirichlet entropy target: target H ≈ 0.5 * log(|A|) as a scale-free prior.
        self.target_entropy = (
            0.5 * float(np.log(max(action_dim, 2)))
            if self._dirichlet
            else -float(action_dim)
        )
        params = (
            list(self.actor.parameters())
            + [self.log_std, self.log_alpha]
            + list(self.q1.parameters())
            + list(self.q2.parameters())
        )
        self.opt = torch.optim.Adam(params, lr=lr)

    def checkpoint_state(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "log_std": self.log_std.detach().clone(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_t": self.q1_t.state_dict(),
            "q2_t": self.q2_t.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.log_std.data.copy_(state["log_std"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_t.load_state_dict(state["q1_t"])
        self.q2_t.load_state_dict(state["q2_t"])
        self.log_alpha.data.copy_(state["log_alpha"])


    def _head(self, raw: torch.Tensor) -> torch.Tensor:
        base = getattr(self, "_last_w_base", None)
        return _apply_weight_head(
            raw,
            self.weight_head,
            tilt_gain=self.weight_head_tilt_gain,
            w_base=base,
        )

    def _sample_actor(
        self, obs: torch.Tensor, *, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (weights, log_prob, entropy) under the configured action law."""
        mean = self.actor(obs)
        if self._dirichlet:
            from mascotrl.policy.dirichlet_tilt import (
                concentrations_from_logits,
                dirichlet_sample,
            )

            alpha = concentrations_from_logits(mean)
            u, logp, ent = dirichlet_sample(alpha, deterministic=deterministic)
            from mascotrl.policy.dirichlet_tilt import multiplicative_tilt

            # Score Dir(u); critic / env see the tilted simplex proposal.
            w = multiplicative_tilt(u, kappa=1.0)
            return w, logp, ent
        if deterministic:
            raw = mean
            std = self.log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(raw).sum(dim=-1)
            ent = dist.entropy().sum(dim=-1)
            return self._head(raw), logp, ent
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        logp = dist.log_prob(raw).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return self._head(raw), logp, ent

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        w, _, _ = self._sample_actor(obs, deterministic=deterministic)
        return w

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        alpha = self.log_alpha.exp()
        with torch.no_grad():
            next_a, next_logp, _ = self._sample_actor(next_obs, deterministic=False)
            q_t = torch.min(
                self.q1_t(next_obs, next_a), self.q2_t(next_obs, next_a)
            )
            target = rewards + (1.0 - dones) * self.gamma * (q_t - alpha * next_logp)

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        new_a, logp, entropy = self._sample_actor(obs, deterministic=False)
        q_pi = torch.min(self.q1(obs, new_a), self.q2(obs, new_a))
        actor_loss = (alpha.detach() * logp - q_pi).mean()
        alpha_loss = -(self.log_alpha * (logp.detach() - self.target_entropy).mean())

        loss = q_loss + actor_loss + alpha_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return {
            "loss": float(loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "policy_loss": float(actor_loss.detach()),
            "q_loss": float(q_loss.detach()),
            "entropy": float(entropy.mean().detach()),
            "action_law": self.action_law,
        }


class TD3Agent(_BaseAgent):
    """TD3 with the same ``weight_head`` contract as PPO/DDPG (P4)."""

    name = "td3"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.1,
        hidden: int = 64,
        weight_head: str = "softmax",
        weight_head_tilt_gain: float = 1.0,
        architecture: str = "mlp",
        num_assets: int | None = None,
        d_model: int | None = None,
        seq_len: int = 1,
        d_state: int = 16,
        share_temporal_encoder: bool = True,
        use_surface_image_encoder: bool = False,
        image_channels: int = 0,
        surface_image_embed_dim: int = 16,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        head = str(weight_head or "softmax").lower()
        if head not in _WEIGHT_HEADS:
            raise ValueError(f"unknown weight_head={weight_head!r}")
        self.weight_head = head
        self.weight_head_tilt_gain = float(weight_head_tilt_gain)
        self.action_law = head if head.startswith("dirichlet") else "gaussian_weight_head"
        self.architecture = str(architecture or "mlp").lower()
        body_kwargs = dict(
            architecture=self.architecture,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=hidden,
            num_assets=num_assets,
            d_model=d_model,
            seq_len=seq_len,
            d_state=d_state,
            share_temporal_encoder=share_temporal_encoder,
            use_surface_image_encoder=use_surface_image_encoder,
            image_channels=image_channels,
            surface_image_embed_dim=surface_image_embed_dim,
            with_critic=False,
        )
        self.actor = _actor_body(**body_kwargs)
        self.actor_t = _actor_body(**body_kwargs)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1 = _QNet(obs_dim, action_dim, hidden)
        self.q2 = _QNet(obs_dim, action_dim, hidden)
        self.q1_t = _QNet(obs_dim, action_dim, hidden)
        self.q2_t = _QNet(obs_dim, action_dim, hidden)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.opt = torch.optim.Adam(
            list(self.actor.parameters())
            + list(self.q1.parameters())
            + list(self.q2.parameters()),
            lr=lr,
        )
        self._step = 0

    def checkpoint_state(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_t": self.actor_t.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_t": self.q1_t.state_dict(),
            "q2_t": self.q2_t.state_dict(),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_t.load_state_dict(state["actor_t"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_t.load_state_dict(state["q1_t"])
        self.q2_t.load_state_dict(state["q2_t"])

    def _head(self, raw: torch.Tensor) -> torch.Tensor:
        base = getattr(self, "_last_w_base", None)
        return _apply_weight_head(
            raw,
            self.weight_head,
            tilt_gain=self.weight_head_tilt_gain,
            w_base=base,
        )

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        raw = self.actor(obs)
        if not deterministic:
            raw = raw + self.policy_noise * torch.randn_like(raw)
        return self._head(raw)

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-0.5, 0.5)
            next_a = self._head(self.actor_t(next_obs) + noise)
            q_t = torch.min(
                self.q1_t(next_obs, next_a), self.q2_t(next_obs, next_a)
            )
            target = rewards + (1.0 - dones) * self.gamma * q_t

        q_loss = F.mse_loss(self.q1(obs, actions), target) + F.mse_loss(
            self.q2(obs, actions), target
        )
        actor_loss = -self.q1(obs, self._head(self.actor(obs))).mean()
        loss = q_loss + actor_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for src, dst in (
                (self.actor, self.actor_t),
                (self.q1, self.q1_t),
                (self.q2, self.q2_t),
            ):
                for p, pt in zip(src.parameters(), dst.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        self._step += 1
        return {
            "loss": float(loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "policy_loss": float(actor_loss.detach()),
            "q_loss": float(q_loss.detach()),
        }


class DDPGAgent(_BaseAgent):
    """Lillicrap 2015 DDPG: deterministic actor, single critic, soft target updates.

    A strict simplification of :class:`TD3Agent` (no twin critics, no
    delayed policy updates, no target-policy smoothing noise) -- DDPG
    predates all three TD3 fixes. Honours cfg ``weight_head`` (default
    ``tanh_l1``) via the shared ``_apply_weight_head`` contract (B-DDPG fix).
    """

    name = "ddpg"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        explore_noise: float = 0.1,
        hidden: int = 64,
        weight_head: str = "tanh_l1",
        weight_head_tilt_gain: float = 1.0,
        architecture: str = "mlp",
        num_assets: int | None = None,
        d_model: int | None = None,
        seq_len: int = 1,
        d_state: int = 16,
        share_temporal_encoder: bool = True,
        use_surface_image_encoder: bool = False,
        image_channels: int = 0,
        surface_image_embed_dim: int = 16,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.explore_noise = float(explore_noise)
        head = str(weight_head or "tanh_l1").lower()
        if head not in _WEIGHT_HEADS:
            raise ValueError(f"unknown weight_head={weight_head!r}")
        self.weight_head = head
        self.weight_head_tilt_gain = float(weight_head_tilt_gain)
        self.action_law = head if head.startswith("dirichlet") else "gaussian_weight_head"
        self.architecture = str(architecture or "mlp").lower()
        body_kwargs = dict(
            architecture=self.architecture,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=hidden,
            num_assets=num_assets,
            d_model=d_model,
            seq_len=seq_len,
            d_state=d_state,
            share_temporal_encoder=share_temporal_encoder,
            use_surface_image_encoder=use_surface_image_encoder,
            image_channels=image_channels,
            surface_image_embed_dim=surface_image_embed_dim,
            with_critic=False,
        )
        self.actor = _actor_body(**body_kwargs)
        self.actor_t = _actor_body(**body_kwargs)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q = _QNet(obs_dim, action_dim, hidden)
        self.q_t = _QNet(obs_dim, action_dim, hidden)
        self.q_t.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.q.parameters()), lr=lr
        )

    def _head(self, raw: torch.Tensor) -> torch.Tensor:
        base = getattr(self, "_last_w_base", None)
        return _apply_weight_head(
            raw, self.weight_head, tilt_gain=self.weight_head_tilt_gain, w_base=base
        )

    def checkpoint_state(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_t": self.actor_t.state_dict(),
            "q": self.q.state_dict(),
            "q_t": self.q_t.state_dict(),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_t.load_state_dict(state["actor_t"])
        self.q.load_state_dict(state["q"])
        self.q_t.load_state_dict(state["q_t"])

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        raw = self.actor(obs)
        if not deterministic:
            raw = raw + self.explore_noise * torch.randn_like(raw)
        return self._head(raw)

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            next_a = self._head(self.actor_t(next_obs))
            q_t = self.q_t(next_obs, next_a)
            target = rewards + (1.0 - dones) * self.gamma * q_t

        q_loss = F.mse_loss(self.q(obs, actions), target)
        actor_loss = -self.q(obs, self._head(self.actor(obs))).mean()
        loss = q_loss + actor_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for src, dst in ((self.actor, self.actor_t), (self.q, self.q_t)):
                for p, pt in zip(src.parameters(), dst.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return {
            "loss": float(loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "policy_loss": float(actor_loss.detach()),
            "q_loss": float(q_loss.detach()),
        }


class MCPGAgent(_BaseAgent):
    """Monte Carlo Policy Gradient (REINFORCE with a learned baseline).

    Distinct from :class:`PPOAgent`: a single unclipped on-policy gradient
    step per call (no importance ratio, no minibatch epochs), and the
    training signal is the undiscounted-bootstrap Monte Carlo advantage
    (``compute_gae`` with ``gae_lambda=1.0``, i.e. a full backward
    cumulative sum of rewards within each episode against a value
    baseline) rather than PPO's clipped, lambda-blended GAE (C4).
    """

    name = "mcpg"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        hidden: int = 64,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        weight_head: str = "tanh_l1",
        architecture: str = "mlp",
        num_assets: int | None = None,
        d_model: int | None = None,
        seq_len: int = 1,
        d_state: int = 16,
        share_temporal_encoder: bool = True,
        use_surface_image_encoder: bool = False,
        image_channels: int = 0,
        surface_image_embed_dim: int = 16,
        actor_final_gain: float = 0.1,
        weight_head_temperature: float = 1.0,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        head = str(weight_head or "tanh_l1").lower()
        if head not in _WEIGHT_HEADS:
            raise ValueError(f"unknown weight_head={weight_head!r}")
        self.weight_head = head
        self.weight_head_temperature = float(weight_head_temperature)
        self.architecture = str(architecture or "mlp").lower()
        if self.architecture == "mlp":
            self.net: nn.Module = _ActorCritic(
                obs_dim, action_dim, hidden, actor_final_gain=actor_final_gain
            )
        else:
            if num_assets is None or d_model is None:
                raise ValueError(
                    f"architecture={architecture!r} requires num_assets and "
                    "d_model (needs use_equity_feature_cube=true so the "
                    "asset-major observation layout is known)"
                )
            self.net = _AssetTemporalActorCritic(
                num_assets=int(num_assets),
                d_model=int(d_model),
                action_dim=action_dim,
                seq_len=int(seq_len),
                d_state=int(d_state),
                temporal_backend=self.architecture,
                share_temporal_encoder=bool(share_temporal_encoder),
                use_surface_image_encoder=bool(use_surface_image_encoder),
                image_channels=int(image_channels),
                surface_image_embed_dim=int(surface_image_embed_dim),
            )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def raw_to_weights(self, raw: torch.Tensor) -> torch.Tensor:
        return _apply_weight_head(
            raw, self.weight_head, temperature=self.weight_head_temperature
        )

    def _raw_to_weights(self, raw: torch.Tensor) -> torch.Tensor:
        return self.raw_to_weights(raw)

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net.mean(obs)
        std = torch.clamp(self.net.log_std, -5.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act_and_logp_raw(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample pre-head logits so research train stores raw actions, not weights."""
        dist = self._dist(obs)
        raw = dist.mean if deterministic else dist.rsample()
        logp = dist.log_prob(raw).sum(dim=-1)
        return raw, logp

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        raw, _ = self.act_and_logp_raw(obs, deterministic=deterministic)
        return self.raw_to_weights(raw)

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        values = self.net.value(obs)
        with torch.no_grad():
            next_values = self.net.value(next_obs)
            _, mc_returns = compute_gae(
                rewards, values.detach(), next_values, dones, gamma=self.gamma, gae_lambda=1.0
            )
        dist = self._dist(obs)
        logp = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        advantage = (mc_returns - values).detach()
        actor_loss = -(logp * advantage).mean()
        critic_loss = F.mse_loss(values, mc_returns)
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        )
        self.opt.step()
        return {
            "loss": float(loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "policy_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "entropy": float(entropy.detach()),
            "grad_norm": grad_norm,
            "policy_grad_norm": grad_norm,
        }


class RRLAgent(_BaseAgent):
    """Moody & Saffell (2001) Recurrent/Direct Reinforcement Learning.

    The original RRL trades a single instrument with a fully differentiable
    position-recurrence, so ``dU/dtheta`` has a closed form. That closed
    form needs the environment's per-step P&L to be a *known differentiable
    function of the position* -- information the generic ``_BaseAgent``
    interface (pre-computed scalar ``rewards``) does not expose. This
    adapter keeps RRL's two defining properties instead: (1) **no value
    baseline / critic** (a pure direct-policy method, unlike PPO/MCPG), and
    (2) the training signal is Moody's **online differential Sharpe ratio**
    of realized rewards (:class:`src.eval.differential_sharpe.DifferentialSharpe`),
    not raw or discounted reward -- via the REINFORCE score-function
    identity ``E[D_t * grad log pi(a_t|s_t)]`` in place of the paper's
    closed-form gradient (C4; documented approximation, not silent).
    """

    name = "rrl"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        hidden: int = 64,
        eta: float = 0.01,
        max_grad_norm: float = 0.5,
        weight_head: str = "tanh_l1",
        architecture: str = "mlp",
        num_assets: int | None = None,
        d_model: int | None = None,
        seq_len: int = 1,
        d_state: int = 16,
        share_temporal_encoder: bool = True,
        use_surface_image_encoder: bool = False,
        image_channels: int = 0,
        surface_image_embed_dim: int = 16,
        actor_final_gain: float = 0.1,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.eta = float(eta)
        self.max_grad_norm = float(max_grad_norm)
        head = str(weight_head or "tanh_l1").lower()
        if head not in _WEIGHT_HEADS:
            raise ValueError(f"unknown weight_head={weight_head!r}")
        self.weight_head = head
        self.architecture = str(architecture or "mlp").lower()
        self.actor = _actor_body(
            architecture=self.architecture,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=hidden,
            num_assets=num_assets,
            d_model=d_model,
            seq_len=seq_len,
            d_state=d_state,
            share_temporal_encoder=share_temporal_encoder,
            use_surface_image_encoder=use_surface_image_encoder,
            image_channels=image_channels,
            surface_image_embed_dim=surface_image_embed_dim,
            with_critic=False,
            actor_final_gain=actor_final_gain,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=lr)

    def checkpoint_state(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "log_std": self.log_std.detach().clone(),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.log_std.data.copy_(state["log_std"])

    def raw_to_weights(self, raw: torch.Tensor) -> torch.Tensor:
        return _apply_weight_head(raw, self.weight_head)

    def _raw_to_weights(self, raw: torch.Tensor) -> torch.Tensor:
        return self.raw_to_weights(raw)

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        std = torch.clamp(self.log_std, -5.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act_and_logp_raw(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the pre-head Gaussian sample (B-RRL fix)."""
        dist = self._dist(obs)
        raw = dist.mean if deterministic else dist.rsample()
        logp = dist.log_prob(raw).sum(dim=-1)
        return raw, logp

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        raw, _ = self.act_and_logp_raw(obs, deterministic=deterministic)
        return self.raw_to_weights(raw)

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        from mascotrl.eval.differential_sharpe import DifferentialSharpe

        ds = DifferentialSharpe(eta=self.eta)
        signal = torch.zeros_like(rewards)
        r_np = rewards.detach().cpu().numpy()
        d_np = dones.detach().cpu().numpy()
        for i in range(r_np.shape[0]):
            if i > 0 and d_np[i - 1] > 0.5:
                ds = DifferentialSharpe(eta=self.eta)
            signal[i] = ds.step(float(r_np[i]))
        if float(signal.std(unbiased=False)) > 1e-8:
            signal = (signal - signal.mean()) / (signal.std(unbiased=False) + 1e-8)

        dist = self._dist(obs)
        logp = dist.log_prob(actions).sum(dim=-1)
        actor_loss = -(logp * signal).mean()
        self.opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + [self.log_std], self.max_grad_norm
            )
        )
        self.opt.step()
        return {
            "loss": float(actor_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "policy_loss": float(actor_loss.detach()),
            "mean_diff_sharpe": float(signal.mean().detach()),
            "grad_norm": grad_norm,
            "policy_grad_norm": grad_norm,
        }


class DQNAgent(_BaseAgent):
    """Independent per-asset discrete-position DQN (Zhang 2020 style).

    The action space is factored: each of the ``action_dim`` assets
    independently chooses a discrete position level from ``levels``
    (default ``(-1, 0, 1)`` -- short/flat/long), and the Q-network emits
    ``action_dim * len(levels)`` values reshaped to ``(K, n_levels)``. This
    is the standard independent-Q-learning simplification for a
    multi-dimensional discrete action space (no joint ``|levels|^K`` table).
    ``act()`` returns the chosen levels L1-normalized into a portfolio
    weight vector; ``actions`` passed to ``train_epoch`` are the raw level
    *values* (not L1-normalized), so the TD update can recover which
    discrete level (hence which Q-head) was actually taken (C4).
    """

    name = "dqn"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden: int = 64,
        levels: tuple[float, ...] = (-1.0, 0.0, 1.0),
        epsilon: float = 0.1,
        target_update_every: int = 100,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.levels = tuple(float(v) for v in levels)
        self.n_levels = len(self.levels)
        self.epsilon = float(epsilon)
        self.target_update_every = max(1, int(target_update_every))
        self.q = _mlp(obs_dim, action_dim * self.n_levels, hidden)
        self.q_t = _mlp(obs_dim, action_dim * self.n_levels, hidden)
        self.q_t.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self._levels_t = torch.tensor(self.levels, dtype=torch.float32)
        self._updates = 0

    def _q_values(self, net: nn.Module, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        return net(obs).view(b, self.action_dim, self.n_levels)

    def act(self, obs: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        with torch.no_grad():
            qv = self._q_values(self.q, obs)
            idx = qv.argmax(dim=-1)
            if not deterministic and self.epsilon > 0.0:
                rand_mask = torch.rand(idx.shape) < self.epsilon
                rand_idx = torch.randint(0, self.n_levels, idx.shape)
                idx = torch.where(rand_mask, rand_idx, idx)
            levels = self._levels_t[idx]
        denom = levels.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return levels / denom

    def act_and_logp_raw(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (raw discrete levels, logp=0) for the research collect path.

        ``train_epoch`` expects raw level values (e.g. -1/0/1), not L1 weights.
        """
        with torch.no_grad():
            qv = self._q_values(self.q, obs)
            idx = qv.argmax(dim=-1)
            if not deterministic and self.epsilon > 0.0:
                rand_mask = torch.rand(idx.shape) < self.epsilon
                rand_idx = torch.randint(0, self.n_levels, idx.shape)
                idx = torch.where(rand_mask, rand_idx, idx)
            levels = self._levels_t[idx]
        logp = torch.zeros(levels.shape[0], dtype=torch.float32)
        return levels, logp

    def raw_to_weights(self, raw: torch.Tensor) -> torch.Tensor:
        """L1-normalize raw discrete levels into a portfolio weight vector."""
        denom = raw.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        w = raw / denom
        zero_mask = raw.abs().sum(dim=-1) < 1e-8
        w = w.clone()
        w[zero_mask] = 0.0
        return w

    def _levels_to_index(self, actions: torch.Tensor) -> torch.Tensor:
        # actions holds raw level values (e.g. -1/0/1); snap to nearest level.
        diffs = (actions.unsqueeze(-1) - self._levels_t.view(1, 1, -1)).abs()
        return diffs.argmin(dim=-1)

    def train_epoch(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        idx = self._levels_to_index(actions)
        qv = self._q_values(self.q, obs)
        q_sel = qv.gather(-1, idx.unsqueeze(-1)).squeeze(-1).mean(dim=-1)
        with torch.no_grad():
            next_qv = self._q_values(self.q_t, next_obs)
            next_max = next_qv.max(dim=-1).values.mean(dim=-1)
            target = rewards + (1.0 - dones) * self.gamma * next_max
        loss = F.mse_loss(q_sel, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self._updates += 1
        if self._updates % self.target_update_every == 0:
            self.q_t.load_state_dict(self.q.state_dict())
        return {
            "loss": float(loss.detach()),
            "q_loss": float(loss.detach()),
            "mean_q": float(q_sel.mean().detach()),
        }


_REGISTRY: dict[str, type[_BaseAgent]] = {
    "ppo": PPOAgent,
    "sac": SACAgent,
    "td3": TD3Agent,
    "ddpg": DDPGAgent,
    "mcpg": MCPGAgent,
    "rrl": RRLAgent,
    "dqn": DQNAgent,
}
