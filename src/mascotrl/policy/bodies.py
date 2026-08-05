"""Shared policy body construction for the spectrum architecture axis.

Decouples temporal encoders (gru/lstm/transformer/mamba) from the RL
algorithm so PPO and off-policy adapters can share one body builder.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


def _orthogonal_init(module: nn.Module, *, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class MLPPolicyBody(nn.Module):
    """Flat MLP trunk: ``forward(obs) -> action logits``."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 64,
        *,
        actor_final_gain: float = 0.01,
    ) -> None:
        super().__init__()
        self.net = _mlp(int(obs_dim), int(action_dim), int(hidden))
        self.net.apply(lambda m: _orthogonal_init(m, gain=actor_final_gain))
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=actor_final_gain)
            nn.init.zeros_(last.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class AssetTemporalPolicyBody(nn.Module):
    """Asset-major temporal body for non-MLP architecture axis values.

    Reshapes flat ``(B, K*seq*C)`` observations into ``(B, K, seq, C)`` and
    routes through ``AlphaFeatureExtractor`` backends. Exposes ``forward``
    (actor logits) and ``value`` for actor-critic algorithms.
    """

    def __init__(
        self,
        num_assets: int,
        d_model: int,
        action_dim: int,
        *,
        seq_len: int = 1,
        d_state: int = 16,
        temporal_backend: str = "gru",
        share_temporal_encoder: bool = True,
        use_surface_image_encoder: bool = False,
        image_channels: int = 0,
        surface_image_embed_dim: int = 16,
        with_critic: bool = True,
    ) -> None:
        super().__init__()
        from mascotrl.features.extractor import AlphaFeatureExtractor

        self.num_assets = int(num_assets)
        self.d_model = int(d_model)
        self.seq_len = max(1, int(seq_len))
        self.temporal_backend = str(temporal_backend).lower()
        self.use_surface_image_encoder = bool(use_surface_image_encoder)
        self.image_channels = int(image_channels) if self.use_surface_image_encoder else 0
        if self.use_surface_image_encoder:
            if self.image_channels != 11 * 34:
                raise ValueError(
                    "use_surface_image_encoder=true requires image_channels="
                    f"11*34=374, got {self.image_channels}"
                )
            if self.image_channels >= self.d_model:
                raise ValueError(
                    f"image_channels={self.image_channels} must be < "
                    f"d_model={self.d_model}"
                )
            from mascotrl.features.surface_cnn import SurfaceImageEncoder

            self.image_encoder = SurfaceImageEncoder(embed_dim=int(surface_image_embed_dim))
            self.base_d_model = self.d_model - self.image_channels
            extractor_d_model = self.base_d_model + int(surface_image_embed_dim)
        else:
            self.image_encoder = None
            self.base_d_model = self.d_model
            extractor_d_model = self.d_model
        self.extractor = AlphaFeatureExtractor(
            num_assets=self.num_assets,
            d_model=extractor_d_model,
            d_state=int(d_state),
            temporal_backend=self.temporal_backend,
            use_dhgnn=False,
            spatial_mode="none",
            share_temporal_encoder=bool(share_temporal_encoder),
        )
        flat = self.num_assets * extractor_d_model
        self.actor_head = nn.Linear(flat, int(action_dim))
        _orthogonal_init(self.actor_head, gain=0.01)
        self.with_critic = bool(with_critic)
        if self.with_critic:
            self.critic_head = nn.Linear(flat, 1)
            _orthogonal_init(self.critic_head, gain=1.0)
        else:
            self.critic_head = None

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        expected = self.num_assets * self.seq_len * self.d_model
        if obs.shape[-1] != expected:
            raise ValueError(
                f"obs last-dim {obs.shape[-1]} != num_assets*seq_len*d_model "
                f"({self.num_assets}*{self.seq_len}*{self.d_model}={expected}); "
                "architecture != 'mlp' requires use_equity_feature_cube=true "
                "so the asset-major layout is well defined"
            )
        x = obs.reshape(b, self.num_assets, self.seq_len, self.d_model)
        if self.use_surface_image_encoder:
            base = x[..., : self.base_d_model]
            img = x[..., self.base_d_model :]
            img = img.reshape(b * self.num_assets * self.seq_len, 1, 11, 34)
            embed = self.image_encoder(img)
            embed = embed.reshape(b, self.num_assets, self.seq_len, -1)
            x = torch.cat([base, embed], dim=-1)
        z = self.extractor(x, x[:, :, -1, :])
        return z.reshape(b, -1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self._encode(obs))

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        if self.critic_head is None:
            raise RuntimeError("AssetTemporalPolicyBody built without critic")
        return self.critic_head(self._encode(obs)).squeeze(-1)


_TEMPORAL_BACKENDS = frozenset({"gru", "lstm", "transformer", "mamba"})


def build_policy_body(
    architecture: str,
    obs_spec: Mapping[str, Any],
    cfg: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build the shared policy trunk for ``architecture``.

    ``obs_spec`` keys:
      - ``obs_dim``, ``action_dim`` (required)
      - ``num_assets``, ``d_model``, ``seq_len`` (required for non-mlp)
      - ``hidden``, ``d_state``, ``share_temporal_encoder``,
        ``use_surface_image_encoder``, ``image_channels``,
        ``surface_image_embed_dim``, ``with_critic``, ``actor_final_gain``
    """
    cfg = dict(cfg or {})
    arch = str(architecture or "mlp").lower().strip()
    if arch == "mamba2":
        arch = "mamba"
    obs_dim = int(obs_spec["obs_dim"])
    action_dim = int(obs_spec["action_dim"])
    hidden = int(obs_spec.get("hidden") or cfg.get("ppo_hidden") or 64)
    if arch == "mlp":
        return MLPPolicyBody(
            obs_dim,
            action_dim,
            hidden,
            actor_final_gain=float(
                obs_spec.get("actor_final_gain")
                or cfg.get("actor_final_gain")
                or 0.01
            ),
        )
    if arch not in _TEMPORAL_BACKENDS:
        raise ValueError(
            f"unknown architecture={architecture!r}; "
            f"allowed=['mlp'] + {sorted(_TEMPORAL_BACKENDS)}"
        )
    num_assets = obs_spec.get("num_assets")
    d_model = obs_spec.get("d_model")
    if num_assets is None or d_model is None:
        raise ValueError(
            f"architecture={arch!r} requires obs_spec num_assets and d_model "
            "(use_equity_feature_cube=true)"
        )
    return AssetTemporalPolicyBody(
        num_assets=int(num_assets),
        d_model=int(d_model),
        action_dim=action_dim,
        seq_len=int(obs_spec.get("seq_len") or cfg.get("feature_seq_len") or 1),
        d_state=int(obs_spec.get("d_state") or cfg.get("d_state") or 16),
        temporal_backend=arch,
        share_temporal_encoder=bool(
            obs_spec.get(
                "share_temporal_encoder",
                cfg.get("share_temporal_encoder", True),
            )
        ),
        use_surface_image_encoder=bool(
            obs_spec.get(
                "use_surface_image_encoder",
                cfg.get("use_surface_image_encoder", False),
            )
        ),
        image_channels=int(obs_spec.get("image_channels") or 0),
        surface_image_embed_dim=int(
            obs_spec.get("surface_image_embed_dim")
            or cfg.get("surface_image_embed_dim")
            or 16
        ),
        with_critic=bool(obs_spec.get("with_critic", True)),
    )


def body_backend_name(body: nn.Module) -> str:
    """Return the temporal backend id for coverage / decoupling tests."""
    if isinstance(body, MLPPolicyBody):
        return "mlp"
    if isinstance(body, AssetTemporalPolicyBody):
        return body.temporal_backend
    raise TypeError(f"unknown policy body type: {type(body)!r}")
