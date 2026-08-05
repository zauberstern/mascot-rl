"""Unified raw-state encoder for spectrum arms (real features preferred).

Train/eval ``iv_feat`` parity
-----------------------------
Both the CMDP synth path (``CMDPEnv._build_raw_states``) and historical OOS /
finetune paths must build observations via the shared helpers here
(``build_raw_states_from_feature_tensor`` / ``build_raw_states``). Do **not**
re-implement pad/truncate or sinusoid expansion inline in train vs eval — a
diverging ``iv_feat`` channel (last ATM column vs engineered vector) silently
breaks DHGNN incidence and temporal-backend ablations. Prefer calling these
builders; pass the same ``d_model`` and feature tensor layout on both sides.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from mascotrl.arms.spec import ArmSpec

FeatureEncoder = Literal["pad_truncate", "legacy_sinusoid"]


def encode_scalar_series_legacy_sinusoid(base_2d: np.ndarray, *, d_model: int) -> np.ndarray:
    """Sinusoidal IV-style encoding: ``(K, seq)`` → ``(K, seq, d_model)``.

    Ablation-only. Expands one scalar via deterministic sinusoids (rank-1).
    Do not use on the production multi-channel path.
    """
    iv = np.nan_to_num(np.asarray(base_2d, dtype=np.float64), nan=0.0)
    if iv.ndim != 2:
        raise ValueError(f"expected (K, seq), got shape {iv.shape}")
    k, seq = iv.shape
    t_idx = np.arange(seq, dtype=np.float32)[None, :, None]  # (1, seq, 1)
    k_idx = np.arange(k, dtype=np.float32)[:, None, None]  # (K, 1, 1)
    base = iv.astype(np.float32)[:, :, None]
    feats = [base]
    for i in range(1, int(d_model)):
        feats.append(base * np.sin(0.1 * i * t_idx + 0.01 * i * k_idx))
    raw = np.concatenate(feats, axis=-1)[..., : int(d_model)]
    return raw.astype(np.float32, copy=False)


# Back-compat alias for any external callers still using the private name.
_encode_scalar_series = encode_scalar_series_legacy_sinusoid


def build_raw_states_from_feature_tensor(feat: np.ndarray, *, d_model: int) -> np.ndarray:
    """Map real feature tensor ``(K, seq, C)`` → ``(K, seq, d_model)``.

    Concatenate channels then pad with zeros if ``C < d_model``, or truncate to
    the first ``d_model`` channels if ``C > d_model``. No sinusoid expansion.
    """
    x = np.nan_to_num(np.asarray(feat, dtype=np.float64), nan=0.0)
    if x.ndim != 3:
        raise ValueError(f"expected (K, seq, C), got shape {x.shape}")
    k, seq, c = x.shape
    d = int(d_model)
    if d <= 0:
        raise ValueError(f"d_model must be positive, got {d}")
    out = np.zeros((k, seq, d), dtype=np.float32)
    n = min(c, d)
    if n > 0:
        out[:, :, :n] = x[:, :, :n].astype(np.float32, copy=False)
    return out


def _equity_base_series(
    equity_feat_hist: np.ndarray | None,
    atm_iv_hist: np.ndarray,
    *,
    n_eq: int,
) -> np.ndarray:
    """Resolve equity-slot scalar series ``(n_eq, seq)`` for the legacy encoder."""
    if equity_feat_hist is not None:
        eq = np.asarray(equity_feat_hist, dtype=np.float64)
        if eq.ndim == 3:
            if eq.shape[0] != n_eq:
                raise ValueError(
                    f"equity_feat_hist K={eq.shape[0]} != equity_slots={n_eq}"
                )
            return eq[:, :, 0]
        if eq.ndim == 2:
            if eq.shape[0] != n_eq:
                raise ValueError(
                    f"equity_feat_hist K={eq.shape[0]} != equity_slots={n_eq}"
                )
            return eq
        raise ValueError(f"equity_feat_hist must be 2-D or 3-D, got {eq.shape}")

    atm = np.asarray(atm_iv_hist, dtype=np.float64)
    if atm.ndim != 2:
        raise ValueError(f"atm_iv_hist must be (K, seq), got {atm.shape}")
    if atm.shape[0] == n_eq:
        return atm
    if atm.shape[0] > n_eq:
        return atm[-n_eq:]
    raise ValueError(
        f"atm_iv_hist K={atm.shape[0]} insufficient for equity_slots={n_eq}"
    )


def _as_feature_tensor(
    hist: np.ndarray,
    *,
    n_slots: int | None = None,
) -> np.ndarray:
    """Promote 2-D ``(K, seq)`` or 3-D ``(K, seq, C)`` to a feature tensor."""
    x = np.asarray(hist, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3:
        raise ValueError(f"feature hist must be 2-D or 3-D, got {x.shape}")
    if n_slots is not None and x.shape[0] != n_slots:
        raise ValueError(f"feature hist K={x.shape[0]} != slots={n_slots}")
    return x


def _encode_path(
    *,
    scalar_2d: np.ndarray | None,
    feature_3d: np.ndarray | None,
    d_model: int,
    feature_encoder: FeatureEncoder,
) -> np.ndarray:
    if feature_encoder == "legacy_sinusoid":
        if scalar_2d is None:
            raise ValueError("legacy_sinusoid requires a scalar (K, seq) series")
        return encode_scalar_series_legacy_sinusoid(scalar_2d, d_model=d_model)
    if feature_3d is not None:
        return build_raw_states_from_feature_tensor(feature_3d, d_model=d_model)
    if scalar_2d is not None:
        # Ablation / backward-compat: scalar ATM-only → legacy sinusoid.
        return encode_scalar_series_legacy_sinusoid(scalar_2d, d_model=d_model)
    raise ValueError("no scalar or feature tensor provided for encoding")


def build_raw_states(
    atm_iv_hist: np.ndarray,
    *,
    d_model: int,
    arm: "ArmSpec | None" = None,
    equity_feat_hist: np.ndarray | None = None,
    feature_hist: np.ndarray | None = None,
    feature_encoder: FeatureEncoder | str = "pad_truncate",
    forbid_atm_equity_proxy: bool = False,
) -> np.ndarray:
    """Build slot-major raw states ``(n_slots, seq, d_model)``.

    Prefer real multi-channel ``feature_hist`` ``(K, seq, C)`` with pad/truncate
    to ``d_model``. Legacy sinusoidal expansion is retained for ablation when
    ``feature_encoder='legacy_sinusoid'`` or when only a scalar ATM series is
    available (no ``feature_hist`` / multi-channel equity features).

    For ``arm.id == "eq"`` with ``forbid_atm_equity_proxy=True`` (Alpha v2),
    ``equity_feat_hist`` or ``feature_hist`` is required — ATM/IV proxies banned.
    """
    atm = np.asarray(atm_iv_hist, dtype=np.float64)
    if atm.ndim != 2:
        raise ValueError(f"atm_iv_hist must be (K, seq), got {atm.shape}")

    enc: FeatureEncoder
    if feature_encoder in ("pad_truncate", "legacy_sinusoid"):
        enc = feature_encoder  # type: ignore[assignment]
    else:
        raise ValueError(
            f"unknown feature_encoder={feature_encoder!r}; "
            "expected 'pad_truncate' or 'legacy_sinusoid'"
        )

    force_legacy = enc == "legacy_sinusoid"

    if arm is None or arm.id == "opt":
        feat = None if force_legacy else (
            _as_feature_tensor(feature_hist) if feature_hist is not None else None
        )
        return _encode_path(
            scalar_2d=atm,
            feature_3d=feat,
            d_model=d_model,
            feature_encoder="legacy_sinusoid" if (force_legacy or feat is None) else "pad_truncate",
        )

    if arm.id == "eq":
        n_eq = int(arm.equity_slots)
        has_eq = equity_feat_hist is not None or feature_hist is not None
        if forbid_atm_equity_proxy and not has_eq:
            raise ValueError(
                "eq arm Alpha v2 forbids ATM/IV equity proxy; pass equity_feat_hist"
            )
        if force_legacy:
            base = _equity_base_series(equity_feat_hist, atm, n_eq=n_eq)
            return encode_scalar_series_legacy_sinusoid(base, d_model=d_model)
        # Production: real tensor path when multi-channel / equity features exist.
        if feature_hist is not None:
            feat = _as_feature_tensor(feature_hist, n_slots=n_eq)
        elif equity_feat_hist is not None:
            feat = _as_feature_tensor(equity_feat_hist, n_slots=n_eq)
        else:
            # ATM proxy ablation path (explicitly allowed when forbid flag is off).
            return encode_scalar_series_legacy_sinusoid(
                _equity_base_series(None, atm, n_eq=n_eq), d_model=d_model
            )
        return build_raw_states_from_feature_tensor(feat, d_model=d_model)

    if arm.id == "mix":
        n_opt = int(arm.option_slots)
        n_eq = int(arm.equity_slots)
        if atm.shape[0] < n_opt:
            raise ValueError(
                f"atm_iv_hist K={atm.shape[0]} < option_slots={n_opt}"
            )
        opt_base = atm[:n_opt]
        if equity_feat_hist is None and feature_hist is None and forbid_atm_equity_proxy:
            raise ValueError(
                "mix arm Alpha v2 forbids ATM equity proxy; pass equity_feat_hist"
            )
        if force_legacy:
            if equity_feat_hist is not None:
                eq_proxy = atm
            elif atm.shape[0] >= n_opt + n_eq:
                eq_proxy = atm[n_opt : n_opt + n_eq]
            else:
                eq_proxy = opt_base if opt_base.shape[0] == n_eq else atm[:n_eq]
            eq_base = _equity_base_series(equity_feat_hist, eq_proxy, n_eq=n_eq)
            opt_raw = encode_scalar_series_legacy_sinusoid(opt_base, d_model=d_model)
            eq_raw = encode_scalar_series_legacy_sinusoid(eq_base, d_model=d_model)
            return np.concatenate([opt_raw, eq_raw], axis=0)

        # Opt block: feature_hist for all slots, or ATM-only legacy for options.
        if feature_hist is not None:
            feat_all = _as_feature_tensor(feature_hist)
            if feat_all.shape[0] < n_opt + n_eq:
                raise ValueError(
                    f"feature_hist K={feat_all.shape[0]} < mix slots={n_opt + n_eq}"
                )
            opt_raw = build_raw_states_from_feature_tensor(
                feat_all[:n_opt], d_model=d_model
            )
            eq_raw = build_raw_states_from_feature_tensor(
                feat_all[n_opt : n_opt + n_eq], d_model=d_model
            )
            return np.concatenate([opt_raw, eq_raw], axis=0)

        opt_raw = encode_scalar_series_legacy_sinusoid(opt_base, d_model=d_model)
        if equity_feat_hist is not None:
            eq_feat = _as_feature_tensor(equity_feat_hist, n_slots=n_eq)
            eq_raw = build_raw_states_from_feature_tensor(eq_feat, d_model=d_model)
        else:
            if atm.shape[0] >= n_opt + n_eq:
                eq_proxy = atm[n_opt : n_opt + n_eq]
            else:
                eq_proxy = opt_base if opt_base.shape[0] == n_eq else atm[:n_eq]
            eq_raw = encode_scalar_series_legacy_sinusoid(eq_proxy, d_model=d_model)
        return np.concatenate([opt_raw, eq_raw], axis=0)

    raise ValueError(f"unknown arm id={getattr(arm, 'id', arm)!r}")
