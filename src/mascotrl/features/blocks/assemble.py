"""Assemble equity feature cube from returns (+ optional extras)."""
from __future__ import annotations

import numpy as np

from src.features.blocks.cross_section import apply_cross_section_normalize
from src.features.blocks.experimental import build_experimental_block
from src.features.blocks.fundamentals_pit import build_fundamentals_pit_block
from src.features.blocks.iv_surface import build_borrow_block, build_iv_surface_block
from src.features.blocks.jkp_lottery import build_jkp_lottery_block
from src.features.blocks.liquidity import build_liquidity_block
from src.features.blocks.microstructure import build_microstructure_block
from src.features.blocks.normalize import expanding_causal_zscore
from src.features.blocks.option_flow import build_option_flow_block
from src.features.blocks.range_volatility import build_range_volatility_block
from src.features.blocks.returns_momentum import build_returns_momentum_block, momentum_12_1
from src.features.blocks.sentiment import build_sentiment_block
from src.features.blocks.volatility_vrp import build_volatility_vrp_block
from src.features.groups import resolve_excludes


def _dedupe_channel_names(
    blocks: list[tuple[np.ndarray, list[str]]],
) -> list[tuple[np.ndarray, list[str]]]:
    """Fail-closed on duplicate names; rename borrow collision to borrow_rate_fee."""
    seen: set[str] = set()
    out: list[tuple[np.ndarray, list[str]]] = []
    for cube, names in blocks:
        new_names: list[str] = []
        for n in names:
            if n in seen:
                if n == "borrow_rate":
                    alt = "borrow_rate_fee"
                    if alt in seen:
                        raise ValueError(f"duplicate feature channel {n!r} (and {alt!r})")
                    new_names.append(alt)
                    seen.add(alt)
                else:
                    raise ValueError(f"duplicate feature channel {n!r}")
            else:
                new_names.append(n)
                seen.add(n)
        out.append((cube, new_names))
    return out


def assemble_equity_feature_cube(
    returns: np.ndarray,
    extras: dict | None = None,
    *,
    normalize: bool = True,
    exclude_channels: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build ``(T, K, C)`` equity-core cube from ``stk_ret`` alone (+ extras).

    ``extras`` may include legacy keys plus::
      - ``ohlc``, ``microstructure``, ``fundamentals_pit``, ``sentiment``,
        ``option_flow``, ``jkp``, ``gics_industry``
      - ``feature_groups_exclude`` / ``feature_channels_exclude`` (resolved
        if ``exclude_channels`` is None)
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"returns must be (T, K), got {r.shape}")
    extras = extras or {}

    macro_raw: np.ndarray | None = None
    macro_names: list[str] | None = None

    mom_block = build_returns_momentum_block(
        r,
        factors=(
            extras.get("factors")
            if extras.get("include_residual_momentum")
            else None
        ),
    )
    vol_block = build_volatility_vrp_block(r, iv=extras.get("iv"))
    liq_block = build_liquidity_block(r, dollar_volume=extras.get("dollar_volume"))
    blocks: list[tuple[np.ndarray, list[str]]] = [mom_block, vol_block, liq_block]

    if extras.get("ohlc") is not None:
        blocks.append(build_range_volatility_block(extras.get("ohlc")))
    if extras.get("microstructure") is not None:
        blocks.append(build_microstructure_block(extras.get("microstructure")))
    if extras.get("fundamentals_pit") is not None:
        blocks.append(build_fundamentals_pit_block(extras.get("fundamentals_pit")))
    if extras.get("sentiment") is not None:
        blocks.append(build_sentiment_block(extras.get("sentiment")))
    if extras.get("option_flow") is not None:
        blocks.append(build_option_flow_block(extras.get("option_flow")))
    if extras.get("jkp") is not None or extras.get("include_jkp_lottery"):
        blocks.append(
            build_jkp_lottery_block(
                r,
                jkp=extras.get("jkp"),
                factors=extras.get("factors"),
            )
        )

    if extras.get("macro") is not None:
        m = np.asarray(extras["macro"], dtype=np.float64)
        if m.ndim == 2:
            if m.shape[0] != r.shape[0]:
                raise ValueError(
                    f"macro (T,F) T={m.shape[0]} != returns T={r.shape[0]}"
                )
        elif m.ndim == 3:
            if m.shape[0] != r.shape[0] or m.shape[1] != r.shape[1]:
                raise ValueError(
                    f"macro (T,K,F)={m.shape} != returns (T,K)={r.shape}"
                )
        else:
            raise ValueError(f"macro must be (T,F) or (T,K,F), got {m.shape}")
        names = extras.get("macro_names")
        if names is None:
            names = [f"macro_{i}" for i in range(int(m.shape[-1]))]
        else:
            names = list(names)
            if len(names) != int(m.shape[-1]):
                names = [f"macro_{i}" for i in range(int(m.shape[-1]))]
        macro_raw = m
        macro_names = names
    if extras.get("include_iv_surface") or extras.get("iv_surface") is not None:
        blocks.append(
            build_iv_surface_block(
                r,
                extras.get("iv_surface"),
                channel_names=extras.get("iv_surface_names"),
            )
        )
    if extras.get("include_borrow") or extras.get("borrow") is not None:
        blocks.append(build_borrow_block(r, extras.get("borrow")))
    if extras.get("include_fundamentals"):
        if extras.get("fundamentals") is None:
            raise ValueError(
                "include_fundamentals=true but extras['fundamentals'] is "
                "None; prefer extras['fundamentals_pit'] from "
                "feature_panels, or pass a real (T,K)/(T,K,C) array."
            )
        fund = np.asarray(extras["fundamentals"], dtype=np.float64)
        if fund.ndim == 2:
            fund = fund[..., None]
        names = [f"fund_{i}" for i in range(int(fund.shape[-1]))]
        blocks.append((np.nan_to_num(fund, nan=0.0), names))
    if extras.get("include_surface_image_encoder"):
        if extras.get("kelly_images") is None:
            raise ValueError(
                "include_surface_image_encoder=true but extras['kelly_images'] "
                "is None; materialize (T,K,11,34) Kelly IV images first "
                "or set include_surface_image_encoder=false."
            )
        img = np.asarray(extras["kelly_images"], dtype=np.float64)
        if img.ndim != 4 or img.shape[0] != r.shape[0] or img.shape[1] != r.shape[1]:
            raise ValueError(
                f"kelly_images must be (T, K, 11, 34) matching returns "
                f"{r.shape}, got {img.shape}"
            )
        img_flat = img.reshape(img.shape[0], img.shape[1], -1)
        names = [f"kelly_px_{i}" for i in range(int(img_flat.shape[-1]))]
        blocks.append((img_flat, names))

    # Experimental last. Default on when any new-family extras are present
    # (full feature net); returns-only cubes stay unchanged for back-compat.
    want_exp = extras.get("include_experimental")
    if want_exp is None:
        want_exp = any(
            extras.get(k) is not None
            for k in (
                "ohlc",
                "microstructure",
                "sentiment",
                "option_flow",
                "fundamentals_pit",
                "jkp",
                "gics_industry",
            )
        )
    if want_exp:
        amihud = None
        if liq_block[1] and "amihud" in liq_block[1]:
            amihud = liq_block[0][..., liq_block[1].index("amihud")]
        iv_surf = extras.get("iv_surface")
        iv_map = iv_surf if isinstance(iv_surf, dict) else None
        exp_cube, exp_names = build_experimental_block(
            r,
            ohlc=extras.get("ohlc"),
            microstructure=extras.get("microstructure"),
            sentiment=extras.get("sentiment"),
            option_flow=extras.get("option_flow"),
            iv_surface=iv_map,
            gics_industry=extras.get("gics_industry"),
            mom_12_1=momentum_12_1(r),
            amihud=amihud,
        )
        if exp_cube.shape[-1] > 0:
            blocks.append((exp_cube, exp_names))

    blocks = [(c, n) for c, n in blocks if c.size > 0 and c.shape[-1] > 0]
    blocks = _dedupe_channel_names(blocks)

    cubes = [c for c, _ in blocks]
    names: list[str] = []
    for c, n in blocks:
        names.extend(n)
    if not cubes and macro_raw is None:
        t, k = r.shape
        return np.zeros((t, k, 0), dtype=np.float64), []
    cube = np.concatenate(cubes, axis=-1) if cubes else np.zeros(
        (r.shape[0], r.shape[1], 0), dtype=np.float64
    )
    slot_mask = extras.get("slot_valid_mask")
    if slot_mask is None:
        slot_mask = extras.get("slot_mask")
    if slot_mask is not None:
        sm = np.asarray(slot_mask, dtype=bool)
        if sm.shape == (r.shape[0], r.shape[1]):
            cube = np.asarray(cube, dtype=np.float64).copy()
            cube[~sm, :] = np.nan
    if normalize and cube.shape[-1] > 0:
        cube = apply_cross_section_normalize(cube)
    if macro_raw is not None:
        m = np.asarray(macro_raw, dtype=np.float64)
        if m.ndim == 2:
            m_tf = m
        else:
            m_tf = m[:, 0, :]
        m_z = expanding_causal_zscore(m_tf)
        m_block = np.broadcast_to(
            m_z[:, None, :], (r.shape[0], r.shape[1], m_z.shape[1])
        ).copy()
        cube = np.concatenate([cube, m_block], axis=-1)
        names.extend(list(macro_names or []))

    drop = exclude_channels
    if drop is None:
        drop = resolve_excludes(
            extras.get("feature_groups_exclude"),
            extras.get("feature_channels_exclude"),
            names,
        )
    if drop:
        keep_idx = [i for i, n in enumerate(names) if n not in drop]
        names = [names[i] for i in keep_idx]
        cube = cube[..., keep_idx] if keep_idx else cube[..., :0]
    return cube.astype(np.float64, copy=False), names
