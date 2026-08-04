"""P5: OLPS stub disclosure honesty for gate3 peer counting."""
from __future__ import annotations

import numpy as np

from src.eval.kahn_breadth import kahn_pack
from src.eval.olps import (
    olps_claim_names,
    olps_stub_names,
    olps_weights,
)
from src.eval.spectrum_gates import compute_gate3


_EXPECTED_STUBS = frozenset({"corn", "bnn", "ons", "anticor", "cwmr", "rmr"})


def test_olps_stub_names_are_documented_eg_fallbacks() -> None:
    stubs = olps_stub_names()
    assert stubs == _EXPECTED_STUBS
    # Stubs are not claim peers.
    claims = set(olps_claim_names())
    assert stubs.isdisjoint(claims)
    # Real algorithms remain claimable.
    for name in ("bah", "eg", "up", "olmar", "pamr", "best_stock"):
        assert name in claims


def test_stub_results_stamp_olps_stub_fallback() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.02, size=(30, 4))
    for stub in olps_stub_names():
        w, meta = olps_weights(stub, r, return_meta=True)
        assert isinstance(w, np.ndarray)
        assert meta.get("olps_stub_fallback") is True
        assert meta.get("stub") is True
        assert meta.get("fallback") == "eg"


def test_gate3_excludes_stub_olps_from_peer_counts() -> None:
    # Stub clones of EG must not inflate n_baselines / n_beaten.
    peers = {
        "equal_weight": 0.4,
        "olps:eg": 0.5,
        "olps:pamr": 0.55,
        "olps:corn": 0.1,  # stub EG fallback — exclude
        "olps:ons": 0.1,  # stub — exclude
        "olps:anticor": 0.1,
        "ceiling:kelly_cnn": 0.45,
    }
    out = compute_gate3(0.6, peers)
    assert "olps:corn" not in out["baselines"]
    assert "olps:ons" not in out["baselines"]
    assert "olps:anticor" not in out["baselines"]
    assert "olps:eg" in out["baselines"]
    assert "olps:pamr" in out["baselines"]
    assert out["n_baselines"] == 4
    # Beats every non-stub peer (0.6 > 0.55 best).
    assert out["n_beaten"] == 4
    assert out["pass"] is True
    assert out["best_baseline"] == "olps:pamr"


def test_gate3_bare_stub_names_also_excluded() -> None:
    out = compute_gate3(1.0, {"eg": 0.5, "corn": 0.2, "bnn": 0.2})
    assert out["n_baselines"] == 1
    assert set(out["baselines"]) == {"eg"}


def test_kahn_refuse_is_not_numeric_breadth_win() -> None:
    """refused_until_panel_returns must not be treated as a breadth claim win."""
    pack = kahn_pack(
        np.zeros((0, 0)),
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
        factor_alpha_positive=True,  # would otherwise allow scale; refuse overrides
        saturation_flag=False,
        k=10,
        ic_after_cost=0.05,
    )
    assert pack["status"] == "refused_until_panel_returns"
    assert "predicted_ir" not in pack
    assert pack["k_scale_claim_allowed"] is False
    breadth_claim = pack.get("status") == "ok" and bool(pack.get("k_scale_claim_allowed"))
    assert breadth_claim is False
