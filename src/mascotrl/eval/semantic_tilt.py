"""Firm-text semantic book embeddings for behaviour interpretation only.

Never feeds capital gates. Text is a static 2026-08-18 lake snapshot, not PIT.
Do not import this module from obs_builder, feature_cube, or any train path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.paths import LAKE_ROOT
from src.data.surface_signals import _canonical_secid_key

FIRM_TEXT_ASOF = "2026-08-18"
FIRM_TEXT_PIT = False
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _REPO_ROOT / "logs" / "artifacts" / "semantic_tilt" / "firm_embeddings.parquet"
_EMPTY_MARKERS = frozenset({"", "nan", "none", "<na>", "nat"})
_TEXT_COLS = (
    "TR.BusinessSummary",
    "TR.BusinessSummary_p4",
    "TR.CommonName",
    "TR.CommonName_p4",
    "TR.InstrumentDescription",
    "TR.InstrumentDescription_p4",
)
_EPS = 1e-12


def nan_semantic_tilt(reason: str) -> dict[str, Any]:
    return {
        "semantic_rotation_rate": float("nan"),
        "semantic_pc1_mean": float("nan"),
        "semantic_pc2_mean": float("nan"),
        "semantic_pc3_mean": float("nan"),
        "data_availability_reason": str(reason or ""),
        "text_asof": FIRM_TEXT_ASOF,
        "text_pit": False,
        "embed_backend": None,
    }


def _is_empty_text(val: Any) -> bool:
    if val is None:
        return True
    try:
        if isinstance(val, float) and not np.isfinite(val):
            return True
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return not s or s.lower() in _EMPTY_MARKERS


def load_firm_text_map(lake_root: Path | str | None = None) -> dict[str, Any]:
    """Load static firm descriptions from lseg_ric_map (not PIT)."""
    lake = Path(lake_root) if lake_root is not None else Path(LAKE_ROOT)
    path = lake / "macro" / "lseg_ric_map.parquet"
    if not path.is_file():
        return {
            "texts": {},
            "asof": FIRM_TEXT_ASOF,
            "pit": False,
            "reason": "lseg_ric_map_missing",
            "n": 0,
            "source": "lseg_ric_map",
        }
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
    except Exception:
        available = set()
    wanted = ["secid", *_TEXT_COLS]
    cols = [c for c in wanted if c in available]
    if "secid" not in cols:
        return {
            "texts": {},
            "asof": FIRM_TEXT_ASOF,
            "pit": False,
            "reason": "secid_col_missing",
            "n": 0,
            "source": "lseg_ric_map",
        }
    df = pd.read_parquet(path, columns=cols)
    texts: dict[str, str] = {}
    for _, row in df.iterrows():
        sid = _canonical_secid_key(row.get("secid"))
        if sid in ("None", "") or sid in texts:
            # Prefer first non-empty; skip overwrite if already filled.
            if sid in texts and texts[sid]:
                continue
        chosen = ""
        for col in _TEXT_COLS:
            if col not in row.index:
                continue
            if not _is_empty_text(row[col]):
                chosen = str(row[col]).strip()
                break
        if chosen:
            texts[sid] = chosen
        elif sid not in texts:
            texts[sid] = ""

    # Optional issuer fallback for empty texts.
    secnmd = lake / "macro" / "om_secnmd.parquet"
    if secnmd.is_file():
        try:
            sn = pd.read_parquet(secnmd)
            if "secid" in sn.columns and "issuer" in sn.columns:
                sn = sn.copy()
                sn["secid"] = sn["secid"].map(_canonical_secid_key)
                if "effect_date" in sn.columns:
                    sn = sn.sort_values("effect_date")
                sn = sn.drop_duplicates("secid", keep="last")
                for _, row in sn.iterrows():
                    sid = str(row["secid"])
                    if texts.get(sid):
                        continue
                    if not _is_empty_text(row.get("issuer")):
                        texts[sid] = str(row["issuer"]).strip()
        except Exception:
            pass

    # Drop empty strings from the returned map.
    texts = {k: v for k, v in texts.items() if v}
    return {
        "texts": texts,
        "asof": FIRM_TEXT_ASOF,
        "pit": False,
        "n": len(texts),
        "source": "lseg_ric_map",
    }


def _embed_tfidf_svd(docs: Sequence[str]) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_docs = len(docs)
    if n_docs == 0:
        return np.zeros((0, 0), dtype=np.float64)
    vec = TfidfVectorizer(max_features=512, min_df=1, stop_words="english")
    X = vec.fit_transform(list(docs))
    n_comp = int(min(32, max(2, n_docs - 1)))
    if n_docs < 2:
        dense = np.asarray(X.toarray(), dtype=np.float64)
        return dense
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    return np.asarray(svd.fit_transform(X), dtype=np.float64)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64)
    if m.size == 0:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.clip(norms, _EPS, None)
    return m / norms


def _load_embed_cache(
    cache_path: Path, *, secids: Sequence[str], model_id: str
) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        df = pd.read_parquet(cache_path)
    except Exception:
        return None
    if "secid" not in df.columns or "model_id" not in df.columns:
        return None
    if set(df["model_id"].astype(str).unique()) != {str(model_id)}:
        return None
    cached = [str(s) for s in df["secid"].tolist()]
    if set(cached) != set(secids) or len(cached) != len(secids):
        return None
    ecols = [c for c in df.columns if str(c).startswith("e") and str(c)[1:].isdigit()]
    ecols = sorted(ecols, key=lambda c: int(str(c)[1:]))
    if not ecols:
        return None
    # Reorder to sorted(secids) order.
    df = df.set_index("secid").loc[list(secids)]
    mat = df[ecols].to_numpy(dtype=np.float64)
    backend = str(df["backend"].iloc[0]) if "backend" in df.columns else "cache"
    return {
        "secids": list(secids),
        "matrix": mat,
        "model_id": str(model_id),
        "backend": backend,
        "asof": FIRM_TEXT_ASOF,
        "pit": False,
    }


def _write_embed_cache(
    cache_path: Path,
    *,
    secids: Sequence[str],
    mat: np.ndarray,
    model_id: str,
    backend: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    d = int(mat.shape[1]) if mat.ndim == 2 else 0
    rows: dict[str, Any] = {
        "secid": list(secids),
        "model_id": [str(model_id)] * len(secids),
        "backend": [str(backend)] * len(secids),
        "asof": [FIRM_TEXT_ASOF] * len(secids),
    }
    for j in range(d):
        rows[f"e{j}"] = mat[:, j].tolist()
    pd.DataFrame(rows).to_parquet(cache_path, index=False)


def embed_descriptions(
    texts: Mapping[str, str],
    *,
    cache_path: Path | None = None,
    model_id: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Embed firm texts; sentence-transformers if available else TF-IDF+SVD."""
    secids = sorted(str(k) for k in texts.keys())
    docs = [str(texts[s]) for s in secids]
    path = Path(cache_path) if cache_path is not None else _DEFAULT_CACHE
    cached = _load_embed_cache(path, secids=secids, model_id=model_id)
    if cached is not None:
        return cached
    backend = "tfidf_svd"
    mat: np.ndarray
    if not docs:
        mat = np.zeros((0, 0), dtype=np.float64)
    else:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_id, device="cpu")
            mat = np.asarray(
                model.encode(docs, show_progress_bar=False), dtype=np.float64
            )
            backend = "sentence_transformers"
        except Exception:
            mat = _embed_tfidf_svd(docs)
            backend = "tfidf_svd"
        mat = _l2_normalize(mat)
    try:
        _write_embed_cache(
            path, secids=secids, mat=mat, model_id=model_id, backend=backend
        )
    except Exception:
        pass
    return {
        "secids": secids,
        "matrix": mat,
        "model_id": str(model_id),
        "backend": backend,
        "asof": FIRM_TEXT_ASOF,
        "pit": False,
    }


def align_embeddings_to_secids(
    embed_blob: Mapping[str, Any], secids: Sequence[str]
) -> np.ndarray:
    """Return (K, D) matrix aligned to ``secids``; missing -> zero vector."""
    keys = list(embed_blob.get("secids") or [])
    mat = np.asarray(embed_blob.get("matrix") if embed_blob.get("matrix") is not None else [], dtype=np.float64)
    if mat.ndim != 2 or mat.size == 0 or not keys:
        return np.zeros((len(secids), 0), dtype=np.float64)
    d = int(mat.shape[1])
    lookup = {str(k): mat[i] for i, k in enumerate(keys)}
    out = np.zeros((len(secids), d), dtype=np.float64)
    for i, sid in enumerate(secids):
        row = lookup.get(_canonical_secid_key(sid)) or lookup.get(str(sid))
        if row is not None:
            out[i] = row
    return out


def semantic_tilt_metrics(
    weights: np.ndarray, embeddings: np.ndarray
) -> dict[str, Any]:
    """Book embedding path stats from W @ E (interpretation only)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    e = np.asarray(embeddings, dtype=np.float64)
    if e.ndim != 2:
        return nan_semantic_tilt("shape_mismatch")
    t, k = w.shape
    if e.shape[0] != k or t < 1 or k < 1 or e.shape[1] < 1:
        return nan_semantic_tilt("shape_mismatch")
    d = int(e.shape[1])
    e_bar = w @ e
    # Rotation rate.
    rates: list[float] = []
    for i in range(1, t):
        a = e_bar[i - 1]
        b = e_bar[i]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < _EPS or nb < _EPS:
            continue
        cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        rates.append(1.0 - cos)
    rotation = float(np.mean(rates)) if rates else float("nan")
    # PCA on universe cloud E, project book path.
    pc_means = [float("nan"), float("nan"), float("nan")]
    try:
        from sklearn.decomposition import PCA

        n_comp = int(min(3, d, max(1, k - 1)))
        if n_comp >= 1 and k >= 2:
            pca = PCA(n_components=n_comp, random_state=0)
            pca.fit(e)
            proj = (e_bar - pca.mean_) @ pca.components_.T
            pc_means = [float(np.mean(proj[:, i])) for i in range(n_comp)]
            while len(pc_means) < 3:
                pc_means.append(float("nan"))
    except Exception:
        pc_means = [float("nan"), float("nan"), float("nan")]
    return {
        "semantic_rotation_rate": rotation,
        "semantic_pc1_mean": pc_means[0],
        "semantic_pc2_mean": pc_means[1],
        "semantic_pc3_mean": pc_means[2],
        "data_availability_reason": "",
        "text_asof": FIRM_TEXT_ASOF,
        "text_pit": False,
        "embed_backend": None,
    }
