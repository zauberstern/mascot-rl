"""Trained-model registry: ModelCard + save/list/verify bundles."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from mascotrl._root import REPO_ROOT

ROOT = REPO_ROOT
DEFAULT_ZOO = ROOT / "logs" / "artifacts" / "models"


@dataclass
class ModelCard:
    model_id: str
    family: str  # research_single_agent | happo
    algo: str
    train_world: str = ""
    architecture: str = ""
    objective: str = ""
    arm: str = "eq"
    obs_dim: int = 0
    action_dim: int = 0
    n_assets: int = 0
    seed: int = 0
    fold_id: int | None = None
    run_config_hash: str = ""
    estimand_id: str = ""
    cpcv_config: dict[str, Any] | None = None
    sharpe_mean: float | None = None
    created_utc: str = ""
    mascotrl_git_sha: str = ""
    weights_sha256: str = ""
    source_artifact: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelCard":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in allowed})


def zoo_root(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else Path(
        os.environ.get("MASCOTRL_MODEL_ZOO", str(DEFAULT_ZOO))
    )


def make_model_id(
    *,
    family: str,
    algo: str,
    arm: str,
    seed: int,
    run_config_hash: str,
) -> str:
    h = (run_config_hash or "nohash")[:8]
    return f"{family}-{algo}-{arm}-{int(seed)}-{h}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def save_model_bundle(
    weights_payload: Mapping[str, Any],
    card: ModelCard,
    *,
    root: str | Path | None = None,
    deploy_config: Mapping[str, Any] | None = None,
    obs_schema: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``logs/artifacts/models/<model_id>/{weights.pt,card.json}``."""
    z = zoo_root(root)
    d = z / card.model_id
    d.mkdir(parents=True, exist_ok=True)
    wpath = d / "weights.pt"
    torch.save(dict(weights_payload), wpath)
    card.weights_sha256 = _sha256_file(wpath)
    if not card.created_utc:
        card.created_utc = datetime.now(timezone.utc).isoformat()
    if not card.mascotrl_git_sha:
        card.mascotrl_git_sha = _git_sha()
    (d / "card.json").write_text(
        json.dumps(card.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if deploy_config is not None:
        (d / "deploy_config.json").write_text(
            json.dumps(dict(deploy_config), indent=2, sort_keys=True, default=str)
            + "\n",
            encoding="utf-8",
        )
    if obs_schema is not None:
        (d / "obs_schema.json").write_text(
            json.dumps(dict(obs_schema), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return d


def load_card(model_id: str, *, root: str | Path | None = None) -> ModelCard:
    path = zoo_root(root) / model_id / "card.json"
    if not path.is_file():
        raise FileNotFoundError(f"model card not found: {path}")
    return ModelCard.from_dict(json.loads(path.read_text(encoding="utf-8")))


def verify_bundle(model_id: str, *, root: str | Path | None = None) -> ModelCard:
    """Fail closed on missing files or sha256 drift."""
    d = zoo_root(root) / model_id
    card = load_card(model_id, root=root)
    wpath = d / "weights.pt"
    if not wpath.is_file():
        raise FileNotFoundError(f"weights missing for {model_id}: {wpath}")
    digest = _sha256_file(wpath)
    if card.weights_sha256 and digest != card.weights_sha256:
        raise RuntimeError(
            f"weights sha256 mismatch for {model_id}: "
            f"card={card.weights_sha256} disk={digest}"
        )
    return card


def list_models(
    *,
    root: str | Path | None = None,
    algo: str | None = None,
    arm: str | None = None,
    family: str | None = None,
    min_sharpe: float | None = None,
) -> list[ModelCard]:
    z = zoo_root(root)
    if not z.is_dir():
        return []
    out: list[ModelCard] = []
    for d in sorted(z.iterdir()):
        if not d.is_dir() or not (d / "card.json").is_file():
            continue
        try:
            card = load_card(d.name, root=root)
        except Exception:
            continue
        if algo and card.algo != algo:
            continue
        if arm and card.arm != arm:
            continue
        if family and card.family != family:
            continue
        if min_sharpe is not None and (
            card.sharpe_mean is None or float(card.sharpe_mean) < float(min_sharpe)
        ):
            continue
        out.append(card)
    return out


def write_model_zoo_index(*, root: str | Path | None = None) -> Path:
    z = zoo_root(root)
    z.mkdir(parents=True, exist_ok=True)
    cards = list_models(root=root)
    lines = [
        "# MascotRL Model Zoo",
        "",
        f"Bundles under `{z}`.",
        "",
        "| model_id | family | algo | arm | seed | sharpe |",
        "|---|---|---|---|---:|---:|",
    ]
    for c in cards:
        sh = "" if c.sharpe_mean is None else f"{float(c.sharpe_mean):.4f}"
        lines.append(
            f"| `{c.model_id}` | {c.family} | {c.algo} | {c.arm} | {c.seed} | {sh} |"
        )
    lines.append("")
    out = z / "MODEL_ZOO.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
