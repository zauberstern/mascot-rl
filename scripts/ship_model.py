#!/usr/bin/env python3
"""Ship a verified model bundle as a user-runnable artifact directory."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.models.export import export_onnx
from src.models.registry import verify_bundle, zoo_root


_PREDICT_PY = '''"""Minimal ONNX inference helper (copy with the ship/ directory)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_schema(ship_dir: str | Path) -> dict:
    return json.loads((Path(ship_dir) / "obs_schema.json").read_text(encoding="utf-8"))


def act_weights(obs: np.ndarray, ship_dir: str | Path = ".") -> np.ndarray:
    import onnxruntime as ort

    ship = Path(ship_dir)
    sess = ort.InferenceSession(
        str(ship / "policy.onnx"), providers=["CPUExecutionProvider"]
    )
    x = np.nan_to_num(np.asarray(obs, dtype=np.float32).reshape(1, -1), nan=0.0)
    raw = sess.run(None, {"obs": x})[0].reshape(-1)
    w = np.nan_to_num(raw, nan=0.0)
    denom = float(np.sum(np.abs(w)))
    if denom > 1e-8:
        return w / denom
    k = max(int(w.size), 1)
    return np.full(k, 1.0 / k, dtype=np.float64)


if __name__ == "__main__":
    import sys

    dim = int(json.loads((Path(".") / "obs_schema.json").read_text())["obs_dim"])
    obs = np.zeros(dim, dtype=np.float32)
    print(act_weights(obs))
'''


def ship_model(model_id: str, *, out_dir: Path, root: Path | None = None) -> Path:
    card = verify_bundle(model_id, root=root)
    bundle = zoo_root(root) / model_id
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "weights.pt",
        "card.json",
        "obs_schema.json",
        "deploy_config.json",
        "policy.onnx",
        "onnx_card.json",
    ):
        src = bundle / name
        if src.is_file():
            shutil.copy2(src, out / name)
    if not (out / "policy.onnx").is_file():
        export_onnx(model_id, root=root)
        shutil.copy2(bundle / "policy.onnx", out / "policy.onnx")
        if (bundle / "onnx_card.json").is_file():
            shutil.copy2(bundle / "onnx_card.json", out / "onnx_card.json")
    if not (out / "obs_schema.json").is_file():
        schema = {
            "obs_dim": int(card.obs_dim),
            "action_dim": int(card.action_dim),
            "n_assets": int(card.n_assets or card.action_dim),
        }
        (out / "obs_schema.json").write_text(
            json.dumps(schema, indent=2) + "\n", encoding="utf-8"
        )
    (out / "predict.py").write_text(_PREDICT_PY, encoding="utf-8")
    manifest = {
        "model_id": model_id,
        "family": card.family,
        "algo": card.algo,
    }
    (out / "ship_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ship a model zoo bundle for deployment")
    ap.add_argument("model_id")
    ap.add_argument("--out", type=Path, default=Path("ship"))
    ap.add_argument("--zoo-root", type=Path, default=None)
    args = ap.parse_args()
    path = ship_model(args.model_id, out_dir=args.out, root=args.zoo_root)
    print(path)


if __name__ == "__main__":
    main()
