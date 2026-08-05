"""ONNX export and deployment verification for model zoo bundles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.registry import ModelCard, load_card, verify_bundle, zoo_root


def _policy_forward_module(agent: Any) -> torch.nn.Module:
    for attr in ("net", "actor", "q"):
        mod = getattr(agent, attr, None)
        if isinstance(mod, torch.nn.Module):
            return mod
    raise RuntimeError("no torch module on agent for ONNX export")


class _PolicyOnnxWrapper(torch.nn.Module):
    """Export actor mean (pre weight-head) for deterministic ONNX inference."""

    def __init__(self, agent: Any):
        super().__init__()
        self._agent = agent
        self._mod = _policy_forward_module(agent)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()
        if hasattr(self._mod, "mean"):
            return self._mod.mean(x)
        if hasattr(self._agent, "raw_to_weights"):
            raw = self._mod(x) if callable(self._mod) else self._mod.mean(x)
            return self._agent.raw_to_weights(raw)
        return self._mod(x)


def export_onnx(
    model_id: str,
    *,
    root: str | Path | None = None,
    opset_version: int = 17,
    rtol: float = 1e-5,
) -> Path:
    """Export ``policy.onnx`` into the model bundle; verify vs eager torch."""
    from src.models.inference import load_policy

    card = verify_bundle(model_id, root=root)
    agent, _ = load_policy(model_id, root=root)
    bundle_dir = zoo_root(root) / model_id
    obs_dim = int(card.obs_dim)
    dummy = torch.randn(1, obs_dim, dtype=torch.float32)
    wrapper = _PolicyOnnxWrapper(agent)
    wrapper.eval()

    onnx_path = bundle_dir / "policy.onnx"
    try:
        import onnx
        import onnxruntime as ort

        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            dynamo=True,
            input_names=["obs"],
            output_names=["action"],
            opset_version=opset_version,
        )
        onnx.checker.check_model(str(onnx_path))
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"obs": dummy.numpy()})[0]
        with torch.no_grad():
            eager = wrapper(dummy).detach().cpu().numpy()
        if not np.allclose(ort_out, eager, rtol=rtol, atol=rtol):
            raise RuntimeError(
                f"ONNX vs eager mismatch for {model_id}: "
                f"max_diff={float(np.max(np.abs(ort_out - eager)))}"
            )
    except ImportError as exc:
        raise ImportError(
            "export_onnx requires onnx, onnxruntime, onnxscript; "
            "pip install onnx onnxruntime onnxscript"
        ) from exc

    meta = {
        "model_id": model_id,
        "opset_version": opset_version,
        "obs_dim": obs_dim,
        "family": card.family,
        "algo": card.algo,
    }
    (bundle_dir / "onnx_card.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return onnx_path


def write_deploy_sidecars(
    card: ModelCard,
    *,
    deploy_config: dict[str, Any],
    obs_schema: dict[str, Any],
    root: str | Path | None = None,
) -> Path:
    bundle_dir = zoo_root(root) / card.model_id
    (bundle_dir / "deploy_config.json").write_text(
        json.dumps(deploy_config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "obs_schema.json").write_text(
        json.dumps(obs_schema, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return bundle_dir
