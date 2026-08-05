"""Central structured logging for MascotRL end-to-end runs."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


_CONFIGURED = False


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    name: str = "mascotrl",
    *,
    file_level: int | None = None,
    console_level: int | None = None,
) -> logging.Logger:
    """
    Configure root-style mascotrl logger.

    Console defaults to INFO (readable train progress).
    File defaults to DEBUG when a log_file is set (maximum dossier detail).
    """
    global _CONFIGURED
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(console_level if console_level is not None else level)
    logger.addHandler(sh)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        # truncate each run so dossiers are self-contained
        fh = logging.FileHandler(path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(file_level if file_level is not None else logging.DEBUG)
        logger.addHandler(fh)

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    _CONFIGURED = True
    return logger


def get_logger(name: str = "mascotrl") -> logging.Logger:
    if not _CONFIGURED:
        return setup_logging(name=name)
    return logging.getLogger(name)


@contextmanager
def log_span(logger: logging.Logger, label: str, **fields: Any) -> Iterator[dict]:
    """Time a block and log start/end with optional metrics dict mutation."""
    extras = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("▶ START %s %s", label, extras)
    t0 = time.perf_counter()
    metrics: dict[str, Any] = {}
    try:
        yield metrics
    except Exception:
        dt = time.perf_counter() - t0
        logger.exception("✖ FAIL  %s after %.3fs", label, dt)
        raise
    else:
        dt = time.perf_counter() - t0
        extra = " ".join(f"{k}={v}" for k, v in metrics.items())
        logger.info("✔ DONE  %s in %.3fs %s", label, dt, extra)


def log_tensor(logger: logging.Logger, name: str, t: Any, level: int = logging.INFO) -> None:
    try:
        import torch

        if isinstance(t, torch.Tensor):
            flat = t.detach().float().reshape(-1)
            logger.log(
                level,
                "%s shape=%s dtype=%s mean=%.6g std=%.6g min=%.6g max=%.6g "
                "absmean=%.6g nonzero=%d/%d",
                name,
                tuple(t.shape),
                t.dtype,
                float(flat.mean()) if flat.numel() else 0.0,
                float(flat.std()) if flat.numel() > 1 else 0.0,
                float(flat.min()) if flat.numel() else 0.0,
                float(flat.max()) if flat.numel() else 0.0,
                float(flat.abs().mean()) if flat.numel() else 0.0,
                int((flat != 0).sum()) if flat.numel() else 0,
                int(flat.numel()),
            )
            return
    except Exception:
        pass
    logger.log(level, "%s=%r", name, t)


def log_dict(logger: logging.Logger, title: str, data: dict[str, Any], level: int = logging.DEBUG) -> None:
    logger.log(level, "%s: %s", title, {k: data[k] for k in sorted(data)})
