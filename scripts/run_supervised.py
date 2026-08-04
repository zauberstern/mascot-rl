#!/usr/bin/env python3
"""Supervise a long-running command: heartbeat status file + exit capture.

Usage::

    python scripts/run_supervised.py --status logs/jobs/eq100.status.json \\
        -- python -u scripts/materialize_oos_arctic.py ...

Or attach to an already-running PID::

    python scripts/run_supervised.py --pid 12345 --status logs/jobs/foo.status.json

Status JSON is rewritten every ``--heartbeat-seconds`` so a crashed agent can
recover by reading the file. Exit code is stamped on termination.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def supervise_pid(
    pid: int,
    status_path: Path,
    *,
    heartbeat_seconds: float,
    cmd: list[str] | None = None,
) -> int:
    payload: dict[str, Any] = {
        "pid": int(pid),
        "cmd": cmd or [],
        "started_at": _utc(),
        "heartbeat_at": _utc(),
        "state": "running",
        "exit_code": None,
    }
    _write_status(status_path, payload)
    while _alive(pid):
        payload["heartbeat_at"] = _utc()
        payload["state"] = "running"
        _write_status(status_path, payload)
        time.sleep(float(heartbeat_seconds))
    # Best-effort: waitpid if we are the parent; else assume unknown exit.
    exit_code: int | None = None
    try:
        waited_pid, status = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1
    except ChildProcessError:
        exit_code = None
    payload["heartbeat_at"] = _utc()
    payload["ended_at"] = _utc()
    payload["state"] = "exited"
    payload["exit_code"] = exit_code
    _write_status(status_path, payload)
    return int(exit_code if exit_code is not None else 0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--status",
        type=Path,
        required=True,
        help="Path to status JSON (rewritten on each heartbeat).",
    )
    p.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
    )
    p.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Attach to an existing PID instead of spawning a command.",
    )
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command after --  e.g. -- python -u scripts/foo.py",
    )
    args = p.parse_args(argv)
    status_path = args.status if args.status.is_absolute() else ROOT / args.status

    if args.pid is not None:
        return supervise_pid(
            int(args.pid),
            status_path,
            heartbeat_seconds=float(args.heartbeat_seconds),
        )

    cmd = list(args.command or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        p.error("pass a command after -- or use --pid")

    # Start in its own session so agent/shell death does not kill the child.
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    log_path = status_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _forward() -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
                fh.flush()

    import threading

    t = threading.Thread(target=_forward, daemon=True)
    t.start()

    payload: dict[str, Any] = {
        "pid": int(proc.pid),
        "cmd": cmd,
        "started_at": _utc(),
        "heartbeat_at": _utc(),
        "state": "running",
        "exit_code": None,
        "log": str(log_path),
    }
    _write_status(status_path, payload)

    def _on_signal(signum: int, _frame: Any) -> None:
        payload["state"] = "signal"
        payload["signal"] = int(signum)
        payload["heartbeat_at"] = _utc()
        _write_status(status_path, payload)
        try:
            os.killpg(proc.pid, signum)
        except OSError:
            proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    while proc.poll() is None:
        payload["heartbeat_at"] = _utc()
        payload["state"] = "running"
        _write_status(status_path, payload)
        time.sleep(float(args.heartbeat_seconds))

    t.join(timeout=5.0)
    exit_code = int(proc.returncode if proc.returncode is not None else 1)
    payload["heartbeat_at"] = _utc()
    payload["ended_at"] = _utc()
    payload["state"] = "exited"
    payload["exit_code"] = exit_code
    _write_status(status_path, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
