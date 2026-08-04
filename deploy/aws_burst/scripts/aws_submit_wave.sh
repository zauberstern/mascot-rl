#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")/../../.." && pwd)/.venv/bin/python" \
  "$(cd "$(dirname "$0")/../../.." && pwd)/scripts/aws_submit_wave.py" "$@"
