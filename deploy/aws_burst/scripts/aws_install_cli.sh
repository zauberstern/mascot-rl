#!/usr/bin/env bash
# AWS-0: install AWS CLI v2 into ~/.local (no root required when possible).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DEST="${AWS_CLI_INSTALL_DIR:-$HOME/.local}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
cd "$tmpdir"
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip
./aws/install -i "$DEST/aws-cli" -b "$DEST/bin" --update
export PATH="$DEST/bin:$PATH"
aws --version
echo "Configure three profiles (volsurf-burst-1..3) then run:"
echo "  $ROOT/deploy/aws_burst/scripts/aws_verify_profiles.sh"
