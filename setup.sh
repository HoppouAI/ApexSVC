#!/usr/bin/env bash
# parity setup for non-windows. mostly here so devs on linux/mac can poke at it.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found on PATH. install from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

echo "[1/3] Creating venv and installing dependencies..."
uv venv
uv sync

echo "[2/3] Cloning upstream NeuCoSVC2 for vendor extraction..."
clone_dir="third_party/_clone/NeuCoSVC"
if [ ! -d "$clone_dir" ]; then
    mkdir -p third_party/_clone
    git clone --branch NeuCoSVC2 --depth 1 https://github.com/thuhcsi/NeuCoSVC.git "$clone_dir"
else
    echo "  already cloned at $clone_dir, skipping."
fi

echo "[3/3] Downloading model weights..."
uv run python scripts/download_models.py

echo "Done. Try: uv run svc --help"
