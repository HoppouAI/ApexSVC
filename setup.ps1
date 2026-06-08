#Requires -Version 5.1
# one-shot env + model setup for windows. creates .venv with uv, installs deps,
# clones NeuCoSVC2 source for vendor extraction, downloads model weights.

$ErrorActionPreference = "Stop"

# locate uv (assume on PATH, otherwise tell user)
$uv = (Get-Command uv -ErrorAction SilentlyContinue)
if (-not $uv) {
    Write-Error "uv not found on PATH. Install from https://docs.astral.sh/uv/getting-started/installation/ and retry."
}

Write-Host "[1/3] Creating venv and installing dependencies via uv..." -ForegroundColor Cyan
uv venv
uv sync

Write-Host "[2/3] Cloning upstream NeuCoSVC2 for vendor extraction..." -ForegroundColor Cyan
$cloneDir = "third_party/_clone/NeuCoSVC"
if (-not (Test-Path $cloneDir)) {
    New-Item -ItemType Directory -Force -Path "third_party/_clone" | Out-Null
    git clone --branch NeuCoSVC2 --depth 1 https://github.com/thuhcsi/NeuCoSVC.git $cloneDir
} else {
    Write-Host "  already cloned at $cloneDir, skipping."
}

Write-Host "[3/3] Downloading model weights..." -ForegroundColor Cyan
uv run python scripts/download_models.py

Write-Host "Done. Try: uv run svc --help" -ForegroundColor Green
