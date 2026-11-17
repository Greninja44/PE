# Windows 10/11 setup (NVIDIA GPU or CPU-only). Run from the repository root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Creates backend\.venv with Python 3.12 via uv (system Python untouched), installs PyTorch (CUDA 12.8 build if an
# NVIDIA GPU is present, otherwise CPU), the backend and frontend dependencies, backend\.env with a fresh admin token,
# and runs the device self-test. Model weights are downloaded separately (scripts\download_models.py).
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$Root = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "uv not found. Install it, open a NEW PowerShell window, and re-run this script:" -ForegroundColor Yellow
  Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
  exit 1
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
  Write-Host "Node.js/npm not found. Install Node.js LTS from https://nodejs.org and re-run." -ForegroundColor Yellow
  exit 1
}

Set-Location "$Root\backend"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { uv venv --python 3.12 .venv }

$gpu = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($gpu) {
  Write-Host "NVIDIA GPU detected - installing PyTorch with CUDA 12.8 (supports RTX 20xx-50xx)"
  uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cu128
} else {
  Write-Host "No NVIDIA GPU detected - installing CPU PyTorch (everything works, image generation is slow)" -ForegroundColor Yellow
  uv pip install --python $py torch torchvision
}
uv pip install --python $py -r requirements-gpu.txt -r requirements-dev.txt

if (-not (Test-Path .env)) {
  Copy-Item .env.example .env
  $tok = & $py -c "import secrets;print(secrets.token_urlsafe(32))"
  (Get-Content .env) -replace '^PE_ADMIN_TOKEN=.*', "PE_ADMIN_TOKEN=$tok" | Set-Content -Encoding utf8 .env
  Write-Host "created backend\.env with a fresh admin token"
}

& $py -m app.workers.agent doctor

Set-Location "$Root\frontend"
npm ci --no-audit --no-fund
Set-Location $Root
Write-Host ""
Write-Host "Next:" -ForegroundColor Green
Write-Host "  backend\.venv\Scripts\python.exe scripts\download_models.py      # ~9 GB, once, resumable"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\dev.ps1           # API + local worker + dashboard"
