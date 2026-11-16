# GPU worker setup for Windows 11 with an NVIDIA RTX 50-series laptop GPU.
# Run from the repository root in PowerShell:  powershell -ExecutionPolicy Bypass -File scripts\setup_gpu_worker.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location "$Root\backend"
nvidia-smi | Out-Null
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt -r requirements-dev.txt
if (-not (Test-Path .env)) {
  Copy-Item .env.example .env
  $tok = .\.venv\Scripts\python.exe -c "import secrets;print(secrets.token_urlsafe(32))"
  (Get-Content .env) -replace '^PE_ADMIN_TOKEN=.*', "PE_ADMIN_TOKEN=$tok" -replace '^PE_DATABASE_URL=.*', 'PE_DATABASE_URL=sqlite:///./data/pe.db' | Set-Content .env
  Write-Host "created backend\.env with a fresh admin token"
}
.\.venv\Scripts\python.exe -m app.workers.agent doctor
Set-Location "$Root\frontend"
npm ci --no-audit --no-fund
