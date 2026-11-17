# Windows: start the API, the local inference worker (CUDA / CPU) and the dashboard, each in its own window.
#   powershell -ExecutionPolicy Bypass -File scripts\dev.ps1                 # API on :8000
#   powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 -Port 8010      # if 8000 is taken
#   powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 -NoWorker       # no local worker
# (-Worker is accepted for backward compatibility; the local worker now starts by default.)
param([int]$Port = 8000, [switch]$NoWorker, [switch]$Worker)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Api = "http://127.0.0.1:$Port"

$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  $owner = (Get-Process -Id $busy[0].OwningProcess -ErrorAction SilentlyContinue).ProcessName
  Write-Host "Port $Port is already in use by '$owner'. Use another port, e.g.: scripts\dev.ps1 -Port 8010" -ForegroundColor Red
  exit 1
}

$env:PYTHONUTF8 = "1"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PYTHONUTF8='1'; Set-Location '$Root\backend'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port $Port"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PE_API_URL='$Api'; Set-Location '$Root\frontend'; npx vite --host 127.0.0.1 --port 5173"
if (-not $NoWorker) {
  Start-Sleep -Seconds 3
  # 'local' waits for the API, registers with PE_ADMIN_TOKEN from backend\.env, then runs jobs.
  Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:PYTHONUTF8='1'; Set-Location '$Root\backend'; .\.venv\Scripts\python.exe -m app.workers.agent local --server $Api"
}
Write-Host "Dashboard: http://localhost:5173   API docs: $Api/docs"
