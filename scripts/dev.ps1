# Windows: start API, dashboard and (optionally) the GPU worker, each in its own window.
#   powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 [-Worker]
param([switch]$Worker)
$Root = Split-Path -Parent $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root\backend'; .\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root\frontend'; npx vite --host 127.0.0.1 --port 5173"
if ($Worker) {
  Start-Sleep -Seconds 4
  Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root\backend'; .\.venv\Scripts\python.exe -m app.workers.agent run --server http://127.0.0.1:8000"
}
Write-Host "Dashboard: http://127.0.0.1:5173   API docs: http://127.0.0.1:8000/docs"
