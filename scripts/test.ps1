# Windows: backend tests + frontend type-check and production build.
#   powershell -ExecutionPolicy Bypass -File scripts\test.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location "$Root\backend"
.\.venv\Scripts\python.exe -m pytest -p no:warnings
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Set-Location "$Root\frontend"
npm run build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Set-Location $Root
Write-Host "All checks passed." -ForegroundColor Green
