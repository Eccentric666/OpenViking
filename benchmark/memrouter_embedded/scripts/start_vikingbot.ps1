# Start VikingBot gateway with UTF-8 encoding fix
$ErrorActionPreference = "Continue"

# Fix Windows console encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$python = "D:\ProgramFiles\anaconda3\envs\openviking\python.exe"
$ovConfig = "D:\Code\cursorProject\OpenViking\benchmark\memrouter_embedded\config\ov+graph.conf"
$logFile = "D:\Code\cursorProject\OpenViking\benchmark\memrouter_embedded\logs\vikingbot.log"

# Ensure config exists at default location
$defaultConfigDir = "$env:USERPROFILE\.openviking"
$defaultConfig = "$defaultConfigDir\ov.conf"
if (-not (Test-Path $defaultConfig)) {
    New-Item -ItemType Directory -Force -Path $defaultConfigDir | Out-Null
    Copy-Item $ovConfig $defaultConfig -Force
    Write-Host "Copied config to $defaultConfig" -ForegroundColor Cyan
}

# Redirect stdout/stderr to file to avoid Rich console encoding issues on Windows
$env:OPENVIKING_CONFIG_FILE = $ovConfig

Write-Host "Starting VikingBot gateway on 127.0.0.1:18790..." -ForegroundColor Cyan
Write-Host "Log: $logFile" -ForegroundColor Gray

# Use pythonw or python with output redirection to avoid console encoding issues
Start-Process -FilePath $python -ArgumentList @(
    "-c",
    "import sys; sys.stdout=open('$logFile','w',encoding='utf-8'); sys.stderr=sys.stdout; import vikingbot.cli.commands; vikingbot.cli.commands.app()",
    "gateway",
    "--config", $ovConfig,
    "--host", "127.0.0.1",
    "--port", "18790"
) -NoNewWindow -WorkingDirectory "D:\Code\cursorProject\OpenViking\bot"

Write-Host "VikingBot started in background. Check log with: Get-Content '$logFile' -Wait" -ForegroundColor Green
