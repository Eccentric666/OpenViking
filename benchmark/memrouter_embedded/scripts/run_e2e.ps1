#Requires -Version 5.1
#
# MemRouter Embedded in OpenViking — E2E evaluation launcher
# =============================================================================
# MemRouter logic is now INSIDE OpenViking Server.
# No MEMROUTER_ENABLED, ECHOMEM_PATH, or MEMROUTER_CONFIG env vars needed.
# All config is in ov.conf under the "memrouter" section.
#
# Usage:
#   .\scripts\run_e2e.ps1 -ForceMemorySearch -Judge -JudgeToken "sk-xxxxx"
# =============================================================================

[CmdletBinding()]
param(
    [switch]$ForceMemorySearch,
    [switch]$Judge,
    [string]$JudgeToken = "",
    [string]$JudgeBaseUrl = "https://api.minimaxi.com/anthropic",
    [string]$JudgeModel = "MiniMax-M2.7",
    [int]$LimitSamples = 0,
    [int]$LimitQuestions = 0,
    [string]$Category = "",
    [switch]$SkipServerStart,
    [string]$FixedRunDir = "",
    [string]$CaseIdsFile = ""
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
$scriptDir     = $PSScriptRoot
$benchmarkRoot = Split-Path -Parent $scriptDir                  # memrouter_embedded
$ovRoot        = (Resolve-Path "$benchmarkRoot\..\..").Path       # OpenViking
$echomemRoot   = "D:\Code\cursorProject\EchoMem"

$dataDir       = "$benchmarkRoot\data"
$configsDir    = "$benchmarkRoot\config"
$scriptsDir    = "$benchmarkRoot\scripts"

$runsRoot      = (Resolve-Path "$benchmarkRoot\..\runs").Path

# Data files
$dataset      = "$ovRoot\benchmark\locomo_e2e\locomo10.json"
$routeLabels  = "$benchmarkRoot\data\locomo_e2e_route_labels.v3.jsonl"

# Config files
$ovConf        = "$configsDir\ov.conf"

# Route events (configured in ov.conf, but also track here)
$routeEventsPath = "$benchmarkRoot\logs\route_events.jsonl"

# Check required files
$missing = @()
foreach ($f in @($dataset, $routeLabels, $ovConf)) {
    if (-not (Test-Path $f)) {
        $missing += $f
    }
}
if ($missing.Count -gt 0) {
    Write-Error "Missing required files:`n$($missing -join '`n')"
    exit 1
}

# ------------------------------------------------------------------
# Run directory
# ------------------------------------------------------------------
if ($FixedRunDir) {
    $runDir = $FixedRunDir
} else {
    $ts     = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDir = "$runsRoot\memrouter_embedded\${ts}_locomo_e2e"
}
$logsDir = "$runDir\logs"
$resDir  = "$runDir\results"

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $resDir  | Out-Null

# Clear stale route events
if (Test-Path $routeEventsPath) {
    Remove-Item $routeEventsPath -Force
}

# ------------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------------
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8       = "1"
$env:NO_COLOR         = "1"
$env:PYTHONPATH       = "$ovRoot\openviking\lib;$ovRoot;$ovRoot\bot;$echomemRoot"
$env:OPENVIKING_CONFIG_FILE = $ovConf

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MemRouter Embedded in OpenViking — E2E" -ForegroundColor Cyan
Write-Host "========================================"
Write-Host "OV root        : $ovRoot"
Write-Host "EchoMem root   : $echomemRoot"
Write-Host "Benchmark root : $benchmarkRoot"
Write-Host "Run directory  : $runDir"
Write-Host "Route events   : $routeEventsPath"
Write-Host ""

# ------------------------------------------------------------------
# Helper: cleanup ports
# ------------------------------------------------------------------
function Stop-E2EServers {
    foreach ($port in @(1933, 18790)) {
        try {
            Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | ForEach-Object {
                Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
                Write-Host "  Killed process on port $port (PID $($_.OwningProcess))" -ForegroundColor DarkGray
            }
        } catch {}
    }
}

function Wait-ForPort {
    param([string]$HostName = "127.0.0.1", [int]$Port, [int]$TimeoutSec = 30)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        try {
            $client = New-Object System.Net.Sockets.TcpClient
            $client.Connect($HostName, $Port)
            $client.Close()
            return $true
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

# ------------------------------------------------------------------
# Start services
# ------------------------------------------------------------------
$gwProc = $null
$ovProc = $null

if (-not $SkipServerStart) {
    Write-Host "Cleaning up stale processes..." -ForegroundColor DarkGray
    Stop-E2EServers

    Write-Host "[1/4] Starting VikingBot gateway (port 18790)..." -ForegroundColor Yellow
    $gwCmd = @"
cd "$ovRoot\bot"
`$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUTF8='1'; `$env:NO_COLOR='1'
`$env:PYTHONPATH='$ovRoot\openviking\lib;$ovRoot;$ovRoot\bot;$echomemRoot'
`$env:OPENVIKING_CONFIG_FILE='$ovConf'
python -m vikingbot gateway --config '$ovConf' --host 127.0.0.1 --port 18790 2>>'$logsDir\vikingbot.gateway.log'
"@
    $gwProc = Start-Process powershell -ArgumentList "-Command", $gwCmd -PassThru -WindowStyle Hidden

    Write-Host "[2/4] Waiting for gateway (max 30s)..." -ForegroundColor Yellow
    if (Wait-ForPort -Port 18790) {
        Write-Host "      Gateway ready." -ForegroundColor Green
    } else {
        Write-Warning "Gateway did not start in time."
    }

    Write-Host "[3/4] Starting OpenViking server (port 1933)..." -ForegroundColor Yellow
    $ovCmd = @"
cd "$ovRoot"
`$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUTF8='1'; `$env:NO_COLOR='1'
`$env:PYTHONPATH='$ovRoot\openviking\lib;$ovRoot;$ovRoot\bot;$echomemRoot'
`$env:OPENVIKING_CONFIG_FILE='$ovConf'
python -m uvicorn openviking.server.app:create_app --factory --host 127.0.0.1 --port 1933 2>>'$logsDir\openviking.server.log'
"@
    $ovProc = Start-Process powershell -ArgumentList "-Command", $ovCmd -PassThru -WindowStyle Hidden

    Write-Host "[4/4] Waiting for OV /health (max 30s)..." -ForegroundColor Yellow
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:1933/health" -Method GET -TimeoutSec 2 -ErrorAction Stop
            if ($resp.status -eq "ok" -or $resp.healthy -eq $true) {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if ($ready) {
        Write-Host "      OpenViking server ready." -ForegroundColor Green
    } else {
        Write-Warning "OV server health check did not pass in 30s."
    }
} else {
    Write-Host "[+] Skipping server start (assumes already running)." -ForegroundColor Yellow
}

# ------------------------------------------------------------------
# Build evaluator arguments
# ------------------------------------------------------------------
$argList = @(
    "--dataset", $dataset,
    "--route-labels", $routeLabels,
    "--route-events-path", $routeEventsPath,
    "--ov-config", $ovConf,
    "--openviking-root", $ovRoot,
    "--output-base", "$benchmarkRoot\runs",
    "--fixed-run-dir", $runDir,
    "--ov-chat-endpoint", "http://127.0.0.1:18790",
    "--ov-api-key", "ov-test-key-12345",
    "--ov-account", "default"
)

if ($ForceMemorySearch) { $argList += "--force-memory-search" }
if ($LimitSamples -gt 0) { $argList += @("--limit-samples", $LimitSamples) }
if ($LimitQuestions -gt 0) { $argList += @("--limit-questions", $LimitQuestions) }
if ($Category) { $argList += @("--category", $Category) }
if ($CaseIdsFile) { $argList += @("--case-ids-file", $CaseIdsFile) }
if ($Judge) {
    if (-not $JudgeToken) { Write-Error "--Judge requires --JudgeToken"; exit 1 }
    $argList += @("--judge", "--judge-token", $JudgeToken,
                  "--judge-base-url", $JudgeBaseUrl, "--judge-model", $JudgeModel)
}

# ------------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------------
$evalScript = "$scriptsDir\eval_locomo_ov_with_memrouter_e2e.py"
Write-Host "Running evaluator..." -ForegroundColor Yellow
Write-Host "      python $evalScript"
Write-Host ""

cd $ovRoot
python "$evalScript" @argList
$exitCode = $LASTEXITCODE

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "Evaluation completed successfully." -ForegroundColor Green
} else {
    Write-Host "Evaluator exited with code $exitCode" -ForegroundColor Red
}
Write-Host "Run directory  : $runDir"
Write-Host "Results dir    : $resDir"
Write-Host "Route events   : $routeEventsPath"
Write-Host "Report         : $resDir\report.md"
Write-Host "========================================"

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------
if (-not $SkipServerStart) {
    Write-Host "Stopping background servers..." -ForegroundColor DarkGray
    if ($ovProc -ne $null) { Stop-Process -Id $ovProc.Id -Force -ErrorAction SilentlyContinue }
    if ($gwProc -ne $null) { Stop-Process -Id $gwProc.Id -Force -ErrorAction SilentlyContinue }
    Stop-E2EServers
    Write-Host "Servers stopped." -ForegroundColor DarkGray
}

exit $exitCode