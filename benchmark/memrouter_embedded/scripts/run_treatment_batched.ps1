#Requires -Version 5.1
#
# Treatment Batched Evaluation Runner
# =====================================
# Runs Treatment (OV + MemRouter + Graph) eval in batches of N questions.
# Before each batch: cleans up and restarts OV Server + VikingBot.
# After each batch: checks for timeouts, retries if needed.
#
# Usage:
#   .\run_treatment_batched.ps1 -JudgeToken "sk-xxxxx"
#
# Optional:
#   -BatchSize 20        (default 20)
#   -ChatTimeout 120     (default 120s)
#   -MaxRetries 2        (default 2 retries per batch)
#   -PythonExe "python"   (default "python")
#   -TotalQuestions 81   (default 81)
# =====================================

[CmdletBinding()]
param(
    [int]$BatchSize = 20,
    [int]$TotalQuestions = 81,
    [int]$ChatTimeout = 120,
    [string]$JudgeToken = "",
    [int]$MaxRetries = 2,
    [string]$PythonExe = "D:\ProgramFiles\anaconda3\envs\openviking\python.exe",
    [string]$FixedMasterDir = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
$scriptDir     = $PSScriptRoot
$benchmarkRoot = Split-Path -Parent $scriptDir
$ovRoot        = (Resolve-Path "$benchmarkRoot\..\..").Path
$echomemRoot   = "D:\Code\cursorProject\EchoMem"

# Config
$ovConf          = "$benchmarkRoot\config\ov+graph.conf"
$dataset         = "$benchmarkRoot\data\conv30_exclude_cat5.json"
$routeLabels     = "$benchmarkRoot\data\locomo_e2e_route_labels.v4.jsonl"
$routeEventsPath = "$benchmarkRoot\logs\route_events.jsonl"

# Results
if ($FixedMasterDir) {
    $masterRunDir = $FixedMasterDir
} else {
    $timestamp    = Get-Date -Format "yyyyMMdd_HHmmss"
    $masterRunDir = "$benchmarkRoot\results\ablation\treatment_${timestamp}_batched"
}
$batchResultsDir = "$masterRunDir\batch_results"
$masterLogsDir   = "$masterRunDir\logs"

New-Item -ItemType Directory -Force -Path $masterRunDir    | Out-Null
New-Item -ItemType Directory -Force -Path $batchResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $masterLogsDir   | Out-Null

# ------------------------------------------------------------------
# Build batch list: array of [start, end]
# ------------------------------------------------------------------
$batches = @()
for ($i = 1; $i -le $TotalQuestions; $i += $BatchSize) {
    $end = [Math]::Min($i + $BatchSize - 1, $TotalQuestions)
    $batches += ,@($i, $end)
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Baseline Batched Evaluation Runner" -ForegroundColor Cyan
Write-Host "========================================"
Write-Host "Total questions : $TotalQuestions"
Write-Host "Batch size      : $BatchSize"
Write-Host "Chat timeout    : ${ChatTimeout}s"
Write-Host "Max retries     : $MaxRetries"
Write-Host "Python          : $PythonExe"
Write-Host "Batches         : $($batches.Count)"
for ($bi = 0; $bi -lt $batches.Count; $bi++) {
    Write-Host "  Batch $($bi+1): Q$($batches[$bi][0])-$($batches[$bi][1])"
}
Write-Host "Master dir      : $masterRunDir"
Write-Host "========================================"

if (-not $JudgeToken) {
    Write-Error "Missing required parameter: -JudgeToken"
    exit 1
}

if (-not (Test-Path $dataset)) {
    Write-Error "Dataset not found: $dataset"
    exit 1
}
if (-not (Test-Path $routeLabels)) {
    Write-Error "Route labels not found: $routeLabels"
    exit 1
}
if (-not (Test-Path $ovConf)) {
    Write-Error "OV config not found: $ovConf"
    exit 1
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
function Stop-AllServers {
    Write-Host "[CLEANUP] Stopping all servers..." -ForegroundColor DarkGray

    # Kill by port
    foreach ($port in @(1933, 18790)) {
        try {
            Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | ForEach-Object {
                $pid_ = $_.OwningProcess
                try {
                    Stop-Process -Id $pid_ -Force -ErrorAction Stop
                    Write-Host "  Killed PID $pid_ on port $port" -ForegroundColor DarkGray
                } catch {
                    Write-Host "  Failed to kill PID $pid_ on port $port : $_" -ForegroundColor DarkRed
                }
            }
        } catch {}
    }

    # Extra: kill any python processes with uvicorn or vikingbot in command line
    Get-Process -Name "python" -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $cmd = (Get-WmiObject Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
            if ($cmd -match "uvicorn|vikingbot|openviking.server") {
                Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
                Write-Host "  Killed python PID $($_.Id) ($cmd)" -ForegroundColor DarkGray
            }
        } catch {}
    }

    Start-Sleep -Seconds 3

    # Verify ports are free
    foreach ($port in @(1933, 18790)) {
        $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($conn) {
            Write-Warning "Port $port still in use by PID $($conn.OwningProcess)!"
        } else {
            Write-Host "  Port $port is free." -ForegroundColor DarkGray
        }
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

function Start-AllServers {
    param([string]$BatchLogDir)

    New-Item -ItemType Directory -Force -Path $BatchLogDir | Out-Null

    # Clear stale route events so each batch gets fresh events
    if (Test-Path $routeEventsPath) {
        Remove-Item $routeEventsPath -Force
        Write-Host "  Cleared stale route_events.jsonl" -ForegroundColor DarkGray
    }

    # --- Start VikingBot gateway ---
    Write-Host "[START] VikingBot gateway (port 18790)..." -ForegroundColor Yellow
    $gwLog = "$BatchLogDir\vikingbot.log"
    $gwCmd = @"
cd "$ovRoot\bot"
`$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUTF8='1'; `$env:NO_COLOR='1'
`$env:PYTHONPATH='$ovRoot\openviking\lib;$ovRoot;$ovRoot\bot;$echomemRoot'
`$env:OPENVIKING_CONFIG_FILE='$ovConf'
$PythonExe -m vikingbot gateway --config '$ovConf' --host 127.0.0.1 --port 18790 2>>'$gwLog'
"@
    $script:gwProc = Start-Process powershell -ArgumentList "-Command", $gwCmd -PassThru -WindowStyle Hidden

    if (Wait-ForPort -Port 18790 -TimeoutSec 30) {
        Write-Host "        Gateway ready." -ForegroundColor Green
    } else {
        Write-Warning "Gateway did NOT become ready in 30s!"
    }

    # --- Start OpenViking server ---
    Write-Host "[START] OpenViking server (port 1933)..." -ForegroundColor Yellow
    $ovLog = "$BatchLogDir\ov_server.log"
    $ovCmd = @"
cd "$ovRoot"
`$env:PYTHONIOENCODING='utf-8'; `$env:PYTHONUTF8='1'; `$env:NO_COLOR='1'
`$env:PYTHONPATH='$ovRoot\openviking\lib;$ovRoot;$ovRoot\bot;$echomemRoot'
`$env:OPENVIKING_CONFIG_FILE='$ovConf'
$PythonExe -m uvicorn openviking.server.app:create_app --factory --host 127.0.0.1 --port 1933 2>>'$ovLog'
"@
    $script:ovProc = Start-Process powershell -ArgumentList "-Command", $ovCmd -PassThru -WindowStyle Hidden

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
        Write-Host "        OV Server ready." -ForegroundColor Green
    } else {
        Write-Warning "OV Server health check did NOT pass in 30s!"
    }

    # Extra warmup
    Start-Sleep -Seconds 5
    Write-Host "        Services warmed up." -ForegroundColor Green
}

function Invoke-BatchEval {
    param([int]$StartQ, [int]$EndQ, [string]$BatchLogDir)

    $argList = @(
        "$scriptDir\eval_locomo_ov_with_graph_e2e.py",
        "--dataset", $dataset,
        "--route-labels", $routeLabels,
        "--route-events-path", $routeEventsPath,
        "--ov-config", $ovConf,
        "--openviking-root", $ovRoot,
        "--output-base", $batchResultsDir,
        "--ov-chat-endpoint", "http://127.0.0.1:18790",
        "--ov-search-endpoint", "http://127.0.0.1:1933",
        "--ov-api-key", "ov-test-key-12345",
        "--ov-account", "default",
        "--chat-timeout", $ChatTimeout,
        "--start-question", $StartQ,
        "--end-question", $EndQ,
        "--force-memory-search",
        "--no-strict-preflight",
        "--judge",
        "--judge-token", $JudgeToken,
        "--judge-base-url", "https://api.minimaxi.com/anthropic",
        "--judge-model", "MiniMax-M2.7"
    )

    Write-Host "[EVAL]  Q$StartQ-$EndQ ..." -ForegroundColor Yellow
    $evalLog = "$BatchLogDir\eval.log"

    # Set environment for eval
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8       = "1"
    $env:NO_COLOR         = "1"
    $env:PYTHONPATH       = "$ovRoot\openviking\lib;$ovRoot;$ovRoot\bot;$echomemRoot"
    $env:OPENVIKING_CONFIG_FILE = $ovConf

    if ($DryRun) {
        Write-Host "  [DRY RUN] Would execute:" -ForegroundColor Magenta
        Write-Host "    $PythonExe $($argList -join ' ')" -ForegroundColor Magenta
        return 0
    }

    # Run eval — capture exit code FIRST before any pipeline that might override $LASTEXITCODE
    $output = & $PythonExe @argList 2>&1
    $exitCode = $LASTEXITCODE
    $output | Out-File -FilePath $evalLog -Encoding UTF8
    # Also stream key lines to console so user sees progress
    $output | ForEach-Object { if ($_ -match "^\[|^Case |ERROR=|completed|Summary") { Write-Host $_ } }
    return $exitCode
}

function Get-LatestBatchRunDir {
    $runDirs = Get-ChildItem -Path $batchResultsDir -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($runDirs.Count -eq 0) { return $null }
    return $runDirs[0].FullName
}

function Test-BatchHasTimeouts {
    param([string]$RunDir)
    if (-not $RunDir) { return $false }
    $routeResults = "$RunDir\results\route_results.jsonl"
    if (-not (Test-Path $routeResults)) {
        Write-Host "  No route_results.jsonl found at $routeResults" -ForegroundColor DarkYellow
        return $false
    }

    $timeoutCount = 0
    Get-Content $routeResults -ErrorAction SilentlyContinue | ForEach-Object {
        $line = $_.Trim()
        if (-not $line) { return }
        try {
            $row = $line | ConvertFrom-Json -ErrorAction Stop
            $err = $row.error
            if ($err -and ($err -match "timeout|504|Gateway Timeout|ReadTimeout|ConnectTimeout")) {
                $timeoutCount++
            }
        } catch {}
    }

    if ($timeoutCount -gt 0) {
        Write-Host "  Found $timeoutCount timeout(s) in batch results." -ForegroundColor Red
        return $true
    }
    Write-Host "  No timeouts detected." -ForegroundColor Green
    return $false
}

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
$allBatchRunDirs = @()
$overallSuccess = $true

for ($bi = 0; $bi -lt $batches.Count; $bi++) {
    $batch     = $batches[$bi]
    $startQ    = $batch[0]
    $endQ      = $batch[1]
    $batchNum  = $bi + 1

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Batch $batchNum / $($batches.Count) : Q$startQ-$endQ" -ForegroundColor Cyan
    Write-Host "========================================"

    # Check if this batch was already completed in a previous run
    $metaFile = "$masterRunDir\batch_$batchNum`_meta.json"
    if (Test-Path $metaFile) {
        $prevMeta = Get-Content $metaFile | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($prevMeta -and $prevMeta.success -eq $true -and $prevMeta.run_dir -and (Test-Path $prevMeta.run_dir)) {
            Write-Host "  [SKIP] Batch $batchNum already completed. Using existing results." -ForegroundColor Green
            $allBatchRunDirs += $prevMeta.run_dir
            continue
        }
    }

    $batchLogDir = "$masterLogsDir\batch_$batchNum`_Q${startQ}-${endQ}"

    # ---- Cleanup + Restart ----
    Stop-AllServers
    Start-AllServers -BatchLogDir $batchLogDir

    # ---- Run eval with retry ----
    $retryCount = 0
    $batchSuccess = $false
    $batchRunDir = $null

    do {
        $exitCode = Invoke-BatchEval -StartQ $startQ -EndQ $endQ -BatchDir $batchLogDir
        $batchRunDir = Get-LatestBatchRunDir

        $hasTimeouts = Test-BatchHasTimeouts -RunDir $batchRunDir

        if ($exitCode -eq 0 -and -not $hasTimeouts) {
            $batchSuccess = $true
            Write-Host "  Batch $batchNum completed successfully." -ForegroundColor Green
            break
        }

        if ($retryCount -lt $MaxRetries) {
            $retryCount++
            Write-Host "  -> Retrying batch $batchNum (attempt $retryCount/$MaxRetries)..." -ForegroundColor Yellow
            Stop-AllServers
            Start-Sleep -Seconds 5
            Start-AllServers -BatchLogDir $batchLogDir
        } else {
            Write-Host "  -> Batch $batchNum FAILED after $MaxRetries retries." -ForegroundColor Red
            $overallSuccess = $false
            break
        }
    } while ($retryCount -le $MaxRetries)

    # Record batch metadata
    $meta = @{
        batch_num   = $batchNum
        start_q     = $startQ
        end_q       = $endQ
        run_dir     = $batchRunDir
        success     = $batchSuccess
        retries     = $retryCount
        exit_code   = $exitCode
        has_timeouts= (Test-BatchHasTimeouts -RunDir $batchRunDir)
    }
    $meta | ConvertTo-Json -Depth 3 | Set-Content "$masterRunDir\batch_$batchNum`_meta.json" -Encoding UTF8

    if ($batchRunDir) {
        $allBatchRunDirs += $batchRunDir
    }

    # ---- Post-batch cleanup ----
    Stop-AllServers
    Write-Host "  Batch $batchNum cleanup done." -ForegroundColor DarkGray
}

# ------------------------------------------------------------------
# Merge results
# ------------------------------------------------------------------
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Merging batch results..." -ForegroundColor Cyan
Write-Host "========================================"

if ($DryRun) {
    Write-Host "[DRY RUN] Would merge: $($allBatchRunDirs -join ', ')" -ForegroundColor Magenta
} elseif ($allBatchRunDirs.Count -gt 0) {
    $mergeArgs = @(
        "$scriptDir\merge_batch_results.py",
        "--master-dir", $masterRunDir,
        "--batch-dirs", ($allBatchRunDirs -join ",")
    )

    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8       = "1"
    & $PythonExe @mergeArgs
} else {
    Write-Warning "No batch results to merge."
}

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($overallSuccess) {
    Write-Host "ALL BATCHES COMPLETED SUCCESSFULLY!" -ForegroundColor Green
} else {
    Write-Host "SOME BATCHES FAILED — CHECK LOGS" -ForegroundColor Red
}
Write-Host "========================================"
Write-Host "Master dir : $masterRunDir"
Write-Host "Batches    : $($batches.Count)"
Write-Host "Results    : $masterRunDir\results"
if (-not $DryRun) {
    Get-ChildItem "$masterRunDir\batch_*_meta.json" -ErrorAction SilentlyContinue | ForEach-Object {
        $m = Get-Content $_ | ConvertFrom-Json
        $status = if ($m.success) { "OK" } else { "FAIL" }
        Write-Host "  Batch $($m.batch_num) Q$($m.start_q)-$($m.end_q) : $status (retries=$($m.retries))"
    }
}
Write-Host "========================================"

exit $(if ($overallSuccess) { 0 } else { 1 })
