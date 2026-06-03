# Baseline small verify (5 questions) - test search endpoint fix
# Usage: Set $env:JUDGE_TOKEN before running, or pass via -JudgeToken parameter.
param(
    [string]$JudgeToken = $env:JUDGE_TOKEN,
    [string]$ProjectRoot = $env:OV_PROJECT_ROOT
)

if (-not $JudgeToken) {
    Write-Error "JudgeToken is required. Set `$env:JUDGE_TOKEN or pass -JudgeToken."
    exit 1
}
if (-not $ProjectRoot) {
    # Default to parent of scripts directory
    $ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
}

$ErrorActionPreference = "Continue"

$logDir = "$ProjectRoot/benchmark/memrouter_embedded/results/small_verify_baseline"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = "$logDir/run.log"

$python = (Get-Command python).Source
$script = "$ProjectRoot/benchmark/memrouter_embedded/scripts/eval_locomo_ov_with_memrouter_e2e.py"

& $python $script `
  --dataset "$ProjectRoot/benchmark/memrouter_embedded/data/conv30_exclude_cat5.json" `
  --route-labels "$ProjectRoot/benchmark/memrouter_embedded/data/locomo_e2e_route_labels.v4.jsonl" `
  --route-events-path "$ProjectRoot/benchmark/memrouter_embedded/logs/route_events.jsonl" `
  --ov-config "$ProjectRoot/benchmark/memrouter_embedded/config/ov_baseline.conf" `
  --ov-chat-endpoint "http://127.0.0.1:18790" `
  --ov-search-endpoint "http://127.0.0.1:1933" `
  --ov-api-key "ov-test-key-12345" `
  --ov-account "default" `
  --ov-user "default" `
  --openviking-root "$ProjectRoot" `
  --output-base "$logDir" `
  --force-memory-search `
  --judge `
  --judge-token "$JudgeToken" `
  --judge-base-url "https://api.minimaxi.com/anthropic" `
  --judge-model "anthropic/MiniMax-M2.7" `
  --start-question 1 `
  --end-question 5 2>&1 | Tee-Object -FilePath $logFile

Write-Host "`nDone. Log saved to: $logFile" -ForegroundColor Green
