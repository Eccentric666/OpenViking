# LoCoMo + OpenViking with Graph Backend — E2E Runner
# Usage:
#   .\run_e2e_graph.ps1 -LimitQuestions 5 -ForceMemorySearch
#   .\run_e2e_graph.ps1 -Category 3 -ForceMemorySearch -Judge -JudgeToken "sk-xxxxx"

param(
    [string]$Category = "",
    [int]$LimitQuestions = 0,
    [int]$LimitSamples = 0,
    [switch]$ForceMemorySearch = $false,
    [switch]$Judge = $false,
    [string]$JudgeToken = "",
    [string]$JudgeModel = "MiniMax-M2.7",
    [string]$Dataset = "",
    [string]$RouteLabels = "",
    [string]$OVConfig = "",
    [string]$Workspace = "",
    [string]$OutputBase = "",
    [switch]$StrictPreflight = $true
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Resolve-Path "$ScriptDir\..\..\..").Path
$BenchmarkDir = "$ProjectRoot\benchmark\memrouter_embedded"

# Defaults
if (-not $Dataset) {
    $Dataset = "$BenchmarkDir\data\locomo10.json"
}
if (-not $RouteLabels) {
    $RouteLabels = "$BenchmarkDir\data\locomo_e2e_route_labels.v3.jsonl"
}
if (-not $OVConfig) {
    $OVConfig = "$BenchmarkDir\config\ov+graph.conf"
}
if (-not $Workspace) {
    $Workspace = "$BenchmarkDir\workspace"
}
if (-not $OutputBase) {
    $OutputBase = "$ProjectRoot\benchmark\runs"
}

$PyScript = "$BenchmarkDir\scripts\eval_locomo_ov_with_graph_e2e.py"

$ArgsList = @(
    "--dataset", $Dataset,
    "--route-labels", $RouteLabels,
    "--ov-config", $OVConfig,
    "--workspace", $Workspace,
    "--output-base", $OutputBase,
    "--ov-chat-endpoint", "http://127.0.0.1:1933",
    "--ov-api-key", "ov-test-key-12345"
)

if ($Category) {
    $ArgsList += @("--category", $Category)
}
if ($LimitQuestions -gt 0) {
    $ArgsList += @("--limit-questions", $LimitQuestions)
}
if ($LimitSamples -gt 0) {
    $ArgsList += @("--limit-samples", $LimitSamples)
}
if ($ForceMemorySearch) {
    $ArgsList += "--force-memory-search"
}
if ($Judge) {
    $ArgsList += "--judge"
    if ($JudgeToken) {
        $ArgsList += @("--judge-token", $JudgeToken)
    }
    $ArgsList += @("--judge-model", $JudgeModel)
}
if (-not $StrictPreflight) {
    $ArgsList += "--no-strict-preflight"
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "LoCoMo + OV Graph Backend E2E Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Dataset:       $Dataset"
Write-Host "Route labels:  $RouteLabels"
Write-Host "OV config:     $OVConfig"
Write-Host "Workspace:     $Workspace"
Write-Host "Output base:   $OutputBase"
Write-Host "========================================"

$Python = "python"
if (Get-Command "uv" -ErrorAction SilentlyContinue) {
    $Python = "uv run python"
}

& $Python $PyScript @ArgsList

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Evaluation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[OK] Evaluation complete." -ForegroundColor Green
