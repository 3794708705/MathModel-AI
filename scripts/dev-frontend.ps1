[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 5173
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$frontendRoot = Join-Path $repoRoot "frontend"
$nodeModules = Join-Path $frontendRoot "node_modules"
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot "package.json") -PathType Leaf)) {
    throw "Frontend directory could not be resolved from scripts/dev-frontend.ps1."
}
if (-not (Test-Path -LiteralPath $nodeModules -PathType Container)) {
    throw "Frontend dependencies are missing. Run 'npm install' in $frontendRoot first."
}
if ($null -eq $npm) {
    throw "npm.cmd was not found on PATH. Install Node.js before starting the frontend."
}
$listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -ne $listener) {
    throw "Frontend port $Port is already in use by process $($listener.OwningProcess). Stop it or pass -Port <port>."
}
$apiBaseUrl = $env:VITE_API_BASE_URL
if ([string]::IsNullOrWhiteSpace($apiBaseUrl)) {
    $apiBaseUrl = "http://127.0.0.1:8000"
}

Write-Host "MathModel AI frontend starting..."
Write-Host "Frontend: http://${HostAddress}:$Port"
Write-Host "API:      $apiBaseUrl"
$frontendOrigin = "http://${HostAddress}:$Port"
$backendReachable = $false
try {
    $health = Invoke-WebRequest -Uri "$($apiBaseUrl.TrimEnd('/'))/health/live" -TimeoutSec 2 -UseBasicParsing
    if ($health.StatusCode -ge 200 -and $health.StatusCode -lt 300) {
        $backendReachable = $true
        Write-Host "Backend:  connected" -ForegroundColor Green
    }
}
catch {
    Write-Warning "Backend is not reachable at $apiBaseUrl. Start it in another PowerShell with: .\scripts\dev-backend.ps1"
}
if ($backendReachable) {
    try {
        $preflight = Invoke-WebRequest `
            -Uri "$($apiBaseUrl.TrimEnd('/'))/health/live" `
            -Method Options `
            -Headers @{
                Origin = $frontendOrigin
                "Access-Control-Request-Method" = "GET"
            } `
            -TimeoutSec 2 `
            -UseBasicParsing
        $allowedOrigin = $preflight.Headers["Access-Control-Allow-Origin"]
        if ($allowedOrigin -ne $frontendOrigin) {
            throw "Access-Control-Allow-Origin was '$allowedOrigin'."
        }
        Write-Host "CORS:     $frontendOrigin allowed" -ForegroundColor Green
    }
    catch {
        throw "Backend at $apiBaseUrl does not allow frontend origin $frontendOrigin. Add it to MM_CORS_ORIGINS and restart the backend. $($_.Exception.Message)"
    }
}
Write-Host "This process runs in the foreground. Press Ctrl+C to stop."
Write-Host ""

Push-Location $frontendRoot
try {
    & $npm.Source run dev -- --host $HostAddress --port $Port
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
