[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pyproject = Join-Path $repoRoot "pyproject.toml"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pyproject -PathType Leaf)) {
    throw "Repository root could not be resolved from scripts/dev-backend.ps1."
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository .venv is missing. Run 'uv sync --dev' from $repoRoot first."
}
$listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -ne $listener) {
    throw "Backend port $Port is already in use by process $($listener.OwningProcess). Stop it or pass -Port <port>."
}

$arguments = @(
    "-m", "mathmodel_ai", "serve",
    "--host", $HostAddress,
    "--port", $Port.ToString()
)
if (-not $NoReload) {
    $arguments += "--reload"
}

Write-Host "MathModel AI backend starting..."
Write-Host "Backend: http://${HostAddress}:$Port"
Write-Host "Docs:    http://${HostAddress}:$Port/docs"
Write-Host "This process runs in the foreground. Press Ctrl+C to stop."
Write-Host ""

Push-Location $repoRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
