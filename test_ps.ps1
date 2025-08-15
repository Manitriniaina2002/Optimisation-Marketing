Write-Host "=== SYSTEM INFORMATION ===" -ForegroundColor Cyan
Write-Host "`n1. Operating System:"
Get-ComputerInfo | Select-Object CsCaption, CsOSName, CsOSVersion, CsOSArchitecture | Format-List

Write-Host "`n2. Environment Variables:" -ForegroundColor Cyan
$env:PATH -split ';' | Where-Object { $_ -ne '' } | ForEach-Object { Write-Host "  $_" }

Write-Host "`n3. Python Installation:" -ForegroundColor Cyan
$pythonPaths = @(
    "C:\Python*",
    "${env:LOCALAPPDATA}\Programs\Python\Python*",
    "${env:ProgramFiles}\Python*",
    "${env:ProgramFiles(x86)}\Python*"
)

$pythonFound = $false
foreach ($path in $pythonPaths) {
    if (Test-Path $path) {
        $pythonFound = $true
        Write-Host "  Found Python at: $path"
        Get-ChildItem -Path $path -Directory | ForEach-Object {
            Write-Host "    - $($_.Name)"
        }
    }
}

if (-not $pythonFound) {
    Write-Host "  No Python installations found in common locations."
}

Write-Host "`n4. Current Directory Contents:" -ForegroundColor Cyan
Get-ChildItem -Path . -Force | Format-Table Name, Length, LastWriteTime

Write-Host "`n5. Running a Simple Python Command:" -ForegroundColor Cyan
try {
    $pythonOutput = python -c "import sys; print(f'Python {sys.version}')" 2>&1
    Write-Host "  $pythonOutput"
} catch {
    Write-Host "  Error running Python: $_" -ForegroundColor Red
}

Write-Host "`n=== TEST COMPLETE ===" -ForegroundColor Green
Read-Host "`nPress Enter to exit..."
