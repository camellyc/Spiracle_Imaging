<#
    sync_from_drive.ps1

    Mirrors the latest top-level analysis files from the Google Drive
    "Analysis Code" folder into this git repo, then commits and pushes.

    Exclusions (never copied):
      - legacy/         (subfolder, skipped because we only copy top-level files)
      - __pycache__/    (subfolder, skipped)
      - desktop.ini     (Google Drive metadata)

    Usage (from anywhere):
      powershell -ExecutionPolicy Bypass -File C:\Users\Lylah\spiracle_imaging\sync_from_drive.ps1
    Or, from inside the repo folder:
      .\sync_from_drive.ps1
#>

$ErrorActionPreference = 'Stop'

# Source: the Google Drive shared "Analysis Code" folder
$Src = 'h:\.shortcut-targets-by-id\10pxdlRXtzFB-abwDGi0jOGOFFNm3pmFK\Tuthill Lab Shared\Yichen\Spiracle\Analysis Code'

# Destination: this repo (the folder this script lives in)
$Dst = $PSScriptRoot

# Repo housekeeping files that must never be removed or overwritten by the sync
$Keep = @('.gitignore', 'sync_from_drive.ps1', 'README.md')

if (-not (Test-Path $Src)) {
    throw "Source folder not found (is Google Drive mounted?): $Src"
}

# 1. Remove existing top-level content files so deletions in Drive propagate.
Get-ChildItem -Path $Dst -File |
    Where-Object { $Keep -notcontains $_.Name } |
    Remove-Item -Force

# 2. Copy top-level files from Drive, excluding desktop.ini.
#    -File with no -Recurse means subfolders (legacy, __pycache__) are skipped.
Get-ChildItem -Path $Src -File |
    Where-Object { $_.Name -ne 'desktop.ini' } |
    ForEach-Object { Copy-Item -Path $_.FullName -Destination $Dst -Force }

Write-Host "Synced files from Drive." -ForegroundColor Green

# 3. Commit and push only if something changed.
Set-Location $Dst
git add -A
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "No changes to push." -ForegroundColor Yellow
} else {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm'
    git commit -m "Sync from Drive: $stamp"
    git push
    Write-Host "Pushed updates to GitHub." -ForegroundColor Green
}
