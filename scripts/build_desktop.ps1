param(
    [string]$Entry = "desktop_app.py",
    [string]$Name = "NexusSearchDesktop"
)

$ErrorActionPreference = "Stop"

pyinstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name $Name `
  $Entry
