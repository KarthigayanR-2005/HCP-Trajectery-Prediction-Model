# Setup script for HCP Project
Write-Host "Starting environment setup for HCP project..." -ForegroundColor Green

# 1. Create directory tree
$dirs = @(
    "data/nuscenes",
    "data/waymo",
    "hcp",
    "mtr_core",
    "fusion",
    "outputs/route_graphs",
    "outputs/maps",
    "outputs/motion_states",
    "eval",
    "paper",
    "backend",
    "ui"
)

foreach ($dir in $dirs) {
    $path = Join-Path -Path $PSScriptRoot -ChildPath $dir
    if (-not (Test-Path -Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Cyan
    } else {
        Write-Host "Directory already exists: $dir" -ForegroundColor Yellow
    }
}

# 2. Setup packages
Write-Host "Installing dependencies..." -ForegroundColor Green
python -m pip install -r (Join-Path -Path $PSScriptRoot -ChildPath "requirements.txt")

Write-Host "Setup completed successfully!" -ForegroundColor Green
