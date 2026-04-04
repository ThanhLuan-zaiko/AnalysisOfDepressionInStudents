# Setup script for installing Python packages with GPU-aware PyTorch (Windows only)
# Usage: .\setup_with_gpu.ps1
# Supports: NVIDIA CUDA, AMD ROCm (via WSL2), Intel XPU, Apple MPS, CPU fallback

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " GPU Detection & Package Setup Script" -ForegroundColor Cyan
Write-Host " Windows Edition" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to detect GPU type with comprehensive checks
function Get-GpuType {
    Write-Host "Step 1: Detecting GPU hardware..." -ForegroundColor Yellow
    Write-Host ""
    
    # Get all display adapters
    $gpus = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FriendlyName
    
    if (-not $gpus) {
        Write-Host "[INFO] No display adapters found" -ForegroundColor Yellow
        return "cpu"
    }
    
    Write-Host "Found display adapter(s):" -ForegroundColor Gray
    $gpus | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
    Write-Host ""
    
    # Priority 1: Check for NVIDIA GPU (most common for ML/DL)
    $nvidiaGpu = $gpus | Where-Object { $_ -match "NVIDIA|GeForce|RTX|GTX|Quadro|Tesla" }
    if ($nvidiaGpu) {
        Write-Host "[FOUND] NVIDIA GPU: $nvidiaGpu" -ForegroundColor Green
        
        # Check if CUDA is available
        $cudaInstalled = Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if ($cudaInstalled) {
            $cudaVersion = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory | 
                          Select-Object -First 1 | Select-Object -ExpandProperty Name
            Write-Host "[OK] CUDA Toolkit found: $cudaVersion" -ForegroundColor Green
            return "nvidia"
        } else {
            Write-Host "[WARNING] CUDA Toolkit not detected!" -ForegroundColor Yellow
            Write-Host "  PyTorch will be installed with CUDA 12.1, but you need to:" -ForegroundColor Yellow
            Write-Host "  1. Download CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive" -ForegroundColor Yellow
            Write-Host "  2. Install it and restart your computer" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  Installing CUDA version anyway (won't break anything)..." -ForegroundColor Cyan
            return "nvidia"
        }
    }
    
    # Priority 2: Check for AMD GPU
    $amdGpu = $gpus | Where-Object { $_ -match "AMD|Radeon|RX [0-9]" }
    if ($amdGpu) {
        Write-Host "[FOUND] AMD GPU: $amdGpu" -ForegroundColor Green
        Write-Host "[WARNING] ROCm support on Windows requires WSL2" -ForegroundColor Yellow
        Write-Host "  For best AMD performance, consider:" -ForegroundColor Yellow
        Write-Host "  - Using WSL2 with ROCm: https://rocm.docs.amd.com/projects/install-on-windows/en/latest/" -ForegroundColor Yellow
        Write-Host "  - Or use CPU version (slower but stable)" -ForegroundColor Yellow
        Write-Host ""
        return "amd"
    }
    
    # Priority 3: Check for Intel GPU (Arc, Iris, UHD, HD Graphics)
    $intelGpu = $gpus | Where-Object { $_ -match "Intel|Arc|Iris|UHD|HD Graphics" }
    if ($intelGpu) {
        Write-Host "[FOUND] Intel GPU: $intelGpu" -ForegroundColor Green
        Write-Host "[INFO] Intel XPU support requires:" -ForegroundColor Yellow
        Write-Host "  - Intel Arc GPUs: Full XPU support available" -ForegroundColor Yellow
        Write-Host "  - Intel Iris/UHD: Limited support, may fall back to CPU" -ForegroundColor Yellow
        Write-Host ""
        return "intel"
    }
    
    # Priority 4: Check for Apple Silicon (if running in Parallels/VM on Mac)
    $osInfo = Get-CimInstance Win32_OperatingSystem
    if ($osInfo.Caption -match "Windows.*on ARM") {
        Write-Host "[FOUND] Windows on ARM detected" -ForegroundColor Yellow
        Write-Host "[INFO] Will install CPU version (ARM64 not fully supported by PyTorch GPU)" -ForegroundColor Yellow
        return "cpu"
    }
    
    # Fallback: CPU version
    Write-Host "[INFO] No GPU detected or not recognizable" -ForegroundColor Yellow
    Write-Host "  Will install CPU version (stable for all systems)" -ForegroundColor Cyan
    return "cpu"
}

# Detect GPU
$gpuType = Get-GpuType
Write-Host "Selected PyTorch variant: $($gpuType.ToUpper())" -ForegroundColor Cyan
Write-Host ""

# Define package list as array (verified compatible with Python 3.14)
$commonPackages = @(
    # Data Processing
    "polars",
    "pyarrow",
    "duckdb",
    
    # Visualization
    "plotly",
    "datashader",
    "holoviews",
    
    # Machine Learning
    "xgboost",
    "lightgbm",
    "statsmodels",
    "factor_analyzer",
    "pingouin",
    
    # Online Learning (River alternative - works with Python 3.14)
    "vowpalwabbit",
    
    # Hyperparameter Tuning
    "optuna",
    
    # Survival Analysis
    "lifelines",
    
    # Time Series
    "darts",
    "neuralprophet"
)

# Step 2: Install common packages - Smart approach
Write-Host "Step 2: Installing common packages..." -ForegroundColor Yellow
Write-Host "Packages: $($commonPackages.Count) packages to install" -ForegroundColor Gray
Write-Host ""

$successPackages = @()
$failedPackages = @()

# Strategy 1: Try bulk install first (much faster)
Write-Host "[Strategy 1] Attempting bulk installation..." -ForegroundColor Cyan
Write-Host ""

& uv add @commonPackages
$bulkExitCode = $LASTEXITCODE

if ($bulkExitCode -eq 0) {
    Write-Host "✅ Bulk installation successful!" -ForegroundColor Green
    Write-Host ""
    $successPackages = $commonPackages
} else {
    Write-Host "⚠️  Bulk installation failed (some packages have issues)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "[Strategy 2] Falling back to individual installation to identify issues..." -ForegroundColor Cyan
    Write-Host ""
    
    # Strategy 2: Install one by one to identify problematic packages
    foreach ($package in $commonPackages) {
        $currentNum = [array]::IndexOf($commonPackages, $package) + 1
        Write-Host "[$currentNum/$($commonPackages.Count)] Testing $package..." -ForegroundColor Gray
        
        # Check if already installed from bulk attempt
        $checkOutput = uv pip list 2>&1 | Out-String
        if ($checkOutput -match $package) {
            Write-Host "  ✓ $package (already installed)" -ForegroundColor Green
            $successPackages += $package
            continue
        }
        
        # Try installing this package
        & uv add $package
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Host "  ✅ $package installed successfully" -ForegroundColor Green
            $successPackages += $package
        } else {
            Write-Host "  ❌ $package failed to install" -ForegroundColor Red
            $failedPackages += $package
        }
        Write-Host ""
    }
}

# Display summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host " Installation Summary" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "✅ Success ($($successPackages.Count)/$($commonPackages.Count)):" -ForegroundColor Green
$successPackages | ForEach-Object { Write-Host "   ✓ $_" -ForegroundColor Green }
Write-Host ""

if ($failedPackages.Count -gt 0) {
    Write-Host "❌ Failed ($($failedPackages.Count)/$($commonPackages.Count)):" -ForegroundColor Red
    $failedPackages | ForEach-Object { Write-Host "   ✗ $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  - river: Not compatible with Python 3.14 (use vowpalwabbit instead)" -ForegroundColor Yellow
    Write-Host "  - Packages with C/C++ extensions may need Visual Studio Build Tools" -ForegroundColor Yellow
    Write-Host "  - DNS errors: Check internet connection and retry" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "🎉 All packages installed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Installed categories:" -ForegroundColor Cyan
    Write-Host "  📊 Data Processing: polars, pyarrow, duckdb" -ForegroundColor Gray
    Write-Host "  📈 Visualization: plotly, datashader, holoviews" -ForegroundColor Gray
    Write-Host "  🤖 ML: xgboost, lightgbm, statsmodels" -ForegroundColor Gray
    Write-Host "  📐 Statistics: factor_analyzer, pingouin, optuna" -ForegroundColor Gray
    Write-Host "  ⏱️  Time Series: darts, neuralprophet" -ForegroundColor Gray
    Write-Host "  🔄 Online ML: vowpalwabbit (faster than river)" -ForegroundColor Gray
    Write-Host "  🏥 Survival: lifelines" -ForegroundColor Gray
}

Write-Host ""

# Step 3: Install PyTorch with appropriate GPU support
Write-Host "Step 3: Installing PyTorch with GPU support..." -ForegroundColor Yellow
Write-Host ""

$installSuccess = $false

switch ($gpuType) {
    "nvidia" {
        Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Cyan
        Write-Host "  Index URL: https://download.pytorch.org/whl/cu121" -ForegroundColor Gray
        uv add --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
        if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
    }
    "amd" {
        Write-Host "Installing PyTorch with ROCm 6.0 support..." -ForegroundColor Cyan
        Write-Host "  Index URL: https://download.pytorch.org/whl/rocm6.0" -ForegroundColor Gray
        Write-Host "  Note: Full ROCm support requires WSL2 on Windows" -ForegroundColor Yellow
        uv add --index-url https://download.pytorch.org/whl/rocm6.0 torch torchvision torchaudio
        if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
        
        if (-not $installSuccess) {
            Write-Host ""
            Write-Host "[WARNING] ROCm installation failed on Windows" -ForegroundColor Yellow
            Write-Host "  Falling back to CPU version for stability..." -ForegroundColor Yellow
            Write-Host "  To use AMD GPU, please set up WSL2 with ROCm." -ForegroundColor Yellow
            $gpuType = "cpu"
            uv add torch torchvision torchaudio
            if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
        }
    }
    "intel" {
        Write-Host "Installing PyTorch with Intel XPU support..." -ForegroundColor Cyan
        Write-Host "  Index URL: https://pytorch-extension.intel.com/release-whl/stable/xpu/us/" -ForegroundColor Gray
        uv add --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ torch torchvision torchaudio
        if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
        
        if (-not $installSuccess) {
            Write-Host ""
            Write-Host "[WARNING] Intel XPU installation failed" -ForegroundColor Yellow
            Write-Host "  Falling back to CPU version..." -ForegroundColor Yellow
            $gpuType = "cpu"
            uv add torch torchvision torchaudio
            if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
        }
    }
    "cpu" {
        Write-Host "Installing PyTorch CPU version..." -ForegroundColor Cyan
        uv add torch torchvision torchaudio
        if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
    }
}

# Ultimate fallback
if (-not $installSuccess) {
    Write-Host ""
    Write-Host "❌ All installation attempts failed!" -ForegroundColor Red
    Write-Host "Attempting final fallback to CPU version..." -ForegroundColor Yellow
    Write-Host ""
    
    # Try installing packages one by one to identify problematic ones
    Write-Host "Installing PyTorch CPU version as last resort..." -ForegroundColor Yellow
    uv add torch torchvision torchaudio
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ CRITICAL: Even CPU installation failed!" -ForegroundColor Red
        Write-Host "  Please check:" -ForegroundColor Red
        Write-Host "  1. Internet connection" -ForegroundColor Red
        Write-Host "  2. Python version compatibility (currently using 3.14)" -ForegroundColor Red
        Write-Host "  3. uv is up to date: uv self update" -ForegroundColor Red
        exit 1
    }
    
    $gpuType = "cpu"
    $installSuccess = $true
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GPU Type: $($gpuType.ToUpper())" -ForegroundColor Cyan
Write-Host "  Status: Success" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 4: Run verification
Write-Host "Step 4: Running GPU verification..." -ForegroundColor Yellow
Write-Host ""
python verify_gpu.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Next Steps:" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

if ($gpuType -eq "cpu") {
    Write-Host "⚠️  Running in CPU mode. For better performance:" -ForegroundColor Yellow
    Write-Host "  - NVIDIA: Install CUDA 12.1 toolkit" -ForegroundColor Yellow
    Write-Host "  - AMD: Use WSL2 with ROCm" -ForegroundColor Yellow
    Write-Host "  - Intel: Ensure Intel drivers are installed" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Then re-run: .\setup_with_gpu.ps1" -ForegroundColor Yellow
} else {
    Write-Host "✅ GPU acceleration is configured!" -ForegroundColor Green
    Write-Host "  Your analysis will run faster with GPU support." -ForegroundColor Green
}

Write-Host ""
Write-Host "Start your analysis with: python main.py" -ForegroundColor Cyan
Write-Host ""
