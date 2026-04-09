# Setup script for installing Python packages with GPU-aware PyTorch (Windows only)
# Usage: .\setup_with_gpu.ps1
# Supports: NVIDIA CUDA, AMD ROCm (via WSL2), Intel XPU, Apple MPS, CPU fallback
# Requires: Administrator privileges to bypass Smart App Control restrictions

$ErrorActionPreference = "Stop"

# ============================================================
# Administrator Privilege Check & Auto-Elevation
# ============================================================
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    return $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    Write-Host "[!] Administrator privileges required!" -ForegroundColor Red
    Write-Host "    Smart App Control (SAC) may block Python 3.14 DLLs without admin rights." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    Attempting to re-launch with Administrator privileges..." -ForegroundColor Cyan
    Write-Host ""

    $scriptPath = $MyInvocation.MyCommand.Path
    if (-not $scriptPath) {
        $scriptPath = $PSCommandPath
    }

    try {
        Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$scriptPath`""
        Write-Host "[OK] Admin window launched. Please use the new window." -ForegroundColor Green
        exit 0
    } catch {
        Write-Host "[FAILED] Could not elevate to Administrator." -ForegroundColor Red
        Write-Host "  Please manually run: Right-click -> 'Run as Administrator'" -ForegroundColor Yellow
        Write-Host "  Continuing without admin rights (SAC blocks may occur)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
    }
} else {
    Write-Host "[OK] Running with Administrator privileges" -ForegroundColor Green
}

Write-Host ""

# ============================================================
# Smart App Control (SAC) Mitigation
# ============================================================
function Mitigate-SAC-Blocks {
    Write-Host "[SAC] Checking Smart App Control status..." -ForegroundColor Cyan

    try {
        $sacStatus = Get-CimInstance -Namespace 'root\Microsoft\Windows\DeviceGuard' -ClassName 'Win32_DeviceGuard' -ErrorAction SilentlyContinue
        $sacRunning = $sacStatus.SecurityServicesRunning -contains 2
        $codeIntegrity = $sacStatus.SecurityServicesConfigured -contains 1

        if ($sacRunning) {
            Write-Host "[WARN] Smart App Control is ACTIVE" -ForegroundColor Yellow
            Write-Host "       This may block unsigned Python DLLs (_overlapped.pyd, etc.)" -ForegroundColor Yellow
            Write-Host ""

            # Try to unblock common Python DLLs
            $pythonBasePaths = @(
                "$env:USERPROFILE\AppData\Roaming\uv\python",
                "$env:LOCALAPPDATA\uv\python",
                (Resolve-Path ".venv\Scripts\python.exe" -ErrorAction SilentlyContinue | Split-Path),
                (Resolve-Path ".venv\Lib\site-packages" -ErrorAction SilentlyContinue | Select-Object -First 1)
            )

            $unblockedCount = 0
            foreach ($basePath in $pythonBasePaths) {
                if ($basePath -and (Test-Path $basePath)) {
                    try {
                        $pydFiles = Get-ChildItem -Path $basePath -Recurse -Filter "*.pyd" -ErrorAction SilentlyContinue
                        foreach ($file in $pydFiles) {
                            Unblock-File -Path $file.FullName -ErrorAction SilentlyContinue
                            $unblockedCount++
                        }
                    } catch {}
                }
            }

            if ($unblockedCount -gt 0) {
                Write-Host "[OK] Attempted to unblock $unblockedCount Python DLL files" -ForegroundColor Green
            }

            Write-Host ""
            Write-Host "[CRITICAL] Smart App Control cannot be bypassed by script!" -ForegroundColor Red
            Write-Host "           You MUST disable it manually:" -ForegroundColor Red
            Write-Host "    1. Windows Security -> App & browser control -> Smart App Control -> Turn off" -ForegroundColor Gray
            Write-Host "    2. Restart your computer" -ForegroundColor Gray
            Write-Host ""
        } else {
            # SAC is OFF - just unblock files for good measure, no warning needed
            $pythonBasePaths = @(
                "$env:USERPROFILE\AppData\Roaming\uv\python",
                "$env:LOCALAPPDATA\uv\python"
            )

            $unblockedCount = 0
            foreach ($basePath in $pythonBasePaths) {
                if ($basePath -and (Test-Path $basePath)) {
                    try {
                        $pydFiles = Get-ChildItem -Path $basePath -Recurse -Filter "*.pyd" -ErrorAction SilentlyContinue
                        foreach ($file in $pydFiles) {
                            Unblock-File -Path $file.FullName -ErrorAction SilentlyContinue
                            $unblockedCount++
                        }
                    } catch {}
                }
            }

            if ($unblockedCount -gt 0) {
                Write-Host "[OK] Smart App Control is OFF. Unblocked $unblockedCount DLL files for safety." -ForegroundColor Green
            } else {
                Write-Host "[OK] Smart App Control is OFF" -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "[INFO] Could not check SAC status" -ForegroundColor Gray
    }
}

function Invoke-LoggedNativeCommand {
    param(
        [scriptblock]$Command,
        [string]$Indent = "    ",
        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )

    # Some build tools log progress to stderr; don't let PowerShell treat that as a fatal error.
    $previousErrorActionPreference = $ErrorActionPreference
    $hasNativeCommandPreference = $null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)
    if ($hasNativeCommandPreference) {
        $previousNativeCommandPreference = $PSNativeCommandUseErrorActionPreference
    }

    try {
        $script:ErrorActionPreference = "Continue"
        if ($hasNativeCommandPreference) {
            $script:PSNativeCommandUseErrorActionPreference = $false
        }

        & $Command 2>&1 | ForEach-Object {
            $line = if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.ToString()
            } else {
                "$_"
            }

            if (-not [string]::IsNullOrWhiteSpace($line)) {
                Write-Host "$Indent$line" -ForegroundColor $Color
            }
        }

        return $LASTEXITCODE
    } finally {
        $script:ErrorActionPreference = $previousErrorActionPreference
        if ($hasNativeCommandPreference) {
            $script:PSNativeCommandUseErrorActionPreference = $previousNativeCommandPreference
        }
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " GPU Detection & Package Setup Script" -ForegroundColor Cyan
Write-Host " Windows Edition (Admin Mode)" -ForegroundColor Yellow
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
        return @{ Type = "cpu"; CudaVersion = $null }
    }

    Write-Host "Found display adapter(s):" -ForegroundColor Gray
    $gpus | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
    Write-Host ""

    # Priority 1: Check for NVIDIA GPU (most common for ML/DL)
    $nvidiaGpu = $gpus | Where-Object { $_ -match "NVIDIA|GeForce|RTX|GTX|Quadro|Tesla" }
    if ($nvidiaGpu) {
        Write-Host "[FOUND] NVIDIA GPU: $nvidiaGpu" -ForegroundColor Green

        # Check nvidia-smi for driver CUDA support (most reliable)
        $driverMaxCuda = $null
        try {
            $smiOutput = & nvidia-smi 2>&1
            $driverMaxCuda = ($smiOutput | Select-String "CUDA Version: (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }) | Select-Object -First 1
        } catch {}

        # Also check CUDA Toolkit
        $cudaToolkitVersion = $null
        $cudaInstalled = Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if ($cudaInstalled) {
            $cudaToolkitVersion = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory |
                                  Select-Object -First 1 | Select-Object -ExpandProperty Name
        }

        # Determine the effective CUDA version (minimum of toolkit and driver support)
        $effectiveCuda = $null
        if ($driverMaxCuda) {
            Write-Host "[OK] NVIDIA driver supports CUDA up to: $driverMaxCuda" -ForegroundColor Green
            $effectiveCuda = $driverMaxCuda
        } elseif ($cudaToolkitVersion) {
            Write-Host "[OK] CUDA Toolkit found: $cudaToolkitVersion" -ForegroundColor Green
            Write-Host "[INFO] nvidia-smi not available, using toolkit version" -ForegroundColor Yellow
            $effectiveCuda = $cudaToolkitVersion -replace 'v', ''
        } else {
            Write-Host "[WARNING] No CUDA found. Install CUDA toolkit or update drivers." -ForegroundColor Yellow
            return @{ Type = "nvidia"; CudaVersion = $null; WheelVariant = "cu126" }
        }

        # Map effective CUDA version to PyTorch wheel variant
        # Use the version that matches driver capability, not necessarily toolkit
        $cudaMajor = [int]$effectiveCuda.Split('.')[0]
        $cudaMinor = [int]$effectiveCuda.Split('.')[1]

        $wheelVariant = "cu126"  # Default for modern setups
        if ($cudaMajor -ge 13) {
            # CUDA 13.x support depends on driver version
            # Driver 570+ typically supports cu130, but older GPUs (Pascal/Turing) may need cu126
            # Check if we have a very recent driver (570+)
            try {
                $driverVersionRaw = ($smiOutput | Select-String "Driver Version: (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }) | Select-Object -First 1
                $driverMajor = [int]($driverVersionRaw -replace '\..*$', '')
                if ($driverMajor -ge 570) {
                    $wheelVariant = "cu130"
                    Write-Host "[INFO] CUDA 13.x + Driver 570+ detected - Using cu130" -ForegroundColor Green
                } else {
                    $wheelVariant = "cu126"
                    Write-Host "[INFO] CUDA 13.x detected but driver < 570 - Using cu126 for broader GPU compatibility" -ForegroundColor Yellow
                }
            } catch {
                $wheelVariant = "cu126"
                Write-Host "[INFO] CUDA 13.x detected - Using cu126 (safe fallback)" -ForegroundColor Yellow
            }
        } elseif ($cudaMajor -eq 12) {
            if ($cudaMinor -ge 6) {
                $wheelVariant = "cu126"
            } elseif ($cudaMinor -ge 4) {
                $wheelVariant = "cu124"
            } else {
                $wheelVariant = "cu121"
            }
        } elseif ($cudaMajor -eq 11) {
            $wheelVariant = "cu118"
        }

        Write-Host "[INFO] Selected wheel variant: $wheelVariant (matches driver capability)" -ForegroundColor Green
        return @{ Type = "nvidia"; CudaVersion = $effectiveCuda; WheelVariant = $wheelVariant }
    }

    # Priority 2: Check for AMD GPU
    $amdGpu = $gpus | Where-Object { $_ -match "AMD|Radeon|RX [0-9]" }
    if ($amdGpu) {
        Write-Host "[FOUND] AMD GPU: $amdGpu" -ForegroundColor Green

        # Check if running in WSL2 (ROCm works better on Linux)
        $isWSL = Test-Path "/proc/version"
        if ($isWSL) {
            Write-Host "[OK] Running in WSL2 - Full ROCm support available!" -ForegroundColor Green

            # Check Python version for ROCm compatibility
            $pythonVersion = python --version 2>&1 | Select-String -Pattern "3\.(\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }

            # ROCm supports Python 3.12-3.14 in newer versions
            if ([int]$pythonVersion -ge 12 -and [int]$pythonVersion -le 14) {
                Write-Host "[OK] Python 3.$pythonVersion compatible with ROCm" -ForegroundColor Green
                return @{ Type = "amd"; CudaVersion = $null; WheelVariant = "rocm" }
            } else {
                Write-Host "[WARNING] Python 3.$pythonVersion may not be fully supported by ROCm" -ForegroundColor Yellow
                Write-Host "  Falling back to CPU version..." -ForegroundColor Yellow
                return @{ Type = "cpu"; CudaVersion = $null; WheelVariant = $null }
            }
        } else {
            # Native Windows - ROCm support is limited
            Write-Host "[WARNING] ROCm on native Windows has limited support" -ForegroundColor Yellow
            Write-Host "  Options:" -ForegroundColor Yellow
            Write-Host "  - Use WSL2 for full ROCm: wsl --install" -ForegroundColor Yellow
            Write-Host "  - Or use CPU version (slower)" -ForegroundColor Yellow
            Write-Host ""
            return @{ Type = "amd"; CudaVersion = $null; WheelVariant = "rocm" }
        }
    }

    # Priority 3: Check for Intel GPU (Arc, Iris, UHD, HD Graphics)
    $intelGpu = $gpus | Where-Object { $_ -match "Intel|Arc|Iris|UHD|HD Graphics" }
    if ($intelGpu) {
        $intelModel = $intelGpu | Select-Object -First 1
        Write-Host "[FOUND] Intel GPU: $intelModel" -ForegroundColor Green

        # Intel Arc = full XPU support, Intel UHD/Iris = limited
        if ($intelModel -match "Arc") {
            Write-Host "[OK] Intel Arc GPU detected - Full XPU support available" -ForegroundColor Green
            return @{ Type = "intel"; CudaVersion = $null; WheelVariant = "xpu"; GpuModel = $intelModel }
        } else {
            Write-Host "[INFO] Intel integrated GPU detected - Limited XPU support" -ForegroundColor Yellow
            Write-Host "  Intel Iris/UHD may fall back to CPU for some operations" -ForegroundColor Yellow
            return @{ Type = "intel"; CudaVersion = $null; WheelVariant = "xpu"; GpuModel = $intelModel }
        }
    }

    # Priority 4: Check for Apple Silicon (if running in Parallels/VM on Mac)
    $osInfo = Get-CimInstance Win32_OperatingSystem
    if ($osInfo.Caption -match "Windows.*on ARM") {
        Write-Host "[FOUND] Windows on ARM detected" -ForegroundColor Yellow
        Write-Host "[INFO] Will install CPU version (ARM64 not fully supported by PyTorch GPU)" -ForegroundColor Yellow
        return @{ Type = "cpu"; CudaVersion = $null; WheelVariant = $null }
    }

    # Fallback: CPU version
    Write-Host "[INFO] No GPU detected or not recognizable" -ForegroundColor Yellow
    Write-Host "  Will install CPU version (stable for all systems)" -ForegroundColor Cyan
    return @{ Type = "cpu"; CudaVersion = $null; WheelVariant = $null }
}

# Detect GPU
$gpuInfo = Get-GpuType
$gpuType = $gpuInfo.Type
Write-Host "Selected PyTorch variant: $($gpuType.ToUpper())" -ForegroundColor Cyan
if ($gpuInfo.WheelVariant) {
    Write-Host "Wheel variant: $($gpuInfo.WheelVariant)" -ForegroundColor Cyan
}
Write-Host ""

# Run SAC mitigation before installing packages
Mitigate-SAC-Blocks
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
    "catboost",
    "statsmodels",
    "factor_analyzer",
    "pingouin",
    "prince",
    "pygam",

    # Scikit-learn ecosystem (required by many ML packages)
    "scikit-learn",
    "shap",
    "numpy",
    "pandas",
    "joblib",

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
    Write-Host "  🤖 ML: xgboost, lightgbm, catboost, statsmodels, pygam" -ForegroundColor Gray
    Write-Host "  📐 Statistics: factor_analyzer, pingouin, prince, optuna" -ForegroundColor Gray
    Write-Host "  🔍 Explainability: shap" -ForegroundColor Gray
    Write-Host "  📦 Core ML: scikit-learn, numpy, pandas, joblib" -ForegroundColor Gray
    Write-Host "  ⏱️  Time Series: darts, neuralprophet" -ForegroundColor Gray
    Write-Host "  🔄 Online ML: vowpalwabbit" -ForegroundColor Gray
    Write-Host "  🏥 Survival: lifelines" -ForegroundColor Gray
}

Write-Host ""

# Step 3: Install PyTorch with appropriate GPU support
Write-Host "Step 3: Installing PyTorch with GPU support..." -ForegroundColor Yellow
Write-Host ""

$installSuccess = $false

switch ($gpuType) {
    "nvidia" {
        $wheelVariant = $gpuInfo.WheelVariant
        $indexUrl = "https://download.pytorch.org/whl/$wheelVariant"
        Write-Host "Installing PyTorch with $wheelVariant support..." -ForegroundColor Cyan
        Write-Host "  Index URL: $indexUrl" -ForegroundColor Gray
        Write-Host ""

        # CRITICAL: Use pip install with explicit index to avoid CPU fallback
        # uv add may resolve to CPU version from PyPI if dependency conflicts occur
        Write-Host "  Force installing GPU wheels (avoiding CPU fallback)..." -ForegroundColor Gray
        & uv pip install torch torchvision torchaudio --index-url $indexUrl --force-reinstall --no-deps
        $installExitCode = $LASTEXITCODE

        if ($installExitCode -eq 0) {
            $installSuccess = $true
            Write-Host "  ✅ PyTorch GPU packages installed successfully" -ForegroundColor Green
        } else {
            Write-Host "  ❌ GPU installation failed" -ForegroundColor Red
            Write-Host "  Checking CUDA compatibility..." -ForegroundColor Yellow

            # Check driver max CUDA support via nvidia-smi
            $smiOutput = & nvidia-smi 2>&1
            $driverMaxCuda = ($smiOutput | Select-String "CUDA Version: (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }) | Select-Object -First 1

            if ($driverMaxCuda) {
                Write-Host "  Driver supports max CUDA: $driverMaxCuda" -ForegroundColor Yellow

                # Map to appropriate wheel variant
                $driverMajor = [int]$driverMaxCuda.Split('.')[0]
                $driverMinor = [int]$driverMaxCuda.Split('.')[1]

                $fallbackVariant = "cu126"  # Default fallback
                if ($driverMajor -ge 12) {
                    if ($driverMinor -ge 6) { $fallbackVariant = "cu126" }
                    elseif ($driverMinor -ge 4) { $fallbackVariant = "cu124" }
                    elseif ($driverMinor -ge 1) { $fallbackVariant = "cu121" }
                } elseif ($driverMajor -ge 11) {
                    $fallbackVariant = "cu118"
                }

                if ($fallbackVariant -ne $wheelVariant) {
                    Write-Host "  Retrying with $fallbackVariant wheels (driver-compatible)..." -ForegroundColor Cyan
                    $fallbackUrl = "https://download.pytorch.org/whl/$fallbackVariant"
                    & uv pip install torch torchvision torchaudio --index-url $fallbackUrl --force-reinstall --no-deps
                    if ($LASTEXITCODE -eq 0) {
                        $installSuccess = $true
                        Write-Host "  ✅ PyTorch installed with $fallbackVariant (driver-compatible)" -ForegroundColor Green
                    }
                }
            }
        }

        # Verify GPU works
        if ($installSuccess) {
            Write-Host "  Verifying GPU functionality..." -ForegroundColor Gray
            $testOutput = & .venv\Scripts\python.exe -c "import torch; print('cuda_available' if torch.cuda.is_available() else 'cuda_unavailable'); print(torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>&1

            if ($testOutput -match "cuda_available") {
                Write-Host "  ✅ CUDA verification PASSED!" -ForegroundColor Green
            } else {
                Write-Host "  ⚠️  CUDA not available despite GPU wheels installed" -ForegroundColor Yellow
                Write-Host "  Possible causes:" -ForegroundColor Yellow
                Write-Host "    - NVIDIA driver outdated (update from nvidia.com)" -ForegroundColor Yellow
                Write-Host "    - CUDA toolkit version mismatch" -ForegroundColor Yellow
                Write-Host "    - GPU compute capability too old (< 3.5)" -ForegroundColor Yellow
            }
        } else {
            Write-Host ""
            Write-Host "  ❌ CRITICAL: Could not install PyTorch with CUDA support" -ForegroundColor Red
            Write-Host "  Installing CPU version as last resort..." -ForegroundColor Yellow
            $gpuType = "cpu"
            & uv pip install torch torchvision torchaudio --no-deps
            if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
        }
    }
    "amd" {
        Write-Host "Installing PyTorch with AMD ROCm support..." -ForegroundColor Cyan
        Write-Host ""

        # Check if running in WSL2
        $isWSL = Test-Path "/proc/version"

        if ($isWSL) {
            # WSL2 - Full ROCm support via pip
            Write-Host "[OK] Running in WSL2 - Installing ROCm via pip..." -ForegroundColor Green

            # Try PyTorch ROCm wheels first
            Write-Host "  Installing PyTorch ROCm..." -ForegroundColor Gray
            & uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 --force-reinstall --no-deps
            if ($LASTEXITCODE -eq 0) {
                $installSuccess = $true
                Write-Host "  ✅ PyTorch ROCm installed" -ForegroundColor Green
            }

            if (-not $installSuccess) {
                Write-Host "  Falling back to CPU version..." -ForegroundColor Yellow
                $gpuType = "cpu"
                & uv pip install torch torchvision torchaudio --no-deps
                if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
            }
        } else {
            # Native Windows - ROCm has limited support
            Write-Host "[WARNING] ROCm on native Windows has limited support" -ForegroundColor Yellow
            Write-Host "  Recommended: Use WSL2 for full ROCm support (wsl --install)" -ForegroundColor Yellow
            Write-Host ""

            # Check Python version
            $pythonVersion = python --version 2>&1 | Select-String -Pattern "3\.(\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }

            if ([int]$pythonVersion -eq 12) {
                # Python 3.12 - try official ROCm wheels
                $rocmVersion = "7.2.1"
                $rocmBaseUrl = "https://repo.radeon.com/rocm/windows/rocm-rel-$rocmVersion"

                Write-Host "  Attempting ROCm SDK installation for Python 3.12..." -ForegroundColor Gray
                & uv pip install `
                    "$rocmBaseUrl/rocm_sdk_core-$rocmVersion-py3-none-win_amd64.whl" `
                    "$rocmBaseUrl/rocm_sdk_devel-$rocmVersion-py3-none-win_amd64.whl"

                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  Installing PyTorch with ROCm..." -ForegroundColor Gray
                    & uv pip install `
                        "$rocmBaseUrl/torch-2.9.1%2Brocm$rocmVersion-cp312-cp312-win_amd64.whl" `
                        "$rocmBaseUrl/torchaudio-2.9.1%2Brocm$rocmVersion-cp312-cp312-win_amd64.whl" `
                        "$rocmBaseUrl/torchvision-0.24.1%2Brocm$rocmVersion-cp312-cp312-win_amd64.whl"

                    if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
                }
            } else {
                # Python 3.13+ - ROCm not available on Windows
                Write-Host "[WARNING] ROCm only supports Python 3.12 on Windows" -ForegroundColor Yellow
                Write-Host "  Current Python: 3.$pythonVersion" -ForegroundColor Yellow
                Write-Host "  Options:" -ForegroundColor Yellow
                Write-Host "    1. Downgrade to Python 3.12 for ROCm support" -ForegroundColor Yellow
                Write-Host "    2. Use WSL2 for full ROCm support" -ForegroundColor Yellow
                Write-Host "    3. Use CPU version (slower)" -ForegroundColor Yellow
                Write-Host ""
            }

            if (-not $installSuccess) {
                Write-Host "  Falling back to CPU version..." -ForegroundColor Yellow
                $gpuType = "cpu"
                & uv pip install torch torchvision torchaudio --no-deps
                if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
            }
        }
    }
    "intel" {
        Write-Host "Installing PyTorch with Intel XPU support..." -ForegroundColor Cyan
        Write-Host "  Method: Intel Extension for PyTorch (IPEX)" -ForegroundColor Gray
        Write-Host ""

        # Check if Intel Arc (discrete) or integrated GPU
        $gpuModel = $gpuInfo.GpuModel
        if ($gpuModel -match "Arc") {
            Write-Host "[OK] Intel Arc GPU detected - Full XPU support" -ForegroundColor Green
        } else {
            Write-Host "[INFO] Intel integrated GPU ($gpuModel) - Limited XPU support" -ForegroundColor Yellow
            Write-Host "  Some operations may fall back to CPU" -ForegroundColor Yellow
        }

        # Step 1: Install base PyTorch (CPU) first - required for IPEX
        $baseTorchInstalled = $false
        Write-Host "  Step 1: Installing base PyTorch..." -ForegroundColor Gray
        & uv pip install torch torchvision torchaudio --no-deps
        $baseExitCode = $LASTEXITCODE

        if ($baseExitCode -ne 0) {
            Write-Host "  ❌ Failed to install base PyTorch" -ForegroundColor Red
            Write-Host "  Trying alternative install method..." -ForegroundColor Yellow
            & uv add torch torchvision torchaudio
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ❌ All installation attempts failed" -ForegroundColor Red
                return
            }
            $baseTorchInstalled = $true
        } else {
            $baseTorchInstalled = $true
        }

        $pythonMinorVersion = $null
        if (Test-Path ".venv\Scripts\python.exe") {
            $pythonVersionText = & .venv\Scripts\python.exe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($pythonVersionText -match "^3\.(\d+)$") {
                $pythonMinorVersion = [int]$Matches[1]
            }
        }

        if ($pythonMinorVersion -and $pythonMinorVersion -gt 13) {
            Write-Host "  [INFO] Intel Extension for PyTorch currently has no wheels for Python 3.$pythonMinorVersion" -ForegroundColor Yellow
            Write-Host "  Using CPU-only PyTorch from Step 1" -ForegroundColor Yellow
            Write-Host "  For Intel XPU support, use Python 3.12 or 3.13" -ForegroundColor Yellow
            $gpuType = "cpu"
            $installSuccess = $baseTorchInstalled
        } else {
            # Step 2: Install Intel Extension for PyTorch (IPEX) for XPU support
            Write-Host "  Step 2: Installing Intel Extension for PyTorch (IPEX)..." -ForegroundColor Gray
            & uv add intel-extension-for-pytorch
            $ipexExitCode = $LASTEXITCODE

            if ($ipexExitCode -eq 0) {
                $installSuccess = $true
                Write-Host "  ✅ IPEX installed successfully" -ForegroundColor Green

                # Verify XPU
                Write-Host "  Verifying XPU functionality..." -ForegroundColor Gray
                $testOutput = & .venv\Scripts\python.exe -c "
import torch
try:
    import intel_extension_for_pytorch as ipex
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print('xpu_available')
    else:
        print('xpu_fallback_to_cpu')
except ImportError:
    print('ipex_not_found')
" 2>&1

                if ($testOutput -match "xpu_available") {
                    Write-Host "  ✅ Intel XPU verification PASSED!" -ForegroundColor Green
                } elseif ($testOutput -match "xpu_fallback_to_cpu") {
                    Write-Host "  ⚠️  IPEX installed but XPU not available - will use CPU" -ForegroundColor Yellow
                    Write-Host "  This is expected for Intel Iris/UHD integrated GPUs" -ForegroundColor Yellow
                    $gpuType = "cpu"
                } else {
                    Write-Host "  ⚠️  IPEX import failed - check Intel driver installation" -ForegroundColor Yellow
                }
            } else {
                Write-Host "  ⚠️  IPEX installation failed" -ForegroundColor Yellow
                Write-Host "  Will use CPU-only PyTorch (Intel GPU acceleration unavailable)" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "  To enable Intel GPU support:" -ForegroundColor Yellow
                Write-Host "  1. Install Intel Arc drivers: https://www.intel.com/content/www/us/en/support/articles/000005630/graphics.html" -ForegroundColor Yellow
                Write-Host "  2. Ensure you have Intel Arc discrete GPU (Iris/UHD have limited support)" -ForegroundColor Yellow
                Write-Host "  3. Try: pip install intel-extension-for-pytorch" -ForegroundColor Yellow

                if ($baseTorchInstalled) {
                    Write-Host "  CPU-only PyTorch from Step 1 is already available" -ForegroundColor Yellow
                    $gpuType = "cpu"
                    $installSuccess = $true
                }
            }
        }
    }
    "cpu" {
        Write-Host "Installing PyTorch CPU version..." -ForegroundColor Cyan
        & uv add torch torchvision torchaudio
        if ($LASTEXITCODE -eq 0) { $installSuccess = $true }
    }
}

# Ultimate fallback
if (-not $installSuccess) {
    Write-Host ""
    Write-Host "❌ All installation attempts failed!" -ForegroundColor Red
    Write-Host "Attempting final fallback to CPU version..." -ForegroundColor Yellow
    Write-Host ""

    Write-Host "Installing PyTorch CPU version as last resort..." -ForegroundColor Yellow
    & uv pip install torch torchvision torchaudio

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

# ============================================================
# Step 4: Build Rust Engine (rust_engine) for fast GAM training
# ============================================================
Write-Host "Step 4: Building Rust engine (rust_engine)..." -ForegroundColor Yellow
Write-Host ""

$rustEnginePath = Join-Path $PSScriptRoot "rust_engine"
$rustAvailable = $false
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    $rustAvailable = $true
    $cargoVersion = cargo --version 2>&1
    Write-Host "[OK] Rust/Cargo found: $cargoVersion" -ForegroundColor Green
} else {
    Write-Host "[INFO] Rust/Cargo not installed" -ForegroundColor Yellow
    Write-Host "  rust_engine will fallback to pyGAM for GAM training" -ForegroundColor Gray
    Write-Host "  To enable: https://rustup.rs/" -ForegroundColor Gray
}

if ($rustAvailable -and (Test-Path $rustEnginePath)) {
    # Install maturin via pip (avoid dependency conflicts)
    Write-Host "[1/3] Installing maturin build tool..." -ForegroundColor Cyan
    & uv pip install maturin --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ maturin installed" -ForegroundColor Green
    } else {
        Write-Host "  ❌ maturin installation failed" -ForegroundColor Red
    }

    # Build and install rust_engine
    Write-Host "[2/3] Building rust_engine (release mode)..." -ForegroundColor Cyan
    Push-Location $rustEnginePath
    $rustExitCode = Invoke-LoggedNativeCommand -Command { uv run maturin develop --release }
    Pop-Location

    if ($rustExitCode -eq 0) {
        Write-Host "[3/3] ✅ rust_engine built and installed successfully!" -ForegroundColor Green
        Write-Host "  GAM training will use Rust engine (~15-50x faster than pyGAM)" -ForegroundColor Green
    } else {
        Write-Host "[WARN] rust_engine build failed (exit code: $rustExitCode)" -ForegroundColor Yellow
        Write-Host "  Falling back to pyGAM for GAM training" -ForegroundColor Gray
    }
} elseif ($rustAvailable -and -not (Test-Path $rustEnginePath)) {
    Write-Host "[INFO] Rust installed but rust_engine/ folder not found" -ForegroundColor Yellow
}

Write-Host ""

# Post-installation SAC mitigation (unblock newly installed DLLs)
Write-Host "[SAC] Running post-installation DLL unblock..." -ForegroundColor Cyan
Mitigate-SAC-Blocks
Write-Host ""

# Step 5: Run verification
Write-Host "Step 5: Running GPU & Rust engine verification..." -ForegroundColor Yellow
Write-Host ""

# Use venv python explicitly to ensure we test the installed packages
$venvPython = ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython verify_gpu.py

    # Test rust_engine
    Write-Host ""
    Write-Host "[Rust] Testing rust_engine..." -ForegroundColor Cyan
    & $venvPython -c "
try:
    from rust_engine import PyGAMClassifier
    print('[OK] rust_engine loaded successfully')
    g = PyGAMClassifier(n_splines=3, optimize_lambda=False)
    print('[OK] GAMClassifier ready')
except ImportError as e:
    print(f'[WARN] rust_engine not available: {e}')
    print('  Will use pyGAM fallback for GAM training')
" 2>&1
} else {
    Write-Host "[WARN] .venv python not found, using system python..." -ForegroundColor Yellow
    python verify_gpu.py
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GPU Type: $($gpuType.ToUpper())" -ForegroundColor Cyan
Write-Host "  Rust Engine: $(if (Test-Path (Join-Path $PSScriptRoot 'rust_engine')) { 'Configured' } else { 'Not found' })" -ForegroundColor Cyan
Write-Host "  Status: Success" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
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

if (Test-Path (Join-Path $PSScriptRoot 'rust_engine')) {
    Write-Host "✅ Rust engine (rust_engine) is ready!" -ForegroundColor Green
    Write-Host "  GAM training will be ~15-50x faster than pyGAM" -ForegroundColor Green
}

Write-Host ""
Write-Host "Start your analysis with:" -ForegroundColor Cyan
Write-Host "  uv run python main.py --models" -ForegroundColor Gray
Write-Host ""
