# ===========================================================================
# QUAD - Windows Bootstrap (resolves the chicken-and-egg)
#
# install.sh / setup.sh are bash scripts and Windows doesn't ship bash. This
# PowerShell script:
#   1. Detects whether bash is already available (Git Bash / WSL / MSYS)
#   2. If not, installs Git for Windows via winget (which bundles Git Bash)
#   3. Hands off to install.sh with all the original arguments
#
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
#   powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1 -QairtArchive C:\Downloads\qairt.zip
#   powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1 -MockOnly
#
# Or, if your PowerShell execution policy already allows scripts:
#   .\bootstrap.ps1
#
# This file is intentionally ASCII-only so it parses correctly on a stock
# Windows PowerShell 5.1 install (which reads UTF-8 without BOM as cp1252).
# ===========================================================================

[CmdletBinding()]
param(
    [string]$QairtArchive = "",
    [switch]$MockOnly,
    [switch]$SkipTests,
    [switch]$Real,
    [switch]$Clean,
    [switch]$NoInstallBash,
    [switch]$Force,
    [string]$Adapters = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot

# Soft nudge toward PowerShell 7+ when running on classic 5.1
# (PSVersionTable.PSEdition is 'Desktop' on 5.x, 'Core' on 7+)
if ($PSVersionTable.PSEdition -eq 'Desktop') {
    Write-Host ""
    Write-Host "  [NOTE] Running on Windows PowerShell $($PSVersionTable.PSVersion). PowerShell 7+ is" -ForegroundColor DarkYellow
    Write-Host "         recommended for better encoding + faster startup. Install via:" -ForegroundColor DarkYellow
    Write-Host "             winget install Microsoft.PowerShell" -ForegroundColor DarkYellow
    Write-Host "         (this script still works on 5.1)" -ForegroundColor DarkYellow
}

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "==========================================================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "==========================================================================" -ForegroundColor Cyan
}
function Write-Step { param([string]$Text); Write-Host "`n--- $Text ---" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Text); Write-Host "  [OK]    $Text" -ForegroundColor Green }
function Write-Info { param([string]$Text); Write-Host "  [INFO]  $Text" -ForegroundColor Blue }
function Write-Warn { param([string]$Text); Write-Host "  [WARN]  $Text" -ForegroundColor Yellow }
function Write-Err  { param([string]$Text); Write-Host "  [ERROR] $Text" -ForegroundColor Red }

Write-Header "QUAD Windows Bootstrap"

# --- Step 1: Find or install bash --------------------------------------------

Write-Step "Step 1: Locate bash"

function Find-Bash {
    $cmd = Get-Command bash -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $candidates = @(
        "C:\Program Files\Git\bin\bash.exe",
        "C:\Program Files\Git\usr\bin\bash.exe",
        "C:\Program Files (x86)\Git\bin\bash.exe",
        "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe",
        "$env:ProgramFiles\Git\bin\bash.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }

    $wsl = Get-Command wsl -ErrorAction SilentlyContinue
    if ($wsl) { return "wsl-bash" }

    return $null
}

$bash = Find-Bash
if ($bash -and -not $Force) {
    Write-Ok "bash found: $bash"
    if ($bash -eq "wsl-bash") {
        Write-Warn "Only WSL-bash detected. Install Git for Windows for a smoother experience"
        Write-Warn "WSL paths require translation; this installer assumes native Windows paths"
    }
} else {
    if ($NoInstallBash) {
        Write-Err "bash not found and -NoInstallBash was passed"
        Write-Err "Install Git for Windows manually from https://git-scm.com/download/win"
        exit 1
    }

    Write-Info "bash not found. Installing Git for Windows (which bundles Git Bash)"

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    $wingetWorked = $false
    if ($winget) {
        Write-Info "winget detected, installing Git.Git from the winget source"
        $wargs = @(
            "install", "--id", "Git.Git",
            "--source", "winget",
            "--silent",
            "--accept-source-agreements",
            "--accept-package-agreements"
        )
        try {
            & winget @wargs
            if ($LASTEXITCODE -eq 0) {
                $wingetWorked = $true
            } else {
                Write-Warn "winget exit code $LASTEXITCODE, falling back to direct download"
            }
        } catch {
            Write-Warn ("winget failed: " + $_.Exception.Message)
        }
    }

    if (-not $wingetWorked) {
        Write-Info "Downloading Git for Windows installer directly"
        $url = "https://github.com/git-for-windows/git/releases/latest/download/Git-2.45.2-64-bit.exe"
        $tmpInstaller = Join-Path $env:TEMP "git-for-windows-installer.exe"
        try {
            Invoke-WebRequest -Uri $url -OutFile $tmpInstaller -UseBasicParsing
        } catch {
            Write-Err ("Direct download failed: " + $_.Exception.Message)
            Write-Err "Install manually from https://git-scm.com/download/win then re-run this script"
            exit 1
        }
        Write-Info "Running installer (silent)"
        Start-Process -FilePath $tmpInstaller -ArgumentList "/VERYSILENT", "/NORESTART" -Wait
        Remove-Item -Force $tmpInstaller -ErrorAction SilentlyContinue
    }

    # Re-resolve PATH (the just-installed Git/cmd directories may not be in this session yet)
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    $bash = Find-Bash
    if (-not $bash) {
        Write-Err "Git for Windows installed but bash still not found on PATH"
        Write-Err "Open a new PowerShell window and re-run: .\bootstrap.ps1"
        exit 1
    }
    Write-Ok "Git for Windows installed; bash at $bash"
}

# --- Step 2: Sanity-check the toolchain --------------------------------------

Write-Step "Step 2: Verify toolchain"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command python3 -ErrorAction SilentlyContinue }
if ($python) {
    $pyVer = & $python.Source --version 2>&1
    Write-Ok "Python: $pyVer at $($python.Source)"
} else {
    Write-Err "Python not found. Install Python 3.10+ from https://python.org/downloads"
    Write-Err "Or via winget: winget install Python.Python.3.12"
    exit 1
}

$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    Write-Ok "git: $($git.Source)"
} else {
    Write-Warn "git not on PATH (expected with Git for Windows)"
}

# --- Step 2b: Visual C++ Redistributable -------------------------------------
#
# QAIRT's host-side Python tools (qairt-converter, qairt-quantizer, the
# *-onnx-converter scripts) link against MSVC runtime DLLs. On Windows ARM64
# (Snapdragon X Elite Copilot+ PCs) Python is emulated x86, so the x86
# redistributable is the one that's almost always missing — its absence shows
# up as "ImportError: DLL load failed while importing libDlModelToolsPy" out
# of qti.aisw.dlc_utils. We install all three (x86 / x64 / arm64) idempotently:
# Microsoft's installer is a no-op when the present version is already newer.

Write-Step "Step 2b: Visual C++ Redistributable (x86 + x64 + arm64)"

function Test-VCRedistInstalled {
    param([Parameter(Mandatory)][ValidateSet('X86','X64','arm64')][string]$Arch)
    $key = "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\$Arch"
    if (-not (Test-Path $key)) {
        # WOW6432 view (32-bit hive) when running 64-bit PowerShell against the X86 redist
        $key = "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\$Arch"
        if (-not (Test-Path $key)) { return $null }
    }
    try {
        $v = (Get-ItemProperty -Path $key -Name Version -ErrorAction Stop).Version
        return $v
    } catch {
        return $null
    }
}

function Install-VCRedist {
    param(
        [Parameter(Mandatory)][ValidateSet('X86','X64','arm64')][string]$Arch,
        [string]$BaseUrl = "https://aka.ms/vs/17/release"
    )
    $existing = Test-VCRedistInstalled -Arch $Arch
    if ($existing) {
        Write-Ok ("VC++ {0} redistributable already installed: {1}" -f $Arch, $existing)
        return
    }
    $fileArch = $Arch.ToLower()
    $url = "$BaseUrl/vc_redist.$fileArch.exe"
    $tmp = Join-Path $env:TEMP "vc_redist.$fileArch.exe"
    Write-Info "Downloading VC++ $Arch redistributable from $url"
    try {
        Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
    } catch {
        Write-Warn ("Download failed for $Arch redistributable: " + $_.Exception.Message)
        Write-Warn "qairt-converter may fail until you install it manually:"
        Write-Warn "    https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist"
        return
    }
    Write-Info "Running installer (silent, /quiet /norestart)"
    $proc = Start-Process -FilePath $tmp -ArgumentList "/install","/quiet","/norestart" -Wait -PassThru
    Remove-Item -Force $tmp -ErrorAction SilentlyContinue
    # 0 = success; 1638 = newer version already installed; 3010 = success, reboot required
    switch ($proc.ExitCode) {
        0    { Write-Ok  "VC++ $Arch redistributable installed" }
        1638 { Write-Ok  "VC++ $Arch redistributable: newer version already present" }
        3010 { Write-Ok  "VC++ $Arch redistributable installed (reboot recommended later)" }
        default {
            Write-Warn ("VC++ {0} installer exit code: {1} (continuing)" -f $Arch, $proc.ExitCode)
        }
    }
}

# Always do x86 — that's the one Snapdragon X Elite emulated Python needs.
# x64 covers the rare case of a x64 native Python on ARM64; arm64 covers
# native ARM64 Python (still in pre-release for many distros at time of writing).
Install-VCRedist -Arch X86
Install-VCRedist -Arch X64
$arch = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Architecture)
if ($arch -eq 12) {  # 12 = ARM64
    Install-VCRedist -Arch arm64

    # Snapdragon X Elite advisory: emulated x86_64 Python loads the wrong
    # .pyd from QAIRT's lib/python tree (windows-arm64ec instead of
    # windows-x86_64) because qti.aisw.dlc_utils.__init__.py keys off
    # platform.processor() which always returns "ARMv8…" on this hardware.
    # Native ARM64 Python sidesteps the issue entirely. Surface this once
    # so the user knows what they're walking into before qairt-converter
    # fails with libDlModelToolsPy ImportError.
    $pyArch = & python -c "import sysconfig; print(sysconfig.get_platform())" 2>$null
    if ($pyArch -and ($pyArch -match "amd64|x86")) {
        Write-Warn "Detected x86_64 Python on ARM64 Windows (Prism emulation)."
        Write-Warn "QAIRT host tools (qairt-converter etc.) prefer native ARM64 Python."
        Write-Warn "If qairt-converter fails with 'libDlModelToolsPy ImportError', install a"
        Write-Warn "native ARM64 Python from python.org/downloads/windows (look for"
        Write-Warn "'Windows arm64' installer) and re-run bootstrap.ps1."
    }
}

# --- Step 3: Hand off to install.sh ------------------------------------------

Write-Step "Step 3: Run install.sh"

$installArgs = @()
if ($MockOnly)   { $installArgs += "--mock-only" }
if ($SkipTests)  { $installArgs += "--skip-tests" }
if ($Real)       { $installArgs += "--real" }
if ($Clean)      { $installArgs += "--clean" }
if ($Adapters)   { $installArgs += @("--adapters", $Adapters) }
if ($QairtArchive) {
    if (-not (Test-Path $QairtArchive)) {
        Write-Err "QairtArchive path does not exist: $QairtArchive"
        exit 1
    }
    $installArgs += @("--qairt-archive", $QairtArchive)
}

$installScript = Join-Path $ScriptDir "install.sh"
if (-not (Test-Path $installScript)) {
    Write-Err "install.sh not found at $installScript"
    Write-Err "Run this script from the QUAD repo root."
    exit 1
}

Push-Location $ScriptDir
try {
    if ($bash -eq "wsl-bash") {
        Write-Info ("Handing off to: wsl bash ./install.sh " + ($installArgs -join ' '))
        & wsl bash ./install.sh @installArgs
    } else {
        Write-Info ("Handing off to: " + $bash + " ./install.sh " + ($installArgs -join ' '))
        & $bash ./install.sh @installArgs
    }
    $rc = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($rc -ne 0) {
    Write-Err "install.sh exited with code $rc"
    exit $rc
}

Write-Header "Bootstrap Complete"
Write-Host "  Next steps in a NEW shell (Git Bash recommended):" -ForegroundColor Cyan
Write-Host "    cd $ScriptDir"
Write-Host "    source ./activate.sh"
Write-Host "    quad mode"
Write-Host "    quad quickstart"
Write-Host ""
