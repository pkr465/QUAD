# Activate QUAD real mode on this Snapdragon X Elite (Windows 11 ARM64) box.
#
# Usage from a new PowerShell session:
#   . C:\work\05\QUAD\activate.ps1
#
# Sets QAIRT_SDK_ROOT, QUAD_ADAPTER_MODE=real, prepends the QAIRT bin/lib dirs
# to PATH (including bin-shims that wrap the extensionless host Python tools
# like qairt-converter / qairt-quantizer / *-onnx-converter so shutil.which
# can find them on Windows), and sets ADSP_LIBRARY_PATH for HTP/DSP execution.

$ErrorActionPreference = "Stop"

$QuadRoot = "C:\work\05\QUAD"
$SdkRoot  = Join-Path $QuadRoot "sdks\qairt-2.46.0.260424"
$Shims    = Join-Path $QuadRoot "sdks\bin-shims"

if (-not (Test-Path $SdkRoot)) {
    Write-Error "QAIRT not found at $SdkRoot. Run bootstrap.ps1 -QairtArchive <zip> first."
    return
}

$env:QAIRT_SDK_ROOT    = $SdkRoot
$env:QUAD_ADAPTER_MODE = "real"

# Prepend SDK paths. bin-shims must come before x86_64-windows-msvc so the
# .cmd wrappers win over the extensionless POSIX scripts.
$prepend = @(
    $Shims,
    (Join-Path $SdkRoot "bin\aarch64-windows-msvc"),
    (Join-Path $SdkRoot "bin\x86_64-windows-msvc"),
    (Join-Path $SdkRoot "lib\aarch64-windows-msvc"),
    (Join-Path $SdkRoot "lib\x86_64-windows-msvc")
)
$env:PATH = ($prepend -join ";") + ";$env:PATH"

# Python module path for qti.* / snpe.* host helpers.
$pyLib = Join-Path $SdkRoot "lib\python"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$pyLib;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $pyLib
}

# X Elite Hexagon NPU is V73; HTP/DSP runtime needs ADSP_LIBRARY_PATH.
$env:ADSP_LIBRARY_PATH = "$SdkRoot\lib\hexagon-v73;$SdkRoot\lib\aarch64-windows-msvc"

Write-Host "QUAD real mode activated:"
Write-Host "  QAIRT_SDK_ROOT     = $env:QAIRT_SDK_ROOT"
Write-Host "  QUAD_ADAPTER_MODE  = $env:QUAD_ADAPTER_MODE"
Write-Host "  ADSP_LIBRARY_PATH  = $env:ADSP_LIBRARY_PATH"
Write-Host ""
Write-Host "Try:  quad mode   (or)   quad doctor --real-mode"
