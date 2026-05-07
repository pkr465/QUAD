@echo off
REM ===========================================================================
REM   QUAD - Windows Bootstrap (cmd.exe shim)
REM
REM   Some users prefer not to invoke PowerShell directly. This .bat just
REM   delegates to bootstrap.ps1 with the bypass execution policy so the
REM   script runs even on locked-down Windows installs.
REM
REM   Prefers pwsh.exe (PowerShell 7+) when available; falls back to the
REM   built-in Windows PowerShell 5.1 (powershell.exe) otherwise.
REM
REM   Usage:
REM     bootstrap.bat
REM     bootstrap.bat --qairt-archive "C:\Users\me\Downloads\qairt.zip"
REM     bootstrap.bat --mock-only
REM     bootstrap.bat --real
REM     bootstrap.bat --clean
REM ===========================================================================

setlocal EnableDelayedExpansion

REM --- Pick the newest PowerShell available --------------------------------
REM pwsh.exe is PowerShell 7+ (cross-platform, ships via winget). It's faster,
REM has better encoding handling, and is what Microsoft now recommends.
REM powershell.exe is the built-in 5.1 - always present, but legacy.

set "PS_EXE="
for %%P in (pwsh.exe) do (
    if not "%%~$PATH:P"=="" set "PS_EXE=%%~$PATH:P"
)
if not defined PS_EXE (
    for %%P in (powershell.exe) do (
        if not "%%~$PATH:P"=="" set "PS_EXE=%%~$PATH:P"
    )
)

if not defined PS_EXE (
    echo [ERROR] Neither pwsh.exe ^(PowerShell 7+^) nor powershell.exe found on PATH.
    echo         Install PowerShell 7 from https://aka.ms/powershell or via:
    echo             winget install Microsoft.PowerShell
    exit /B 1
)

REM --- Translate POSIX-style flags to PS parameter style -------------------
set "PS_ARGS="

:parse
if "%~1"=="" goto run
if /I "%~1"=="--qairt-archive" (
    set "PS_ARGS=!PS_ARGS! -QairtArchive ""%~2"""
    shift
    shift
    goto parse
)
if /I "%~1"=="--mock-only"        ( set "PS_ARGS=!PS_ARGS! -MockOnly"       & shift & goto parse )
if /I "%~1"=="--skip-tests"       ( set "PS_ARGS=!PS_ARGS! -SkipTests"      & shift & goto parse )
if /I "%~1"=="--no-install-bash"  ( set "PS_ARGS=!PS_ARGS! -NoInstallBash"  & shift & goto parse )
if /I "%~1"=="--force"            ( set "PS_ARGS=!PS_ARGS! -Force"          & shift & goto parse )
if /I "%~1"=="--clean"            ( set "PS_ARGS=!PS_ARGS! -Clean"          & shift & goto parse )
if /I "%~1"=="--real"             ( set "PS_ARGS=!PS_ARGS! -Real"           & shift & goto parse )
if /I "%~1"=="--adapters" (
    set "PS_ARGS=!PS_ARGS! -Adapters ""%~2"""
    shift
    shift
    goto parse
)
echo Unknown argument: %~1
echo Usage: bootstrap.bat [--qairt-archive PATH] [--mock-only] [--real]
echo                      [--clean] [--skip-tests] [--no-install-bash] [--force]
exit /B 2

:run
echo [bootstrap.bat] Using "%PS_EXE%"
"%PS_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"!PS_ARGS!
exit /B %ERRORLEVEL%
