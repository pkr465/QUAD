@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  QUAD — Windows Bootstrap (cmd.exe shim)
REM
REM  Some users prefer not to invoke PowerShell directly. This .bat just
REM  delegates to bootstrap.ps1 with the bypass execution policy so the
REM  script runs even on locked-down Windows installs.
REM
REM  Usage:
REM    bootstrap.bat
REM    bootstrap.bat --qairt-archive "C:\Users\me\Downloads\qairt.zip"
REM    bootstrap.bat --mock-only
REM ═══════════════════════════════════════════════════════════════════════════

setlocal

REM Translate POSIX-style flags to PS parameter style
set "PS_ARGS="
:parse
if "%~1"=="" goto run
if /I "%~1"=="--qairt-archive" (
    set "PS_ARGS=%PS_ARGS% -QairtArchive ""%~2"""
    shift
    shift
    goto parse
)
if /I "%~1"=="--mock-only"   ( set "PS_ARGS=%PS_ARGS% -MockOnly"   & shift & goto parse )
if /I "%~1"=="--skip-tests"  ( set "PS_ARGS=%PS_ARGS% -SkipTests"  & shift & goto parse )
if /I "%~1"=="--no-install-bash" ( set "PS_ARGS=%PS_ARGS% -NoInstallBash" & shift & goto parse )
if /I "%~1"=="--force"       ( set "PS_ARGS=%PS_ARGS% -Force"      & shift & goto parse )
if /I "%~1"=="--adapters"    (
    set "PS_ARGS=%PS_ARGS% -Adapters ""%~2"""
    shift
    shift
    goto parse
)
echo Unknown argument: %~1
echo Usage: bootstrap.bat [--qairt-archive PATH] [--mock-only] [--skip-tests] [--no-install-bash]
exit /B 2

:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"%PS_ARGS%
exit /B %ERRORLEVEL%
