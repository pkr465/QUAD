#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# QUAD — Environment & Configuration Setup
# ═══════════════════════════════════════════════════════════════════════════════
#
# Creates an isolated virtual environment, installs all dependencies,
# generates config files from examples, and verifies the installation.
#
# Usage:
#   ./setup.sh              # Standard setup (mock mode, dev deps)
#   ./setup.sh --real       # Also install real-hardware dependencies
#   ./setup.sh --no-tests   # Skip test verification
#   ./setup.sh --clean      # Remove existing .venv first
#   ./setup.sh --help
#
# Windows (no bash yet?)
#   This script needs bash. If you're on a fresh Windows machine without
#   Git Bash / WSL, run from PowerShell:
#       .\bootstrap.ps1
#   …or from cmd.exe:
#       bootstrap.bat
#   bootstrap.ps1 installs Git for Windows (which bundles Git Bash) via
#   winget, then re-runs install.sh with whatever flags you passed.
#
# After setup, activate the venv with:
#   source .venv/bin/activate
#   quad doctor
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Windows compatibility guard ──────────────────────────────────────────────
# If we're somehow running under cmd.exe / PowerShell directly (no MSYS / WSL),
# `BASH_VERSION` will be unset and `set -o pipefail` will already have failed.
# But on legitimate-but-old bash (e.g. macOS 3.2), warn before going further.
if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "ERROR: This script requires bash. You appear to be running under sh/dash/zsh."
    echo "       On Windows, run bootstrap.ps1 (or bootstrap.bat) which auto-installs"
    echo "       Git for Windows and re-runs this script."
    exit 1
fi
if [[ "${BASH_VERSINFO[0]:-0}" -lt 4 ]]; then
    echo "WARNING: bash ${BASH_VERSION} is old; QUAD's setup scripts need bash 4+."
    echo "         On macOS, install via 'brew install bash'."
    echo "         On Windows, run bootstrap.ps1 to install Git for Windows."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON="${PYTHON:-python3}"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}✅  $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️   $*${NC}"; }
info() { echo -e "${CYAN}ℹ️   $*${NC}"; }
fail() { echo -e "${RED}❌  $*${NC}"; exit 1; }
step() { echo -e "\n${BOLD}──────────────────────────────────${NC}"; echo -e "${BOLD}$*${NC}"; }

# ── Parse args ────────────────────────────────────────────────────────────────
INSTALL_REAL=false
RUN_TESTS=true
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --real)      INSTALL_REAL=true; shift ;;
        --no-tests)  RUN_TESTS=false; shift ;;
        --clean)     CLEAN=true; shift ;;
        --help|-h)
            echo ""
            echo -e "${BOLD}QUAD setup.sh — Environment & Configuration Setup${NC}"
            echo ""
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --real       Install real-hardware dependencies (asyncssh, paramiko, onnx)"
            echo "  --no-tests   Skip test suite verification"
            echo "  --clean      Remove existing .venv before setup"
            echo "  --help       Show this help"
            echo ""
            echo "After setup:"
            echo "  source .venv/bin/activate"
            echo "  quad doctor"
            echo "  quad quickstart"
            exit 0 ;;
        *) warn "Unknown argument: $1"; shift ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║  QUAD — Qualcomm Unified Agent for Developers   ║${NC}"
echo -e "${BOLD}${CYAN}║  Environment & Configuration Setup               ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

cd "$SCRIPT_DIR"

# ── Step 1: Python version check ──────────────────────────────────────────────
step "Step 1: Checking Python version"
PYTHON_BIN="$(command -v "$PYTHON" || true)"
[[ -z "$PYTHON_BIN" ]] && fail "Python not found. Install Python 3.10+ and ensure it is in PATH."

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION#*.}"

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
    fail "Python $PY_VERSION found. QUAD requires Python 3.10+. Install 3.11 recommended."
fi
ok "Python $PY_VERSION found at $PYTHON_BIN"

# ── Step 2: Create virtual environment ───────────────────────────────────────
step "Step 2: Creating virtual environment"

if [[ "$CLEAN" == "true" && -d "$VENV_DIR" ]]; then
    info "Removing existing .venv (--clean)"
    rm -rf "$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
    warn ".venv already exists — skipping creation (use --clean to recreate)"
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    ok "Virtual environment created at $VENV_DIR"
fi

# Activate venv for the rest of this script
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
PIP="${VENV_DIR}/bin/pip"
PYTHON_VENV="${VENV_DIR}/bin/python"
ok "Virtual environment activated"

# ── Step 3: Upgrade pip ───────────────────────────────────────────────────────
step "Step 3: Upgrading pip"
"$PIP" install --upgrade pip --quiet
ok "pip upgraded to $("$PIP" --version | awk '{print $2}')"

# ── Step 4: Install QUAD and dependencies ─────────────────────────────────────
step "Step 4: Installing QUAD with dev dependencies"
info "Installing quad-agent[dev] ..."
"$PIP" install -e ".[dev]" --quiet
ok "quad-agent installed (mock mode ready)"

if [[ "$INSTALL_REAL" == "true" ]]; then
    info "Installing real-hardware dependencies ..."
    "$PIP" install -e ".[real]" --quiet
    ok "Real-hardware deps installed (asyncssh, paramiko, onnx)"
fi

# Verify key packages
for pkg in pytest pytest_asyncio fastmcp pydantic jinja2 numpy typer; do
    "$PYTHON_VENV" -c "import $pkg" 2>/dev/null && ok "$pkg importable" || warn "$pkg not importable"
done

# ── Step 5: Generate config files ────────────────────────────────────────────
step "Step 5: Generating configuration files"

# quad.toml
if [[ -f "quad.toml" ]]; then
    info "quad.toml already exists — skipping"
else
    if [[ -f "configs/quad.toml.example" ]]; then
        cp "configs/quad.toml.example" "quad.toml"
        ok "quad.toml created from configs/quad.toml.example"
    else
        # Generate minimal quad.toml
        cat > quad.toml <<'TOML'
[server]
adapter_mode = "mock"
log_level = "info"
model_output_dir = "./output"
template_dir = "./templates"

[adapters.qnn]
sdk_path = ""
version = "2.x"

[adapters.snpe]
sdk_path = ""

[adapters.hexagon]
sdk_path = ""
tools_path = ""

[platforms.windows]
enabled = true
device_type = "local"

[platforms.linux]
enabled = true
device_type = "remote"
ssh_host = ""
ssh_user = "root"
ssh_key = ""

[platforms.android]
enabled = true
device_type = "adb"
serial = ""
TOML
        ok "quad.toml created (minimal config, mock mode)"
    fi
fi

# .env
if [[ -f ".env" ]]; then
    info ".env already exists — skipping"
else
    if [[ -f ".env.example" ]]; then
        cp ".env.example" ".env"
        ok ".env created from .env.example"
        warn "Edit .env to set QAIRT_SDK_ROOT, QAI_HUB_API_KEY, etc. for real mode"
    else
        cat > .env <<'ENV'
# QUAD Environment Variables
# See docs/PREREQUISITES.md for how to obtain each value.

# ── SDK Paths ─────────────────────────────────────────────────────────────────
QAIRT_SDK_ROOT=          # e.g. /opt/qairt/2.45.0.260326
QNN_SDK_ROOT=            # e.g. /opt/qnn/2.28.0
SNPE_ROOT=               # e.g. /opt/snpe-2.x
HEXAGON_SDK_ROOT=        # e.g. /opt/hexagon-sdk-5.x
ANDROID_NDK_ROOT=        # e.g. /opt/android-ndk-r26

# ── DSP Runtime ───────────────────────────────────────────────────────────────
ADSP_LIBRARY_PATH=       # e.g. /opt/qairt/2.45.0/lib/aarch64-android;/vendor/lib/rfsa/adsp

# ── Cloud Services ────────────────────────────────────────────────────────────
QAI_HUB_API_KEY=         # Qualcomm AI Hub token (hub.qai.qualcomm.com)

# ── Device Connection ─────────────────────────────────────────────────────────
ANDROID_SERIAL=          # ADB device serial (adb devices)
TARGET_IP=               # SSH target IP for Linux device (QCS2210)
TARGET_USER=root
TARGET_SSH_KEY=          # Path to SSH key

# ── QUAD Overrides ────────────────────────────────────────────────────────────
QUAD_ADAPTER_MODE=       # "mock" or "real" (overrides quad.toml)
QUAD_LOG_LEVEL=          # "debug", "info", "warning"
ENV
        ok ".env created from template"
        warn "Edit .env to configure SDK paths and API keys for real mode"
    fi
fi

# Create output directory
mkdir -p output
ok "output/ directory ready"

# ── Step 6: Verify installation ───────────────────────────────────────────────
step "Step 6: Verifying installation"

# Import check
"$PYTHON_VENV" -c "import quad; print(f'quad v{quad.__version__} importable')" && ok "quad package OK" || fail "quad import failed"

# CLI check
# Check quad-server is installed (don't run it — FastMCP blocks on stdio)
if [[ -f "${VENV_DIR}/bin/quad-server" ]]; then
    ok "quad-server CLI installed"
else
    warn "quad-server CLI not found (run: pip install -e .[dev])"
fi

# Doctor check
echo ""
info "Running quad doctor ..."
"$PYTHON_VENV" -c "
from quad.cli.doctor import run_doctor
report = run_doctor()
for c in report.checks:
    icon = '  ✅' if c.status == 'pass' else ('  ⚠️ ' if c.status == 'warn' else '  ❌')
    print(f'{icon} {c.name}: {c.message[:90]}')
n_pass = len([c for c in report.checks if c.status=='pass'])
n_warn = len([c for c in report.checks if c.status=='warn'])
n_fail = len([c for c in report.checks if c.status=='fail'])
print(f'\n  Summary: {n_pass} pass, {n_warn} warn (expected — SDK not installed), {n_fail} fail')
"

# ── Step 7: Run tests ─────────────────────────────────────────────────────────
if [[ "$RUN_TESTS" == "true" ]]; then
    step "Step 7: Running test suite"
    info "Running pytest (this may take ~15 seconds) ..."

    if "$PYTHON_VENV" -m pytest tests/ -q --tb=short 2>&1 | tail -5; then
        PASSED=$("$PYTHON_VENV" -m pytest tests/ -q --tb=no 2>&1 | grep -oE '[0-9]+ passed' | head -1 || echo "? passed")
        ok "Tests: $PASSED"
    else
        warn "Some tests failed — check output above"
    fi
else
    step "Step 7: Tests skipped (--no-tests)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  ✅  QUAD setup complete!                        ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Activate your environment:${NC}"
echo -e "  ${CYAN}source .venv/bin/activate${NC}"
echo ""
echo -e "${BOLD}Try QUAD:${NC}"
echo -e "  ${CYAN}quad doctor${NC}           # Re-run diagnostics"
echo -e "  ${CYAN}quad quickstart${NC}       # Interactive zero-to-inference wizard"
echo -e "  ${CYAN}quad benchmark${NC}        # Run standard benchmarks (mock mode)"
echo -e "  ${CYAN}quad configure${NC}        # Configure SDK paths (when SDK installed)"
echo ""
echo -e "${BOLD}Run the MCP server:${NC}"
echo -e "  ${CYAN}quad-server --config quad.toml${NC}"
echo ""
if [[ "$INSTALL_REAL" == "false" ]]; then
    echo -e "${YELLOW}To enable real hardware mode:${NC}"
    echo -e "  1. Edit ${CYAN}.env${NC} with SDK paths"
    echo -e "  2. Run ${CYAN}./setup.sh --real${NC} to install hardware deps"
    echo -e "  3. Edit ${CYAN}quad.toml${NC}: set ${CYAN}adapter_mode = \"real\"${NC}"
    echo -e "  4. Run ${CYAN}source activate_qairt.sh${NC} to activate SDK tools"
    echo ""
fi
