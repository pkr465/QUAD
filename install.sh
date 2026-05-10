#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# QUAD — Qualcomm Unified Agent for Developers
# Global Installation Script (Orchestrator)
#
# Installs the QUAD platform and calls modular adapter setup scripts:
#   scripts/adapters/setup_qairt.sh   — QAIRT/SNPE SDK (model conversion, inference)
#   scripts/adapters/setup_qnn.sh     — QNN-specific configuration
#   scripts/adapters/setup_hexagon.sh — Hexagon SDK (custom DSP kernels)
#
# Usage:
#   ./install.sh                                  # Full: platform + adapters
#   ./install.sh --qairt-archive ~/Downloads/qairt-2.45.0.zip  # One-step real setup
#   ./install.sh --mock-only                      # Platform only (no SDK setup)
#   ./install.sh --adapters qairt,qnn             # Specific adapters only
#   ./install.sh --skip-tests                     # Skip verification
#   ./install.sh --help                           # Show options
#
# Windows (no bash yet?)
#   Run from PowerShell:           .\bootstrap.ps1
#   Or from cmd.exe:               bootstrap.bat
#   bootstrap.ps1 installs Git for Windows (which bundles Git Bash) via
#   winget, then re-runs this script with the same arguments. It also
#   installs the Visual C++ 2015-2022 Redistributable (x86 + x64 + arm64
#   on ARM64 hosts) so QAIRT host tools (qairt-converter, qairt-quantizer,
#   *-onnx-converter) can load qti.aisw.dlc_utils.libDlModelToolsPy.
#
# On Linux/macOS qairt-converter runs on a glibc-native x86_64 Python, so
# no equivalent system-package step is needed here. If qairt-converter
# fails to import on Linux, your distro is missing libstdc++ — install
# build-essential (Debian) or gcc-c++ (RHEL/Fedora).
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── bash compatibility guard ────────────────────────────────────────────────
if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "ERROR: install.sh requires bash. On Windows, run .\\bootstrap.ps1 first." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="${PYTHON:-python3}"
ADAPTERS_DIR="${SCRIPT_DIR}/scripts/adapters"

# Source helpers
source "${SCRIPT_DIR}/scripts/helpers.sh"

# ── Parse arguments ──
MOCK_ONLY=false
SKIP_TESTS=false
CLEAN_VENV=false
INSTALL_REAL_EXTRAS=false
SERVER_ONLY=false
ADAPTERS_LIST="qairt,qnn,hexagon,target,udo"  # Default: all
QAIRT_ARCHIVE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mock-only)      MOCK_ONLY=true; shift ;;
        --skip-tests)     SKIP_TESTS=true; shift ;;
        --clean)          CLEAN_VENV=true; shift ;;
        --real)           INSTALL_REAL_EXTRAS=true; shift ;;
        --server-only)    SERVER_ONLY=true; shift ;;
        --adapters)       ADAPTERS_LIST="$2"; shift 2 ;;
        --qairt-archive)  QAIRT_ARCHIVE="$2"; shift 2 ;;
        --help|-h)
            echo ""
            echo -e "${BOLD}QUAD — One-Step Installer${NC}"
            echo ""
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --qairt-archive PATH   Install the SDK from a downloaded archive"
            echo "                         (.zip / .tar.gz / .tgz) — recommended for first-time"
            echo "                         setup with the developer-portal download."
            echo "  --mock-only            Skip SDK setup; install QUAD in mock mode only."
            echo "  --server-only          Install QUAD server only (skip Claude Code .claude/"
            echo "                         settings.json + skills provisioning). Pair with"
            echo "                         install-client.sh on the IDE machine for split deploys."
            echo "  --real                 Also install real-hardware Python extras"
            echo "                         (asyncssh, paramiko, onnx — for remote-target deploys)"
            echo "  --clean                Remove existing .venv/ before creating (full reinstall)"
            echo "  --adapters LIST        Comma-separated adapter list"
            echo "                         (default: qairt,qnn,hexagon,target,udo)"
            echo "  --skip-tests           Skip the post-install test verification"
            echo "  --help                 Show this help"
            echo ""
            echo "Quickest paths to a working real-hardware setup:"
            echo ""
            echo "  1) After downloading the SDK from the Qualcomm developer portal:"
            echo "       ./install.sh --qairt-archive ~/Downloads/qairt-2.45.0.260326.zip"
            echo ""
            echo "  2) If you already dropped the archive in ~/Downloads/, just:"
            echo "       ./install.sh"
            echo ""
            echo "  3) For CI / org-managed mirrors with a pre-stored token:"
            echo "       export QAIRT_DOWNLOAD_URL=https://your-mirror/qairt.zip"
            echo "       export QAIRT_DOWNLOAD_TOKEN=<bearer-token>"
            echo "       ./install.sh"
            echo ""
            echo "  4) No SDK available right now (still want a working dev env):"
            echo "       ./install.sh --mock-only"
            echo ""
            echo "The installer always succeeds — if no SDK is found, QUAD falls back"
            echo "to mock mode and prints clear next-step instructions including the"
            echo "download URLs:"
            echo ""
            echo "  https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk"
            echo "  https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai"
            echo ""
            echo -e "After install, ${BOLD}quad sdk install <archive>${NC} can be used to add or"
            echo "update the SDK at any time."
            echo ""
            exit 0
            ;;
        *)  log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  QUAD — Qualcomm Unified Agent for Developers${NC}"
echo -e "${BOLD}  Global Installer${NC}"
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
if [ "$MOCK_ONLY" = true ]; then
    echo -e "  Mode: ${YELLOW}Mock only${NC} (no SDK adapters)"
else
    echo -e "  Mode: ${GREEN}Full install${NC} (platform + adapters: ${ADAPTERS_LIST})"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Python
# ═══════════════════════════════════════════════════════════════════════════
log_section "Step 1: Python Environment"

PY_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    log_error "Python 3.10+ required (found $PY_VERSION)"
    echo "    PYTHON=/path/to/python3.11 ./install.sh"
    exit 1
fi
log_ok "Python $PY_VERSION"

# Snapdragon X Elite (Windows-on-ARM) detection: an x86_64 Python running
# through Prism emulation will fail QAIRT host-tool imports because
# qti.aisw.dlc_utils picks the wrong .pyd path. We can only detect this
# here (install.sh runs through Git Bash on Windows); the actual install
# of the native ARM64 Python is in bootstrap.ps1 / bootstrap.bat.
PY_PLATFORM=$($PYTHON -c "import sysconfig; print(sysconfig.get_platform())" 2>/dev/null)
HOST_MACHINE=$($PYTHON -c "import platform; print(platform.uname().machine)" 2>/dev/null)
if [[ "$HOST_MACHINE" == "ARM64" || "$HOST_MACHINE" == "AARCH64" ]] \
        && [[ "$PY_PLATFORM" == *amd64* || "$PY_PLATFORM" == *x86* ]]; then
    log_warn "x86_64 Python detected on ARM64 Windows (Prism emulation)."
    log_warn "QAIRT host tools (qairt-converter, qairt-quantizer) will fail with"
    log_warn "ImportError on libDlModelToolsPy because dlc_utils picks the wrong .pyd."
    log_warn "Recommended: re-run from PowerShell — bootstrap.ps1 will install"
    log_warn "Python.Python.3.12 --architecture arm64 via winget, then re-run install.sh."
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: QUAD Platform Package
# ═══════════════════════════════════════════════════════════════════════════
log_section "Step 2: QUAD Platform"

if [ "$CLEAN_VENV" = true ] && [ -d "$VENV_DIR" ]; then
    log_info "Removing existing .venv/ (--clean)"
    rm -rf "$VENV_DIR"
fi

if [ -d "$VENV_DIR" ]; then
    log_ok "Virtual environment exists (.venv/)"
else
    log_info "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
    log_ok "Created .venv/"
fi

# Activate (handle both POSIX and Windows-Python venv layouts)
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/Scripts/activate"
else
    log_error "Could not find venv activate script in $VENV_DIR"
    exit 1
fi

pip install --upgrade pip --quiet 2>/dev/null
pip install -e ".[dev]" --quiet 2>/dev/null
log_ok "quad-agent + dev dependencies installed"

if [ "$INSTALL_REAL_EXTRAS" = true ]; then
    log_info "Installing real-hardware Python extras (asyncssh, paramiko, onnx, psutil)..."
    pip install -e ".[real]" --quiet 2>/dev/null && \
        log_ok "Real-hardware extras installed" || \
        log_warn "Some real-hardware extras failed (check pip output with --verbose)"
    # Pull the *latest compatible* releases of the profiling-critical
    # packages (psutil >= 5.9 in pyproject; 6.x adds Windows ARM64 wheels
    # we want for RSS sampling on Snapdragon X Elite). --upgrade is a
    # no-op when the venv is already on the newest release.
    log_info "Refreshing profiling deps to latest compatible (psutil + onnx + httpx)..."
    pip install --upgrade --quiet psutil onnx httpx 2>/dev/null && \
        log_ok "Profiling deps refreshed" || \
        log_warn "Some profiling deps failed to upgrade"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Configuration
# ═══════════════════════════════════════════════════════════════════════════
log_section "Step 3: Configuration"

if [ ! -f "$SCRIPT_DIR/quad.toml" ]; then
    cp "$SCRIPT_DIR/configs/quad.toml.example" "$SCRIPT_DIR/quad.toml"
    log_ok "Created quad.toml"
else
    log_ok "quad.toml exists"
fi

if [ "$SERVER_ONLY" = true ]; then
    log_info "--server-only: skipping client provisioning."
    log_info "  To set up Claude Code on this machine later:"
    log_info "    quad-client install --transport=stdio-local"
    log_info "  Or on a different (lightweight) machine:"
    log_info "    ./install-client.sh --transport=stdio-ssh --ssh-host=<this-machine>"
elif [ -f "$SCRIPT_DIR/.claude/settings.json" ]; then
    log_ok ".claude/settings.json (MCP auto-detection)"
else
    # Delegate to `quad client install` — single source of truth for the
    # Claude Code provisioning. This also installs the bundled skills.
    if python -m quad.cli.main client install --client claude_code 2>/dev/null; then
        log_ok "Provisioned Claude Code client (settings.json + skills)"
    else
        # Fallback for the very first bootstrap (before the package is
        # importable) — write a minimal settings.json by hand.
        mkdir -p "$SCRIPT_DIR/.claude"
        cat > "$SCRIPT_DIR/.claude/settings.json" << 'SETTINGS'
{
  "permissions": {
    "allow": [
      "mcp__quad__hardware_detect",
      "mcp__quad__convert_model",
      "mcp__quad__profile_workload",
      "mcp__quad__orchestrate_workload",
      "mcp__quad__generate_code"
    ]
  },
  "mcpServers": {
    "quad": {
      "command": "python",
      "args": ["-m", "quad.mcp.server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "QUAD_ADAPTER_MODE": "mock"
      }
    }
  }
}
SETTINGS
        log_warn "quad client install failed — wrote a minimal .claude/settings.json fallback"
        log_warn "Run 'quad client install --force' once the package is importable to also install skills"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: VS Code Setup
# ═══════════════════════════════════════════════════════════════════════════
log_section "Step 4: VS Code Setup"

VSCODE_DIR="$SCRIPT_DIR/.vscode"
mkdir -p "$VSCODE_DIR"

# ── settings.json — Python interpreter, CMake, C++ include paths ──
cat > "$VSCODE_DIR/settings.json" << VSCODE_SETTINGS
{
  "python.defaultInterpreterPath": "${VENV_DIR}/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "[python]": { "editor.defaultFormatter": "charliermarsh.ruff" },
  "ruff.lint.args": ["--config=pyproject.toml"],
  "files.exclude": {
    "**/__pycache__": true, ".venv": true, "**/*.egg-info": true, "qairt": true
  },
  "cmake.configureOnOpen": true,
  "cmake.buildDirectory": "\${workspaceFolder}/build",
  "C_Cpp.default.includePath": [
    "${QAIRT_SDK_ROOT:-\${workspaceFolder}/qairt/2.45.0.260326}/include/SNPE",
    "${QAIRT_SDK_ROOT:-\${workspaceFolder}/qairt/2.45.0.260326}/include/QNN"
  ],
  "terminal.integrated.env.linux": {
    "QAIRT_SDK_ROOT": "${QAIRT_SDK_ROOT:-${SCRIPT_DIR}/qairt/2.45.0.260326}",
    "SNPE_ROOT": "${QAIRT_SDK_ROOT:-${SCRIPT_DIR}/qairt/2.45.0.260326}",
    "PATH": "${QAIRT_SDK_ROOT:-${SCRIPT_DIR}/qairt/2.45.0.260326}/bin/x86_64-linux-clang:\${env:PATH}"
  },
  "terminal.integrated.env.osx": {
    "QAIRT_SDK_ROOT": "${QAIRT_SDK_ROOT:-${SCRIPT_DIR}/qairt/2.45.0.260326}",
    "SNPE_ROOT": "${QAIRT_SDK_ROOT:-${SCRIPT_DIR}/qairt/2.45.0.260326}"
  }
}
VSCODE_SETTINGS
log_ok "settings.json — Python interpreter, SDK paths, CMake"

# ── extensions.json — Recommended extensions prompt ──
cat > "$VSCODE_DIR/extensions.json" << 'VSCODE_EXT'
{
  "recommendations": [
    "ms-python.python",
    "charliermarsh.ruff",
    "ms-python.debugpy",
    "ms-vscode.cpptools",
    "ms-vscode.cmake-tools",
    "ms-python.mypy-type-checker",
    "redhat.vscode-yaml",
    "tamasfe.even-better-toml"
  ]
}
VSCODE_EXT
log_ok "extensions.json — recommended extensions"

# ── tasks.json — Build, test, serve, convert, deploy tasks ──
cat > "$VSCODE_DIR/tasks.json" << 'VSCODE_TASKS'
{
  "version": "2.0.0",
  "tasks": [
    { "label": "QUAD: Run Tests",          "type": "shell", "command": "make test",               "group": {"kind": "test", "isDefault": true}, "problemMatcher": [] },
    { "label": "QUAD: Start Server (Mock)","type": "shell", "command": "./launch.sh --verbose",   "isBackground": true, "problemMatcher": [] },
    { "label": "QUAD: Start Server (Real)","type": "shell", "command": "./launch.sh --real",      "isBackground": true, "problemMatcher": [] },
    { "label": "QUAD: Lint",               "type": "shell", "command": "make lint",               "group": "build",     "problemMatcher": [] },
    { "label": "QUAD: Format",             "type": "shell", "command": "make format",             "problemMatcher": [] },
    { "label": "QUAD: Quickstart",         "type": "shell", "command": "source .venv/bin/activate && python -m quad.cli.main quickstart", "problemMatcher": [] },
    { "label": "QUAD: Doctor",             "type": "shell", "command": "source .venv/bin/activate && python -m quad.cli.main doctor",     "problemMatcher": [] },
    { "label": "QUAD: Benchmark",          "type": "shell", "command": "source .venv/bin/activate && python -m quad.cli.main benchmark",  "problemMatcher": [] },
    {
      "label": "SNPE: Convert Model",
      "type": "shell",
      "command": "source ./activate.sh && qairt-converter --input_network ${input:modelPath}",
      "problemMatcher": []
    },
    {
      "label": "SNPE: Run Inference",
      "type": "shell",
      "command": "source ./activate.sh && snpe-net-run --container ${input:dlcPath} --input_list ${input:inputList} --output_dir ./output ${input:runtime}",
      "problemMatcher": []
    },
    {
      "label": "SNPE: Build C++ (CMake)",
      "type": "shell",
      "command": "source ./activate.sh && mkdir -p build && cd build && cmake .. && cmake --build . --config Release",
      "group": "build",
      "problemMatcher": ["$gcc"]
    },
    {
      "label": "Deploy: Model to Target",
      "type": "shell",
      "command": "./deploy.sh ${input:modelPath} --runtime ${input:runtime}",
      "problemMatcher": []
    }
  ],
  "inputs": [
    { "id": "modelPath",  "type": "promptString",  "description": "Model file (.onnx or .dlc)", "default": "model.onnx" },
    { "id": "dlcPath",    "type": "promptString",  "description": ".dlc file path",             "default": "model.dlc" },
    { "id": "inputList",  "type": "promptString",  "description": "input_list.txt path",        "default": "input_list.txt" },
    { "id": "runtime",    "type": "pickString",    "description": "Runtime",
      "options": ["--use_cpu", "--use_gpu", "--use_dsp", "--use_aip"],
      "default": "--use_cpu" }
  ]
}
VSCODE_TASKS
log_ok "tasks.json — QUAD, SNPE, build, deploy tasks"

# ── launch.json — Debug configurations ──
cat > "$VSCODE_DIR/launch.json" << 'VSCODE_LAUNCH'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "QUAD: MCP Server (Mock)",
      "type": "debugpy", "request": "launch",
      "module": "quad.server.main",
      "env": { "QUAD_ADAPTER_MODE": "mock" },
      "console": "integratedTerminal"
    },
    {
      "name": "QUAD: MCP Server (Real)",
      "type": "debugpy", "request": "launch",
      "module": "quad.server.main",
      "env": { "QUAD_ADAPTER_MODE": "real" },
      "console": "integratedTerminal"
    },
    {
      "name": "QUAD: Run Tests",
      "type": "debugpy", "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v", "--tb=short"],
      "console": "integratedTerminal"
    },
    {
      "name": "QUAD: Quickstart",
      "type": "debugpy", "request": "launch",
      "module": "quad.cli.main",
      "args": ["quickstart"],
      "console": "integratedTerminal"
    }
  ]
}
VSCODE_LAUNCH
log_ok "launch.json — debug configurations (Mock, Real, Tests, Quickstart)"

# ── Install recommended extensions if code CLI is available ──
if command -v code &> /dev/null; then
    log_info "Installing recommended VS Code extensions..."
    EXTENSIONS=(
        "ms-python.python"
        "charliermarsh.ruff"
        "ms-python.debugpy"
        "ms-vscode.cpptools"
        "ms-vscode.cmake-tools"
    )
    for ext in "${EXTENSIONS[@]}"; do
        code --install-extension "$ext" --force &>/dev/null && log_ok "$ext" || log_warn "$ext (failed, install manually)"
    done
else
    log_warn "VS Code CLI (code) not found — install extensions manually"
    log_info "Recommended: ms-python.python, charliermarsh.ruff, ms-vscode.cpptools, ms-vscode.cmake-tools"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: SDK Setup (one-step, multi-strategy)
# ═══════════════════════════════════════════════════════════════════════════
SDK_INSTALLED=false
RESOLVED_SDK_ROOT=""
RESOLVED_SDK_VERSION=""
RESOLVED_SDK_FLAVOR=""

if [ "$MOCK_ONLY" = true ]; then
    log_section "Step 5: SDK Setup (Skipped — mock-only)"
    log_info "All tools will use simulated responses."
else
    # source setup_sdk.sh so it can export RESOLVED_SDK_ROOT into our scope
    chmod +x "${SCRIPT_DIR}/scripts/setup_sdk.sh"
    # shellcheck source=scripts/setup_sdk.sh
    source "${SCRIPT_DIR}/scripts/setup_sdk.sh"

    if setup_sdk --qairt-archive "${QAIRT_ARCHIVE}"; then
        SDK_INSTALLED=true
        # RESOLVED_SDK_ROOT/VERSION/FLAVOR populated by setup_sdk
    fi

    # Optional: run additional adapters (qnn-specific, hexagon, target, udo)
    # only after the core QAIRT setup succeeded
    if [ "$SDK_INSTALLED" = true ]; then
        IFS=',' read -ra ADAPTERS <<< "$ADAPTERS_LIST"
        ADAPTERS_INSTALLED=0
        for adapter in "${ADAPTERS[@]}"; do
            adapter=$(echo "$adapter" | tr -d ' ')
            # qairt is handled by setup_sdk.sh — skip its legacy script
            if [ "$adapter" = "qairt" ]; then
                ADAPTERS_INSTALLED=$((ADAPTERS_INSTALLED + 1))
                continue
            fi
            setup_script="${ADAPTERS_DIR}/setup_${adapter}.sh"
            if [ -f "$setup_script" ]; then
                chmod +x "$setup_script"
                # shellcheck source=/dev/null
                source "$setup_script"
                setup_func="setup_${adapter}"
                if declare -f "$setup_func" > /dev/null 2>&1; then
                    if $setup_func; then
                        ADAPTERS_INSTALLED=$((ADAPTERS_INSTALLED + 1))
                    fi
                fi
            fi
        done
        log_info "Adapters configured: ${ADAPTERS_INSTALLED}/${#ADAPTERS[@]}"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Model Framework Packages
# ═══════════════════════════════════════════════════════════════════════════
log_section "Step 6: Model Packages"

pip install --quiet onnx>=1.14 numpy>=1.24 pillow>=10.0 2>/dev/null || true
log_ok "Core packages: onnx, numpy, pillow"

if [ "$MOCK_ONLY" = false ]; then
    pip install --quiet onnxruntime>=1.16 2>/dev/null && log_ok "onnxruntime" || log_warn "onnxruntime failed"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Verification
# ═══════════════════════════════════════════════════════════════════════════
TEST_RESULT=""
if [ "$SKIP_TESTS" = true ]; then
    log_section "Step 7: Verification (Skipped)"
else
    log_section "Step 7: Verification"
    log_info "Running test suite..."
    echo ""
    TEST_RESULT=$(pytest tests/ -q --tb=no 2>&1 | tail -1)
    echo "    $TEST_RESULT"
    echo ""
    if echo "$TEST_RESULT" | grep -q "passed"; then
        log_ok "Tests passing"
    else
        log_warn "Some tests may have issues"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Generate activate.sh
# ═══════════════════════════════════════════════════════════════════════════
log_section "Step 8: Activation Script"

cat > "${SCRIPT_DIR}/activate.sh" << 'ACTIVATE_HEADER'
#!/usr/bin/env bash
# QUAD Environment Activation — source this in each new terminal
# Usage: source ./activate.sh
ACTIVATE_HEADER

cat >> "${SCRIPT_DIR}/activate.sh" << ACTIVATE_BODY

# Python virtual environment
if [ -d "${VENV_DIR}" ]; then
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
    elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
        source "${VENV_DIR}/Scripts/activate"
    fi
fi

# QAIRT/SNPE SDK — resolved at activation via sdk_manager so this works
# correctly even if the SDK was added/updated after install.
if command -v python >/dev/null 2>&1; then
    eval "\$(python - <<'PY' 2>/dev/null
from quad.sdk_manager import resolve_sdk_root
info = resolve_sdk_root()
if info:
    print(f'export QAIRT_SDK_ROOT="{info.root}"')
    print(f'export QNN_SDK_ROOT="{info.root}"')
    print(f'export SNPE_ROOT="{info.root}"')
    if info.bin_dir:
        # cross-platform PATH separator handled by sdk_manager — use POSIX here
        print(f'export PATH="{info.bin_dir}:\$PATH"')
PY
)" || true
fi
ACTIVATE_BODY

if [ -n "${HEXAGON_TOOLS_DIR:-}" ]; then
    echo "export HEXAGON_TOOLS_DIR=\"${HEXAGON_TOOLS_DIR}\"" >> "${SCRIPT_DIR}/activate.sh"
fi

cat >> "${SCRIPT_DIR}/activate.sh" << 'ACTIVATE_FOOTER'

echo ""
echo "  QUAD environment activated"
echo "  Python: $(python --version 2>&1)"
echo "  Mode: $(grep -m1 adapter_mode quad.toml 2>/dev/null || echo 'mock')"
[ -n "${QAIRT_SDK_ROOT:-}" ] && echo "  SDK: ${QAIRT_SDK_ROOT}"
echo ""
echo "  Commands: ./launch.sh | quad quickstart | quad doctor | make test"
echo ""
ACTIVATE_FOOTER

chmod +x "${SCRIPT_DIR}/activate.sh"
log_ok "Created activate.sh"

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  Installation Complete!${NC}"
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Platform:  quad-agent installed"
if [ "$SDK_INSTALLED" = true ] && [ -n "$RESOLVED_SDK_ROOT" ]; then
    echo -e "  QAIRT SDK: ${GREEN}${RESOLVED_SDK_FLAVOR} ${RESOLVED_SDK_VERSION} (real mode available)${NC}"
    echo "             root: ${RESOLVED_SDK_ROOT}"
else
    echo -e "  QAIRT SDK: ${YELLOW}Not installed — running in MOCK mode${NC}"
    echo "             To enable real mode: ./install.sh --qairt-archive <path>"
fi
[ -n "$TEST_RESULT" ] && echo "  Tests:     $TEST_RESULT"
echo ""
echo -e "  ${BOLD}To get started:${NC}"
echo ""
echo "    source ./activate.sh         # Activate environment"
if [ "$SDK_INSTALLED" = true ]; then
    echo "    quad mode                    # Confirms 'real-mode: READY'"
    echo "    quad doctor --real-mode      # Strict pre-flight on the SDK"
fi
echo "    ./launch.sh                  # Start MCP server"
echo "    quad quickstart              # Interactive wizard"
echo "    quad sdk status              # Show resolved SDK info"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
