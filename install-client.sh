#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# QUAD — Lightweight CLIENT installer
#
# This is the IDE-machine installer. It does NOT install the QUAD MCP
# server, the SDK adapters, the compiler, or any of the heavy
# dependencies. It installs:
#
#   * The 'quad-client' Python package (typer + httpx + the client
#     subpackage) — total dep set is < 5 MB
#   * Claude Code's .claude/settings.json + bundled .claude/skills/
#   * Tested connection to wherever the MCP server is running
#
# Three deployment topologies are supported. The script asks which one
# you want (or pass --transport):
#
#   1. stdio-local  — server + client on the same machine (default).
#                     The MCP server gets installed too via install.sh
#                     (this script delegates if --transport=stdio-local).
#
#   2. stdio-ssh    — server runs on a remote machine; this client
#                     proxies through SSH. Lightweight client; you must
#                     have key-based SSH to the server.
#
#   3. sse-http     — server runs as a hosted HTTP/SSE service.
#                     Lightweight client; auth via bearer token in env.
#
# Examples:
#   ./install-client.sh                                # interactive prompts
#   ./install-client.sh --transport=stdio-local       # full local install
#   ./install-client.sh --transport=stdio-ssh \
#       --ssh-user=pavanr --ssh-host=test-laptop.lan
#   ./install-client.sh --transport=sse-http \
#       --sse-url=https://mcp.example.com/sse
#   ./install-client.sh --transport=sse-http \
#       --sse-url=... --sse-auth-token-env=QUAD_MCP_TOKEN
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# Source colour helpers if available
if [ -f "${SCRIPT_DIR}/scripts/helpers.sh" ]; then
    source "${SCRIPT_DIR}/scripts/helpers.sh"
else
    log_section() { echo ""; echo "=== $1 ==="; echo ""; }
    log_info()    { echo "  [INFO]  $1"; }
    log_ok()      { echo "  [OK]    $1"; }
    log_warn()    { echo "  [WARN]  $1"; }
    log_error()   { echo "  [ERROR] $1"; }
fi

# ── Defaults / parse args ─────────────────────────────────────────────────

TRANSPORT=""
SSH_USER=""
SSH_HOST=""
SSH_PORT=22
SSH_KEY=""
SSH_SERVER_CMD="python -m quad.mcp.server"
SSE_URL=""
SSE_AUTH_TOKEN_ENV=""
ADAPTER_MODE="mock"
FORCE=false
SKIP_TEST=false
SKIP_PROMPT=false
PROJECT_ROOT="${PWD}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --transport)            TRANSPORT="$2"; shift 2; SKIP_PROMPT=true ;;
        --transport=*)          TRANSPORT="${1#--transport=}"; shift; SKIP_PROMPT=true ;;
        --ssh-user)             SSH_USER="$2"; shift 2 ;;
        --ssh-user=*)           SSH_USER="${1#--ssh-user=}"; shift ;;
        --ssh-host)             SSH_HOST="$2"; shift 2 ;;
        --ssh-host=*)           SSH_HOST="${1#--ssh-host=}"; shift ;;
        --ssh-port)             SSH_PORT="$2"; shift 2 ;;
        --ssh-port=*)           SSH_PORT="${1#--ssh-port=}"; shift ;;
        --ssh-key)              SSH_KEY="$2"; shift 2 ;;
        --ssh-key=*)            SSH_KEY="${1#--ssh-key=}"; shift ;;
        --ssh-server-cmd)       SSH_SERVER_CMD="$2"; shift 2 ;;
        --sse-url)              SSE_URL="$2"; shift 2 ;;
        --sse-url=*)            SSE_URL="${1#--sse-url=}"; shift ;;
        --sse-auth-token-env)   SSE_AUTH_TOKEN_ENV="$2"; shift 2 ;;
        --sse-auth-token-env=*) SSE_AUTH_TOKEN_ENV="${1#--sse-auth-token-env=}"; shift ;;
        --adapter-mode)         ADAPTER_MODE="$2"; shift 2 ;;
        --force|-f)             FORCE=true; shift ;;
        --skip-test)            SKIP_TEST=true; shift ;;
        --project-root)         PROJECT_ROOT="$2"; shift 2 ;;
        --help|-h)
            grep -E '^# ' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) log_error "Unknown argument: $1"; echo "Run with --help for usage." >&2; exit 2 ;;
    esac
done

log_section "QUAD Client Installer"
echo "  Project root:  ${PROJECT_ROOT}"

# ── Python check ─────────────────────────────────────────────────────────

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    log_error "Python not found on PATH (tried: $PYTHON)"
    log_error "Install Python 3.10+ first. Then re-run this script."
    exit 1
fi
PY_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
log_ok "Python ${PY_VERSION}"

# ── Interactive transport prompt ──────────────────────────────────────────

if [ -z "$TRANSPORT" ]; then
    echo ""
    echo "Where does the QUAD MCP server live?"
    echo "  1) stdio-local  — same machine (default; will run install.sh too)"
    echo "  2) stdio-ssh    — remote machine, via SSH"
    echo "  3) sse-http     — remote machine, hosted HTTP/SSE service"
    echo ""
    read -r -p "Choice [1/2/3, default 1]: " choice
    case "${choice:-1}" in
        1) TRANSPORT="stdio-local" ;;
        2) TRANSPORT="stdio-ssh" ;;
        3) TRANSPORT="sse-http" ;;
        *) log_error "Invalid choice: ${choice}"; exit 2 ;;
    esac
fi
log_ok "Transport: ${TRANSPORT}"

# ── Per-transport prompts ─────────────────────────────────────────────────

if [ "$TRANSPORT" = "stdio-ssh" ]; then
    [ -z "$SSH_USER" ] && read -r -p "SSH username on the server machine: " SSH_USER
    [ -z "$SSH_HOST" ] && read -r -p "SSH hostname or IP: " SSH_HOST
    if [ -z "$SSH_KEY" ]; then
        echo "  (Press Enter to use the default key from your SSH agent / ~/.ssh/id_rsa)"
        read -r -p "Path to SSH private key (optional): " SSH_KEY
    fi
elif [ "$TRANSPORT" = "sse-http" ]; then
    [ -z "$SSE_URL" ] && read -r -p "Server URL (e.g. https://mcp.example.com/sse): " SSE_URL
    if [ -z "$SSE_AUTH_TOKEN_ENV" ]; then
        echo "  (Press Enter to skip if the server is open / no auth required)"
        read -r -p "Env var name holding the bearer token (e.g. QUAD_MCP_TOKEN): " SSE_AUTH_TOKEN_ENV
    fi
fi

# ── Install lightweight client deps ───────────────────────────────────────

log_section "Installing client deps"
log_info "Installing: typer + httpx + quad-agent[client]"
# Use editable install if we're in a checkout, else regular pip install.
# --upgrade on every run pulls the latest compatible release of typer +
# httpx so the client doesn't lag behind the server's enriched schema
# (RSS / utilisation / power fields landed in QUAD post-2026-05-10).
if [ -f "${SCRIPT_DIR}/pyproject.toml" ]; then
    "$PYTHON" -m pip install --quiet --upgrade pip 2>/dev/null
    "$PYTHON" -m pip install --quiet --upgrade -e "${SCRIPT_DIR}[client]" 2>/dev/null
    log_ok "Installed quad-agent[client] from local checkout (editable, latest)"
else
    "$PYTHON" -m pip install --quiet --upgrade pip 2>/dev/null
    "$PYTHON" -m pip install --quiet --upgrade "quad-agent[client]" 2>/dev/null
    log_ok "Installed quad-agent[client] from PyPI (latest)"
fi
# Refresh the two latest-critical client deps explicitly so the client
# can render the server's enriched profiling fields (httpx 0.27+ for the
# streaming response API; typer 0.12+ for nested subcommand groups).
"$PYTHON" -m pip install --quiet --upgrade typer httpx 2>/dev/null && \
    log_ok "Client deps refreshed to latest" || \
    log_warn "Could not refresh client deps (offline?)"

# ── Optional precision profilers (server-side only — informational here) ──
#
# QPM3 + Snapdragon Profiler live on the *server* machine, not the IDE
# machine. We surface their availability here so the IDE-side developer
# knows whether the QUAD server they're talking to will report measured
# vs estimated power / utilisation. For stdio-local installs we can
# query directly; for stdio-ssh / sse-http transports we just print the
# download URLs as guidance.
# Snapdragon X Elite advisory: when the client runs on the SAME box as a
# locally-installed server, the server's qairt-converter needs Python
# 3.10 x86_64 + VS 2022 Build Tools + the dlc_utils patch. The client
# code itself works fine on any Python 3.10+. Surface a one-line note
# so the user can address it once.
PY_VER_LOCAL=$("$PYTHON" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null)
PY_PLAT_LOCAL=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_platform())" 2>/dev/null)
HOST_MACH_LOCAL=$("$PYTHON" -c "import platform; print(platform.uname().machine)" 2>/dev/null)
if [ "$TRANSPORT" = "stdio-local" ] \
        && { [ "$HOST_MACH_LOCAL" = "ARM64" ] || [ "$HOST_MACH_LOCAL" = "AARCH64" ]; } \
        && { echo "$PY_PLAT_LOCAL" | grep -qi amd64 || echo "$PY_PLAT_LOCAL" | grep -qi x86; } \
        && [ "$PY_VER_LOCAL" != "3.10" ]; then
    log_info "Snapdragon X Elite + stdio-local detected. QUAD server's model"
    log_info "conversion needs Python 3.10 x86_64 + VS 2022 Build Tools. Run from"
    log_info "PowerShell next:  .\\bootstrap.ps1  (installs both via winget)."
fi

QPM3_FOUND=$("$PYTHON" -c "from quad.profiler.qpm3 import find_qpm3; print(find_qpm3() or '')" 2>/dev/null || echo "")
SDPTRACE_FOUND=$("$PYTHON" -c "from quad.profiler.sdptrace import find_sdptrace; print(find_sdptrace() or '')" 2>/dev/null || echo "")
if [ -n "$QPM3_FOUND" ] || [ -n "$SDPTRACE_FOUND" ]; then
    log_section "Optional profilers (server-side)"
    [ -n "$QPM3_FOUND" ]     && log_ok "QPM3 ready on this host: $QPM3_FOUND"
    [ -n "$SDPTRACE_FOUND" ] && log_ok "sdptrace ready on this host: $SDPTRACE_FOUND"
elif [ "$TRANSPORT" = "stdio-local" ]; then
    log_info "Precision profilers (optional, server-side):"
    log_info "  QPM3 + Snapdragon Profiler enable measured power + GPU% in"
    log_info "  profile_workload responses. Download: https://www.qualcomm.com/developer/software/snapdragon-profiler"
fi

# Verify quad-client is importable
if ! "$PYTHON" -c "from quad.client.cli import cli" >/dev/null 2>&1; then
    log_error "quad.client.cli not importable after install — something's wrong."
    log_error "Run: $PYTHON -m pip install -e .[client] && $PYTHON -c 'from quad.client.cli import cli'"
    exit 1
fi
log_ok "quad-client CLI ready"

# ── For stdio-local: also install the full server (delegates to install.sh)

if [ "$TRANSPORT" = "stdio-local" ]; then
    log_section "Setting up local MCP server"
    log_info "stdio-local needs the QUAD server too. Delegating to install.sh."
    log_info "(Run with --transport=stdio-ssh or sse-http to skip this step.)"
    if [ -x "${SCRIPT_DIR}/install.sh" ]; then
        # shellcheck source=install.sh
        bash "${SCRIPT_DIR}/install.sh" --mock-only --skip-tests
        log_ok "Server installed (mock mode by default; run 'quad sdk install <archive>' to add the SDK)"
    else
        log_warn "install.sh not found in this directory."
        log_warn "Make sure the QUAD package is installed for stdio-local to work."
    fi
fi

# ── Build the connect-test arg list ───────────────────────────────────────

CT_ARGS=("$TRANSPORT")
if [ "$TRANSPORT" = "stdio-local" ]; then
    CT_ARGS+=(--server-command "$PYTHON")
elif [ "$TRANSPORT" = "stdio-ssh" ]; then
    CT_ARGS+=(--ssh-user "$SSH_USER" --ssh-host "$SSH_HOST" --ssh-port "$SSH_PORT")
    [ -n "$SSH_KEY" ] && CT_ARGS+=(--ssh-key "$SSH_KEY")
    CT_ARGS+=(--server "$SSH_SERVER_CMD")
elif [ "$TRANSPORT" = "sse-http" ]; then
    CT_ARGS+=(--sse-url "$SSE_URL")
    if [ -n "$SSE_AUTH_TOKEN_ENV" ]; then
        TOKEN_VAL="${!SSE_AUTH_TOKEN_ENV:-}"
        if [ -n "$TOKEN_VAL" ]; then
            CT_ARGS+=(--auth-token "$TOKEN_VAL")
        fi
    fi
fi

# ── Connection test ───────────────────────────────────────────────────────

if [ "$SKIP_TEST" = false ]; then
    log_section "Testing connection"
    log_info "Probing ${TRANSPORT}…"
    set +e
    "$PYTHON" -m quad.client.cli connect-test "${CT_ARGS[@]}"
    PROBE_RC=$?
    set -e
    if [ $PROBE_RC -ne 0 ]; then
        log_warn "Connection test failed (exit ${PROBE_RC})."
        if [ "$FORCE" = false ]; then
            echo ""
            echo "Re-run with --force to provision anyway, or fix the issue and retry."
            exit 1
        fi
        log_warn "--force given; provisioning anyway."
    else
        log_ok "Connection test passed"
    fi
fi

# ── Provision Claude Code ─────────────────────────────────────────────────

log_section "Provisioning Claude Code"
INSTALL_ARGS=(--transport "$TRANSPORT" --skip-test --project-root "$PROJECT_ROOT")
[ "$FORCE" = true ] && INSTALL_ARGS+=(--force)

case "$TRANSPORT" in
    stdio-local)
        INSTALL_ARGS+=(--adapter-mode "$ADAPTER_MODE")
        ;;
    stdio-ssh)
        INSTALL_ARGS+=(--ssh-user "$SSH_USER" --ssh-host "$SSH_HOST" --ssh-port "$SSH_PORT")
        [ -n "$SSH_KEY" ] && INSTALL_ARGS+=(--ssh-key "$SSH_KEY")
        INSTALL_ARGS+=(--ssh-server-command "$SSH_SERVER_CMD")
        ;;
    sse-http)
        INSTALL_ARGS+=(--sse-url "$SSE_URL")
        [ -n "$SSE_AUTH_TOKEN_ENV" ] && INSTALL_ARGS+=(--sse-auth-token-env "$SSE_AUTH_TOKEN_ENV")
        ;;
esac

"$PYTHON" -m quad.client.cli install "${INSTALL_ARGS[@]}"

# ── Summary ───────────────────────────────────────────────────────────────

log_section "Done"
echo "  Transport:      ${TRANSPORT}"
echo "  Project root:   ${PROJECT_ROOT}"
echo "  Settings file:  ${PROJECT_ROOT}/.claude/settings.json"
echo "  Skills dir:     ${PROJECT_ROOT}/.claude/skills"
echo ""
echo "  Verify:         quad-client status --project-root '${PROJECT_ROOT}'"
echo "  Re-test conn:   quad-client connect-test ${TRANSPORT} ${CT_ARGS[*]:1}"
echo ""
echo "  Open Claude Code in this project — the QUAD MCP tools will appear in the tool list."
echo "  Try: \"What hardware do I have?\" or \"Convert mobilenetv2.onnx to INT8\""
