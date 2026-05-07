#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# QUAD — setup.sh (deprecated alias for install.sh)
# ═══════════════════════════════════════════════════════════════════════════════
#
# setup.sh has been consolidated into install.sh. This thin shim translates
# the legacy flags and forwards to install.sh, so existing scripts and
# documentation that reference ./setup.sh keep working.
#
# Old flag → new flag:
#   ./setup.sh             → ./install.sh --mock-only
#   ./setup.sh --real      → ./install.sh --real
#   ./setup.sh --no-tests  → ./install.sh --skip-tests
#   ./setup.sh --clean     → ./install.sh --clean
#
# Use install.sh directly for the full feature set, including
# --qairt-archive PATH (one-step real-hardware install).
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "ERROR: setup.sh requires bash. On Windows, run .\\bootstrap.ps1 first." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SH="${SCRIPT_DIR}/install.sh"

if [ ! -x "$INSTALL_SH" ]; then
    if [ -f "$INSTALL_SH" ]; then
        chmod +x "$INSTALL_SH"
    else
        echo "ERROR: install.sh not found at $INSTALL_SH" >&2
        exit 1
    fi
fi

# Translate legacy flags
ARGS=()
DEFAULT_MOCK=true   # setup.sh's default was mock-only behaviour

while [[ $# -gt 0 ]]; do
    case $1 in
        --real)      ARGS+=("--real");        DEFAULT_MOCK=false; shift ;;
        --no-tests)  ARGS+=("--skip-tests");  shift ;;
        --clean)     ARGS+=("--clean");       shift ;;
        --help|-h)
            cat <<HELP

NOTE: setup.sh has been consolidated into install.sh.

This shim still works and forwards to install.sh. For the full set of
options (especially --qairt-archive for one-step real-hardware install),
run:

    ./install.sh --help

Legacy translation:
    ./setup.sh             -> ./install.sh --mock-only
    ./setup.sh --real      -> ./install.sh --real
    ./setup.sh --no-tests  -> ./install.sh --skip-tests
    ./setup.sh --clean     -> ./install.sh --clean
HELP
            exit 0
            ;;
        *)  ARGS+=("$1"); shift ;;
    esac
done

if [ "$DEFAULT_MOCK" = true ]; then
    ARGS=("--mock-only" "${ARGS[@]}")
fi

echo "[setup.sh] Forwarding to: ${INSTALL_SH} ${ARGS[*]}" >&2
exec "$INSTALL_SH" "${ARGS[@]}"
