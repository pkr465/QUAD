#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# QUAD — QNN Adapter Setup (Placeholder)
# Called by install.sh when QNN-specific setup is needed.
#
# Currently QAIRT SDK includes both QNN and SNPE tools.
# This script will handle QNN-specific configuration when needed
# (e.g., QNN-only installations, Windows QNN setup).
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# Source shared helpers if available
if [ -f "${SCRIPT_DIR}/../helpers.sh" ]; then
    source "${SCRIPT_DIR}/../helpers.sh"
else
    log_info()  { echo -e "  \033[0;34m▸\033[0m $1"; }
    log_ok()    { echo -e "  \033[0;32m✓\033[0m $1"; }
    log_warn()  { echo -e "  \033[1;33m⚠\033[0m $1"; }
fi

setup_qnn() {
    echo ""
    echo "  ┌─────────────────────────────────────────────┐"
    echo "  │  QNN Adapter Setup                           │"
    echo "  └─────────────────────────────────────────────┘"
    echo ""

    # QNN tools are included in QAIRT SDK
    if [ -n "${QAIRT_SDK_ROOT:-}" ] && [ -d "${QAIRT_SDK_ROOT}/bin" ]; then
        log_ok "QNN tools available via QAIRT SDK"
        log_ok "qnn-onnx-converter, qnn-net-run, qnn-context-binary-generator"

        # QNN-specific environment
        export QNN_SDK_ROOT="${QAIRT_SDK_ROOT}"
        if [ -d "${QAIRT_SDK_ROOT}/include/QNN" ]; then
            log_ok "QNN headers found at ${QAIRT_SDK_ROOT}/include/QNN/"
        fi
    else
        log_warn "QAIRT SDK not configured. Run setup_qairt.sh first."
        log_warn "QNN tools are part of the QAIRT SDK package."
        return 1
    fi

    log_ok "QNN adapter setup complete"
    return 0
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_qnn
fi
