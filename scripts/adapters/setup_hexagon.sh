#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# QUAD — Hexagon SDK Adapter Setup (Placeholder)
# Called by install.sh when Hexagon DSP development is needed.
#
# Required for: custom op development, DSP kernel programming, HTP tuning.
# Not required for basic model conversion/inference (QAIRT handles that).
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

if [ -f "${SCRIPT_DIR}/../helpers.sh" ]; then
    source "${SCRIPT_DIR}/../helpers.sh"
else
    log_info()  { echo -e "  \033[0;34m▸\033[0m $1"; }
    log_ok()    { echo -e "  \033[0;32m✓\033[0m $1"; }
    log_warn()  { echo -e "  \033[1;33m⚠\033[0m $1"; }
fi

setup_hexagon() {
    echo ""
    echo "  ┌─────────────────────────────────────────────┐"
    echo "  │  Hexagon SDK Adapter Setup                   │"
    echo "  └─────────────────────────────────────────────┘"
    echo ""

    # Check if Hexagon SDK is available via QAIRT
    if [ -n "${QAIRT_SDK_ROOT:-}" ]; then
        if ls "${QAIRT_SDK_ROOT}"/bin/x86_64-linux-clang/hexagon-* > /dev/null 2>&1; then
            export HEXAGON_TOOLS_DIR="${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang"
            log_ok "Hexagon tools found in QAIRT SDK"
            log_ok "HEXAGON_TOOLS_DIR=${HEXAGON_TOOLS_DIR}"
            return 0
        fi
    fi

    # Check for standalone Hexagon SDK
    if [ -n "${HEXAGON_SDK_ROOT:-}" ] && [ -d "${HEXAGON_SDK_ROOT}" ]; then
        log_ok "Hexagon SDK found at ${HEXAGON_SDK_ROOT}"
        return 0
    fi

    log_warn "Hexagon SDK not found."
    log_warn "Install via QPM3 (Qualcomm Package Manager 3):"
    log_warn "  1. Install QPM3 from https://qpm.qualcomm.com"
    log_warn "  2. Search 'Hexagon SDK' in QPM3"
    log_warn "  3. Download version matching your target chipset"
    log_warn ""
    log_warn "Hexagon SDK is optional — needed only for custom DSP kernels."
    return 1
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_hexagon
fi
