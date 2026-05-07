#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# QUAD — UDO Package Setup & Compilation
# Called by install.sh — can also be run standalone.
#
# Sets up and compiles SNPE User-Defined Operations (UDO) packages.
# UDOs allow custom ops to run on CPU, GPU, or DSP (HTP) backends.
#
# Prerequisites:
#   QAIRT_SDK_ROOT   — path to the QAIRT/SNPE SDK (set by setup_qairt.sh)
#   HEXAGON_SDK_ROOT — path to Hexagon SDK (required for DSP targets)
#   ANDROID_NDK_ROOT — path to Android NDK  (required for Android targets)
#
# Reference:
#   $QAIRT_SDK_ROOT/examples/SNPE/NativeCpp/UdoExample/
#   SNPE UDO Tutorial: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# ── Load shared helpers ─────────────────────────────────────────────────────
if [ -f "${SCRIPT_DIR}/../helpers.sh" ]; then
    source "${SCRIPT_DIR}/../helpers.sh"
else
    # Inline fallback when run standalone without the rest of the repo
    log_info()    { echo -e "  \033[0;34m▸\033[0m $1"; }
    log_ok()      { echo -e "  \033[0;32m✓\033[0m $1"; }
    log_warn()    { echo -e "  \033[1;33m⚠\033[0m $1"; }
    log_error()   { echo -e "  \033[0;31m✗\033[0m $1"; }
    log_section() { echo -e "\n\033[0;36m━━━ $1 ━━━\033[0m\n"; }
fi

# ═══════════════════════════════════════════════════════════════════════════
# setup_udo — main entry point
# ═══════════════════════════════════════════════════════════════════════════

setup_udo() {
    echo ""
    echo "  ┌─────────────────────────────────────────────┐"
    echo "  │  UDO Package Setup                           │"
    echo "  └─────────────────────────────────────────────┘"
    echo ""

    # ── Step 1: Verify required environment variables ──────────────────
    log_section "Step 1: Checking environment"

    if [ -z "${QAIRT_SDK_ROOT:-}" ]; then
        log_error "QAIRT_SDK_ROOT is not set."
        log_warn  "Run setup_qairt.sh first, or export QAIRT_SDK_ROOT manually:"
        log_warn  "  export QAIRT_SDK_ROOT=/path/to/qairt/<version>"
        return 1
    fi

    if [ ! -d "${QAIRT_SDK_ROOT}" ]; then
        log_error "QAIRT_SDK_ROOT does not exist: ${QAIRT_SDK_ROOT}"
        return 1
    fi
    log_ok "QAIRT_SDK_ROOT=${QAIRT_SDK_ROOT}"

    if [ -z "${HEXAGON_SDK_ROOT:-}" ]; then
        log_warn "HEXAGON_SDK_ROOT is not set — DSP (HTP) targets will be skipped."
        log_warn "Install the Hexagon SDK via QPM3 and export HEXAGON_SDK_ROOT."
        HEXAGON_AVAILABLE=false
    elif [ ! -d "${HEXAGON_SDK_ROOT}" ]; then
        log_warn "HEXAGON_SDK_ROOT does not exist: ${HEXAGON_SDK_ROOT}"
        HEXAGON_AVAILABLE=false
    else
        log_ok "HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}"
        HEXAGON_AVAILABLE=true
    fi

    # ── Step 2: Verify snpe-udo-package-generator is on PATH ───────────
    log_section "Step 2: Checking snpe-udo-package-generator"

    UDO_GENERATOR=""
    # Look in SDK bin directory first (most common install layout)
    _bin_dir="${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang"
    if [ -f "${_bin_dir}/snpe-udo-package-generator" ]; then
        UDO_GENERATOR="${_bin_dir}/snpe-udo-package-generator"
    elif command -v snpe-udo-package-generator &> /dev/null; then
        UDO_GENERATOR="$(command -v snpe-udo-package-generator)"
    fi

    if [ -z "${UDO_GENERATOR}" ]; then
        log_error "snpe-udo-package-generator not found."
        log_warn  "Expected location: ${_bin_dir}/snpe-udo-package-generator"
        log_warn  "Make sure QAIRT SDK is properly installed and the bin/ directory is on PATH."
        return 1
    fi
    log_ok "Generator: ${UDO_GENERATOR}"

    # ── Step 3: Export SNPE_UDO_ROOT ────────────────────────────────────
    log_section "Step 3: Setting UDO root"

    export SNPE_UDO_ROOT="${QAIRT_SDK_ROOT}/share/SNPE/SnpeUdo"

    if [ -d "${SNPE_UDO_ROOT}" ]; then
        log_ok "SNPE_UDO_ROOT=${SNPE_UDO_ROOT}"
    else
        log_warn "Expected SNPE_UDO_ROOT not found: ${SNPE_UDO_ROOT}"
        log_warn "Check your QAIRT SDK version — path may differ."
        log_warn "Continuing anyway; set SNPE_UDO_ROOT manually if builds fail."
    fi

    # ── Step 4: List available SDK example UDO configs ──────────────────
    log_section "Step 4: SDK example UDO configs"

    UDO_EXAMPLES_DIR="${QAIRT_SDK_ROOT}/examples/SNPE/NativeCpp/UdoExample"

    if [ -d "${UDO_EXAMPLES_DIR}" ]; then
        log_info "Available UDO example configs in SDK:"
        while IFS= read -r cfg; do
            log_info "  ${cfg#${UDO_EXAMPLES_DIR}/}"
        done < <(find "${UDO_EXAMPLES_DIR}" -name "*.json" 2>/dev/null | sort)
    else
        log_warn "UDO examples directory not found: ${UDO_EXAMPLES_DIR}"
        log_warn "Expected layout (QAIRT ≥ 2.x):"
        log_warn "  \$QAIRT_SDK_ROOT/examples/SNPE/NativeCpp/UdoExample/Softmax/"
        log_warn "  \$QAIRT_SDK_ROOT/examples/SNPE/NativeCpp/UdoExample/Conv2D/"
    fi

    # ── Step 5: Show package generation command ──────────────────────────
    log_section "Step 5: Generating a UDO package"

    log_info "To generate a UDO package from a JSON config:"
    echo ""
    echo "    snpe-udo-package-generator -p config.json -o output_dir/"
    echo ""
    log_info "Example using the SDK's Softmax config:"
    echo ""
    echo "    snpe-udo-package-generator \\"
    echo "        -p ${UDO_EXAMPLES_DIR}/Softmax/config/Softmax_Htp.json \\"
    echo "        -o /tmp/SoftmaxUdoPackage/"
    echo ""
    log_info "The generator creates:"
    log_info "  output_dir/<PackageName>/src/        — C++ source stubs"
    log_info "  output_dir/<PackageName>/include/    — Headers"
    log_info "  output_dir/<PackageName>/Makefile    — Build targets"
    log_info "  output_dir/<PackageName>/Android.mk  — NDK build file"

    # ── Step 6: Show compilation targets ────────────────────────────────
    log_section "Step 6: Compilation targets"

    log_info "After generating a package, enter the package directory and run:"
    echo ""

    echo "  # x86 host (for simulation and testing)"
    echo "  make cpu_x86"
    echo ""

    echo "  # Android ARM64 (requires ANDROID_NDK_ROOT)"
    if [ -z "${ANDROID_NDK_ROOT:-}" ]; then
        echo "  # NOTE: ANDROID_NDK_ROOT is not set — this target will fail"
        echo "  # export ANDROID_NDK_ROOT=/path/to/android-ndk-r25c"
    fi
    echo "  make cpu_android"
    echo ""

    echo "  # DSP / HTP (requires HEXAGON_SDK_ROOT)"
    if [ "${HEXAGON_AVAILABLE}" = "false" ]; then
        echo "  # NOTE: HEXAGON_SDK_ROOT is not set — this target will fail"
        echo "  # export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/<version>"
    fi
    echo "  make dsp_x86"
    echo ""

    log_info "Full example (Softmax, all targets):"
    echo ""
    echo "    export PACKAGE_DIR=/tmp/SoftmaxUdoPackage/SoftmaxUdoPackage"
    echo ""
    echo "    # Generate"
    echo "    snpe-udo-package-generator \\"
    echo "        -p ${UDO_EXAMPLES_DIR}/Softmax/config/Softmax_Htp.json \\"
    echo "        -o /tmp/SoftmaxUdoPackage/"
    echo ""
    echo "    # Build x86 (host)"
    echo "    make -C \${PACKAGE_DIR} cpu_x86"
    echo ""
    echo "    # Build Android"
    echo "    make -C \${PACKAGE_DIR} cpu_android"
    echo ""
    echo "    # Build DSP"
    if [ "${HEXAGON_AVAILABLE}" = "true" ]; then
        echo "    make -C \${PACKAGE_DIR} dsp_x86"
    else
        echo "    # make -C \${PACKAGE_DIR} dsp_x86  (skipped — HEXAGON_SDK_ROOT not set)"
    fi
    echo ""

    # ── Summary ──────────────────────────────────────────────────────────
    log_section "UDO Setup Summary"
    log_ok "snpe-udo-package-generator: ${UDO_GENERATOR}"
    log_ok "SNPE_UDO_ROOT: ${SNPE_UDO_ROOT}"
    if [ "${HEXAGON_AVAILABLE}" = "true" ]; then
        log_ok "DSP/HTP target: available (HEXAGON_SDK_ROOT set)"
    else
        log_warn "DSP/HTP target: unavailable (HEXAGON_SDK_ROOT not set)"
    fi
    if [ -n "${ANDROID_NDK_ROOT:-}" ] && [ -d "${ANDROID_NDK_ROOT}" ]; then
        log_ok "Android target: available (ANDROID_NDK_ROOT set)"
    else
        log_warn "Android target: ANDROID_NDK_ROOT not set — Android builds will fail"
    fi

    log_ok "UDO setup complete"
    return 0
}

# ── Export for use by install.sh ─────────────────────────────────────────────
export SNPE_UDO_ROOT="${SNPE_UDO_ROOT:-}"

# Run if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_udo
fi
