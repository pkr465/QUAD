#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# QUAD — QAIRT/SNPE Adapter Setup
# Called by install.sh — can also be run standalone.
#
# Sets up the Qualcomm AI Runtime (QAIRT) which includes SNPE + QNN tools.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
QAIRT_VERSION="${QAIRT_VERSION:-2.45.0.260326}"
QAIRT_DIR="${PROJECT_ROOT}/qairt"
QAIRT_SDK_ROOT="${QAIRT_DIR}/${QAIRT_VERSION}"
QAIRT_DOWNLOAD_URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QAIRT_VERSION}/"

# Source shared helpers if available
if [ -f "${SCRIPT_DIR}/../helpers.sh" ]; then
    source "${SCRIPT_DIR}/../helpers.sh"
else
    # Inline fallback if run standalone
    log_info()  { echo -e "  \033[0;34m▸\033[0m $1"; }
    log_ok()    { echo -e "  \033[0;32m✓\033[0m $1"; }
    log_warn()  { echo -e "  \033[1;33m⚠\033[0m $1"; }
    log_error() { echo -e "  \033[0;31m✗\033[0m $1"; }
fi

# ── Check if already installed ──
setup_qairt() {
    echo ""
    echo "  ┌─────────────────────────────────────────────┐"
    echo "  │  QAIRT/SNPE Adapter Setup (v${QAIRT_VERSION})  │"
    echo "  └─────────────────────────────────────────────┘"
    echo ""

    # Step 1: Check if SDK is present
    if [ -d "${QAIRT_SDK_ROOT}/bin" ]; then
        log_ok "QAIRT SDK present at qairt/${QAIRT_VERSION}/"
    else
        log_info "QAIRT SDK not found. Downloading..."
        _download_sdk
    fi

    # Step 2: Verify SDK structure
    if [ ! -d "${QAIRT_SDK_ROOT}/bin" ]; then
        log_warn "SDK not available — skipping QAIRT adapter setup"
        log_warn "Download manually: ${QAIRT_DOWNLOAD_URL}"
        return 1
    fi

    # Step 3: Set environment variables
    _set_environment

    # Step 4: System dependencies (Linux)
    if [[ "$(uname)" == "Linux" ]] && [ "${SKIP_SDK_DEPS:-false}" != "true" ]; then
        _install_system_deps
    fi

    # Step 5: Python dependencies
    _install_python_deps

    # Step 6: Verify tools
    _verify_tools

    # Step 7: Update quad.toml
    _update_config

    log_ok "QAIRT/SNPE adapter setup complete"
    return 0
}

_download_sdk() {
    local download_file="${PROJECT_ROOT}/qairt-sdk-${QAIRT_VERSION}.zip"

    echo ""
    echo "    Download URL: ${QAIRT_DOWNLOAD_URL}"
    echo ""

    # Try wget
    if command -v wget &> /dev/null; then
        if wget -q --show-progress -O "${download_file}" "${QAIRT_DOWNLOAD_URL}" 2>/dev/null; then
            log_info "Extracting SDK..."
            unzip -q "${download_file}" -d "${PROJECT_ROOT}" 2>/dev/null
            rm -f "${download_file}"
            log_ok "SDK downloaded and extracted"
            return 0
        fi
        rm -f "${download_file}"
    fi

    # Try curl
    if command -v curl &> /dev/null; then
        if curl -fsSL -o "${download_file}" "${QAIRT_DOWNLOAD_URL}" 2>/dev/null; then
            log_info "Extracting SDK..."
            unzip -q "${download_file}" -d "${PROJECT_ROOT}" 2>/dev/null
            rm -f "${download_file}"
            log_ok "SDK downloaded and extracted"
            return 0
        fi
        rm -f "${download_file}"
    fi

    log_warn "Automatic download failed (may require Qualcomm account)"
    log_warn "Please download manually from the URL above"
    log_warn "Then extract: unzip <file>.zip -d ${PROJECT_ROOT}"
    return 1
}

_set_environment() {
    export QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT}"
    export QNN_SDK_ROOT="${QAIRT_SDK_ROOT}"
    export SNPE_ROOT="${QAIRT_SDK_ROOT}"
    export PATH="${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}"
    export LD_LIBRARY_PATH="${QAIRT_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${QAIRT_SDK_ROOT}/lib/python/:${PYTHONPATH:-}"

    log_ok "Environment: QAIRT_SDK_ROOT=${QAIRT_SDK_ROOT}"
}

_install_system_deps() {
    if [ -f "${QAIRT_SDK_ROOT}/bin/check-linux-dependency.sh" ]; then
        log_info "Installing system dependencies (may require sudo)..."
        chmod +x "${QAIRT_SDK_ROOT}/bin/check-linux-dependency.sh"
        sudo "${QAIRT_SDK_ROOT}/bin/check-linux-dependency.sh" 2>/dev/null || \
            log_warn "Some system deps may need manual installation"
    fi
}

_install_python_deps() {
    if [ -f "${QAIRT_SDK_ROOT}/bin/check-python-dependency" ]; then
        log_info "Installing QAIRT Python dependencies..."
        python "${QAIRT_SDK_ROOT}/bin/check-python-dependency" 2>/dev/null || \
            log_warn "Some Python deps may need manual install"
    fi

    # Core packages for model conversion
    pip install --quiet onnx>=1.14 onnxruntime>=1.16 numpy>=1.24 2>/dev/null || true
    log_ok "Model framework packages installed"
}

_verify_tools() {
    log_info "Verifying SDK tools..."
    local found=0 total=0

    for tool in qairt-converter qairt-quantizer snpe-net-run snpe-dlc-info qnn-net-run qnn-onnx-converter; do
        total=$((total + 1))
        if [ -f "${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang/${tool}" ]; then
            log_ok "${tool}"
            found=$((found + 1))
        else
            log_warn "${tool} — not found"
        fi
    done

    log_info "Tools verified: ${found}/${total}"
}

_update_config() {
    local config="${PROJECT_ROOT}/quad.toml"
    if [ -f "${config}" ]; then
        sed -i.bak 's|adapter_mode = "mock"|adapter_mode = "real"|' "${config}" 2>/dev/null || true
        sed -i.bak "s|sdk_path = \"\"|sdk_path = \"${QAIRT_SDK_ROOT}\"|" "${config}" 2>/dev/null || true
        rm -f "${config}.bak"
        log_ok "quad.toml updated (adapter_mode=real)"
    fi
}

# Export for use by install.sh
export QAIRT_SDK_ROOT
export QNN_SDK_ROOT
export SNPE_ROOT

# Run if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_qairt
fi
