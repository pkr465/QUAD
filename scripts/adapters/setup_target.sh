#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# QUAD — Target Device Setup & Deployment
# Handles SSH setup, file transfer, and remote execution on target devices.
#
# Called by install.sh or run standalone:
#   ./scripts/adapters/setup_target.sh
#
# Supports: Linux (x86/ARM), Android (ADB), OE Linux (Yocto)
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
    log_error() { echo -e "  \033[0;31m✗\033[0m $1"; }
fi

# ── Configuration (override via env vars or quad.toml) ──
TARGET_IP="${TARGET_IP:-}"
TARGET_USER="${TARGET_USER:-root}"
TARGET_DEVICE_ARCH="${TARGET_DEVICE_ARCH:-}"
DESTINATION="${DESTINATION:-/tmp/snpeexample/}"
HEXAGON_VERSION="${HEXAGON_VERSION:-}"

setup_target() {
    echo ""
    echo "  ┌─────────────────────────────────────────────┐"
    echo "  │  Target Device Setup                         │"
    echo "  └─────────────────────────────────────────────┘"
    echo ""

    # Check prerequisites
    if [ -z "${QAIRT_SDK_ROOT:-}" ]; then
        log_error "QAIRT_SDK_ROOT not set. Run setup_qairt.sh first."
        return 1
    fi

    # Interactive setup if not configured
    if [ -z "$TARGET_IP" ]; then
        _configure_target
    fi

    # Verify connectivity
    _verify_connection

    # Detect target architecture
    if [ -z "$TARGET_DEVICE_ARCH" ]; then
        _detect_architecture
    fi

    # Create destination on target
    _prepare_destination

    # Transfer SNPE runtime files
    _transfer_runtime

    # Transfer DSP/HTP files if needed
    if [ -n "$HEXAGON_VERSION" ]; then
        _transfer_dsp_files
    fi

    log_ok "Target device setup complete"
    log_info "Target: ${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
    log_info "Arch: ${TARGET_DEVICE_ARCH}"
    return 0
}

_configure_target() {
    echo ""
    log_info "Target device configuration required."
    echo ""
    read -p "    Target device IP address: " TARGET_IP
    read -p "    Target username [${TARGET_USER}]: " input_user
    TARGET_USER="${input_user:-$TARGET_USER}"
    read -p "    Destination path [${DESTINATION}]: " input_dest
    DESTINATION="${input_dest:-$DESTINATION}"
    echo ""

    # Save to environment for this session
    export TARGET_IP TARGET_USER DESTINATION
}

_verify_connection() {
    log_info "Verifying SSH connection to ${TARGET_USER}@${TARGET_IP}..."
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "${TARGET_USER}@${TARGET_IP}" "echo ok" &>/dev/null; then
        log_ok "SSH connection verified"
    else
        log_warn "SSH connection failed. Ensure:"
        log_warn "  1. Target device has openssh-server installed"
        log_warn "  2. Both devices are on the same network"
        log_warn "  3. SSH key is configured (ssh-copy-id ${TARGET_USER}@${TARGET_IP})"
        echo ""
        echo "    On target device:"
        echo "      sudo apt install openssh-server"
        echo "      sudo systemctl enable ssh && sudo systemctl start ssh"
        echo ""
        return 1
    fi
}

_detect_architecture() {
    log_info "Detecting target architecture..."
    local arch=$(ssh "${TARGET_USER}@${TARGET_IP}" "uname -m" 2>/dev/null)
    local os_id=$(ssh "${TARGET_USER}@${TARGET_IP}" "grep ^ID= /etc/os-release 2>/dev/null | cut -d= -f2" 2>/dev/null)

    case "${arch}" in
        x86_64)
            TARGET_DEVICE_ARCH="x86_64-linux-clang"
            ;;
        aarch64|arm64)
            # Determine GCC version for proper lib selection
            local gcc_ver=$(ssh "${TARGET_USER}@${TARGET_IP}" "gcc --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+' | head -1" 2>/dev/null)
            case "${gcc_ver}" in
                11.*)  TARGET_DEVICE_ARCH="aarch64-oe-linux-gcc11.2" ;;
                9.4*)  TARGET_DEVICE_ARCH="aarch64-ubuntu-gcc9.4" ;;
                9.3*)  TARGET_DEVICE_ARCH="aarch64-oe-linux-gcc9.3" ;;
                8.*)   TARGET_DEVICE_ARCH="aarch64-oe-linux-gcc8.2" ;;
                *)     TARGET_DEVICE_ARCH="aarch64-oe-linux-gcc11.2" ;;  # Default
            esac
            ;;
        *)
            log_warn "Unknown architecture: ${arch}. Defaulting to x86_64-linux-clang"
            TARGET_DEVICE_ARCH="x86_64-linux-clang"
            ;;
    esac

    export TARGET_DEVICE_ARCH
    log_ok "Architecture: ${TARGET_DEVICE_ARCH}"
}

_prepare_destination() {
    log_info "Creating destination: ${DESTINATION}"
    ssh "${TARGET_USER}@${TARGET_IP}" "mkdir -p ${DESTINATION}" 2>/dev/null
    log_ok "Destination ready"
}

_transfer_runtime() {
    log_info "Transferring SNPE runtime files..."

    local sdk="${QAIRT_SDK_ROOT}"

    # Core runtime library
    if [ -f "${sdk}/lib/${TARGET_DEVICE_ARCH}/libSNPE.so" ]; then
        scp -q "${sdk}/lib/${TARGET_DEVICE_ARCH}/libSNPE.so" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
        log_ok "libSNPE.so"
    else
        log_warn "libSNPE.so not found for ${TARGET_DEVICE_ARCH}"
    fi

    # snpe-net-run executable
    if [ -f "${sdk}/bin/${TARGET_DEVICE_ARCH}/snpe-net-run" ]; then
        scp -q "${sdk}/bin/${TARGET_DEVICE_ARCH}/snpe-net-run" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
        log_ok "snpe-net-run"
    else
        log_warn "snpe-net-run not found for ${TARGET_DEVICE_ARCH}"
    fi

    # Additional SNPE libraries (non-QNN)
    local lib_dir="${sdk}/lib/${TARGET_DEVICE_ARCH}"
    if [ -d "$lib_dir" ]; then
        for lib in $(ls "${lib_dir}/" 2>/dev/null | grep -v Qnn | grep "\.so$"); do
            scp -q "${lib_dir}/${lib}" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}" 2>/dev/null || true
        done
        log_ok "Additional SNPE libraries transferred"
    fi
}

_transfer_dsp_files() {
    log_info "Transferring DSP/HTP files (Hexagon v${HEXAGON_VERSION})..."

    local sdk="${QAIRT_SDK_ROOT}"
    local hex_arch="hexagon-v${HEXAGON_VERSION}"
    local ver="${HEXAGON_VERSION}"

    # ── Skel library: v65/v66 use "Dsp" prefix; v68+ use "Htp" prefix ──
    # See: SNPE DSP Runtime Environment documentation
    local skel_name=""
    local skel_file=""
    if [ "${ver}" = "65" ] || [ "${ver}" = "66" ]; then
        skel_name="libSnpeDspV${ver}Skel.so"
        skel_file="${sdk}/lib/${hex_arch}/unsigned/${skel_name}"
    else
        # v68, v69, v73, v75, v79, v81 → HTP prefix
        skel_name="libSnpeHtpV${ver}Skel.so"
        skel_file="${sdk}/lib/${hex_arch}/unsigned/${skel_name}"
    fi

    if [ -f "$skel_file" ]; then
        scp -q "$skel_file" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
        log_ok "${skel_name}"

        # Windows DSP Signature Verification (Snapdragon X Elite+):
        # The .cat catalog file MUST be in the SAME folder as the .so.
        # Missing .cat → "Unable to load Skel Library. transportStatus: 9"
        # Only applicable for HTP versions (v68+); v65/v66 have no .cat.
        if [ "${ver}" != "65" ] && [ "${ver}" != "66" ]; then
            local cat_name="libqnnhtpv${ver}.cat"
            local cat_file="${sdk}/lib/${hex_arch}/unsigned/${cat_name}"
            if [ -f "$cat_file" ]; then
                scp -q "$cat_file" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
                log_ok "${cat_name} (Windows signature catalog — same folder as skel)"
            fi
        fi
    else
        log_warn "Skel not found: ${skel_file}"
    fi

    # ── Stub library (always "Dsp" prefix regardless of version) ──
    local stub="${sdk}/lib/${TARGET_DEVICE_ARCH}/libSnpeDspV${ver}Stub.so"
    if [ -f "$stub" ]; then
        scp -q "$stub" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
        log_ok "libSnpeDspV${ver}Stub.so"
    fi

    # ── HTP prepare library (v68+) ──
    local htp="${sdk}/lib/${TARGET_DEVICE_ARCH}/libSnpeHtpPrepare.so"
    if [ -f "$htp" ]; then
        scp -q "$htp" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
        log_ok "libSnpeHtpPrepare.so"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# Deployment function — called by QUAD tools to deploy a specific model
# ═══════════════════════════════════════════════════════════════════════════
deploy_model_to_target() {
    local model_path="${1}"
    local input_list="${2:-}"

    if [ -z "$TARGET_IP" ] || [ -z "$TARGET_USER" ]; then
        log_error "Target not configured. Run: setup_target"
        return 1
    fi

    log_info "Deploying model to target..."

    # Transfer model
    scp -q "${model_path}" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
    log_ok "Model: $(basename ${model_path})"

    # Transfer input list if provided
    if [ -n "$input_list" ] && [ -f "$input_list" ]; then
        scp -q "${input_list}" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
        log_ok "Input list: $(basename ${input_list})"

        # Transfer referenced input files
        while IFS= read -r line || [ -n "$line" ]; do
            [[ "$line" =~ ^#.*$ ]] && continue  # Skip comments
            [[ -z "$line" ]] && continue         # Skip empty lines
            if [ -f "$line" ]; then
                scp -q "$line" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}" 2>/dev/null || true
            fi
        done < "$input_list"
        log_ok "Input data transferred"
    fi

    log_ok "Deployment complete: ${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
}

# ═══════════════════════════════════════════════════════════════════════════
# Remote execution — run snpe-net-run on target device
# ═══════════════════════════════════════════════════════════════════════════
execute_on_target() {
    local model_file="${1}"
    local input_list="${2:-target_raw_list.txt}"
    local runtime="${3:-}"  # --use_cpu, --use_gpu, --use_dsp
    local output_dir="${4:-output}"

    if [ -z "$TARGET_IP" ] || [ -z "$TARGET_USER" ]; then
        log_error "Target not configured. Run: setup_target"
        return 1
    fi

    log_info "Executing inference on target..."

    # Build remote command
    local cmd="cd ${DESTINATION} && "
    cmd+="export PATH=\$PATH:${DESTINATION} && "
    cmd+="export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${DESTINATION} && "

    # Set ADSP_LIBRARY_PATH for DSP/AIP runtimes
    # IMPORTANT: Uses semicolons (not colons) and must be quoted
    # Three mandatory paths required for Android/Linux targets
    # Automotive Linux needs different paths (/usr/lib/rfsa/adsp instead)
    if [[ "$runtime" == *"dsp"* ]] || [[ "$runtime" == *"aip"* ]]; then
        local target_os="${TARGET_OS:-linux}"  # "android", "linux", "automotive"
        local adsp_paths="${DESTINATION}"
        if [ "$target_os" = "automotive" ]; then
            adsp_paths="${adsp_paths};/usr/lib/rfsa/adsp;/dsp"
        else
            # Android + embedded Linux (default)
            adsp_paths="${adsp_paths};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
        fi
        cmd+="export ADSP_LIBRARY_PATH=\"${adsp_paths}\" && "
    fi

    cmd+="./snpe-net-run "
    cmd+="--container \"./${model_file}\" "
    cmd+="--input_list \"./${input_list}\" "
    cmd+="--output_dir \"./${output_dir}\" "
    [ -n "$runtime" ] && cmd+="${runtime} "

    # Execute
    local result
    result=$(ssh "${TARGET_USER}@${TARGET_IP}" "$cmd" 2>&1)
    echo "$result"

    if echo "$result" | grep -q "Successfully executed"; then
        log_ok "Inference complete"
        return 0
    else
        log_warn "Inference may have issues — check output above"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# Retrieve results from target
# ═══════════════════════════════════════════════════════════════════════════
retrieve_results() {
    local output_dir="${1:-output}"
    local local_dir="${2:-.}"

    log_info "Retrieving results from target..."
    scp -rq "${TARGET_USER}@${TARGET_IP}:${DESTINATION}/${output_dir}" "${local_dir}/"
    log_ok "Results saved to ${local_dir}/${output_dir}/"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_target
fi
