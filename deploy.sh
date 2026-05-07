#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# QUAD — Deploy & Run Model on Target Device
#
# End-to-end workflow: Convert → Deploy → Execute → Retrieve Results
#
# Usage:
#   ./deploy.sh model.onnx                    # Full pipeline (default: DSP)
#   ./deploy.sh model.dlc --runtime cpu       # Pre-converted model on CPU
#   ./deploy.sh model.onnx --runtime gpu --quantize int8
#   ./deploy.sh --help
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/helpers.sh"

# ── Defaults ──
MODEL_PATH=""
RUNTIME="--use_dsp"
QUANTIZE=""
OUTPUT_DIR="./results"
SKIP_CONVERT=false

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --runtime)
            case $2 in
                cpu) RUNTIME="--use_cpu" ;;
                gpu) RUNTIME="--use_gpu" ;;
                dsp|npu) RUNTIME="--use_dsp" ;;
                aip|htp) RUNTIME="--use_aip" ;;
                *) log_error "Unknown runtime: $2"; exit 1 ;;
            esac
            shift 2 ;;
        --quantize)   QUANTIZE="$2"; shift 2 ;;
        --output)     OUTPUT_DIR="$2"; shift 2 ;;
        --skip-convert) SKIP_CONVERT=true; shift ;;
        --help|-h)
            echo ""
            echo -e "${BOLD}QUAD — Deploy & Run Model on Target Device${NC}"
            echo ""
            echo "Usage: ./deploy.sh <model_path> [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  model_path          Path to model (.onnx, .pt, .dlc)"
            echo ""
            echo "Options:"
            echo "  --runtime TYPE      cpu, gpu, dsp/npu, aip/htp (default: dsp)"
            echo "  --quantize LEVEL    int8 or int4 (default: none/fp32)"
            echo "  --output DIR        Local output directory (default: ./results)"
            echo "  --skip-convert      Model is already .dlc, skip conversion"
            echo "  --help              Show this help"
            echo ""
            echo "Environment (set before running or via activate.sh):"
            echo "  TARGET_IP           Target device IP address"
            echo "  TARGET_USER         SSH username (default: root)"
            echo "  TARGET_DEVICE_ARCH  Target arch (auto-detected if not set)"
            echo "  HEXAGON_VERSION     Hexagon DSP version (e.g., 68, 73)"
            echo "  QAIRT_SDK_ROOT      Path to QAIRT SDK"
            echo ""
            echo "Examples:"
            echo "  ./deploy.sh model.onnx                         # ONNX → DLC → DSP"
            echo "  ./deploy.sh model.onnx --runtime gpu           # Run on GPU"
            echo "  ./deploy.sh model.onnx --quantize int8         # INT8 quantized"
            echo "  ./deploy.sh model.dlc --skip-convert --runtime dsp"
            echo ""
            echo "Full workflow:"
            echo "  1. Converts model to .dlc (unless --skip-convert)"
            echo "  2. Transfers model + runtime to target device via SSH"
            echo "  3. Executes snpe-net-run on target"
            echo "  4. Retrieves results back to host"
            echo ""
            exit 0
            ;;
        *)
            if [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$1"
            else
                log_error "Unexpected argument: $1"
                exit 1
            fi
            shift ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    log_error "Model path required. Usage: ./deploy.sh <model_path>"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    log_error "Model not found: $MODEL_PATH"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  QUAD — Deploy & Run: $(basename $MODEL_PATH)${NC}"
echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# ── Check prerequisites ──
if [ -z "${QAIRT_SDK_ROOT:-}" ]; then
    log_error "QAIRT_SDK_ROOT not set. Run: source ./activate.sh"
    exit 1
fi

if [ -z "${TARGET_IP:-}" ]; then
    log_error "TARGET_IP not set."
    echo "    Set target device: export TARGET_IP=<device_ip>"
    echo "    Or run: source ./scripts/adapters/setup_target.sh && setup_target"
    exit 1
fi

DESTINATION="${DESTINATION:-/tmp/snpeexample/}"
TARGET_USER="${TARGET_USER:-root}"
TARGET_DEVICE_ARCH="${TARGET_DEVICE_ARCH:-x86_64-linux-clang}"

echo "  Model:    $(basename $MODEL_PATH)"
echo "  Runtime:  $RUNTIME"
echo "  Target:   ${TARGET_USER}@${TARGET_IP}:${DESTINATION}"
echo "  Arch:     ${TARGET_DEVICE_ARCH}"
[ -n "$QUANTIZE" ] && echo "  Quantize: $QUANTIZE"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Convert to DLC (if needed)
# ═══════════════════════════════════════════════════════════════════════════
DLC_PATH="$MODEL_PATH"

if [ "$SKIP_CONVERT" = false ] && [[ "$MODEL_PATH" != *.dlc ]]; then
    log_section "Step 1: Convert to DLC"

    DLC_PATH="${MODEL_PATH%.*}.dlc"

    log_info "Converting: $(basename $MODEL_PATH) → $(basename $DLC_PATH)"
    "${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang/qairt-converter" \
        --input_network "$MODEL_PATH" \
        2>&1 | grep -v "^$" | head -20

    if [ ! -f "$DLC_PATH" ]; then
        # Try finding the output
        DLC_PATH=$(find "$(dirname $MODEL_PATH)" -name "*.dlc" -newer "$MODEL_PATH" | head -1)
        if [ -z "$DLC_PATH" ]; then
            log_error "Conversion failed — no .dlc output found"
            exit 1
        fi
    fi
    log_ok "Converted: $(basename $DLC_PATH) ($(du -h $DLC_PATH | cut -f1))"

    # Quantize if requested
    if [ -n "$QUANTIZE" ]; then
        log_info "Quantizing to ${QUANTIZE}..."
        QUANTIZED_DLC="${DLC_PATH%.*}_${QUANTIZE}.dlc"

        # Create a minimal input list for calibration
        CALIB_DIR=$(mktemp -d)
        python3 -c "import numpy as np; np.random.randn(1,3,224,224).astype('float32').tofile('${CALIB_DIR}/calib.raw')"
        echo "${CALIB_DIR}/calib.raw" > "${CALIB_DIR}/calib_list.txt"

        "${QAIRT_SDK_ROOT}/bin/x86_64-linux-clang/qairt-quantizer" \
            --input_dlc "$DLC_PATH" \
            --input_list "${CALIB_DIR}/calib_list.txt" \
            --output_dlc "$QUANTIZED_DLC" \
            2>&1 | head -10

        if [ -f "$QUANTIZED_DLC" ]; then
            DLC_PATH="$QUANTIZED_DLC"
            log_ok "Quantized: $(basename $DLC_PATH) ($(du -h $DLC_PATH | cut -f1))"
        else
            log_warn "Quantization failed — using unquantized model"
        fi
        rm -rf "$CALIB_DIR"
    fi
else
    log_section "Step 1: Convert (Skipped — already .dlc)"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Deploy to Target
# ═══════════════════════════════════════════════════════════════════════════
log_section "Step 2: Deploy to Target"

# Create destination
ssh "${TARGET_USER}@${TARGET_IP}" "mkdir -p ${DESTINATION}" 2>/dev/null

# Transfer model
scp -q "$DLC_PATH" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}/"
log_ok "Model transferred: $(basename $DLC_PATH)"

# Transfer runtime (if not already there)
RUNTIME_CHECK=$(ssh "${TARGET_USER}@${TARGET_IP}" "ls ${DESTINATION}/libSNPE.so 2>/dev/null" || echo "")
if [ -z "$RUNTIME_CHECK" ]; then
    log_info "Transferring SNPE runtime..."
    scp -q "${QAIRT_SDK_ROOT}/lib/${TARGET_DEVICE_ARCH}/libSNPE.so" \
        "${TARGET_USER}@${TARGET_IP}:${DESTINATION}/" 2>/dev/null || log_warn "libSNPE.so transfer failed"
    scp -q "${QAIRT_SDK_ROOT}/bin/${TARGET_DEVICE_ARCH}/snpe-net-run" \
        "${TARGET_USER}@${TARGET_IP}:${DESTINATION}/" 2>/dev/null || log_warn "snpe-net-run transfer failed"

    # DSP libraries — skel naming depends on Hexagon version:
    # v65/v66 → libSnpeDspV{XX}Skel.so  (Dsp prefix)
    # v68+    → libSnpeHtpV{XX}Skel.so  (Htp prefix)
    if [ -n "${HEXAGON_VERSION:-}" ]; then
        local hex_arch="hexagon-v${HEXAGON_VERSION}"
        local skel_name=""
        if [ "${HEXAGON_VERSION}" = "65" ] || [ "${HEXAGON_VERSION}" = "66" ]; then
            skel_name="libSnpeDspV${HEXAGON_VERSION}Skel.so"
        else
            skel_name="libSnpeHtpV${HEXAGON_VERSION}Skel.so"
        fi
        scp -q "${QAIRT_SDK_ROOT}/lib/${hex_arch}/unsigned/${skel_name}" \
            "${TARGET_USER}@${TARGET_IP}:${DESTINATION}/" 2>/dev/null || true

        # Windows DSP Signature Verification (Snapdragon X Elite+):
        # The .cat catalog file MUST be deployed to the SAME folder as the .so.
        # Failure to do so causes: "Unable to load Skel Library. transportStatus: 9"
        # Catalog naming: libqnnhtpvXX.cat (all lowercase, v68+, no Windows skel for v65/v66)
        if [ "${HEXAGON_VERSION}" != "65" ] && [ "${HEXAGON_VERSION}" != "66" ]; then
            local cat_name="libqnnhtpv${HEXAGON_VERSION}.cat"
            local cat_path="${QAIRT_SDK_ROOT}/lib/${hex_arch}/unsigned/${cat_name}"
            if [ -f "$cat_path" ]; then
                scp -q "$cat_path" "${TARGET_USER}@${TARGET_IP}:${DESTINATION}/" 2>/dev/null || true
                log_ok "${cat_name} (Windows signature catalog)"
            fi
        fi

        scp -q "${QAIRT_SDK_ROOT}/lib/${TARGET_DEVICE_ARCH}/libSnpeDspV${HEXAGON_VERSION}Stub.so" \
            "${TARGET_USER}@${TARGET_IP}:${DESTINATION}/" 2>/dev/null || true
        scp -q "${QAIRT_SDK_ROOT}/lib/${TARGET_DEVICE_ARCH}/libSnpeHtpPrepare.so" \
            "${TARGET_USER}@${TARGET_IP}:${DESTINATION}/" 2>/dev/null || true
    fi
    log_ok "Runtime deployed"
else
    log_ok "Runtime already on target"
fi

# Create dummy input for test run
log_info "Creating test input..."
ssh "${TARGET_USER}@${TARGET_IP}" "cd ${DESTINATION} && python3 -c \"import numpy as np; np.random.randn(1,3,224,224).astype('float32').tofile('test_input.raw')\" && echo 'test_input.raw' > input_list.txt" 2>/dev/null \
    || log_warn "Could not create test input on target (python3 may not be installed)"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Execute on Target
# ═══════════════════════════════════════════════════════════════════════════
log_section "Step 3: Execute Inference"

DLC_FILENAME=$(basename "$DLC_PATH")

REMOTE_CMD="cd ${DESTINATION} && "
REMOTE_CMD+="export PATH=\$PATH:${DESTINATION} && "
REMOTE_CMD+="export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${DESTINATION} && "

# DSP environment
if [[ "$RUNTIME" == *"dsp"* ]] || [[ "$RUNTIME" == *"aip"* ]]; then
    REMOTE_CMD+="export ADSP_LIBRARY_PATH=\"${DESTINATION};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp\" && "
fi

REMOTE_CMD+="chmod +x ./snpe-net-run && "
REMOTE_CMD+="./snpe-net-run --container ./${DLC_FILENAME} --input_list ./input_list.txt --output_dir ./output ${RUNTIME}"

log_info "Running: snpe-net-run ${RUNTIME} on $(basename $DLC_PATH)"
echo ""
EXEC_OUTPUT=$(ssh "${TARGET_USER}@${TARGET_IP}" "$REMOTE_CMD" 2>&1) || true
echo "$EXEC_OUTPUT"
echo ""

if echo "$EXEC_OUTPUT" | grep -q "Successfully executed"; then
    log_ok "Inference completed successfully"
else
    log_warn "Inference may have issues — check output above"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Retrieve Results
# ═══════════════════════════════════════════════════════════════════════════
log_section "Step 4: Retrieve Results"

mkdir -p "$OUTPUT_DIR"
scp -rq "${TARGET_USER}@${TARGET_IP}:${DESTINATION}/output" "${OUTPUT_DIR}/" 2>/dev/null \
    && log_ok "Results saved to ${OUTPUT_DIR}/output/" \
    || log_warn "No output to retrieve (inference may not have produced results)"

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${GREEN}  Done!${NC}"
echo ""
echo "  Model:   $(basename $DLC_PATH)"
echo "  Target:  ${TARGET_USER}@${TARGET_IP}"
echo "  Runtime: $RUNTIME"
echo "  Results: ${OUTPUT_DIR}/output/"
echo ""
