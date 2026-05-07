#!/usr/bin/env bash
# QUAD — Qualcomm Unified Agent for Developers
# Launch script — starts the MCP server in mock mode
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── Parse arguments ──
MODE="${1:-mock}"
TRANSPORT="${2:-stdio}"
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --real)    MODE="real"; shift ;;
        --mock)    MODE="mock"; shift ;;
        --sse)     TRANSPORT="sse"; shift ;;
        --stdio)   TRANSPORT="stdio"; shift ;;
        --verbose) VERBOSE="--verbose"; shift ;;
        --help|-h)
            echo "Usage: ./launch.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mock       Run in mock mode (default, no hardware needed)"
            echo "  --real       Run in real mode (requires Qualcomm SDKs)"
            echo "  --stdio      Use stdio transport (default, for Claude Code)"
            echo "  --sse        Use SSE transport (for web/IDE plugins)"
            echo "  --verbose    Enable debug logging"
            echo "  --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./launch.sh                    # Mock mode, stdio (Claude Code)"
            echo "  ./launch.sh --mock --verbose   # Mock mode with debug logs"
            echo "  ./launch.sh --sse              # SSE transport for IDE plugins"
            echo ""
            echo "Environment Variables:"
            echo "  QUAD_ADAPTER_MODE   Override adapter mode (mock|real)"
            echo "  QUAD_LOG_LEVEL      Override log level (debug|info|warning|error)"
            echo "  QAI_HUB_API_KEY    Qualcomm AI Hub API key (for real mode)"
            exit 0
            ;;
        *)  shift ;;
    esac
done

# ── Activate venv ──
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found. Run ./install.sh first."
    exit 1
fi

# ── Set environment ──
export QUAD_ADAPTER_MODE="${QUAD_ADAPTER_MODE:-$MODE}"

if [ -n "$VERBOSE" ]; then
    export QUAD_LOG_LEVEL="debug"
fi

# ── Display banner ──
echo "════════════════════════════════════════════════════════════" >&2
echo "  QUAD — Qualcomm Unified Agent for Developers" >&2
echo "  Mode: $QUAD_ADAPTER_MODE | Transport: $TRANSPORT" >&2
echo "════════════════════════════════════════════════════════════" >&2
echo "" >&2
echo "  Tools available:" >&2
echo "    • hardware_detect   — Detect Qualcomm chipset & compute units" >&2
echo "    • convert_model     — Convert ONNX/PyTorch/TF to QNN/SNPE" >&2
echo "    • profile_workload  — Profile latency, power, memory" >&2
echo "    • orchestrate_workload — Allocate layers to CPU/GPU/NPU" >&2
echo "    • generate_code     — Generate inference code (C++/Python/Kotlin)" >&2
echo "" >&2

if [ "$QUAD_ADAPTER_MODE" = "mock" ]; then
    echo "  [Mock Mode] All tools return simulated data." >&2
    echo "  No hardware or Qualcomm SDKs required." >&2
else
    echo "  [Real Mode] Tools will invoke actual Qualcomm SDKs." >&2
    echo "  Ensure QNN_SDK_ROOT / SNPE_ROOT are configured in quad.toml" >&2
fi

echo "" >&2
echo "  Starting server..." >&2
echo "════════════════════════════════════════════════════════════" >&2

# ── Launch server ──
cd "$SCRIPT_DIR"

if [ "$TRANSPORT" = "sse" ]; then
    exec python -m quad.server.main --transport sse
else
    exec python -m quad.server.main
fi
