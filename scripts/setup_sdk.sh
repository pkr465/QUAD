#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# QUAD — One-step SDK setup
#
# Tries every legitimate strategy in order so a developer running
# ./install.sh ends up with a working QAIRT SDK install whenever
# possible:
#
#   1. --qairt-archive <path>          (explicit, highest priority)
#   2. QAIRT_SDK_ROOT already populated and valid → no-op
#   3. `quad sdk discover` finds an install (vendor defaults, ./sdks, etc.)
#   4. Auto-detect ~/Downloads/qairt*.zip / ~/Downloads/snpe*.zip
#   5. QAIRT_DOWNLOAD_URL + QAIRT_DOWNLOAD_TOKEN env vars
#      (developer-supplied auth for CI / org-managed mirrors)
#   6. Print clear instructions + exit gracefully (mock mode still works)
#
# Why not auto-download from qualcomm.com directly?
# Both Qualcomm developer pages gate downloads behind a developer
# account login + EULA acceptance per their license. There is no
# anonymous direct-download URL. This script never bypasses that —
# strategy 5 just lets the developer plug in their own pre-accepted
# token / mirror URL once.
#
# Used standalone or sourced from install.sh.
#
# Environment variables (all optional):
#   QAIRT_DOWNLOAD_URL     — Override download URL (e.g. internal mirror)
#   QAIRT_DOWNLOAD_TOKEN   — Auth token / cookie for the download URL
#                            (sent as 'Authorization: Bearer <token>')
#   QAIRT_DOWNLOADS_DIR    — Where to look for pre-downloaded archives
#                            (default: ~/Downloads)
#   QAIRT_SDK_ROOT         — Skip download if this points at a real install
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOADS_DIR="${QAIRT_DOWNLOADS_DIR:-$HOME/Downloads}"

# Source helpers (or inline fallback)
if [ -f "${SCRIPT_DIR}/helpers.sh" ]; then
    source "${SCRIPT_DIR}/helpers.sh"
else
    log_info()  { echo "  [INFO]  $1"; }
    log_ok()    { echo "  [OK]    $1"; }
    log_warn()  { echo "  [WARN]  $1"; }
    log_error() { echo "  [ERROR] $1"; }
    log_section() { echo ""; echo "=== $1 ==="; echo ""; }
fi

# Output: writes to RESOLVED_SDK_ROOT in caller scope when sourced.
RESOLVED_SDK_ROOT=""
RESOLVED_SDK_VERSION=""
RESOLVED_SDK_FLAVOR=""
RESOLVED_SDK_SOURCE=""

# Pick a python that has QUAD installed (venv preferred)
_pick_python() {
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        echo "${VIRTUAL_ENV}/bin/python"
    elif [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
        echo "${PROJECT_ROOT}/.venv/bin/python"
    elif [ -x "${PROJECT_ROOT}/.venv/Scripts/python.exe" ]; then
        echo "${PROJECT_ROOT}/.venv/Scripts/python.exe"
    else
        echo "${PYTHON:-python3}"
    fi
}

# Strategy 1: --qairt-archive flag → unpack with `quad sdk install`
_try_archive_flag() {
    local archive="$1"
    if [ -z "$archive" ]; then return 1; fi
    if [ ! -f "$archive" ]; then
        log_error "Archive not found: $archive"
        return 1
    fi
    log_info "Installing SDK from --qairt-archive: $(basename "$archive")"
    local py
    py=$(_pick_python)
    if "$py" -m quad.cli.main sdk install "$archive" --overwrite; then
        RESOLVED_SDK_SOURCE="archive-flag"
        return 0
    fi
    return 1
}

# Strategy 2: env var already valid
_try_env_var() {
    if [ -z "${QAIRT_SDK_ROOT:-}" ]; then return 1; fi
    if [ ! -d "${QAIRT_SDK_ROOT}/bin" ]; then
        log_warn "QAIRT_SDK_ROOT=${QAIRT_SDK_ROOT} but no bin/ subdirectory — ignoring"
        return 1
    fi
    RESOLVED_SDK_ROOT="${QAIRT_SDK_ROOT}"
    RESOLVED_SDK_SOURCE="env-var"
    log_ok "QAIRT_SDK_ROOT already set and valid"
    return 0
}

# Strategy 3: discover via sdk_manager
_try_discover() {
    local py
    py=$(_pick_python)
    local out
    if ! out=$("$py" -m quad.cli.main sdk discover 2>&1); then
        return 1
    fi
    # Parse first line marked with `*` (the chosen one)
    local line
    line=$(echo "$out" | grep -m1 '^[[:space:]]*\* ' || true)
    if [ -z "$line" ]; then return 1; fi
    log_ok "Discovered SDK via vendor-defaults / project ./sdks scan"
    return 0
}

# Strategy 4: auto-detect a downloaded archive in ~/Downloads
_try_downloads_dir() {
    if [ ! -d "$DOWNLOADS_DIR" ]; then return 1; fi
    local archive
    archive=$(find "$DOWNLOADS_DIR" -maxdepth 1 -type f \
        \( -name "qairt*.zip" -o -name "qairt*.tar.gz" \
           -o -name "QAIRT*.zip" -o -name "snpe*.zip" \
           -o -name "Qualcomm_AI_Runtime*.zip" \) \
        -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    if [ -z "$archive" ]; then return 1; fi
    log_info "Auto-detected archive in $DOWNLOADS_DIR: $(basename "$archive")"
    _try_archive_flag "$archive"
}

# Strategy 5: developer-supplied download URL + token (CI / mirror pattern)
_try_url_with_token() {
    if [ -z "${QAIRT_DOWNLOAD_URL:-}" ]; then return 1; fi
    if [ -z "${QAIRT_DOWNLOAD_TOKEN:-}" ]; then
        log_warn "QAIRT_DOWNLOAD_URL set but QAIRT_DOWNLOAD_TOKEN missing"
        log_warn "Authenticated download requires both. Skipping."
        return 1
    fi
    local archive="${PROJECT_ROOT}/.tmp_qairt_download.zip"
    log_info "Downloading from \$QAIRT_DOWNLOAD_URL (auth via \$QAIRT_DOWNLOAD_TOKEN)"

    if command -v curl >/dev/null 2>&1; then
        if ! curl --fail --location --silent --show-error \
                --max-time 1800 \
                --header "Authorization: Bearer ${QAIRT_DOWNLOAD_TOKEN}" \
                --output "$archive" "$QAIRT_DOWNLOAD_URL"; then
            log_warn "curl download failed"
            rm -f "$archive"
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if ! wget --quiet --timeout=1800 \
                --header="Authorization: Bearer ${QAIRT_DOWNLOAD_TOKEN}" \
                -O "$archive" "$QAIRT_DOWNLOAD_URL"; then
            log_warn "wget download failed"
            rm -f "$archive"
            return 1
        fi
    else
        log_warn "Neither curl nor wget available for download"
        return 1
    fi

    log_ok "Downloaded $(du -h "$archive" 2>/dev/null | cut -f1) archive"
    _try_archive_flag "$archive"
    local rc=$?
    rm -f "$archive"
    if [ $rc -eq 0 ]; then
        RESOLVED_SDK_SOURCE="url+token"
    fi
    return $rc
}

# Print missing-SDK guidance — uses the same message as the Python module
_print_guidance() {
    echo ""
    echo "  ┌──────────────────────────────────────────────────────────────────┐"
    echo "  │  No QAIRT SDK installed — QUAD will run in MOCK MODE              │"
    echo "  └──────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  Mock mode is a complete working environment — all 5 MCP tools"
    echo "  return realistic simulated responses. To enable real Qualcomm"
    echo "  hardware mode:"
    echo ""
    echo "  Option 1 — One command (after you've downloaded the archive):"
    echo ""
    echo "    ./install.sh --qairt-archive ~/Downloads/qairt-2.45.0.260326.zip"
    echo ""
    echo "  Option 2 — Already-downloaded archive in standard location:"
    echo ""
    echo "    Just put the .zip in ~/Downloads/ and re-run ./install.sh"
    echo ""
    echo "  Option 3 — CI / org-managed mirror (no interactive download):"
    echo ""
    echo "    export QAIRT_DOWNLOAD_URL=https://your-mirror/qairt.zip"
    echo "    export QAIRT_DOWNLOAD_TOKEN=<bearer-token>"
    echo "    ./install.sh"
    echo ""
    echo "  Where to download QAIRT (developer account + EULA required):"
    echo "    https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk"
    echo "  or the legacy SNPE SDK:"
    echo "    https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai"
    echo ""
}

# Main entry point — call from install.sh or run standalone.
# Args: --qairt-archive <path>  (optional)
setup_sdk() {
    local archive_flag=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --qairt-archive) archive_flag="$2"; shift 2 ;;
            --help|-h)
                grep -E '^# ' "$0" | sed 's/^# //'
                return 0
                ;;
            *) shift ;;  # ignore unrecognised — let install.sh handle them
        esac
    done

    log_section "SDK Setup"

    # Try each strategy in priority order; first success wins
    if _try_archive_flag "$archive_flag" \
        || _try_env_var \
        || _try_discover \
        || _try_downloads_dir \
        || _try_url_with_token; then

        # Re-discover so RESOLVED_SDK_ROOT is populated even when the strategy
        # was an install (which extracts but doesn't set vars by itself).
        local py
        py=$(_pick_python)
        local status_json
        if status_json=$("$py" -c "
from quad.sdk_manager import resolve_sdk_root, apply_to_environment
info = resolve_sdk_root()
if info:
    apply_to_environment(info)
    print(info.root)
    print(info.version)
    print(info.flavor)
" 2>/dev/null); then
            RESOLVED_SDK_ROOT=$(echo "$status_json" | sed -n 1p)
            RESOLVED_SDK_VERSION=$(echo "$status_json" | sed -n 2p)
            RESOLVED_SDK_FLAVOR=$(echo "$status_json" | sed -n 3p)
        fi

        if [ -n "$RESOLVED_SDK_ROOT" ]; then
            export QAIRT_SDK_ROOT="$RESOLVED_SDK_ROOT"
            export QNN_SDK_ROOT="$RESOLVED_SDK_ROOT"
            export SNPE_ROOT="$RESOLVED_SDK_ROOT"
            log_ok "SDK ready: $RESOLVED_SDK_FLAVOR $RESOLVED_SDK_VERSION"
            log_ok "Root: $RESOLVED_SDK_ROOT"
            log_ok "Source: $RESOLVED_SDK_SOURCE"
            return 0
        fi
    fi

    _print_guidance
    return 1  # signal "no SDK" — caller decides whether to abort
}

# Run if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_sdk "$@"
fi
