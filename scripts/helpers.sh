#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# QUAD — Shared helper functions for setup scripts
# ═══════════════════════════════════════════════════════════════════════════

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "  ${BLUE}▸${NC} $1"; }
log_ok()      { echo -e "  ${GREEN}✓${NC} $1"; }
log_warn()    { echo -e "  ${YELLOW}⚠${NC} $1"; }
log_error()   { echo -e "  ${RED}✗${NC} $1"; }
log_section() { echo -e "\n${CYAN}━━━ $1 ━━━${NC}\n"; }
