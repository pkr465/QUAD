# QUAD Server Install Guide

> Audience: developers setting up the QUAD MCP server. The server holds
> the QAIRT SDK, the SDK adapter layer, the compiler, the runtime, and
> the codegen templates — it's the heavy half of QUAD. Install this
> wherever you have (or want to have) Qualcomm hardware to talk to.
>
> **For the client side**, see [`docs/CLIENT_INSTALL.md`](CLIENT_INSTALL.md).

## What this installs

| Component | Where | Size |
|---|---|---|
| Python deps (fastmcp, jinja2, structlog, numpy, pydantic, …) | `pip` | ~150 MB |
| QUAD core package (`src/quad/`) | site-packages | ~10 MB |
| Templates (42 Jinja2 files) | bundled in wheel | ~500 KB |
| QAIRT SDK (optional, multi-GB) | `./sdks/qairt-X.Y.Z/` | ~1 GB unpacked |
| `quad-server` + `quad` + `quad-client` entry points | global `pip` scripts | < 1 MB |
| `quad.toml` | project root | ~1 KB |

If you also provision Claude Code locally (default), it adds:
- `.claude/settings.json` (~1 KB)
- `.claude/skills/*.md` (~50 KB, 11 files)

For a server-only deploy that isn't ALSO the developer's IDE
machine, pass `--server-only` to skip the client provisioning.

---

## Three install paths

### Path 1: Full local install (developer laptop)

You have a Snapdragon X Elite Copilot+ PC and want to develop QUAD on
it directly. Server + client + (optionally) QAIRT SDK all on this
box.

```bash
git clone https://github.com/pkr465/QUAD.git
cd QUAD
./install.sh --qairt-archive ~/Downloads/qairt-2.45.0.260326.zip
```

What runs:
1. Creates `.venv/`, installs full dep set
2. Unpacks the QAIRT SDK into `./sdks/qairt-2.45.0.260326/`
3. Sets `QAIRT_SDK_ROOT` etc. in the venv environment
4. Generates `quad.toml` + `.claude/settings.json` + `.claude/skills/`
5. Runs the test suite
6. Reports the install summary

After this:
```bash
source ./activate.sh
quad mode             # → 'real-mode: READY'
quad doctor --real-mode
```

For mock-mode-only (no SDK):
```bash
./install.sh --mock-only
```

### Path 2: Server-only install (lab machine / CI runner)

You have a remote Snapdragon machine that should ONLY host the MCP
server. No Claude Code on this machine.

```bash
git clone https://github.com/pkr465/QUAD.git
cd QUAD
./install.sh --server-only --qairt-archive ~/Downloads/qairt-2.45.0.zip
```

`--server-only` skips:
- `.claude/settings.json` generation
- `.claude/skills/` provisioning

The server is fully ready; client laptops connect to it via
`install-client.sh --transport=stdio-ssh` (see
[`docs/CLIENT_INSTALL.md`](CLIENT_INSTALL.md)).

### Path 3: Managed / hosted MCP service

You're running QUAD as a multi-tenant MCP service behind a
load-balancer / TLS-terminating proxy.

```bash
git clone https://github.com/pkr465/QUAD.git
cd QUAD
./install.sh --server-only --real --qairt-archive ~/Downloads/qairt-2.45.0.zip
# Front it with your own SSE/HTTP wrapper around quad-server
```

QUAD itself doesn't ship a hosted SSE wrapper — that's a deployment
concern (see your reverse proxy / API gateway docs). The MCP server
on the inside is exactly the same as in Path 1; only the network
plumbing differs.

---

## Prerequisites

| OS | Required |
|---|---|
| **Windows (default — Snapdragon X Elite)** | Python 3.10+, PowerShell 7+ recommended (`winget install Microsoft.PowerShell`), Git Bash via `bootstrap.ps1` if not present |
| **macOS** | Python 3.10+ (`brew install python@3.12`) |
| **Linux** | Python 3.10+ (`apt install python3.12` / `dnf install python3.12`) |

For real-hardware mode, also:
- The QAIRT SDK archive (download from the [Qualcomm developer portal](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) — requires a developer account and EULA acceptance)
- Optionally: the SNPE SDK if you need the legacy DLC pipeline

---

## Install flags reference

| Flag | What it does |
|---|---|
| `--qairt-archive PATH` | Unpacks the SDK from `PATH` (`.zip` / `.tar.gz`) into `./sdks/<flavor>-<version>/` |
| `--mock-only` | Skip SDK setup; install QUAD in mock mode only |
| `--server-only` | Skip Claude Code client provisioning; install server bits only |
| `--real` | Also install real-hardware Python extras (`asyncssh`, `paramiko`, `onnx`, `fastapi`, `uvicorn`, `psutil`) |
| `--clean` | Remove existing `.venv/` before creating |
| `--adapters LIST` | Comma-separated adapter list (default: all) |
| `--skip-tests` | Skip the post-install test verification |

---

## SDK acquisition

The installer tries six strategies in order. First success wins.

1. **`--qairt-archive PATH`** (explicit, highest priority)
2. **`QAIRT_SDK_ROOT` already set + valid** (no-op)
3. **`quad sdk discover`** finds an install at `./sdks/`, `~/.quad/sdks/`, vendor defaults
4. **`~/Downloads/qairt*.zip` auto-detected** (set `QAIRT_DOWNLOADS_DIR` to override)
5. **`QAIRT_DOWNLOAD_URL` + `QAIRT_DOWNLOAD_TOKEN`** (CI / org-managed mirrors)
6. **Graceful fallback to mock mode** with clear instructions

This is documented in `scripts/setup_sdk.sh`. The Qualcomm developer
pages gate downloads behind login + EULA — no anonymous direct URL
exists. The installer never bypasses that.

---

## Verifying the server install

```bash
source ./activate.sh
quad mode                    # adapter mode + real-mode readiness
quad sdk status              # which SDK was discovered
quad doctor                  # 16 environment checks
quad doctor --real-mode      # strict pre-flight (exits non-zero on issues)
```

Expected output for a fully-real-mode install:

```
adapter_mode:    real
sdk:             qairt 2.45.0.260326  (project:./sdks)
real-mode:       READY
  reason:        Real mode active. SDK root: ./sdks/qairt-2.45.0.260326
```

For a mock-mode install:
```
adapter_mode:    mock
sdk:             (none — run `quad sdk status` for guidance)
real-mode:       NOT READY
```

---

## Running the server

The MCP server is invoked **by Claude Code** automatically when you've
provisioned the client (see `docs/CLIENT_INSTALL.md`). You don't
typically run it manually.

For debugging or remote-target prep:

```bash
# Stdio mode (what Claude Code uses)
quad-server

# Or via Python module:
python -m quad.mcp.server

# Verify the server starts cleanly without actually spawning Claude Code:
quad-client connect-test stdio-local
```

---

## Updating the server

```bash
git pull
source ./activate.sh
pip install -e .[dev]      # refresh deps
quad doctor --real-mode    # re-verify
```

For SDK updates, point at the new archive:

```bash
quad sdk install ~/Downloads/qairt-2.46.0.zip
```

The installer handles the version directory layout automatically; old
versions stay in `./sdks/` until you `rm -rf sdks/qairt-2.45.0/`.

---

## Comparison: server install vs client install

| | Server (`install.sh`) | Client (`install-client.sh`) |
|---|:-:|:-:|
| Python deps | ~150 MB | ~5 MB |
| Disk footprint | ~1.2 GB (with QAIRT SDK) | ~6 MB |
| Install time | 5-15 min | ~30 sec |
| QAIRT SDK install | optional | no |
| Provides `quad-server` | yes | no |
| Provides `quad-client` | yes | yes |
| Can run the MCP server | yes | no |
| Provisions Claude Code | yes (unless `--server-only`) | yes |

---

## When to choose which topology

| Situation | Recommended |
|---|---|
| One developer, full local setup | `./install.sh --qairt-archive ...` (no `--server-only`) |
| Lab machine with real hardware, lightweight dev laptops | `./install.sh --server-only --qairt-archive ...` on lab; `./install-client.sh --transport=stdio-ssh ...` on laptops |
| CI runner | `./install.sh --server-only --mock-only --skip-tests` |
| Hosted multi-tenant MCP service | `./install.sh --server-only --real --qairt-archive ...` + your own reverse proxy |

---

## See also

- [`docs/CLIENT_INSTALL.md`](CLIENT_INSTALL.md) — Claude Code / IDE side setup
- [`docs/REAL_HARDWARE.md`](REAL_HARDWARE.md) — full SDK enablement playbook
- [`docs/CLIENT_SERVER_ARCHITECTURE.md`](CLIENT_SERVER_ARCHITECTURE.md) — internal layering
- [`docs/SAMPLE_APP_REPORT.md`](SAMPLE_APP_REPORT.md) — real Snapdragon X Elite measurements
