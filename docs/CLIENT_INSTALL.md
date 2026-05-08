# QUAD Client Install Guide

> Audience: developers who want Claude Code (or another MCP-compatible
> IDE) to talk to a QUAD MCP server. The server may be on this same
> machine, on a remote machine over SSH, or hosted as an HTTP/SSE
> service. **You do not need the QUAD server itself for the
> client install** — only when the server runs on this machine.

## What this installs

| Component | Where | Size |
|---|---|---|
| Python deps (`typer`, `httpx`) | `pip` | ~5 MB |
| `quad-client` CLI entry point | global `pip` scripts | < 1 MB |
| `.claude/settings.json` | project root | ~1 KB |
| `.claude/skills/*.md` (11 files) | project root | ~50 KB |

What it does **NOT** install: `fastmcp`, `numpy`, `jinja2`, `structlog`,
the `qairt-converter` SDK, the templates directory, or any of the
adapter/runtime/compiler code. Total client install footprint
is ~6 MB.

---

## The three deployment topologies

```
┌─────────────────────┐     stdio      ┌─────────────────────┐
│  Same machine       │  (subprocess)  │  Same machine       │
│  ┌──────────────┐   │ ─────────────▶ │  ┌──────────────┐   │
│  │ Claude Code  │   │                │  │ MCP server   │   │
│  └──────────────┘   │                │  │ (this repo)  │   │
└─────────────────────┘                │  └──────────────┘   │
   stdio-local                         └─────────────────────┘


┌─────────────────────┐     ssh        ┌─────────────────────┐
│  IDE laptop (light) │  (proxy stdio) │  Snapdragon X Elite │
│  ┌──────────────┐   │ ═════════════▶ │  ┌──────────────┐   │
│  │ Claude Code  │   │                │  │ MCP server   │   │
│  └──────────────┘   │                │  │ + QAIRT SDK  │   │
└─────────────────────┘                │  └──────────────┘   │
   stdio-ssh                           └─────────────────────┘


┌─────────────────────┐    https/sse   ┌─────────────────────┐
│  IDE laptop (light) │   (auth+TLS)   │  Hosted MCP cloud   │
│  ┌──────────────┐   │ ═════════════▶ │  ┌──────────────┐   │
│  │ Claude Code  │   │                │  │ MCP server   │   │
│  └──────────────┘   │                │  │ (managed)    │   │
└─────────────────────┘                │  └──────────────┘   │
   sse-http                            └─────────────────────┘
```

Pick whichever matches your topology — the install steps below cover
all three.

---

## Quickest path

```bash
git clone https://github.com/pkr465/QUAD.git
cd QUAD
./install-client.sh
```

The script asks which transport (1/2/3), prompts for any required
details (SSH host, server URL, etc.), tests the connection, then
provisions Claude Code. If the test fails it explains why.

---

## Per-transport setup

### 1. `stdio-local` — server on the same machine

```bash
./install-client.sh --transport=stdio-local
```

Equivalent to running the full `./install.sh` (which both server +
client live in this directory). The client install kicks off
`install.sh --mock-only --skip-tests` automatically so the server is
ready.

After this:
```bash
quad-client status   # shows: settings.json present, 11 skills installed
quad mode             # adapter_mode + real-mode readiness
```

### 2. `stdio-ssh` — server on a remote Snapdragon machine

The most common topology for "lightweight client laptop, real
hardware in the lab."

**Prerequisites on the developer laptop:**
- Python 3.10+ (lightweight; no SDK needed)
- OpenSSH client (`winget install Microsoft.OpenSSH.Client` on Windows)
- SSH key authentication to the server (no password prompts)

**Prerequisites on the server machine:**
- The full QUAD package installed (`./install.sh`)
- An SSH server accepting your public key

**Install:**
```bash
./install-client.sh \
    --transport=stdio-ssh \
    --ssh-user=pavanr \
    --ssh-host=snapdragon-test.lan \
    --ssh-port=22 \
    --ssh-key=~/.ssh/id_ed25519
```

The script will:
1. Install the lightweight `[client]` deps locally
2. SSH to `pavanr@snapdragon-test.lan` and run
   `python -c 'import quad.mcp.server'` to verify the server is
   importable there
3. Generate `.claude/settings.json` with the SSH command (Claude Code
   will spawn `ssh pavanr@snapdragon-test.lan python -m quad.mcp.server`
   on every tool call)
4. Install the bundled skills

**Generated settings.json shape:**
```json
{
  "mcpServers": {
    "quad": {
      "command": "ssh",
      "args": [
        "-p", "22",
        "-i", "/home/me/.ssh/id_ed25519",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30",
        "pavanr@snapdragon-test.lan",
        "python -m quad.mcp.server"
      ]
    }
  }
}
```

**Common SSH issues:**

| Symptom | Fix |
|---|---|
| `Permission denied (publickey)` | Add your `~/.ssh/id_*.pub` to `~/.ssh/authorized_keys` on the server |
| `Could not resolve hostname` | Check spelling, DNS, or use the IP directly |
| `ModuleNotFoundError: 'quad'` after SSH | Install the QUAD package on the server: `pip install quad-agent` |
| Hangs on connection | `BatchMode=yes` is set — there's no password fallback. Fix key auth. |

### 3. `sse-http` — hosted MCP server

For organisations running a managed MCP server. Claude Code keeps an
HTTP connection open instead of spawning a subprocess.

```bash
./install-client.sh \
    --transport=sse-http \
    --sse-url=https://mcp.example.com/sse \
    --sse-auth-token-env=QUAD_MCP_TOKEN
```

Set the token in your environment first (or in `.env` if the IDE
reads it):
```bash
export QUAD_MCP_TOKEN=<your-bearer-token>
```

Connection test will issue an HTTP GET to the URL and verify
reachability + auth.

---

## Manual install (without `install-client.sh`)

If you want the lightweight client without the bash script:

```bash
# 1. Install the package with the client extras only (no fastmcp, no numpy)
pip install "quad-agent[client]"

# 2. Test the connection
quad-client connect-test stdio-ssh \
    --ssh-user=pavanr --ssh-host=snapdragon-test.lan

# 3. Provision Claude Code
quad-client install --transport=stdio-ssh \
    --ssh-user=pavanr --ssh-host=snapdragon-test.lan
```

---

## Verifying the install

```bash
quad-client status
```

Expected output:
```
Client:           claude_code
settings.json:    present @ /path/to/project/.claude/settings.json
Skills dir:       present @ /path/to/project/.claude/skills
Bundled skills:   11
Installed skills: 11
```

Then open Claude Code in the same project — the QUAD MCP tools should
appear in the tool list within a few seconds. Try:

> "What hardware do I have?"
>
> "Convert mobilenetv2.onnx to INT8 for QNN."

Claude Code will route those to the matching MCP tool via the
installed skill files.

---

## Updating

When QUAD adds new MCP tools or skills, re-run with `--force`:

```bash
quad-client install --transport=<your-transport> --force
```

This overwrites `.claude/settings.json` and the bundled skill files.
User-added skills (any `.md` file you wrote yourself) are preserved.

---

## Uninstalling

```bash
quad-client uninstall
```

Removes the bundled skill files. `.claude/settings.json` is **not**
removed — it may contain user customisations. Delete it manually if
you want a clean slate.

---

## Comparison: client install vs full install

| | Client (`install-client.sh`) | Full (`install.sh`) |
|---|:-:|:-:|
| Python deps | ~5 MB | ~150 MB |
| Disk footprint | ~6 MB | ~1.2 GB (with QAIRT SDK) |
| Install time | ~30 sec | 5-15 min |
| QAIRT SDK install | no | optional via `--qairt-archive` |
| Can run the MCP server itself | no (stdio-ssh / sse-http only) | yes |
| `quad-server` entry point | no | yes |
| `quad sdk install` available | no | yes |
| `quad doctor` available | no (use `quad-client connect-test` instead) | yes |
| Use case | Developer laptop with lab/cloud server | Full local setup, CI, lab machine |

---

## When to choose which topology

| Situation | Recommended |
|---|---|
| One developer, one machine, full setup | `stdio-local` (run `./install.sh`) |
| Lightweight laptop + heavy lab machine | `stdio-ssh` |
| Many developers + central GPU/NPU server | `sse-http` (managed) |
| CI runner | `--server-only` install + skip client |
| Demos / no real hardware | `stdio-local --mock-only` |

---

## See also

- [`docs/SERVER_INSTALL.md`](SERVER_INSTALL.md) — server-side setup
- [`docs/REAL_HARDWARE.md`](REAL_HARDWARE.md) — installing the QAIRT SDK on the server
- [`docs/CLIENT_SERVER_ARCHITECTURE.md`](CLIENT_SERVER_ARCHITECTURE.md) — internal architecture
