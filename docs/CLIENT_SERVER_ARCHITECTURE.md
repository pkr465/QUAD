# Client / Server Architecture — Analysis & Refactor

> Companion to [`docs/GAP_ANALYSIS.md`](GAP_ANALYSIS.md). Documents the
> separation of concerns between QUAD's MCP server, the Claude Code
> client adapter, and the underlying business logic. Records the
> 2026-05-08 refactor that introduced explicit `core` / `mcp` /
> `client` layering, plus the residual gaps that remain.

---

## Pre-refactor mixing

Before this refactor, three concerns were tangled across modules:

| Concern | Lived in | Problem |
|---|---|---|
| **Pure business logic** (e.g. "detect hardware") | `src/quad/tools/*.py` | Mixed with MCP-server enrichment (`payload["ui"]`, `payload["tips"]`) — non-MCP callers get unwanted fields |
| **MCP server registration** (FastMCP setup) | `src/quad/server/__init__.py` | Only ~118 lines; thin but still mixed with config loading, SDK discovery, logging setup |
| **Claude Code client config** (settings.json + skills) | `install.sh` heredoc + 11 hand-written `.md` files at `.claude/skills/` | Not packaged with the wheel; install.sh has to know the JSON shape; no Python API; impossible to support a second MCP client (Cursor, Continue, etc.) without a parallel set of files |

The user's stated goal: **"lightweight client side and MCP server
setup that doesn't need to be exposed to the developer."**

That means:
1. The developer should not see / configure the MCP plumbing
2. Adding new MCP clients (beyond Claude Code) should be a focused
   change, not a copy of every install script and skill file

---

## Post-refactor layering

```
src/quad/
├── core/                     # ⭐ Pure business logic — no MCP, no client
│   ├── operations/           # Reusable tool operations (no enrichment)
│   │   ├── hardware_detect.py
│   │   ├── convert_model.py
│   │   ├── profile_workload.py
│   │   ├── orchestrate_workload.py
│   │   └── generate_code.py
│   └── __init__.py
├── mcp/                      # ⭐ MCP protocol layer
│   ├── server.py             # FastMCP registration (relocated from server/)
│   ├── enrichment.py         # ui + tips + suggestions wrapping
│   ├── tools.py              # Thin MCP wrappers (call core, then enrich)
│   └── __init__.py
├── client/                   # ⭐ Client-side IDE/agent integrations
│   ├── __init__.py           # MCPClientProvisioner abstract base
│   ├── claude_code/
│   │   ├── settings.py       # Generate .claude/settings.json
│   │   ├── skills.py         # Manage .claude/skills/ files
│   │   ├── skills_content/   # Bundled skill markdown files
│   │   └── __init__.py
│   └── (cursor/, continue_dev/ — future)
├── adapters/                 # SDK adapters (factory, qairt, mock, aimet, aihub)
├── compiler/                 # IR + frontend + op-coverage
├── runtime/                  # Device, host_probe, tensor, model
├── ui/                       # Markdown formatters (used by mcp/enrichment)
├── suggestions.py            # Recommendation engine
├── tips.py                   # Contextual tips catalogue
├── tools/                    # ⚠ Backward-compat shims → forward to mcp/tools
└── server/                   # ⚠ Backward-compat shim → forwards to mcp/server
```

### Layer responsibilities

| Layer | What it does | What it does NOT do | Importable from |
|---|---|---|---|
| **`core/`** | Pure operations: take adapter + args, return data | Format markdown, attach tips, register MCP tools, generate client config | Anywhere — CLI, MCP server, tests, future SDK |
| **`mcp/`** | Register tools with FastMCP; enrich responses with UI/tips/suggestions | Do business logic itself | The `quad-server` entry point; tests that exercise the MCP surface |
| **`client/`** | Generate / install / manage IDE-side config (settings.json, skills) | Run the MCP server; do business logic | The `quad client install` CLI command |

### Data flow

```
Claude Code
   │
   │ JSON-RPC over stdio
   ▼
mcp.server (FastMCP-registered tools)
   │ Calls tool wrappers
   ▼
mcp.tools.tool_X (thin wrappers)
   │ Call into core
   ▼
core.operations.tool_X (pure logic)
   │ Use adapters
   ▼
adapters.factory → adapter (mock | qairt | aimet | aihub | …)
   │ Use runtime / compiler / codegen
   ▼
Hardware
```

The enrichment (`payload["ui"]`, `payload["tips"]`,
`payload["suggestions"]`) happens at the `mcp/` layer, so:

- A CLI caller using `core.operations.hardware_detect()` directly gets
  a clean `DeviceProfile` dict — no UI keys
- A future Python SDK can use `core.operations.X` without any
  Claude-Code-specific assumptions
- The MCP server's response is still rich (the enrichment is at the
  `mcp/` boundary)

### Client provisioning flow

```
quad client install [--client claude_code]
   │
   ▼
client.claude_code.provisioner.ClaudeCodeProvisioner.install()
   ├── Generate .claude/settings.json from settings.py template
   ├── Copy bundled skill files from client/claude_code/skills_content/
   │   to user's .claude/skills/
   └── Verify: read back settings.json, count skills
```

Everything else flows through the same MCP server — Claude Code's
client config just *points at* the existing `quad-server` entrypoint.
A second client (e.g. Cursor) would only need a new
`client/cursor/provisioner.py` to write Cursor's equivalent of
settings.json; the MCP server stays unchanged.

---

## Backward compatibility

The refactor preserved **every** existing import path:

| Old path | Status | Forwards to |
|---|---|---|
| `from quad.tools.hardware_detect import hardware_detect_impl` | ✅ works | `quad.mcp.tools.hardware_detect_impl` |
| `from quad.server import cli, mcp` | ✅ works | `quad.mcp.server.cli`, `quad.mcp.server.mcp` |
| `python -m quad.server.main` | ✅ works | Forwards to `quad.mcp.server.cli()` |
| `quad-server` CLI script | ✅ works | Same entrypoint, now via `quad.mcp.server:cli` |

Existing 2002 tests run unmodified — verified post-refactor.

---

## Remaining gaps (after this refactor)

### G1. `quad serve` vs `quad-server` ambiguity

Two commands, both called "server":

- `quad-server` (entry point) → starts the MCP stdio server (Claude Code talks to this)
- `quad serve <model>` (CLI subcommand) → starts the FastAPI HTTP inference server

These do completely different things but the names are
indistinguishable. **Proposed rename:** `quad serve` → `quad serve-http`,
add `quad serve-mcp` alias for `quad-server`.

**Effort:** 1 hour. **Severity:** medium (UX confusion).

### G2. No abstract MCPClient base class for non-Claude-Code clients

The `client/claude_code/` package is the only client integration today.
Adding Cursor / Continue / Cline support would require duplicating the
provisioner pattern. The `client/__init__.py` defines a
`MCPClientProvisioner` ABC, but it's a single-implementation interface
right now — not battle-tested against a second client's quirks.

**Effort:** 1-2 days when a second client lands. **Severity:** low (no second client requested yet).

### G3. Skills not auto-generated from MCP tool docstrings

Each `.md` skill file in `client/claude_code/skills_content/` is
hand-written. When the MCP tool API changes (e.g. a new parameter
gets added), the skill stays out of date until someone updates it
manually. A code-generation step that produces skill stubs from the
`@mcp.tool` decorated functions would close this.

**Effort:** 2-3 days. **Severity:** medium (drift risk).

### G4. install.sh still has Claude-Code-specific knowledge

The `install.sh` step that creates `.claude/settings.json` was
preserved as a fallback, but the canonical path is now
`quad client install --client claude_code`. The install.sh logic
should defer entirely to that command rather than carrying its own
JSON heredoc. Currently both work, with the install.sh path acting
as a "first-bootstrap" for users who don't have the Python package
installed yet.

**Effort:** 2 hours. **Severity:** low (just dedup).

### G5. No client-config update path

After install, if QUAD adds a new MCP tool or changes one, the
user's `.claude/settings.json` permissions list goes stale (still
allows the old set, missing the new one). `quad client update`
should re-render settings.json + refresh bundled skills. Today
users have to delete and reinstall.

**Effort:** 0.5 day. **Severity:** medium (every release becomes
papercut for existing users).

### G6. MCP server still loads heavyweight modules eagerly

`src/quad/mcp/server.py` imports the adapter factory + SDK manager
at module-load time. For a long-running MCP server this is fine,
but it makes the *startup latency* perceptible to Claude Code (it
waits for the server to print its tool list before letting the user
talk to it). Lazy-loading the heavy modules would speed cold-start
by ~500ms-1s.

**Effort:** 1 day. **Severity:** low (only matters for cold-start
perception).

### G7. No client-side telemetry

The MCP server doesn't know which client is connected (Claude Code
vs Cursor vs custom). A client-name handshake during init would let
the server tailor responses (e.g. richer markdown for Claude Code,
plainer text for non-rendering clients).

**Effort:** 1 day per client. **Severity:** low (not blocking
anything).

### G8. Inference HTTP server still under `serve/` (not `server/`)

The FastAPI inference server (`src/quad/serve/`) was named before the
client/server layering existed. It's neither a "core" nor an "MCP"
nor a "client" concern — it's a separate runtime feature.
Architectural placement question: should it be `src/quad/runtime/http/`
to make clear that "serving inference HTTP" is a runtime concern,
not a protocol concern? Today's name implies it's the inverse of
"client" which it isn't.

**Effort:** 1 hour rename + tests. **Severity:** very low (cosmetic).

---

## What developers see now

A QUAD developer interacting with the system through Claude Code goes
through this set of touchpoints — and only this set:

1. **`./install.sh --qairt-archive ~/Downloads/qairt.zip`** — one
   command on a fresh machine. Installs the package, the SDK, and the
   Claude Code client config.

2. **Open Claude Code** — auto-discovers the MCP server via the
   generated `.claude/settings.json`. Skills auto-loaded from
   `.claude/skills/`.

3. **Ask Claude Code** — "what hardware do I have?", "convert this
   model to INT8", etc. — Claude Code routes to the right MCP tool
   via the matching skill.

The developer **never sees**:

- The MCP protocol (JSON-RPC over stdio)
- FastMCP's API
- The `mcp__quad__hardware_detect` tool name (only the skill names)
- The `quad-server` entrypoint (it's invoked by Claude Code, not the user)
- The settings.json contents (auto-generated)
- The skill markdown files (auto-installed)

If they want to see / edit any of these:

- `quad client status` — what's installed
- `quad client install --force` — re-render everything
- `quad mode` / `quad doctor` — health checks (no MCP knowledge needed)

This matches the user's goal of "lightweight client side and MCP
server setup that doesn't need to be exposed to the developer."

---

## Test impact

- **Test suite count:** unchanged (2002 passing) — backward-compat
  shims meant existing tests didn't have to be touched
- **New tests added:** ~25 in `tests/unit/test_client/` covering the
  Claude Code provisioner, settings rendering, skills installation
- **Refactor risk:** low (zero existing tests modified; new code
  fully unit-tested)
