# Copilot Instructions for QUAD

## Project Context
This is a Python MCP server (FastMCP) that abstracts Qualcomm SDKs (QNN, SNPE, Hexagon, Adreno, AIMET) for AI developers. It provides 5 tools via Model Context Protocol: hardware_detect, profile_workload, convert_model, orchestrate_workload, generate_code.

**Architecture**: tools → adapters → SDK/mock backends  
**Stack**: Python 3.11, FastMCP, Pydantic v2, Jinja2, structlog, pytest  
**Design**: Mock-first — everything works without hardware or SDKs installed

## Patterns to Follow
- Adapter pattern: all SDK calls go through `src/quad/adapters/`
- Pydantic models for all data contracts: `src/quad/models/`
- Jinja2 templates for code generation: `templates/`
- `async def` for all tool handlers and adapter methods
- structlog for logging (not print or logging module)
- Factory pattern for adapter selection (mock vs real via config)
- Custom exception hierarchy rooted at `QUADError`

## Key Directories
- `src/quad/server/` — FastMCP entry point and config
- `src/quad/tools/` — MCP tool handlers (one per file)
- `src/quad/adapters/` — SDK abstraction (base ABC + implementations)
- `src/quad/platforms/` — Platform-specific logic (Windows, Linux, Android)
- `src/quad/codegen/` — Jinja2 template engine
- `src/quad/models/` — Pydantic data models
- `templates/` — Code generation templates
- `configs/` — Device profiles, supported ops lists

## Do NOT
- Import QNN/SNPE/Hexagon modules directly in tool code
- Use subprocess with shell=True
- Hardcode SDK paths (use config from quad.toml)
- Skip input validation (Pydantic handles it)
- Add print() statements (use structlog)
- Commit .env files, API keys, or model binaries
