"""Response-enrichment helpers for MCP tool responses.

Each ``enrich_*`` function takes a clean payload from
:mod:`quad.core.operations` and decorates it with three optional keys:

* ``ui``           — markdown summary via :mod:`quad.ui.formatters`
* ``tips``         — 2 contextual tips from :mod:`quad.tips`
* ``suggestions``  — per-bottleneck recommendations (profile only)

Enrichment is best-effort: every call is wrapped in try/except so a
failure in the UI layer can never break the data path. A non-MCP
caller can always import from :mod:`quad.core.operations` directly to
get the same payloads without these keys.
"""

from __future__ import annotations

from typing import Any


def _safe_enrich(payload: dict[str, Any], context: str, formatter: Any | None = None) -> dict[str, Any]:
    """Attach ``ui`` (if formatter given) + ``tips`` to a payload, swallowing errors."""
    try:
        from quad.tips import get_tips_for

        if formatter is not None:
            payload["ui"] = formatter(payload)
        payload["tips"] = [t.text for t in get_tips_for(context, n=2)]
    except Exception:
        pass
    return payload


def enrich_hardware_detect(payload: dict[str, Any]) -> dict[str, Any]:
    """Add ui + tips for hardware_detect output."""
    try:
        from quad.ui import format_device

        return _safe_enrich(payload, "detect", format_device)
    except Exception:
        return payload


def enrich_convert_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Add ui + tips for convert_model output."""
    try:
        from quad.ui import format_conversion

        return _safe_enrich(payload, "convert", format_conversion)
    except Exception:
        return payload


def enrich_profile_workload(
    payload: dict[str, Any],
    profiling_level: str = "detailed",
) -> dict[str, Any]:
    """Add ui + tips + suggestions for profile_workload output."""
    try:
        from quad.suggestions import suggest_optimisations
        from quad.ui import format_profile

        _safe_enrich(payload, "profile", format_profile)

        bottlenecks: list[dict[str, Any]] = []
        if payload.get("linting_layers"):
            bottlenecks = [
                layer for layer in payload["linting_layers"]
                if layer.get("is_bottleneck")
            ]
        suggestions = suggest_optimisations(
            bottlenecks=bottlenecks,
            profiling_level=profiling_level,
        )
        payload["suggestions"] = [s.to_dict() for s in suggestions]
    except Exception:
        pass
    return payload


def enrich_orchestrate_workload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add ui + tips for orchestrate_workload output."""
    try:
        from quad.ui import format_allocation

        return _safe_enrich(payload, "orchestrate", format_allocation)
    except Exception:
        return payload


def enrich_generate_code(payload: dict[str, Any], platform: str, language: str, sdk: str) -> dict[str, Any]:
    """Add ui + tips for generate_code output (custom UI per file list)."""
    try:
        from quad.tips import get_tips_for
        from quad.ui.formatters import format_table

        files = payload.get("source_files", {}) or {}
        rows = [
            [name, len(content.splitlines()), len(content)]
            for name, content in files.items()
        ]
        payload["ui"] = (
            f"### Generated code: {platform} / {language} / {sdk}\n\n"
            + format_table(["File", "Lines", "Bytes"], rows, align=["l", "r", "r"])
            + f"\n\n**Build:** `{payload.get('build_instructions', '?')}`\n"
            + f"**Deps:** {', '.join(payload.get('dependencies', []) or ['—'])}"
        )
        payload["tips"] = [t.text for t in get_tips_for("codegen", n=2)]
    except Exception:
        pass
    return payload
