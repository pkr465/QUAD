"""quad configure — interactive configuration wizard."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import typer

from scripts.helpers import log_info, log_ok, log_warn  # type: ignore[import]

# ── Hexagon version choices ──────────────────────────────────────────────────
HEXAGON_VERSIONS = ["v65", "v66", "v68", "v69", "v73", "v75", "v79", "v81"]

CHIPSET_HINTS = {
    "Snapdragon X Elite": "v75",
    "Snapdragon 8 Elite (SM8750)": "v79",
    "Snapdragon 8 Gen 3 (SM8650)": "v75",
    "Snapdragon 8 Gen 2 (SM8550)": "v73",
    "Snapdragon 8 Gen 1 (SM8450)": "v69",
    "Snapdragon 888 (SM8350)": "v68",
    "QCS2210 (Arduino UNO Q)": "v66",
    "Other / I don't know": "",
}


def run_configure(
    config_path: Path = Path("quad.toml"),
    env_path: Path = Path(".env"),
    non_interactive: bool = False,
) -> None:
    """Interactive configuration wizard — writes quad.toml and .env."""

    print()
    print("═══════════════════════════════════════════════════════")
    print("  QUAD Configuration Wizard")
    print("  Writes: quad.toml (config) + .env (secrets)")
    print("═══════════════════════════════════════════════════════")
    print()

    # ── Load existing values as defaults ──────────────────────────────────────
    existing_toml: dict = {}
    if config_path.exists():
        import tomllib if hasattr(__import__("sys"), "version_info") else None
        try:
            import tomllib  # type: ignore
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(config_path, "rb") as f:
            existing_toml = tomllib.load(f)

    existing_env: dict[str, str] = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                existing_env[k.strip()] = v.strip().strip('"').strip("'")

    def ask(prompt: str, default: str = "", secret: bool = False) -> str:
        if non_interactive:
            return default
        if default:
            display_default = "***" if secret and default else default
            val = typer.prompt(f"  {prompt}", default=display_default)
            return val if val != "***" else default
        return typer.prompt(f"  {prompt}", default=default)

    def ask_bool(prompt: str, default: bool = False) -> bool:
        if non_interactive:
            return default
        return typer.confirm(f"  {prompt}", default=default)

    def ask_choice(prompt: str, choices: list[str], default: str = "") -> str:
        if non_interactive:
            return default
        print(f"\n  {prompt}")
        for i, c in enumerate(choices, 1):
            marker = " (current)" if c == default else ""
            print(f"    {i}. {c}{marker}")
        while True:
            raw = typer.prompt("  Choice", default=str(choices.index(default) + 1) if default in choices else "1")
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            except ValueError:
                if raw in choices:
                    return raw
            print("    Invalid choice, try again.")

    # ── Section 1: Adapter Mode ───────────────────────────────────────────────
    print("[ 1/5 ] Adapter Mode")
    current_mode = existing_toml.get("server", {}).get("adapter_mode", "mock")
    adapter_mode = ask_choice(
        "Run mode?",
        ["mock (development — no hardware needed)", "real (requires Qualcomm SDKs)"],
        default=f"{current_mode} (development — no hardware needed)" if current_mode == "mock" else "real (requires Qualcomm SDKs)",
    )
    adapter_mode = "real" if adapter_mode.startswith("real") else "mock"

    # ── Section 2: SDK paths ──────────────────────────────────────────────────
    print("\n[ 2/5 ] SDK Paths")
    qairt_path = ""
    if adapter_mode == "real":
        default_sdk = (
            existing_toml.get("adapters", {}).get("qairt", {}).get("sdk_path", "")
            or os.environ.get("QAIRT_SDK_ROOT", "")
        )
        qairt_path = ask("QAIRT SDK path (e.g. /path/to/qairt/2.45.0.260326)", default=default_sdk)
    else:
        print("  Skipped (mock mode — no SDK needed)")

    # ── Section 3: Target Device ──────────────────────────────────────────────
    print("\n[ 3/5 ] Target Devices")

    win_enabled = ask_bool("Using Windows on Snapdragon?", default=existing_toml.get("platforms", {}).get("windows", {}).get("enabled", False))

    linux_enabled = ask_bool("Using Linux device (Arduino UNO Q / QCS2210)?", default=existing_toml.get("platforms", {}).get("linux", {}).get("enabled", False))
    ssh_host, ssh_user = "", "root"
    if linux_enabled:
        default_ip = existing_env.get("TARGET_IP", existing_toml.get("platforms", {}).get("linux", {}).get("ssh_host", ""))
        ssh_host = ask("Device IP address", default=default_ip)
        ssh_user = ask("SSH username", default=existing_env.get("TARGET_USER", "root"))

    android_enabled = ask_bool("Using Android device?", default=existing_toml.get("platforms", {}).get("android", {}).get("enabled", False))
    android_serial = ""
    if android_enabled:
        android_serial = ask("ADB device serial (blank = auto-detect)", default=existing_env.get("ANDROID_SERIAL", ""))

    # DSP version
    hexagon_version = ""
    if linux_enabled or android_enabled or win_enabled:
        print("\n  What chipset is your target device?")
        chipset = ask_choice("Chipset", list(CHIPSET_HINTS.keys()))
        hexagon_version = CHIPSET_HINTS.get(chipset, "")
        if not hexagon_version:
            hexagon_version = ask_choice("Hexagon DSP version", HEXAGON_VERSIONS, default="v73")

    # ── Section 4: API Keys & Secrets ─────────────────────────────────────────
    print("\n[ 4/5 ] API Keys & Secrets  (stored in .env, never committed)")
    ai_hub_key = ask(
        "Qualcomm AI Hub API key (blank to skip)",
        default=existing_env.get("QAI_HUB_API_KEY", ""),
        secret=True,
    )
    hexagon_sdk_root = ask(
        "Hexagon SDK root (optional, for UDO/DSP compilation)",
        default=existing_env.get("HEXAGON_SDK_ROOT", ""),
    )
    android_ndk = ask(
        "Android NDK path (optional, for UDO Android compilation)",
        default=existing_env.get("ANDROID_NDK_ROOT", ""),
    )

    # ── Section 5: Preferences ────────────────────────────────────────────────
    print("\n[ 5/5 ] Preferences")
    log_level = ask_choice("Log level", ["info", "debug", "warning", "error"],
                           default=existing_toml.get("server", {}).get("log_level", "info"))
    power_mode = ask_choice("Default power mode", ["balanced", "performance", "efficiency"],
                            default=existing_toml.get("server", {}).get("default_power_mode", "balanced"))

    # ── Write quad.toml ───────────────────────────────────────────────────────
    print()
    print("Writing quad.toml...")

    toml_content = f"""# ═══════════════════════════════════════════════════════════════════════════
# QUAD Configuration — quad.toml
# Auto-generated by: quad configure
# Last updated: {date.today().isoformat()}
# ═══════════════════════════════════════════════════════════════════════════
# IMPORTANT: Do NOT put secrets/API keys here — use .env instead.
# ═══════════════════════════════════════════════════════════════════════════

[server]
adapter_mode = "{adapter_mode}"
log_level = "{log_level}"
log_format = "console"
model_output_dir = "./output"
template_dir = "./templates"
default_power_mode = "{power_mode}"

[adapters.qairt]
sdk_path = "{qairt_path}"
version = "2.45.0"

[adapters.qnn]
sdk_path = "{qairt_path}"

[adapters.snpe]
sdk_path = "{qairt_path}"

[adapters.hexagon]
sdk_path = ""

[adapters.adreno]
sdk_path = ""

[adapters.ai_hub]
api_key_env = "QAI_HUB_API_KEY"
base_url = "https://app.aihub.qualcomm.com/api/v1"

[dsp]
hexagon_version = "{hexagon_version}"
adsp_lib_dir = ""
target_os = "android"

[platforms.windows]
enabled = {"true" if win_enabled else "false"}
device_type = "local"

[platforms.linux]
enabled = {"true" if linux_enabled else "false"}
device_type = "remote"
ssh_host = "{ssh_host}"
ssh_user = "{ssh_user}"
ssh_key = "~/.ssh/id_rsa"
deploy_dest = "/tmp/snpeexample"

[platforms.android]
enabled = {"true" if android_enabled else "false"}
device_serial = "{android_serial}"
adb_path = "adb"
deploy_dest = "/data/local/tmp/snpeexample"
"""

    with open(config_path, "w") as f:
        f.write(toml_content)
    print(f"  ✓ {config_path}")

    # ── Write .env ────────────────────────────────────────────────────────────
    print("Writing .env...")

    env_content = f"""# QUAD Environment Variables
# Auto-generated by: quad configure — {date.today().isoformat()}
# DO NOT COMMIT this file.

# Qualcomm AI Hub API key
QAI_HUB_API_KEY={ai_hub_key}

# QAIRT SDK
QAIRT_SDK_ROOT={qairt_path}
QNN_SDK_ROOT={qairt_path}
SNPE_ROOT={qairt_path}

# Hexagon SDK (for UDO/DSP development)
HEXAGON_SDK_ROOT={hexagon_sdk_root}

# Android NDK (for UDO Android compilation)
ANDROID_NDK_ROOT={android_ndk}

# Target device — Linux (SSH)
TARGET_IP={ssh_host}
TARGET_USER={ssh_user}
TARGET_SSH_KEY=~/.ssh/id_rsa
TARGET_OS=android

# Target device — Android (ADB)
ANDROID_SERIAL={android_serial}

# QUAD runtime overrides (optional)
QUAD_ADAPTER_MODE={adapter_mode}
QUAD_LOG_LEVEL={log_level}
"""

    with open(env_path, "w") as f:
        f.write(env_content)
    print(f"  ✓ {env_path}")

    # ── Update .claude/settings.json ─────────────────────────────────────────
    claude_settings = Path(".claude/settings.json")
    if claude_settings.exists():
        import json
        with open(claude_settings) as f:
            settings = json.load(f)
        if "mcpServers" in settings and "quad" in settings["mcpServers"]:
            settings["mcpServers"]["quad"]["env"]["QUAD_ADAPTER_MODE"] = adapter_mode
            with open(claude_settings, "w") as f:
                json.dump(settings, f, indent=2)
        print(f"  ✓ .claude/settings.json (QUAD_ADAPTER_MODE={adapter_mode})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("═══════════════════════════════════════════════════════")
    print("  Configuration Complete!")
    print()
    print(f"  Mode:     {adapter_mode}")
    if qairt_path:
        print(f"  SDK:      {qairt_path}")
    if hexagon_version:
        print(f"  DSP:      Hexagon {hexagon_version}")
    print(f"  Windows:  {'enabled' if win_enabled else 'disabled'}")
    print(f"  Linux:    {'enabled — ' + ssh_host if linux_enabled else 'disabled'}")
    print(f"  Android:  {'enabled' if android_enabled else 'disabled'}")
    print()
    print("  Next steps:")
    print(f"    source activate.sh && ./launch.sh --{adapter_mode}")
    print("    quad doctor     ← validate configuration")
    print("    quad quickstart ← run interactive demo")
    print("═══════════════════════════════════════════════════════")
