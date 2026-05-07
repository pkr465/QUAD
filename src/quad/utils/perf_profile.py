"""SNPE Low-Level Performance APIs (SNPE 2.22+).

Introduced in SNPE 2.22: SNPEPerfProfile — lifecycle-aware DSP performance control.

Before 2.22: One preset profile applied for ENTIRE lifecycle (init + inference + deinit).
From 2.22:  Different perf settings per lifecycle phase:
  - Before/after initialization
  - Before/after each inference
  - Before/after de-initialization

Key features:
1. Custom profiles: start from scratch OR from a preset (Burst, HP, etc.)
2. Voltage corner control: Bus/Core voltage corners (min/target/max, start/done)
3. DSP hysteresis timer: hold performance vote N ms after inference (reduces RPC calls)
4. Multi-instance sync: control "done" vote so high-perf SNPE doesn't interfere
5. YAML config: --perf_config_yaml for snpe-net-run/snpe-throughput-net-run

Heuristics built into preset profiles:
  RPC polling: BURST, SHP, HP only (FastRPC, not MCDM)
  Hysteresis: 300ms for BURST and SHP (hold vote between back-to-back inferences)
  Async voting thread: all profiles EXCEPT BURST and SHP
  No voting: SYSTEM_SETTINGS (client votes directly via Hexagon SDK)

C API header: SNPEPerfProfile.h
C++ wrapper:  SNPEPerfProfile.hpp
Available:    SNPE 2.22+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# DCVS Voltage Corners
# ══════════════════════════════════════════════════════════════════════════════

class DCVSVoltageCorner(str, Enum):
    """Voltage corner values for DSP DCVS control.

    In decreasing voltage/performance order:
      TURBO > NOM_PLUS > NOM > SVS_PLUS > SVS > SVS2 > MIN > DISABLE
    """
    TURBO_PLUS = "SNPE_DCVS_VOLTAGE_VCORNER_TURBO_PLUS"
    TURBO = "SNPE_DCVS_VOLTAGE_VCORNER_TURBO"
    NOM_PLUS = "SNPE_DCVS_VOLTAGE_VCORNER_NOM_PLUS"
    NOM = "SNPE_DCVS_VOLTAGE_VCORNER_NOM"
    SVS_PLUS = "SNPE_DCVS_VOLTAGE_VCORNER_SVS_PLUS"
    SVS = "SNPE_DCVS_VOLTAGE_VCORNER_SVS"
    SVS2 = "SNPE_DCVS_VOLTAGE_VCORNER_SVS2"
    MIN = "SNPE_DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER"
    DISABLE = "SNPE_DCVS_VOLTAGE_CORNER_DISABLE"  # Disable voting for this corner

    @property
    def performance_level(self) -> int:
        """Relative performance level (higher = better performance, more power)."""
        return {
            "SNPE_DCVS_VOLTAGE_VCORNER_TURBO_PLUS": 8,
            "SNPE_DCVS_VOLTAGE_VCORNER_TURBO": 7,
            "SNPE_DCVS_VOLTAGE_VCORNER_NOM_PLUS": 6,
            "SNPE_DCVS_VOLTAGE_VCORNER_NOM": 5,
            "SNPE_DCVS_VOLTAGE_VCORNER_SVS_PLUS": 4,
            "SNPE_DCVS_VOLTAGE_VCORNER_SVS": 3,
            "SNPE_DCVS_VOLTAGE_VCORNER_SVS2": 2,
            "SNPE_DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER": 1,
            "SNPE_DCVS_VOLTAGE_CORNER_DISABLE": 0,
        }[self.value]


# ══════════════════════════════════════════════════════════════════════════════
# Preset Performance Profiles (extended from 2.22)
# ══════════════════════════════════════════════════════════════════════════════

class SnpePerfPreset(str, Enum):
    """All preset performance profiles available in SNPE 2.22+."""
    BURST = "SNPE_PERFORMANCE_PROFILE_BURST"
    SUSTAINED_HIGH_PERFORMANCE = "SNPE_PERFORMANCE_PROFILE_SUSTAINED_HIGH_PERFORMANCE"
    HIGH_PERFORMANCE = "SNPE_PERFORMANCE_PROFILE_HIGH_PERFORMANCE"
    BALANCED = "SNPE_PERFORMANCE_PROFILE_BALANCED"
    LOW_BALANCED = "SNPE_PERFORMANCE_PROFILE_LOW_BALANCED"
    HIGH_POWER_SAVER = "SNPE_PERFORMANCE_PROFILE_HIGH_POWER_SAVER"
    POWER_SAVER = "SNPE_PERFORMANCE_PROFILE_POWER_SAVER"
    LOW_POWER_SAVER = "SNPE_PERFORMANCE_PROFILE_LOW_POWER_SAVER"
    EXTREME_POWER_SAVER = "SNPE_PERFORMANCE_PROFILE_EXTREME_POWER_SAVER"
    SYSTEM_SETTINGS = "SNPE_PERFORMANCE_PROFILE_SYSTEM_SETTINGS"

    @property
    def has_rpc_polling(self) -> bool:
        """RPC polling enabled (FastRPC only, not MCDM)."""
        return self in (
            SnpePerfPreset.BURST,
            SnpePerfPreset.SUSTAINED_HIGH_PERFORMANCE,
            SnpePerfPreset.HIGH_PERFORMANCE,
        )

    @property
    def has_hysteresis_300ms(self) -> bool:
        """300ms vote hysteresis (hold clocks between inferences)."""
        return self in (
            SnpePerfPreset.BURST,
            SnpePerfPreset.SUSTAINED_HIGH_PERFORMANCE,
        )

    @property
    def has_async_voting(self) -> bool:
        """Async voting thread (improves responsiveness)."""
        return self not in (
            SnpePerfPreset.BURST,
            SnpePerfPreset.SUSTAINED_HIGH_PERFORMANCE,
            SnpePerfPreset.SYSTEM_SETTINGS,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Custom Performance Profile
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SNPEPerfProfile:
    """Custom SNPE performance profile with fine-grained DSP control.

    Can be created from scratch or from a preset as a starting point.
    Applied per lifecycle phase: init, inference, or deinit.

    C API:
        Snpe_SNPEPerfProfile_Handle_t h = Snpe_SNPEPerfProfile_Create();
        Snpe_SNPEPerfProfile_Handle_t h = Snpe_SNPEPerfProfile_CreatePreset(preset);

    C++ Wrapper:
        SNPEPerfProfile p;
        SNPEPerfProfile p(BURST);
    """

    preset: SnpePerfPreset | None = None      # Starting preset (None = from scratch)
    name: str = "custom"

    # ── DSP general settings ──
    enable_async_voting: bool = True
    dsp_hysteresis_time_ms: float = 0.0       # 0 = off; 300ms default for BURST/SHP
    dsp_rpc_polling_time_us: int = 0          # 0 = off; 9999µs for SHP
    dsp_sleep_disable_ms: int = 0

    # ── Bus voltage corners (start = before inference, done = after inference) ──
    bus_vcorner_min_start: DCVSVoltageCorner | None = None
    bus_vcorner_target_start: DCVSVoltageCorner | None = None
    bus_vcorner_max_start: DCVSVoltageCorner | None = None
    bus_vcorner_min_done: DCVSVoltageCorner | None = None
    bus_vcorner_target_done: DCVSVoltageCorner | None = None
    bus_vcorner_max_done: DCVSVoltageCorner | None = None

    # ── Core voltage corners ──
    core_vcorner_min_start: DCVSVoltageCorner | None = None
    core_vcorner_target_start: DCVSVoltageCorner | None = None
    core_vcorner_max_start: DCVSVoltageCorner | None = None
    core_vcorner_min_done: DCVSVoltageCorner | None = None
    core_vcorner_target_done: DCVSVoltageCorner | None = None
    core_vcorner_max_done: DCVSVoltageCorner | None = None

    # ── DSP DCVS (Dynamic Clock and Voltage Scaling) ──
    dcvs_enable_start: bool = False
    dcvs_enable_done: bool = True
    sleep_latency_start_us: int = 100
    sleep_latency_done_us: int = 2000
    high_performance_mode: bool = False

    def disable_done_votes(self) -> "SNPEPerfProfile":
        """Disable all 'done' votes for multi-instance synchronization.

        Use on high-perf SNPE instances to avoid interfering with
        low-power SNPE instances at HTP/NSP vote aggregation.

        This is the pattern shown in the SNPE 2.22 synchronization docs.
        """
        self.bus_vcorner_min_done = DCVSVoltageCorner.DISABLE
        self.bus_vcorner_target_done = DCVSVoltageCorner.DISABLE
        self.bus_vcorner_max_done = DCVSVoltageCorner.DISABLE
        self.core_vcorner_min_done = DCVSVoltageCorner.DISABLE
        self.core_vcorner_target_done = DCVSVoltageCorner.DISABLE
        self.core_vcorner_max_done = DCVSVoltageCorner.DISABLE
        return self

    def set_hysteresis(self, ms: float) -> "SNPEPerfProfile":
        """Set DSP vote hysteresis timer.

        0 = disable (brings clocks down immediately after inference).
        300 = BURST/SHP default (holds clocks 300ms between inferences).
        """
        self.dsp_hysteresis_time_ms = ms
        return self

    @property
    def c_api_create_call(self) -> str:
        """C API initialization code."""
        if self.preset:
            return f"Snpe_SNPEPerfProfile_CreatePreset({self.preset.value});"
        return "Snpe_SNPEPerfProfile_Create();"

    @property
    def cpp_api_create_call(self) -> str:
        """C++ wrapper initialization code."""
        if self.preset:
            preset_short = self.preset.value.replace("SNPE_PERFORMANCE_PROFILE_", "")
            return f"SNPEPerfProfile perfProfile({preset_short});"
        return "SNPEPerfProfile perfProfile;"

    def to_yaml_section(self, phase: str = "execute") -> dict[str, Any]:
        """Convert to YAML dict for --perf_config_yaml format.

        Args:
            phase: "init", "execute", or "deinit"
        """
        d: dict[str, Any] = {
            "DSP_ENABLE_DCVS_START": self.dcvs_enable_start,
            "DSP_ENABLE_DCVS_DONE": self.dcvs_enable_done,
            "DSP_SLEEP_LATENCY_START_US": self.sleep_latency_start_us,
            "DSP_SLEEP_LATENCY_DONE_US": self.sleep_latency_done_us,
            "HIGH_PERFORMANCE_MODE": self.high_performance_mode,
        }
        if self.bus_vcorner_min_start:
            d["BUS_VOLTAGE_CORNER_MIN_START"] = self.bus_vcorner_min_start.value
        if self.bus_vcorner_target_start:
            d["BUS_VOLTAGE_CORNER_TARGET_START"] = self.bus_vcorner_target_start.value
        if self.bus_vcorner_max_start:
            d["BUS_VOLTAGE_CORNER_MAX_START"] = self.bus_vcorner_max_start.value
        if self.bus_vcorner_min_done:
            d["BUS_VOLTAGE_CORNER_MIN_DONE"] = self.bus_vcorner_min_done.value
        if self.bus_vcorner_target_done:
            d["BUS_VOLTAGE_CORNER_TARGET_DONE"] = self.bus_vcorner_target_done.value
        if self.bus_vcorner_max_done:
            d["BUS_VOLTAGE_CORNER_MAX_DONE"] = self.bus_vcorner_max_done.value
        if self.core_vcorner_min_start:
            d["CORE_VOLTAGE_CORNER_MIN_START"] = self.core_vcorner_min_start.value
        if self.core_vcorner_target_start:
            d["CORE_VOLTAGE_CORNER_TARGET_START"] = self.core_vcorner_target_start.value
        if self.core_vcorner_max_start:
            d["CORE_VOLTAGE_CORNER_MAX_START"] = self.core_vcorner_max_start.value
        if self.core_vcorner_min_done:
            d["CORE_VOLTAGE_CORNER_MIN_DONE"] = self.core_vcorner_min_done.value
        if self.core_vcorner_target_done:
            d["CORE_VOLTAGE_CORNER_TARGET_DONE"] = self.core_vcorner_target_done.value
        if self.core_vcorner_max_done:
            d["CORE_VOLTAGE_CORNER_MAX_DONE"] = self.core_vcorner_max_done.value
        return d


@dataclass
class SNPEPerfConfig:
    """Full performance configuration with per-phase profiles.

    Maps to the --perf_config_yaml YAML structure with sections:
      general, init, execute, deinit.
    """
    general_async_voting_enable: bool = True
    general_hysteresis_us: int = 300_000     # 300ms default for BURST/SHP
    general_sleep_disable_ms: int = 0
    general_rpc_polling_us: int = 0

    init: SNPEPerfProfile = field(default_factory=SNPEPerfProfile)
    execute: SNPEPerfProfile = field(default_factory=SNPEPerfProfile)
    deinit: SNPEPerfProfile = field(default_factory=SNPEPerfProfile)

    def to_yaml_dict(self) -> dict[str, Any]:
        """Generate the full YAML config dict for --perf_config_yaml."""
        return {
            "general": {
                "ASYNC_VOTING_ENABLE": self.general_async_voting_enable,
                "DSP_HYSTERESIS_TIME_US": self.general_hysteresis_us,
                "DSP_SLEEP_DISABLE_MS": self.general_sleep_disable_ms,
                "DSP_RPC_POLLING_TIME_US": self.general_rpc_polling_us,
            },
            "init": self.init.to_yaml_section("init"),
            "execute": self.execute.to_yaml_section("execute"),
            "deinit": self.deinit.to_yaml_section("deinit"),
        }

    def to_yaml_string(self) -> str:
        """Generate YAML string for --perf_config_yaml flag."""
        try:
            import yaml
            return yaml.dump(self.to_yaml_dict(), default_flow_style=False)
        except ImportError:
            # Manual YAML serialization for environments without PyYAML
            lines = []
            d = self.to_yaml_dict()
            for section, items in d.items():
                lines.append(f"{section}:")
                for k, v in items.items():
                    v_str = str(v).lower() if isinstance(v, bool) else str(v)
                    lines.append(f"  {k}: {v_str}")
            return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Factory: preset configs
# ══════════════════════════════════════════════════════════════════════════════

def create_sustained_high_performance_config() -> SNPEPerfConfig:
    """Create the SUSTAINED_HIGH_PERFORMANCE preset config (from docs YAML sample)."""
    turbo = DCVSVoltageCorner.TURBO
    svs = DCVSVoltageCorner.SVS
    svs2 = DCVSVoltageCorner.SVS2

    def shp_phase(is_deinit: bool = False) -> SNPEPerfProfile:
        p = SNPEPerfProfile(
            preset=SnpePerfPreset.SUSTAINED_HIGH_PERFORMANCE,
            dcvs_enable_start=False,
            dcvs_enable_done=True,
            sleep_latency_start_us=100,
            sleep_latency_done_us=2000,
            high_performance_mode=True,
            bus_vcorner_min_start=turbo,
            bus_vcorner_target_start=turbo,
            bus_vcorner_max_start=turbo,
            core_vcorner_min_start=turbo,
            core_vcorner_target_start=turbo,
            core_vcorner_max_start=turbo,
        )
        if is_deinit:
            p.bus_vcorner_min_done = DCVSVoltageCorner.MIN
            p.bus_vcorner_target_done = DCVSVoltageCorner.MIN
            p.bus_vcorner_max_done = DCVSVoltageCorner.MIN
            p.core_vcorner_min_done = DCVSVoltageCorner.MIN
            p.core_vcorner_target_done = DCVSVoltageCorner.MIN
            p.core_vcorner_max_done = DCVSVoltageCorner.MIN
        else:
            p.bus_vcorner_min_done = svs2
            p.bus_vcorner_target_done = svs
            p.bus_vcorner_max_done = svs
            p.core_vcorner_min_done = svs2
            p.core_vcorner_target_done = svs
            p.core_vcorner_max_done = svs
        return p

    return SNPEPerfConfig(
        general_async_voting_enable=False,
        general_hysteresis_us=300_000,
        general_sleep_disable_ms=0,
        general_rpc_polling_us=9999,
        init=shp_phase(False),
        execute=shp_phase(False),
        deinit=shp_phase(True),
    )


def create_multi_instance_sync_profile(
    high_perf_preset: SnpePerfPreset = SnpePerfPreset.BURST,
) -> SNPEPerfProfile:
    """Create a profile for the high-perf SNPE instance in multi-SNPE scenarios.

    Disables all 'done' votes so the high-perf instance does NOT interfere
    with low-perf SNPE instances at HTP/NSP vote aggregation level.

    Based on the synchronization example in SNPE 2.22 docs.
    """
    p = SNPEPerfProfile(preset=high_perf_preset, name="multi_instance_sync")
    p.disable_done_votes()
    return p
