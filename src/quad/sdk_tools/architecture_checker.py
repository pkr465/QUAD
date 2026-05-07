"""SNPE Architecture Checker (Experimental) — HTP model optimization analysis.

Based on SNPE Architecture Checker documentation (80-63442-10 Rev AH, Apr 13 2026).

Architecture Checker is a tool for models running on the HTP backend
(quantized 8-bit, 16-bit, and FP16 models). It:
  1. Outputs a CSV listing issues that reduce HTP performance
  2. Optionally applies recommended modifications to the model DLC

Platform usage:
  Linux/WSL:  snpe-architecture-checker -i model.dlc [-o output_path] [-m MODIFY]
  Windows:    python snpe-architecture-checker -i model.dlc [-o output_path] [-m MODIFY]

Prerequisite: SNPE_ROOT environment variable must be set.

Output CSV: <output_path>_architecture_checker.csv
Columns: Graph/Layer name, Issue, Recommendation, Type,
         Input_tensor_name:[dims], Output_tensor_name:[dims],
         Parameters, Previous layer, Next layers,
         Modification, Modification_info

Modifier modes:
  --modify / --modify show  Show possible modifications (dry run, no changes)
  --modify all              Apply all applicable modifications
  --modify apply=r1,r2      Apply specific rules (comma-separated, no spaces)

Note: Architecture Checker with modifier is experimental. For actual
performance improvements, the model may require retraining/redesigning.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

TOOL_NAME = "snpe-architecture-checker"
OUTPUT_SUFFIX = "_architecture_checker.csv"

# Known rule names from documentation
KNOWN_RULES: dict[str, str] = {
    "elwisediv": (
        "Replace ElementWiseDivide with ElementWiseMultiply using the reciprocal value. "
        "ElementWiseDivide usually has poor performance compared to ElementWiseMultiply."
    ),
    "prelu": (
        "Replace PReLU with ReLU. "
        "PReLU has lower HTP parallelism than ReLU (see also Linting profile recommendations)."
    ),
}

# Known issue patterns and their recommendations
KNOWN_ISSUES: dict[str, dict[str, str]] = {
    "16bit_activation": {
        "pattern": "16-bit activation",
        "issue": (
            "This model uses 16-bit activation data. 16-bit activation data takes "
            "twice the amount of memory than 8-bit activation data does."
        ),
        "recommendation": "Try to use a smaller datatype to get better performance. E.g., 8-bit",
    },
    "low_channel_conv": {
        "pattern": "channels.*smaller than 32",
        "issue": (
            "The number of channels in the input/output tensor of this convolution "
            "is low (smaller than 32)."
        ),
        "recommendation": (
            "Try increasing the number of channels in the input/output tensor "
            "to 32 or greater to get better performance."
        ),
    },
    "elementwise_divide": {
        "pattern": "ElementWiseDivide.*poor performance",
        "issue": "ElementWiseDivide usually has poor performance compared to ElementWiseMultiply",
        "recommendation": (
            "Try replacing ElementWiseDivide with ElementWiseMultiply using the "
            "reciprocal value to get better performance"
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class ModifyMode(str, Enum):
    """Architecture Checker --modify flag modes."""
    SHOW = "show"     # Display possible modifications (dry run)
    ALL = "all"       # Apply all applicable modifications
    APPLY = "apply"   # Apply specific rules (comma-separated)


class ModificationStatus(str, Enum):
    """Status of a modification in the output CSV."""
    DONE = "Done"
    NOT_APPLICABLE = "N/A"


# ══════════════════════════════════════════════════════════════════════════════
# Issue Data Models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ArchCheckerIssue:
    """One row from the Architecture Checker CSV output.

    Row may be a graph-level issue (layer_name='Graph') or a layer-level issue.
    Type is 'N/A' for graph-level issues, or the op type (e.g. 'Conv2d').
    Modification is 'Done' if modifier was applied, 'N/A' otherwise.
    """
    row_number: int
    layer_name: str                          # 'Graph' for graph-level issues
    issue: str
    recommendation: str
    op_type: str = "N/A"                     # Conv2d, Eltwise_Binary, etc.
    input_tensors: list[str] = field(default_factory=list)   # "name:[dims]" strings
    output_tensors: list[str] = field(default_factory=list)
    parameters: str = ""                     # Dict-like string from CSV
    previous_layers: list[str] = field(default_factory=list)
    next_layers: list[str] = field(default_factory=list)
    modification: ModificationStatus = ModificationStatus.NOT_APPLICABLE
    modification_info: str = "N/A"

    @property
    def is_graph_level(self) -> bool:
        """True if this issue applies to the whole graph, not a specific layer."""
        return self.layer_name.strip().lower() == "graph"

    @property
    def is_modified(self) -> bool:
        """True if the Architecture Checker successfully applied a modification."""
        return self.modification == ModificationStatus.DONE


@dataclass
class ArchCheckerReport:
    """Full Architecture Checker analysis report for one DLC."""
    input_dlc: str
    output_csv: str
    issues: list[ArchCheckerIssue] = field(default_factory=list)
    # Summary counts
    total_issues: int = 0
    modifications_applied: int = 0
    modifications_not_applicable: int = 0

    @property
    def graph_issues(self) -> list[ArchCheckerIssue]:
        """Issues at the graph level (e.g. 16-bit activation)."""
        return [i for i in self.issues if i.is_graph_level]

    @property
    def layer_issues(self) -> list[ArchCheckerIssue]:
        """Issues at the layer level."""
        return [i for i in self.issues if not i.is_graph_level]

    @property
    def modified_layers(self) -> list[ArchCheckerIssue]:
        """Issues where modifier was successfully applied."""
        return [i for i in self.issues if i.is_modified]

    @property
    def unresolved_issues(self) -> list[ArchCheckerIssue]:
        """Issues where no modification was applied or available."""
        return [i for i in self.issues if not i.is_modified]

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        parts = [
            f"DLC: {Path(self.input_dlc).name}",
            f"Issues: {self.total_issues}",
            f"Modified: {self.modifications_applied}",
            f"Unresolved: {self.modifications_not_applicable}",
        ]
        return " | ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Arg Builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ArchCheckerArgs:
    """CLI arg builder for snpe-architecture-checker.

    Required: input_dlc
    Optional: output_path, modify_mode, modify_rules

    Output file: <output_path>_architecture_checker.csv
    """
    input_dlc: str
    output_path: Optional[str] = None       # Without suffix; tool appends _architecture_checker.csv
    modify_mode: Optional[ModifyMode] = None
    modify_rules: list[str] = field(default_factory=list)  # For ModifyMode.APPLY
    use_python: bool = False                 # True for Windows: "python snpe-architecture-checker"

    @property
    def output_csv_path(self) -> str:
        """The actual output CSV path (with suffix appended by the tool)."""
        if self.output_path:
            return self.output_path + OUTPUT_SUFFIX
        dlc_base = str(Path(self.input_dlc).with_suffix(""))
        return dlc_base + OUTPUT_SUFFIX

    def build(self) -> list[str]:
        """Build the complete CLI argument list."""
        if self.use_python:
            args = ["python", TOOL_NAME]
        else:
            args = [TOOL_NAME]

        args += ["--input_dlc", self.input_dlc]

        if self.output_path:
            args += ["--output_path", self.output_path]

        if self.modify_mode is not None:
            if self.modify_mode == ModifyMode.SHOW:
                args += ["--modify", "show"]
            elif self.modify_mode == ModifyMode.ALL:
                args += ["--modify", "all"]
            elif self.modify_mode == ModifyMode.APPLY:
                if not self.modify_rules:
                    raise ValueError(
                        "modify_mode=APPLY requires at least one rule in modify_rules"
                    )
                # Rules are comma-separated without spaces
                rules_str = ",".join(self.modify_rules)
                args += ["--modify", f"apply={rules_str}"]
            else:
                args.append("--modify")

        return args

    def build_show(self) -> list[str]:
        """Build args for show-modifications mode (dry run)."""
        copy = ArchCheckerArgs(
            input_dlc=self.input_dlc,
            output_path=self.output_path,
            modify_mode=ModifyMode.SHOW,
            use_python=self.use_python,
        )
        return copy.build()

    def build_apply_all(self) -> list[str]:
        """Build args for apply-all-modifications mode."""
        copy = ArchCheckerArgs(
            input_dlc=self.input_dlc,
            output_path=self.output_path,
            modify_mode=ModifyMode.ALL,
            use_python=self.use_python,
        )
        return copy.build()

    def build_apply_rules(self, rules: list[str]) -> list[str]:
        """Build args for applying specific rule names."""
        copy = ArchCheckerArgs(
            input_dlc=self.input_dlc,
            output_path=self.output_path,
            modify_mode=ModifyMode.APPLY,
            modify_rules=rules,
            use_python=self.use_python,
        )
        return copy.build()


# ══════════════════════════════════════════════════════════════════════════════
# CSV Parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_arch_checker_csv(csv_path: str) -> ArchCheckerReport:
    """Parse an Architecture Checker output CSV into structured data.

    CSV columns (in order):
      Row#, Graph/Layer name, Issue, Recommendation, Type,
      Input_tensor_name:[dims], Output_tensor_name:[dims], Parameters,
      Previous layer, Next layers, Modification, Modification_info

    Args:
        csv_path: Path to the *_architecture_checker.csv file

    Returns:
        ArchCheckerReport with parsed issues and summary counts
    """
    path = Path(csv_path)
    issues: list[ArchCheckerIssue] = []

    def _split_tensor_list(value: str) -> list[str]:
        """Split comma-separated tensor/layer names, handling bracketed dims."""
        if not value or value.strip() in ("N/A", ""):
            return []
        # Split on commas not inside brackets
        parts = re.split(r",\s*(?![^\[]*\])", value)
        return [p.strip().strip("'\"[]") for p in parts if p.strip()]

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row

        for row in reader:
            if not row or not any(row):
                continue

            # Pad to 12 columns if shorter
            row = row + [""] * (12 - len(row))

            row_num_str, layer_name, issue, recommendation, op_type = row[:5]
            input_tensors_str, output_tensors_str, parameters = row[5:8]
            prev_layers_str, next_layers_str, modification_str, mod_info = row[8:12]

            try:
                row_num = int(row_num_str.strip())
            except (ValueError, AttributeError):
                row_num = len(issues) + 1

            mod_status = (
                ModificationStatus.DONE
                if modification_str.strip() == "Done"
                else ModificationStatus.NOT_APPLICABLE
            )

            issues.append(ArchCheckerIssue(
                row_number=row_num,
                layer_name=layer_name.strip(),
                issue=issue.strip(),
                recommendation=recommendation.strip(),
                op_type=op_type.strip() or "N/A",
                input_tensors=_split_tensor_list(input_tensors_str),
                output_tensors=_split_tensor_list(output_tensors_str),
                parameters=parameters.strip(),
                previous_layers=_split_tensor_list(prev_layers_str),
                next_layers=_split_tensor_list(next_layers_str),
                modification=mod_status,
                modification_info=mod_info.strip() or "N/A",
            ))

    n_modified = sum(1 for i in issues if i.is_modified)
    return ArchCheckerReport(
        input_dlc="(from csv)",
        output_csv=csv_path,
        issues=issues,
        total_issues=len(issues),
        modifications_applied=n_modified,
        modifications_not_applicable=len(issues) - n_modified,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def find_arch_checker(sdk_root: Optional[str] = None) -> Optional[str]:
    """Locate snpe-architecture-checker binary.

    Searches PATH first, then SNPE_ROOT/bin directories.
    Returns None if not found.
    """
    # Check PATH
    tool = shutil.which(TOOL_NAME)
    if tool:
        return tool

    # Check SDK root
    root = sdk_root or os.environ.get("SNPE_ROOT") or os.environ.get("QAIRT_SDK_ROOT")
    if root:
        candidates = [
            Path(root) / "bin" / "x86_64-linux-clang" / TOOL_NAME,
            Path(root) / "bin" / "x86_64-windows-msvc" / TOOL_NAME,
            Path(root) / "benchmarks" / TOOL_NAME,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    return None


def run_arch_checker(
    args: ArchCheckerArgs,
    *,
    sdk_root: Optional[str] = None,
    timeout: float = 300.0,
) -> ArchCheckerReport:
    """Run snpe-architecture-checker and parse the output CSV.

    Prerequisites:
      - SNPE_ROOT environment variable must be set
      - snpe-architecture-checker must be in PATH or discoverable in SDK root

    Args:
        args: ArchCheckerArgs instance describing what to run
        sdk_root: Override SNPE_ROOT for locating the tool
        timeout: Max seconds to wait

    Returns:
        ArchCheckerReport with parsed issues from the output CSV

    Raises:
        FileNotFoundError: If snpe-architecture-checker is not found
        RuntimeError: If the tool exits non-zero
        TimeoutError: If execution exceeds timeout
    """
    cmd = args.build()

    # On Windows, the tool is run as "python snpe-architecture-checker"
    # If use_python=False but not found as executable, try python wrapper
    if not args.use_python:
        tool = find_arch_checker(sdk_root)
        if tool:
            cmd[0] = tool
        else:
            raise FileNotFoundError(
                f"'{TOOL_NAME}' not found in PATH or SDK root. "
                "Ensure SNPE_ROOT is set: export SNPE_ROOT=/path/to/qairt"
            )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"snpe-architecture-checker timed out after {timeout}s")

    if result.returncode != 0:
        raise RuntimeError(
            f"snpe-architecture-checker failed (exit {result.returncode}):\n"
            f"{result.stderr[:1000]}"
        )

    csv_path = args.output_csv_path
    if not Path(csv_path).exists():
        # Try to find output CSV with tool-generated name
        parent = Path(args.output_path or args.input_dlc).parent
        candidates = sorted(parent.glob("*_architecture_checker.csv"))
        if candidates:
            csv_path = str(candidates[-1])
        else:
            raise FileNotFoundError(
                f"Architecture checker output CSV not found: {csv_path}"
            )

    report = parse_arch_checker_csv(csv_path)
    report.input_dlc = args.input_dlc
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

ARCH_CHECKER_NOTES: dict[str, Any] = {
    "description": (
        "Experimental tool for HTP backend models (quantized 8-bit, 16-bit, FP16). "
        "Lists issues reducing HTP performance and optionally applies modifications."
    ),
    "supported_models": "Quantized 8-bit, quantized 16-bit, FP16 models on HTP",
    "prerequisite": "SNPE_ROOT environment variable must be set",
    "output_csv": (
        "<output_path>_architecture_checker.csv\n"
        "Columns: row#, layer_name, issue, recommendation, type, "
        "input_tensors, output_tensors, parameters, prev_layers, next_layers, "
        "modification, modification_info"
    ),
    "tool_name": TOOL_NAME,
    "output_suffix": OUTPUT_SUFFIX,
    "usage": {
        "linux_wsl": f"snpe-architecture-checker -i model.dlc [-o output] [-m MODIFY]",
        "windows": f"python snpe-architecture-checker -i model.dlc [-o output] [-m MODIFY]",
    },
    "modify_modes": {
        "--modify / --modify show": "Display possible modifications (dry run, no changes to model)",
        "--modify all": "Apply all applicable modifications",
        "--modify apply=r1,r2": "Apply specific rules (comma-separated, no spaces)",
    },
    "known_rules": KNOWN_RULES,
    "known_issues": KNOWN_ISSUES,
    "csv_columns": [
        "Graph / Layer name",
        "Issue",
        "Recommendation",
        "Type",
        "Input_tensor_name:[dims]",
        "Output_tensor_name:[dims]",
        "Parameters",
        "Previous layer",
        "Next layers",
        "Modification",
        "Modification_info",
    ],
    "modification_status": {
        "Done": "Modification was successfully applied to the model",
        "N/A": "No modification applicable or tool not run with --modify flag",
    },
    "layer_name_note": (
        "Layer/tensor names in CSV may differ slightly from original model "
        "but should be similar. Use input/output tensors and prev/next layers "
        "to locate the correct node in the original model."
    ),
    "caveat": (
        "Architecture Checker with modifier is experimental. "
        "For actual performance improvements, the model may require retraining/redesigning."
    ),
    "workflow": [
        "1. Run without --modify to identify all issues",
        "2. Run with --modify show to see applicable rule names",
        "3. Run with --modify all (or specific rules) to apply modifications",
        "4. Re-run on modified model to confirm issues resolved",
        "5. Retrain/redesign as needed for remaining unresolved issues",
    ],
}
