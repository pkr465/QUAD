"""Tests for SNPE Architecture Checker module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quad.sdk_tools.architecture_checker import (
    ARCH_CHECKER_NOTES,
    KNOWN_ISSUES,
    KNOWN_RULES,
    OUTPUT_SUFFIX,
    TOOL_NAME,
    ArchCheckerArgs,
    ArchCheckerIssue,
    ArchCheckerReport,
    ModificationStatus,
    ModifyMode,
    find_arch_checker,
    parse_arch_checker_csv,
    run_arch_checker,
)


# ══════════════════════════════════════════════════════════════════════════════
# Sample CSV content (from documentation)
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_CSV_NO_MODIFY = """\
Row#,Graph / Layer name,Issue,Recommendation,Type,Input_tensor_name:[dims],Output_tensor_name:[dims],Parameters,Previous layer,Next layers,Modification,Modification_info
1,Graph,This model uses 16-bit activation data. 16-bit activation data takes twice the amount of memory than 8-bit activation data does.,Try to use a smaller datatype to get better performance. E.g. 8-bit,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A
2,Layer_name_1,The number of channels in the input/output tensor of this convolution is low (smaller than 32).,Try increasing the number of channels in the input/output tensor to 32 or greater to get better performance.,Conv2d,"input_1:[1, 250, 250, 3], __param_1:[5, 5, 3, 32], convolution_0_bias:[32]","output_1:[1, 123, 123, 32]","{'dilation':'[1, 1]', 'group':1}",['previous_layer_name'],"['next_layer_name1', 'next_layer_name2']",N/A,N/A
"""

SAMPLE_CSV_WITH_MODIFY = """\
Row#,Graph / Layer name,Issue,Recommendation,Type,Input_tensor_name:[dims],Output_tensor_name:[dims],Parameters,Previous layer,Next layers,Modification,Modification_info
1,Layer_name_1,ElementWiseDivide usually has poor performance compared to ElementWiseMultiply,Try replacing ElementWiseDivide with ElementWiseMultiply using the reciprocal value to get better performance.,Eltwise_Binary,"input_1:[1, 52, 52, 6], input_2:[1]","output_1:[1, 52, 52, 6]","{'eltwise_type': 'ElementWiseDivide'}",['previous_layer_name'],"['next_layer_name1', 'next_layer_name2']",Done,ElementWiseDivide has been replaced by ElementWiseMultiply using the reciprocal value
2,Layer_name_2,The number of channels in the input/output tensor of this convolution is low (smaller than 32).,Try increasing the number of channels in the input/output tensor to 32 or greater to get better performance.,Conv2d,"input_3:[1, 250, 250, 3], __param_1:[5, 5, 3, 32], convolution_1_bias:[32]","output_2:[1, 123, 123, 32]","{'dilation':'[1, 1]', 'group':1}",['previous_layer_name'],"['next_layer_name1', 'next_layer_name2']",N/A,N/A
"""


# ══════════════════════════════════════════════════════════════════════════════
# ArchCheckerArgs — CLI Builder
# ══════════════════════════════════════════════════════════════════════════════

class TestArchCheckerArgs:
    def test_minimal_args(self) -> None:
        a = ArchCheckerArgs(input_dlc="model.dlc")
        args = a.build()
        assert TOOL_NAME in args
        assert "--input_dlc" in args
        assert "model.dlc" in args

    def test_output_path(self) -> None:
        a = ArchCheckerArgs("model.dlc", output_path="./archOutput")
        args = a.build()
        assert "--output_path" in args
        assert "./archOutput" in args

    def test_modify_show(self) -> None:
        a = ArchCheckerArgs("model.dlc", modify_mode=ModifyMode.SHOW)
        args = a.build()
        assert "--modify" in args
        assert "show" in args

    def test_modify_all(self) -> None:
        a = ArchCheckerArgs("model.dlc", modify_mode=ModifyMode.ALL)
        args = a.build()
        assert "--modify" in args
        assert "all" in args

    def test_modify_apply_single_rule(self) -> None:
        a = ArchCheckerArgs("model.dlc", modify_mode=ModifyMode.APPLY,
                            modify_rules=["elwisediv"])
        args = a.build()
        assert "--modify" in args
        modify_val = args[args.index("--modify") + 1]
        assert modify_val == "apply=elwisediv"

    def test_modify_apply_multiple_rules(self) -> None:
        a = ArchCheckerArgs("model.dlc", modify_mode=ModifyMode.APPLY,
                            modify_rules=["prelu", "elwisediv"])
        args = a.build()
        modify_val = args[args.index("--modify") + 1]
        # No spaces in rule list as per documentation
        assert modify_val == "apply=prelu,elwisediv"
        assert " " not in modify_val

    def test_modify_apply_no_rules_raises(self) -> None:
        a = ArchCheckerArgs("model.dlc", modify_mode=ModifyMode.APPLY,
                            modify_rules=[])
        with pytest.raises(ValueError, match="modify_rules"):
            a.build()

    def test_windows_python_prefix(self) -> None:
        a = ArchCheckerArgs("model.dlc", use_python=True)
        args = a.build()
        assert args[0] == "python"
        assert args[1] == TOOL_NAME

    def test_linux_no_python_prefix(self) -> None:
        a = ArchCheckerArgs("model.dlc", use_python=False)
        args = a.build()
        assert args[0] == TOOL_NAME

    def test_output_csv_path_with_output_path(self) -> None:
        a = ArchCheckerArgs("model.dlc", output_path="./archOutput")
        assert a.output_csv_path == f"./archOutput{OUTPUT_SUFFIX}"

    def test_output_csv_path_without_output_path(self) -> None:
        a = ArchCheckerArgs("/models/my_model.dlc")
        assert a.output_csv_path.endswith(OUTPUT_SUFFIX)
        assert "my_model" in a.output_csv_path

    def test_build_show_helper(self) -> None:
        a = ArchCheckerArgs("model.dlc", output_path="out")
        args = a.build_show()
        assert "show" in args

    def test_build_apply_all_helper(self) -> None:
        a = ArchCheckerArgs("model.dlc")
        args = a.build_apply_all()
        assert "all" in args

    def test_build_apply_rules_helper(self) -> None:
        a = ArchCheckerArgs("model.dlc")
        args = a.build_apply_rules(["elwisediv"])
        assert "apply=elwisediv" in args

    def test_documentation_sample_command(self) -> None:
        """Reproduce the exact sample command from the documentation."""
        a = ArchCheckerArgs(
            input_dlc="./model.dlc",
            output_path="./archCheckerOutput",
        )
        args = a.build()
        assert "--input_dlc" in args
        assert "./model.dlc" in args
        assert "--output_path" in args
        assert "./archCheckerOutput" in args

    def test_apply_prelu_elwisediv_command(self) -> None:
        """Reproduce: snpe-architecture-checker ... --modify apply=prelu,elwisediv"""
        a = ArchCheckerArgs("./model.dlc", output_path="./archCheckerOutput",
                            modify_mode=ModifyMode.APPLY,
                            modify_rules=["prelu", "elwisediv"])
        args = a.build()
        assert "apply=prelu,elwisediv" in args


# ══════════════════════════════════════════════════════════════════════════════
# ArchCheckerIssue
# ══════════════════════════════════════════════════════════════════════════════

class TestArchCheckerIssue:
    def test_graph_level_detection(self) -> None:
        issue = ArchCheckerIssue(
            row_number=1, layer_name="Graph",
            issue="16-bit activation", recommendation="Use 8-bit",
        )
        assert issue.is_graph_level is True

    def test_layer_level_detection(self) -> None:
        issue = ArchCheckerIssue(
            row_number=2, layer_name="Layer_name_1",
            issue="Low channels", recommendation="Increase channels",
        )
        assert issue.is_graph_level is False

    def test_is_modified_done(self) -> None:
        issue = ArchCheckerIssue(
            row_number=1, layer_name="conv1",
            issue="Div op", recommendation="Use mul",
            modification=ModificationStatus.DONE,
        )
        assert issue.is_modified is True

    def test_is_modified_na(self) -> None:
        issue = ArchCheckerIssue(
            row_number=1, layer_name="conv1",
            issue="Low channels", recommendation="Increase",
            modification=ModificationStatus.NOT_APPLICABLE,
        )
        assert issue.is_modified is False


# ══════════════════════════════════════════════════════════════════════════════
# ArchCheckerReport
# ══════════════════════════════════════════════════════════════════════════════

class TestArchCheckerReport:
    def _make_report(self) -> ArchCheckerReport:
        graph_issue = ArchCheckerIssue(
            row_number=1, layer_name="Graph",
            issue="16-bit activation", recommendation="Use 8-bit",
            modification=ModificationStatus.NOT_APPLICABLE,
        )
        conv_issue = ArchCheckerIssue(
            row_number=2, layer_name="Layer_name_1",
            issue="Low channels", recommendation="Increase to 32",
            op_type="Conv2d",
            modification=ModificationStatus.NOT_APPLICABLE,
        )
        div_issue = ArchCheckerIssue(
            row_number=3, layer_name="Layer_name_2",
            issue="ElementWiseDivide", recommendation="Use Mul",
            op_type="Eltwise_Binary",
            modification=ModificationStatus.DONE,
            modification_info="Replaced with ElementWiseMultiply",
        )
        return ArchCheckerReport(
            input_dlc="model.dlc", output_csv="out.csv",
            issues=[graph_issue, conv_issue, div_issue],
            total_issues=3,
            modifications_applied=1,
            modifications_not_applicable=2,
        )

    def test_graph_issues(self) -> None:
        report = self._make_report()
        assert len(report.graph_issues) == 1
        assert report.graph_issues[0].layer_name == "Graph"

    def test_layer_issues(self) -> None:
        report = self._make_report()
        assert len(report.layer_issues) == 2

    def test_modified_layers(self) -> None:
        report = self._make_report()
        assert len(report.modified_layers) == 1
        assert "Layer_name_2" in report.modified_layers[0].layer_name

    def test_unresolved_issues(self) -> None:
        report = self._make_report()
        assert len(report.unresolved_issues) == 2

    def test_summary_contains_counts(self) -> None:
        report = self._make_report()
        summary = report.summary()
        assert "model.dlc" in summary
        assert "3" in summary  # total issues

    def test_counts_correct(self) -> None:
        report = self._make_report()
        assert report.total_issues == 3
        assert report.modifications_applied == 1
        assert report.modifications_not_applicable == 2


# ══════════════════════════════════════════════════════════════════════════════
# CSV Parser
# ══════════════════════════════════════════════════════════════════════════════

class TestParseArchCheckerCsv:
    def test_parse_no_modify_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "model_architecture_checker.csv"
        csv_file.write_text(SAMPLE_CSV_NO_MODIFY)
        report = parse_arch_checker_csv(str(csv_file))
        assert report.total_issues == 2

    def test_parse_graph_level_issue(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "model_architecture_checker.csv"
        csv_file.write_text(SAMPLE_CSV_NO_MODIFY)
        report = parse_arch_checker_csv(str(csv_file))
        graph_issues = report.graph_issues
        assert len(graph_issues) == 1
        assert "16-bit" in graph_issues[0].issue

    def test_parse_layer_level_issue(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "model_architecture_checker.csv"
        csv_file.write_text(SAMPLE_CSV_NO_MODIFY)
        report = parse_arch_checker_csv(str(csv_file))
        layer_issues = report.layer_issues
        assert len(layer_issues) == 1
        assert layer_issues[0].op_type == "Conv2d"

    def test_parse_graph_level_not_modified(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(SAMPLE_CSV_NO_MODIFY)
        report = parse_arch_checker_csv(str(csv_file))
        assert report.modifications_applied == 0

    def test_parse_with_modify_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "modified_architecture_checker.csv"
        csv_file.write_text(SAMPLE_CSV_WITH_MODIFY)
        report = parse_arch_checker_csv(str(csv_file))
        assert report.total_issues == 2
        assert report.modifications_applied == 1
        assert report.modifications_not_applicable == 1

    def test_parse_modified_row_is_done(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(SAMPLE_CSV_WITH_MODIFY)
        report = parse_arch_checker_csv(str(csv_file))
        done_issues = [i for i in report.issues if i.is_modified]
        assert len(done_issues) == 1
        assert "ElementWiseDivide" in done_issues[0].issue
        assert "reciprocal" in done_issues[0].modification_info.lower()

    def test_parse_na_row_not_modified(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(SAMPLE_CSV_WITH_MODIFY)
        report = parse_arch_checker_csv(str(csv_file))
        na_issues = [i for i in report.issues if not i.is_modified]
        assert len(na_issues) == 1
        assert "channels" in na_issues[0].issue.lower()

    def test_parse_op_type_preserved(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "out.csv"
        csv_file.write_text(SAMPLE_CSV_WITH_MODIFY)
        report = parse_arch_checker_csv(str(csv_file))
        types = {i.op_type for i in report.issues}
        assert "Eltwise_Binary" in types
        assert "Conv2d" in types


# ══════════════════════════════════════════════════════════════════════════════
# find_arch_checker
# ══════════════════════════════════════════════════════════════════════════════

class TestFindArchChecker:
    def test_returns_none_when_not_installed(self) -> None:
        with patch("shutil.which", return_value=None):
            with patch.dict("os.environ", {}, clear=True):
                result = find_arch_checker()
                assert result is None

    def test_returns_path_when_in_path(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/snpe-architecture-checker"):
            result = find_arch_checker()
            assert result == "/usr/bin/snpe-architecture-checker"

    def test_searches_sdk_root(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin" / "x86_64-linux-clang"
        bin_dir.mkdir(parents=True)
        tool = bin_dir / TOOL_NAME
        tool.touch()
        with patch("shutil.which", return_value=None):
            result = find_arch_checker(sdk_root=str(tmp_path))
        assert result == str(tool)


# ══════════════════════════════════════════════════════════════════════════════
# run_arch_checker
# ══════════════════════════════════════════════════════════════════════════════

class TestRunArchChecker:
    def test_raises_file_not_found_when_tool_missing(self, tmp_path: Path) -> None:
        a = ArchCheckerArgs("model.dlc", output_path=str(tmp_path / "out"))
        with patch("shutil.which", return_value=None):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(FileNotFoundError, match="snpe-architecture-checker"):
                    run_arch_checker(a)

    def test_runs_and_parses_output(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "out_architecture_checker.csv"
        csv_path.write_text(SAMPLE_CSV_WITH_MODIFY)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        a = ArchCheckerArgs("model.dlc", output_path=str(tmp_path / "out"))

        with patch("shutil.which", return_value="/usr/bin/snpe-architecture-checker"):
            with patch("subprocess.run", return_value=mock_result):
                report = run_arch_checker(a)

        assert report.total_issues == 2
        assert report.modifications_applied == 1

    def test_raises_runtime_error_on_nonzero(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: DLC not found"

        a = ArchCheckerArgs("model.dlc")
        with patch("shutil.which", return_value="/usr/bin/snpe-architecture-checker"):
            with patch("subprocess.run", return_value=mock_result):
                with pytest.raises(RuntimeError, match="failed"):
                    run_arch_checker(a)


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

class TestArchCheckerNotes:
    def test_known_rules_present(self) -> None:
        assert "elwisediv" in KNOWN_RULES
        assert "prelu" in KNOWN_RULES

    def test_elwisediv_rule_mentions_reciprocal(self) -> None:
        assert "reciprocal" in KNOWN_RULES["elwisediv"].lower()

    def test_known_issues_present(self) -> None:
        for key in ("16bit_activation", "low_channel_conv", "elementwise_divide"):
            assert key in KNOWN_ISSUES

    def test_low_channel_threshold_is_32(self) -> None:
        issue = KNOWN_ISSUES["low_channel_conv"]
        assert "32" in issue["recommendation"]

    def test_16bit_activation_recommendation(self) -> None:
        issue = KNOWN_ISSUES["16bit_activation"]
        assert "8-bit" in issue["recommendation"]

    def test_notes_modify_modes(self) -> None:
        modes = ARCH_CHECKER_NOTES["modify_modes"]
        assert any("show" in k.lower() for k in modes)
        assert any("all" in k.lower() for k in modes)
        assert any("apply" in k.lower() for k in modes)

    def test_notes_csv_columns(self) -> None:
        cols = ARCH_CHECKER_NOTES["csv_columns"]
        assert "Modification" in cols
        assert "Issue" in cols
        assert "Recommendation" in cols

    def test_notes_workflow_steps(self) -> None:
        workflow = ARCH_CHECKER_NOTES["workflow"]
        assert len(workflow) >= 4
        assert any("show" in step.lower() for step in workflow)
        assert any("all" in step.lower() for step in workflow)

    def test_notes_caveat_mentions_retraining(self) -> None:
        assert "retrain" in ARCH_CHECKER_NOTES["caveat"].lower()

    def test_notes_prerequisite_snpe_root(self) -> None:
        assert "SNPE_ROOT" in ARCH_CHECKER_NOTES["prerequisite"]

    def test_output_suffix_constant(self) -> None:
        assert OUTPUT_SUFFIX == "_architecture_checker.csv"

    def test_tool_name_constant(self) -> None:
        assert TOOL_NAME == "snpe-architecture-checker"

    def test_exported_from_sdk_tools_package(self) -> None:
        from quad.sdk_tools import (  # noqa: F401
            ARCH_CHECKER_NOTES,
            ArchCheckerArgs,
            ArchCheckerReport,
            ModifyMode,
            find_arch_checker,
            parse_arch_checker_csv,
        )
