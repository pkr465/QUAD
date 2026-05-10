"""Unit tests for QAIRT/SNPE stdout parsers.

The parsers are pure functions, so the tests use hand-crafted stdout/CSV
fixtures derived from real QAIRT 2.46 documentation and the runtime
output we captured on the Snapdragon X Elite reference box.
"""
from __future__ import annotations

import pytest

from quad.adapters.parsers import (
    parse_qairt_converter_stdout,
    parse_qnn_platform_validator,
    parse_snpe_diagview_csv,
    parse_snpe_diagview_layers,
    parse_snpe_net_run_stdout,
)


# ── snpe-net-run stdout ────────────────────────────────────────────────────


SNPE_NET_RUN_FIXTURE_DETAILED = """\
SNPE Network: Loading model
Using DSP runtime
perf_profile=high_performance
profiling_level=detailed
[INFO] Successfully loaded container
Total Inference Time: 4.27 ms
Forward Propagate: 3.91 ms
Saved 1 output
[INFO] Inference complete
"""

SNPE_NET_RUN_FIXTURE_NETRUN_LABEL = """\
SNPE Network: Loading model
Using HTP runtime
Total Inference Time [NetRun]: 12.05 ms
Saved 1 output
"""

SNPE_NET_RUN_FIXTURE_GPU_US = """\
Using GPU runtime
Total Inference Time: 4270 us
Saved 2 outputs
"""

SNPE_NET_RUN_FIXTURE_FAILURE = """\
SNPE Network: Loading model
ERROR: Failed to initialize accelerator
FATAL: transportStatus: 9
"""


def test_net_run_parses_total_inference_time_ms():
    out = parse_snpe_net_run_stdout(SNPE_NET_RUN_FIXTURE_DETAILED)
    assert out["_parsed"] is True
    assert out["latency_ms"] == pytest.approx(4.27)
    assert out["forward_ms"] == pytest.approx(3.91)
    assert out["runtime"] == "npu"
    assert out["perf_profile"] == "high_performance"
    assert out["n_outputs"] == 1


def test_net_run_parses_netrun_label_variant():
    out = parse_snpe_net_run_stdout(SNPE_NET_RUN_FIXTURE_NETRUN_LABEL)
    assert out["latency_ms"] == pytest.approx(12.05)
    assert out["runtime"] == "npu"


def test_net_run_parses_microseconds_and_normalises_to_ms():
    out = parse_snpe_net_run_stdout(SNPE_NET_RUN_FIXTURE_GPU_US)
    assert out["latency_ms"] == pytest.approx(4.27, rel=1e-3)
    assert out["runtime"] == "gpu"
    assert out["n_outputs"] == 2


def test_net_run_captures_errors_without_crashing():
    out = parse_snpe_net_run_stdout(SNPE_NET_RUN_FIXTURE_FAILURE)
    assert out["latency_ms"] == 0.0
    assert any("ERROR" in e or "FATAL" in e or "transportStatus" in e for e in out["errors"])


def test_net_run_empty_input_is_safe():
    out = parse_snpe_net_run_stdout("")
    assert out["_parsed"] is False
    assert out["latency_ms"] == 0.0
    assert out["errors"] == []


# ── snpe-diagview CSV ──────────────────────────────────────────────────────


DIAGVIEW_CSV_DETAILED = """\
Section,Init metrics
Init,Create Network(s),De-Init
12345,9876,234

Section,Execution metrics
Total Inference Time,Forward Propagate,RPC Execute,Snpe Accelerator,Accelerator
4270,3920,4180,3950,3940

Section,Layer metrics
Layer Times
conv1,123,100,150,DSP
relu1,45,30,60,DSP
pool1,200,180,220,DSP
fc1,500,480,520,DSP
"""


def test_diagview_csv_extracts_total_inference_time():
    out = parse_snpe_diagview_csv(DIAGVIEW_CSV_DETAILED)
    assert out["_parsed"] is True
    assert out["total_inference_us"] == pytest.approx(4270.0)
    assert out["forward_propagate_us"] == pytest.approx(3920.0)
    assert out["rpc_execute_us"] == pytest.approx(4180.0)
    assert out["mean_latency_ms"] == pytest.approx(4.27, rel=1e-3)


def test_diagview_csv_init_metrics_block():
    out = parse_snpe_diagview_csv(DIAGVIEW_CSV_DETAILED)
    assert out["init_us"] == pytest.approx(12345.0)
    assert out["create_network_us"] == pytest.approx(9876.0)
    assert out["deinit_us"] == pytest.approx(234.0)


def test_diagview_layers_round_trip():
    layers = parse_snpe_diagview_layers(DIAGVIEW_CSV_DETAILED)
    assert len(layers) == 4
    assert [l["name"] for l in layers] == ["conv1", "relu1", "pool1", "fc1"]
    # DSP normalises to npu in the runtime vocabulary.
    assert all(l["runtime"] == "npu" for l in layers)
    assert layers[0]["avg_us"] == 123.0
    assert layers[3]["max_us"] == 520.0


def test_diagview_empty_csv_returns_zeros_not_error():
    out = parse_snpe_diagview_csv("")
    assert out["_parsed"] is False
    assert out["mean_latency_ms"] == 0.0


def test_diagview_falls_back_to_forward_propagate_if_total_missing():
    csv_no_total = """\
Section,Execution metrics
Forward Propagate
2500
"""
    out = parse_snpe_diagview_csv(csv_no_total)
    assert out["forward_propagate_us"] == pytest.approx(2500.0)
    assert out["mean_latency_ms"] == pytest.approx(2.5, rel=1e-3)


# ── qnn-platform-validator ─────────────────────────────────────────────────


PV_DSP_GPU = """\
PF_VALIDATOR: DEBUG: Calling PlatformValidator->setBackend
PF_VALIDATOR: DEBUG: Calling PlatformValidator->isBackendHardwarePresent
Backend DSP Prerequisites: Present.
PF_VALIDATOR: DEBUG: Calling PlatformValidator->getCoreVersion
Core Version of the backend DSP: Hexagon Architecture V73
*********** Results Summary ***********
Backend = DSP
{
  Backend Hardware  : Supported
  Backend Libraries : Found
  Library Version   : Not Found
  Core Version      : Hexagon Architecture V73
  Unit Test         : Not Queried
}

Backend GPU Prerequisites: Present.
Library Version of the backend GPU: OpenCL 3.0 Qualcomm(R) Adreno(TM) X1-85 GPU
Core Version of the backend GPU: Adreno(TM) 740
*********** Results Summary ***********
Backend = GPU
{
  Backend Hardware  : Supported
  Library Version   : OpenCL 3.0 Qualcomm(R) Adreno(TM) X1-85 GPU
  Core Version      : Adreno(TM) 740
}
"""


def test_pv_extracts_runtimes_and_versions():
    p = parse_qnn_platform_validator(PV_DSP_GPU)
    assert p["_parsed"] is True
    assert "npu" in p["runtimes"]
    assert "gpu" in p["runtimes"]
    assert p["per_backend"]["DSP"]["hardware"].startswith("Supported")
    assert "Hexagon Architecture V73" in p["per_backend"]["DSP"]["core_version"]
    assert p["per_backend"]["GPU"]["hardware"].startswith("Supported")
    assert "Adreno(TM) X1-85" in p["per_backend"]["GPU"]["lib_version"]
    assert p["npu_arch"] == "V73"
    assert p["gpu_model"].startswith("X1-85") or p["gpu_model"].startswith("740")


def test_pv_unsupported_backend_excluded():
    text = """
*********** Results Summary ***********
Backend = DSP
{
  Backend Hardware  : Not Supported
  Library Version   : Not Found
  Core Version      : Not Found
}
"""
    p = parse_qnn_platform_validator(text)
    assert p["_parsed"] is True
    assert "npu" not in p["runtimes"]
    assert p["per_backend"]["DSP"]["hardware"].lower().startswith("not")


def test_pv_empty_input():
    p = parse_qnn_platform_validator("")
    assert p["_parsed"] is False
    assert p["runtimes"] == []


# ── qairt-converter ────────────────────────────────────────────────────────


CONVERTER_OK = """\
[INFO] Reading ONNX model from input.onnx
[INFO] Building IR graph
[INFO] Supported ops: 78/80
[WARNING] Unsupported ops: GridSample, ScatterND
[INFO] Optimisation pass complete
[INFO] Output: model.bin
Conversion complete
"""

CONVERTER_FAIL = """\
[INFO] Reading ONNX model from missing.onnx
ERROR: File not found: missing.onnx
"""


def test_converter_parses_supported_ops_and_output():
    p = parse_qairt_converter_stdout(CONVERTER_OK)
    assert p["_parsed"] is True
    assert p["success"] is True
    assert p["output_path"] == "model.bin"
    assert p["supported_ops"] == 78
    assert p["total_ops"] == 80
    assert p["supported_ops_pct"] == pytest.approx(97.5)
    assert "GridSample" in p["unsupported_ops"]
    assert "ScatterND" in p["unsupported_ops"]
    assert p["warnings"]


def test_converter_failure_path():
    p = parse_qairt_converter_stdout(CONVERTER_FAIL)
    assert p["success"] is False
    assert any("File not found" in e for e in p["errors"])


def test_converter_empty_input():
    p = parse_qairt_converter_stdout("")
    assert p["_parsed"] is False
    assert p["output_path"] is None
