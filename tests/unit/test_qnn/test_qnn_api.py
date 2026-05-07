"""Tests for QNN SDK API reference and inference pipelines."""

from __future__ import annotations

import pytest

from quad.qnn.api import (
    QNN_API_COMPONENTS,
    QnnApiComponent,
    QnnCategory,
    get_backend_specialized_components,
    get_components_by_category,
    get_pre_backend_components,
)
from quad.qnn.inference_pipeline import (
    BIN_CACHE_NOTES,
    BIN_PIPELINE_STEPS,
    INFERENCE_PIPELINES,
    PIPELINES,
    PipelineStep,
    QnnBackendLib,
    QnnInferencePipeline,
    QnnModelFormat,
    QNN_INFERENCE_NOTES,
    SO_PIPELINE_STEPS,
    TFLITE_DELEGATE_PIPELINE_STEPS,
)


# ══════════════════════════════════════════════════════════════════════════════
# API Component Registry
# ══════════════════════════════════════════════════════════════════════════════

class TestQnnApiComponents:
    def test_all_eleven_components_present(self) -> None:
        expected = {
            "QnnBackend", "QnnDevice", "QnnContext", "QnnGraph", "QnnTensor",
            "QnnOpPackage", "QnnProfile", "QnnLog", "QnnProperty",
            "QnnMem", "QnnSignal",
        }
        assert expected == set(QNN_API_COMPONENTS.keys())

    def test_categories_correct(self) -> None:
        core = {"QnnBackend", "QnnDevice", "QnnContext", "QnnGraph",
                "QnnTensor", "QnnOpPackage"}
        utility = {"QnnProfile", "QnnLog"}
        system = {"QnnProperty", "QnnMem", "QnnSignal"}

        for name in core:
            assert QNN_API_COMPONENTS[name].category == QnnCategory.CORE
        for name in utility:
            assert QNN_API_COMPONENTS[name].category == QnnCategory.UTILITY
        for name in system:
            assert QNN_API_COMPONENTS[name].category == QnnCategory.SYSTEM

    def test_backend_specialized_flags(self) -> None:
        # Specialized: Backend, Device, Context, Graph, OpPackage, Profile
        specialized = {"QnnBackend", "QnnDevice", "QnnContext", "QnnGraph",
                       "QnnOpPackage", "QnnProfile"}
        # NOT specialized: Tensor, Log, Property, Mem, Signal
        not_specialized = {"QnnTensor", "QnnLog", "QnnProperty", "QnnMem", "QnnSignal"}

        for name in specialized:
            assert QNN_API_COMPONENTS[name].backend_specialized is True, name
        for name in not_specialized:
            assert QNN_API_COMPONENTS[name].backend_specialized is False, name

    def test_qnnlog_can_init_before_backend(self) -> None:
        assert QNN_API_COMPONENTS["QnnLog"].init_before_backend is True

    def test_qnnproperty_can_init_before_backend(self) -> None:
        assert QNN_API_COMPONENTS["QnnProperty"].init_before_backend is True

    def test_qnnbackend_cannot_init_before_itself(self) -> None:
        assert QNN_API_COMPONENTS["QnnBackend"].init_before_backend is False

    def test_qnncontext_binary_cache_documented(self) -> None:
        ctx = QNN_API_COMPONENTS["QnnContext"]
        features = " ".join(ctx.key_features)
        assert "bin" in features.lower() or "cache" in features.lower() or "binary" in features.lower()

    def test_qnntensor_scopes_documented(self) -> None:
        tensor = QNN_API_COMPONENTS["QnnTensor"]
        features = " ".join(tensor.key_features)
        assert "scope" in features.lower()

    def test_get_components_by_category_core(self) -> None:
        core = get_components_by_category(QnnCategory.CORE)
        names = {c.name for c in core}
        assert "QnnBackend" in names
        assert "QnnGraph" in names
        assert "QnnLog" not in names  # utility

    def test_get_backend_specialized_components(self) -> None:
        specialized = get_backend_specialized_components()
        names = {c.name for c in specialized}
        assert "QnnBackend" in names
        assert "QnnContext" in names
        assert "QnnTensor" not in names  # not specialized

    def test_get_pre_backend_components(self) -> None:
        pre = get_pre_backend_components()
        names = {c.name for c in pre}
        assert "QnnLog" in names
        assert "QnnProperty" in names
        assert "QnnBackend" not in names


# ══════════════════════════════════════════════════════════════════════════════
# .so Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestSoPipeline:
    def test_six_steps(self) -> None:
        assert len(SO_PIPELINE_STEPS) == 6

    def test_step_numbers_sequential(self) -> None:
        nums = [s.number for s in SO_PIPELINE_STEPS]
        assert nums == list(range(1, 7))

    def test_step1_loads_backend(self) -> None:
        s = SO_PIPELINE_STEPS[0]
        assert "backend" in s.name.lower() or "load" in s.name.lower()
        assert any("dlopen" in c or "getProviders" in c for c in s.key_calls)

    def test_step2_loads_model_so(self) -> None:
        s = SO_PIPELINE_STEPS[1]
        assert any("model.so" in c or "composeGraphs" in c for c in s.key_calls)

    def test_step3_init_order(self) -> None:
        s = SO_PIPELINE_STEPS[2]
        calls = s.key_calls
        # Log must come before Backend
        log_idx = next((i for i, c in enumerate(calls) if "log" in c.lower()), -1)
        backend_idx = next((i for i, c in enumerate(calls) if "backend" in c.lower()), -1)
        assert log_idx < backend_idx

    def test_step4_finalize_required(self) -> None:
        s = SO_PIPELINE_STEPS[3]
        assert any("finalize" in c.lower() for c in s.key_calls)

    def test_step5_execute(self) -> None:
        s = SO_PIPELINE_STEPS[4]
        assert any("execute" in c.lower() for c in s.key_calls)

    def test_step6_free_resources(self) -> None:
        s = SO_PIPELINE_STEPS[5]
        calls = " ".join(s.key_calls).lower()
        assert "free" in calls or "destroy" in calls

    def test_compose_graphs_function_documented(self) -> None:
        """QnnModel_composeGraphs must be called from model.so."""
        all_calls = " ".join(c for s in SO_PIPELINE_STEPS for c in s.key_calls)
        assert "composeGraphs" in all_calls


# ══════════════════════════════════════════════════════════════════════════════
# .bin Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestBinPipeline:
    def test_six_steps(self) -> None:
        assert len(BIN_PIPELINE_STEPS) == 6

    def test_step2_loads_qnnsystem(self) -> None:
        s = BIN_PIPELINE_STEPS[1]
        assert "system" in s.name.lower() or "system" in s.description.lower()
        assert any("System" in c or "system" in c.lower() for c in s.key_calls)

    def test_step3_no_context_create(self) -> None:
        """Context is loaded from binary, not created fresh."""
        s = BIN_PIPELINE_STEPS[2]
        assert not any("contextCreate" in c for c in s.key_calls)

    def test_step4_loads_context_binary(self) -> None:
        s = BIN_PIPELINE_STEPS[3]
        calls = " ".join(s.key_calls)
        assert "FromBinary" in calls or "binary" in calls.lower()

    def test_step4_backend_specific_note(self) -> None:
        s = BIN_PIPELINE_STEPS[3]
        notes = " ".join(s.notes).lower()
        assert "backend" in notes and ("specific" in notes or "match" in notes)

    def test_no_finalize_in_bin_pipeline(self) -> None:
        """Finalization already done — not needed when loading .bin."""
        all_calls = " ".join(c for s in BIN_PIPELINE_STEPS for c in s.key_calls)
        assert "finalize" not in all_calls.lower()

    def test_libqnnsystem_required(self) -> None:
        pipeline = PIPELINES[QnnModelFormat.BIN]
        assert "libQnnSystem.so" in pipeline.required_libraries


# ══════════════════════════════════════════════════════════════════════════════
# TFLite Delegate Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestTfliteDelegatePipeline:
    def test_four_steps(self) -> None:
        assert len(TFLITE_DELEGATE_PIPELINE_STEPS) == 4

    def test_step1_loads_tflite(self) -> None:
        s = TFLITE_DELEGATE_PIPELINE_STEPS[0]
        calls = " ".join(s.key_calls).lower()
        assert "tflite" in calls or "flatbuffer" in calls.lower() or "interpreter" in calls.lower()

    def test_step2_registers_delegate(self) -> None:
        s = TFLITE_DELEGATE_PIPELINE_STEPS[1]
        calls = " ".join(s.key_calls)
        assert "Delegate" in calls or "delegate" in calls.lower()

    def test_step3_uses_invoke(self) -> None:
        s = TFLITE_DELEGATE_PIPELINE_STEPS[2]
        calls = " ".join(s.key_calls)
        assert "Invoke" in calls or "invoke" in calls.lower()

    def test_step4_deletes_delegate(self) -> None:
        s = TFLITE_DELEGATE_PIPELINE_STEPS[3]
        calls = " ".join(s.key_calls)
        assert "Delete" in calls or "delete" in calls.lower()

    def test_qnn_acceleration_transparent_note(self) -> None:
        pipeline = PIPELINES[QnnModelFormat.TFLITE]
        assert "transparent" in pipeline.key_difference.lower()


# ══════════════════════════════════════════════════════════════════════════════
# QnnInferencePipeline registry
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineRegistry:
    def test_all_three_formats_registered(self) -> None:
        for fmt in QnnModelFormat:
            assert fmt in PIPELINES

    def test_so_required_libraries(self) -> None:
        libs = PIPELINES[QnnModelFormat.SO].required_libraries
        assert any("QnnBackend" in lib or "backend" in lib.lower() for lib in libs)
        assert any("model.so" in lib for lib in libs)

    def test_bin_required_libraries(self) -> None:
        libs = PIPELINES[QnnModelFormat.BIN].required_libraries
        assert any("System" in lib for lib in libs)
        assert any(".bin" in lib for lib in libs)

    def test_tflite_required_libraries(self) -> None:
        libs = PIPELINES[QnnModelFormat.TFLITE].required_libraries
        assert any("tflite" in lib.lower() or "Delegate" in lib for lib in libs)

    def test_get_step_returns_correct_step(self) -> None:
        pipeline = PIPELINES[QnnModelFormat.SO]
        step1 = pipeline.get_step(1)
        assert step1 is not None
        assert step1.number == 1

    def test_get_step_returns_none_for_invalid(self) -> None:
        pipeline = PIPELINES[QnnModelFormat.SO]
        assert pipeline.get_step(99) is None

    def test_all_key_calls_non_empty(self) -> None:
        for fmt, pipeline in PIPELINES.items():
            calls = pipeline.all_key_calls()
            assert len(calls) > 0, f"Pipeline {fmt} has no key calls"

    def test_describe_output(self) -> None:
        desc = PIPELINES[QnnModelFormat.SO].describe()
        assert "Step 1" in desc
        assert "SO" in desc or "so" in desc.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Binary Cache Notes
# ══════════════════════════════════════════════════════════════════════════════

class TestBinCacheNotes:
    def test_purpose_mentions_time(self) -> None:
        assert "time" in BIN_CACHE_NOTES["purpose"].lower()

    def test_backend_specific_documented(self) -> None:
        note = BIN_CACHE_NOTES["backend_specific"]
        assert "backend" in note.lower()
        assert "match" in note.lower() or "same" in note.lower()

    def test_loading_api_documented(self) -> None:
        assert "contextCreateFromBinary" in BIN_CACHE_NOTES["loading_api"] or \
               "createFromBinary" in BIN_CACHE_NOTES["loading_api"]

    def test_snpe_equivalent_documented(self) -> None:
        equiv = BIN_CACHE_NOTES["equivalent_to_dlc"]
        assert "snpe" in equiv.lower() or "dlc" in equiv.lower()

    def test_generation_mentions_saver(self) -> None:
        gen = BIN_CACHE_NOTES["generation"]
        assert "Saver" in gen or "saver" in gen.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

class TestQnnInferenceNotes:
    def test_all_formats_documented(self) -> None:
        for fmt in ("so", "bin", "tflite"):
            assert fmt in QNN_INFERENCE_NOTES["supported_formats"]

    def test_init_order_log_first(self) -> None:
        order = QNN_INFERENCE_NOTES["initialization_order"]
        log_idx = next((i for i, s in enumerate(order) if s.startswith("QnnLog")), -1)
        backend_idx = next((i for i, s in enumerate(order) if s.startswith("QnnBackend")), -1)
        assert log_idx != -1 and backend_idx != -1
        assert log_idx < backend_idx

    def test_backend_libraries_documented(self) -> None:
        libs = QNN_INFERENCE_NOTES["backend_libraries"]
        for key in ("cpu", "gpu", "htp", "system", "saver"):
            assert key in libs

    def test_recommended_alternative_qrb(self) -> None:
        alt = QNN_INFERENCE_NOTES["recommended_alternative"]
        assert "qrb_inference_manager" in alt

    def test_so_vs_bin_difference_documented(self) -> None:
        diff = QNN_INFERENCE_NOTES["so_vs_bin_key_difference"]
        assert "compose" in diff.lower()
        assert "binary" in diff.lower() or "bin" in diff.lower()

    def test_backend_libs_enum_matches_notes(self) -> None:
        for lib in QnnBackendLib:
            assert ".so" in lib.value

    def test_exported_from_qnn_package(self) -> None:
        from quad.qnn import (  # noqa: F401
            QNN_API_COMPONENTS,
            QNN_INFERENCE_NOTES,
            PIPELINES,
            QnnModelFormat,
        )
