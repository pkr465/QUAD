"""Tests for QUAD Serve & Deploy — inference server, registry, and deployment."""

import numpy as np
import pytest

from quad.serve.server import (
    ModelServer,
    ServerConfig,
    ModelInfo,
    HealthStatus,
    ServerMetrics,
)
from quad.serve.request import InferenceRequest, InferenceResponse, BatchRequest
from quad.serve.model_registry import ModelRegistry, ModelEntry, ModelConfig
from quad.serve.deploy import deploy_model, DeployResult
from quad.runtime.tensor import Tensor
from quad.runtime.device import Device


# ===========================================================================
# ModelServer — Loading / Unloading
# ===========================================================================


class TestModelLoading:
    """Tests for model load and unload lifecycle."""

    def test_load_model_basic(self):
        server = ModelServer(port=9000)
        server.load_model("yolo", "models/yolo.qbin", device="npu:0")
        assert server.num_models == 1

    def test_load_multiple_models(self):
        server = ModelServer(port=9000)
        server.load_model("yolo", "models/yolo.qbin", device="npu:0")
        server.load_model("resnet", "models/resnet.qbin", device="npu:1")
        assert server.num_models == 2

    def test_load_model_with_version(self):
        server = ModelServer(port=9000)
        server.load_model("yolo", "models/yolo.qbin", device="npu", version=3)
        models = server.list_models()
        assert models[0].version == 3

    def test_load_duplicate_model_raises(self):
        server = ModelServer(port=9000)
        server.load_model("yolo", "models/yolo.qbin", device="npu")
        with pytest.raises(ValueError, match="already loaded"):
            server.load_model("yolo", "models/yolo_v2.qbin", device="npu")

    def test_unload_model(self):
        server = ModelServer(port=9000)
        server.load_model("yolo", "models/yolo.qbin", device="npu")
        server.unload_model("yolo")
        assert server.num_models == 0

    def test_unload_nonexistent_raises(self):
        server = ModelServer(port=9000)
        with pytest.raises(KeyError, match="not loaded"):
            server.unload_model("nonexistent")

    def test_list_models(self):
        server = ModelServer(port=9000)
        server.load_model("yolo", "models/yolo.qbin", device="npu:0")
        server.load_model("resnet", "models/resnet.qbin", device="gpu")
        models = server.list_models()
        names = [m.name for m in models]
        assert "yolo" in names
        assert "resnet" in names


# ===========================================================================
# ModelServer — Inference
# ===========================================================================


class TestInference:
    """Tests for single and batch inference."""

    def setup_method(self):
        self.server = ModelServer(port=9000)
        self.server.load_model("resnet", "models/resnet.qbin", device="npu")
        self.server.start()

    def test_infer_single(self):
        inputs = {"image": np.random.randn(1, 3, 224, 224).astype(np.float32)}
        response = self.server.infer("resnet", inputs)
        assert isinstance(response, InferenceResponse)
        assert "output" in response.outputs
        assert response.latency_ms >= 0
        assert response.model_name == "resnet"

    def test_infer_output_shape(self):
        inputs = {"image": np.random.randn(1, 3, 224, 224).astype(np.float32)}
        response = self.server.infer("resnet", inputs)
        assert response.outputs["output"].shape == (1, 1000)

    def test_infer_batch_output_shape(self):
        inputs = {"image": np.random.randn(4, 3, 224, 224).astype(np.float32)}
        response = self.server.infer("resnet", inputs)
        assert response.outputs["output"].shape == (4, 1000)

    def test_infer_missing_model_raises(self):
        inputs = {"image": np.random.randn(1, 3, 224, 224).astype(np.float32)}
        with pytest.raises(KeyError, match="not loaded"):
            self.server.infer("nonexistent", inputs)

    def test_infer_empty_inputs_raises(self):
        with pytest.raises(ValueError, match="inputs must not be empty"):
            self.server.infer("resnet", {})

    def test_infer_increments_count(self):
        inputs = {"image": np.random.randn(1, 3, 224, 224).astype(np.float32)}
        self.server.infer("resnet", inputs)
        self.server.infer("resnet", inputs)
        assert self.server.total_inferences == 2

    def test_infer_batch(self):
        batch = [
            {"image": np.random.randn(1, 3, 224, 224).astype(np.float32)},
            {"image": np.random.randn(1, 3, 224, 224).astype(np.float32)},
            {"image": np.random.randn(1, 3, 224, 224).astype(np.float32)},
        ]
        responses = self.server.infer_batch("resnet", batch)
        assert len(responses) == 3
        assert all(isinstance(r, InferenceResponse) for r in responses)
        assert self.server.total_inferences == 3

    def test_infer_batch_empty_raises(self):
        with pytest.raises(ValueError, match="batch must not be empty"):
            self.server.infer_batch("resnet", [])


# ===========================================================================
# ModelServer — Health & Metrics
# ===========================================================================


class TestHealthMetrics:
    """Tests for health check and server metrics."""

    def test_health_healthy(self):
        server = ModelServer(port=9000)
        server.load_model("yolo", "models/yolo.qbin", device="npu")
        server.start()
        health = server.health()
        assert health.status == "healthy"
        assert health.models_loaded == 1
        assert health.uptime_s >= 0

    def test_health_degraded_no_models(self):
        server = ModelServer(port=9000)
        server.start()
        health = server.health()
        assert health.status == "degraded"

    def test_health_unhealthy_not_running(self):
        server = ModelServer(port=9000)
        server.load_model("yolo", "models/yolo.qbin", device="npu")
        health = server.health()
        assert health.status == "unhealthy"

    def test_metrics_initial(self):
        server = ModelServer(port=9000)
        server.start()
        m = server.metrics()
        assert m.total_requests == 0
        assert m.avg_latency_ms == 0.0

    def test_metrics_after_inference(self):
        server = ModelServer(port=9000)
        server.load_model("resnet", "models/resnet.qbin", device="npu")
        server.start()
        inputs = {"image": np.random.randn(1, 3, 224, 224).astype(np.float32)}
        server.infer("resnet", inputs)
        server.infer("resnet", inputs)
        m = server.metrics()
        assert m.total_requests == 2
        assert m.avg_latency_ms > 0
        assert m.p99_latency_ms > 0

    def test_metrics_power(self):
        server = ModelServer(port=9000)
        server.load_model("resnet", "models/resnet.qbin", device="npu")
        server.start()
        m = server.metrics()
        assert m.power_mw > 0


# ===========================================================================
# ModelServer — Lifecycle
# ===========================================================================


class TestServerLifecycle:
    """Tests for server start/stop."""

    def test_start_stop(self):
        server = ModelServer(port=9000)
        assert not server.is_running
        server.start()
        assert server.is_running
        server.stop()
        assert not server.is_running

    def test_server_config(self):
        server = ModelServer(port=9001, host="127.0.0.1", power_budget_mw=5000)
        assert server.config.port == 9001
        assert server.config.host == "127.0.0.1"
        assert server.config.power_budget_mw == 5000


# ===========================================================================
# Request / Response Models
# ===========================================================================


class TestRequestResponse:
    """Tests for InferenceRequest, InferenceResponse, and BatchRequest."""

    def test_inference_request_creation(self):
        req = InferenceRequest(
            model_name="resnet",
            inputs={"image": np.zeros((1, 3, 224, 224), dtype=np.float32)},
        )
        assert req.model_name == "resnet"
        assert req.request_id  # auto-generated
        assert req.priority == 0

    def test_inference_request_empty_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            InferenceRequest(
                model_name="",
                inputs={"image": np.zeros((1,), dtype=np.float32)},
            )

    def test_inference_request_empty_inputs_raises(self):
        with pytest.raises(ValueError, match="inputs"):
            InferenceRequest(model_name="resnet", inputs={})

    def test_inference_response_properties(self):
        resp = InferenceResponse(
            outputs={"logits": np.zeros((1, 1000)), "features": np.zeros((1, 512))},
            latency_ms=2.5,
            model_name="resnet",
            request_id="abc123",
        )
        assert resp.output_names == ["logits", "features"]
        assert resp.num_outputs == 2

    def test_batch_request_validation(self):
        reqs = [
            InferenceRequest("resnet", {"x": np.zeros((1,))}),
            InferenceRequest("resnet", {"x": np.zeros((1,))}),
        ]
        batch = BatchRequest(requests=reqs, max_wait_ms=5.0)
        assert batch.batch_size == 2
        assert batch.model_name == "resnet"
        assert batch.validate() is True

    def test_batch_request_mixed_models_invalid(self):
        reqs = [
            InferenceRequest("resnet", {"x": np.zeros((1,))}),
            InferenceRequest("yolo", {"x": np.zeros((1,))}),
        ]
        batch = BatchRequest(requests=reqs)
        assert batch.validate() is False


# ===========================================================================
# Model Registry
# ===========================================================================


class TestModelRegistry:
    """Tests for the model zoo / registry."""

    def test_default_models_loaded(self):
        registry = ModelRegistry()
        models = registry.list_all()
        names = [m.name for m in models]
        assert "mobilenetv2" in names
        assert "resnet50" in names
        assert "yolov8n" in names
        assert "whisper-tiny" in names
        assert "llama-7b" in names

    def test_get_model(self):
        registry = ModelRegistry()
        entry = registry.get("yolov8n")
        assert entry.name == "yolov8n"
        assert "snapdragon-8-gen3" in entry.chipsets

    def test_get_nonexistent_raises(self):
        registry = ModelRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent_model")

    def test_register_new_model(self):
        registry = ModelRegistry()
        entry = registry.register(
            "custom_model",
            "models/custom.qbin",
            metadata={
                "tags": ["custom", "test"],
                "description": "Test model",
                "chipsets": ["snapdragon-8-gen3"],
            },
        )
        assert entry.name == "custom_model"
        assert entry.version == 1
        retrieved = registry.get("custom_model")
        assert retrieved.name == "custom_model"

    def test_register_increments_version(self):
        registry = ModelRegistry()
        registry.register("mymodel", "models/v1.qbin")
        registry.register("mymodel", "models/v2.qbin")
        versions = registry.get_versions("mymodel")
        assert versions == [1, 2]

    def test_search_by_name(self):
        registry = ModelRegistry()
        results = registry.search("yolo")
        assert len(results) == 1
        assert results[0].name == "yolov8n"

    def test_search_by_tag(self):
        registry = ModelRegistry()
        results = registry.search("detection")
        assert any(r.name == "yolov8n" for r in results)

    def test_search_by_description(self):
        registry = ModelRegistry()
        results = registry.search("speech")
        assert any(r.name == "whisper-tiny" for r in results)

    def test_search_no_results(self):
        registry = ModelRegistry()
        results = registry.search("quantum_teleportation")
        assert results == []

    def test_get_versions(self):
        registry = ModelRegistry()
        versions = registry.get_versions("yolov8n")
        assert 3 in versions

    def test_registry_count(self):
        registry = ModelRegistry()
        assert registry.count == 5

    def test_remove_model(self):
        registry = ModelRegistry()
        registry.remove("mobilenetv2")
        assert registry.count == 4
        with pytest.raises(KeyError):
            registry.get("mobilenetv2")


# ===========================================================================
# Deployment
# ===========================================================================


class TestDeployment:
    """Tests for model deployment automation."""

    def test_deploy_local(self):
        result = deploy_model("models/yolo.qbin", target_device="local")
        assert isinstance(result, DeployResult)
        assert result.success is True
        assert "yolo.qbin" in result.deployed_path
        assert result.deployment_time_s >= 0
        assert "local" in result.device_info

    def test_deploy_remote_ssh(self):
        result = deploy_model(
            "models/resnet.qbin",
            target_device="remote",
            ssh_host="user@edge-device.local",
        )
        assert result.success is True
        assert "user@edge-device.local" in result.deployed_path
        assert "remote" in result.device_info

    def test_deploy_remote_no_host_raises(self):
        with pytest.raises(ValueError, match="ssh_host is required"):
            deploy_model("models/resnet.qbin", target_device="remote")

    def test_deploy_android(self):
        result = deploy_model("models/yolo.qbin", target_device="android")
        assert result.success is True
        assert "/data/local/tmp/quad/" in result.deployed_path
        assert "android" in result.device_info

    def test_deploy_invalid_target_raises(self):
        with pytest.raises(ValueError, match="Invalid target_device"):
            deploy_model("models/yolo.qbin", target_device="invalid")


# ===========================================================================
# Integration with quad.runtime
# ===========================================================================


class TestRuntimeIntegration:
    """Test integration with QUAD runtime primitives (Tensor, Device)."""

    def test_device_creation_from_server(self):
        server = ModelServer(port=9000)
        server.load_model("test", "models/test.qbin", device="npu:0")
        model = server.list_models()[0]
        device = Device(model.device)
        assert device.type == "npu"
        assert device.index == 0

    def test_tensor_as_input(self):
        server = ModelServer(port=9000)
        server.load_model("resnet", "models/resnet.qbin", device="npu")
        server.start()

        # Create a Tensor and use its numpy data as input
        t = Tensor([1, 3, 224, 224], device=Device("npu"), dtype="float32")
        inputs = {"image": t.to_numpy()}
        response = self.server_infer(server, inputs)
        assert response.outputs["output"].shape[0] == 1

    def server_infer(self, server, inputs):
        return server.infer("resnet", inputs)
