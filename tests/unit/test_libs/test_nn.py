"""Tests for quad.libs.nn — Neural network primitives."""

import pytest

from quad.libs.nn import (
    Conv2d,
    Linear,
    MultiHeadAttention,
    LayerNorm,
    FusedConvBnRelu,
    FlashAttention,
    list_ops,
)
from quad.runtime.tensor import Tensor


class TestConv2d:
    """Tests for Conv2d operation."""

    def test_basic_conv(self):
        conv = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        x = Tensor.rand(1, 3, 224, 224, device="npu")
        out = conv(x)
        assert out.shape == (1, 64, 224, 224)

    def test_conv_no_padding(self):
        conv = Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        x = Tensor.rand(1, 3, 224, 224, device="npu")
        out = conv(x)
        assert out.shape == (1, 64, 222, 222)

    def test_conv_stride(self):
        conv = Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        x = Tensor.rand(2, 3, 224, 224, device="npu")
        out = conv(x)
        assert out.shape == (2, 64, 112, 112)

    def test_conv_different_kernel(self):
        conv = Conv2d(16, 32, kernel_size=(5, 5), stride=1, padding=2)
        x = Tensor.rand(1, 16, 56, 56, device="npu")
        out = conv(x)
        assert out.shape == (1, 32, 56, 56)

    def test_conv_invalid_input_dims(self):
        conv = Conv2d(3, 64, kernel_size=3)
        x = Tensor.rand(3, 224, 224, device="npu")
        with pytest.raises(ValueError, match="expects 4D input"):
            conv(x)

    def test_conv_wrong_channels(self):
        conv = Conv2d(3, 64, kernel_size=3)
        x = Tensor.rand(1, 16, 224, 224, device="npu")
        with pytest.raises(ValueError, match="Expected 3 input channels"):
            conv(x)

    def test_conv_on_cpu(self):
        conv = Conv2d(3, 64, kernel_size=3, padding=1, device="cpu")
        x = Tensor.rand(1, 3, 32, 32, device="cpu")
        out = conv(x)
        assert out.shape == (1, 64, 32, 32)
        assert out.device.type == "cpu"

    def test_conv_metrics(self):
        conv = Conv2d(3, 64, kernel_size=3, padding=1)
        x = Tensor.rand(1, 3, 8, 8, device="npu")
        conv(x)
        assert conv.metrics.flops > 0
        assert conv.metrics.memory_bytes > 0


class TestLinear:
    """Tests for Linear (fully connected) operation."""

    def test_basic_linear(self):
        fc = Linear(512, 1000)
        x = Tensor.rand(1, 512, device="npu")
        out = fc(x)
        assert out.shape == (1, 1000)

    def test_linear_batch(self):
        fc = Linear(256, 128)
        x = Tensor.rand(32, 256, device="npu")
        out = fc(x)
        assert out.shape == (32, 128)

    def test_linear_3d_input(self):
        fc = Linear(64, 32)
        x = Tensor.rand(4, 16, 64, device="npu")
        out = fc(x)
        assert out.shape == (4, 16, 32)

    def test_linear_wrong_features(self):
        fc = Linear(512, 1000)
        x = Tensor.rand(1, 256, device="npu")
        with pytest.raises(ValueError, match="Expected last dim 512"):
            fc(x)

    def test_linear_on_gpu(self):
        fc = Linear(128, 64, device="gpu")
        x = Tensor.rand(8, 128, device="gpu")
        out = fc(x)
        assert out.shape == (8, 64)
        assert out.device.type == "gpu"

    def test_linear_metrics(self):
        fc = Linear(512, 1000)
        x = Tensor.rand(1, 512, device="npu")
        fc(x)
        assert fc.metrics.flops == 2 * 1 * 1000 * 512


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_self_attention(self):
        mha = MultiHeadAttention(embed_dim=512, num_heads=8)
        x = Tensor.rand(2, 128, 512, device="npu")
        out = mha(x)
        assert out.shape == (2, 128, 512)

    def test_cross_attention(self):
        mha = MultiHeadAttention(embed_dim=256, num_heads=4)
        q = Tensor.rand(1, 64, 256, device="npu")
        k = Tensor.rand(1, 32, 256, device="npu")
        v = Tensor.rand(1, 32, 256, device="npu")
        out = mha(q, k, v)
        assert out.shape == (1, 64, 256)

    def test_invalid_embed_dim(self):
        with pytest.raises(ValueError, match="divisible by num_heads"):
            MultiHeadAttention(embed_dim=100, num_heads=3)

    def test_wrong_input_dims(self):
        mha = MultiHeadAttention(embed_dim=256, num_heads=4)
        x = Tensor.rand(256, device="npu")
        with pytest.raises(ValueError, match="Expected 3D input"):
            mha(x)

    def test_wrong_embed_dim(self):
        mha = MultiHeadAttention(embed_dim=256, num_heads=4)
        x = Tensor.rand(1, 32, 128, device="npu")
        with pytest.raises(ValueError, match="Expected embed_dim=256"):
            mha(x)


class TestLayerNorm:
    """Tests for LayerNorm."""

    def test_basic_layer_norm(self):
        ln = LayerNorm(512)
        x = Tensor.rand(2, 16, 512, device="npu")
        out = ln(x)
        assert out.shape == (2, 16, 512)

    def test_layer_norm_2d(self):
        ln = LayerNorm((32, 32))
        x = Tensor.rand(4, 3, 32, 32, device="npu")
        out = ln(x)
        assert out.shape == (4, 3, 32, 32)

    def test_layer_norm_shape_mismatch(self):
        ln = LayerNorm(512)
        x = Tensor.rand(2, 16, 256, device="npu")
        with pytest.raises(ValueError, match="Expected trailing dims"):
            ln(x)

    def test_layer_norm_on_cpu(self):
        ln = LayerNorm(128, device="cpu")
        x = Tensor.rand(4, 128, device="cpu")
        out = ln(x)
        assert out.shape == (4, 128)
        assert out.device.type == "cpu"


class TestFusedConvBnRelu:
    """Tests for FusedConvBnRelu."""

    def test_fused_basic(self):
        fused = FusedConvBnRelu(3, 64, kernel_size=3)
        x = Tensor.rand(1, 3, 224, 224, device="npu")
        out = fused(x)
        # Same padding: output H = input H
        assert out.shape == (1, 64, 224, 224)

    def test_fused_batch(self):
        fused = FusedConvBnRelu(64, 128, kernel_size=3)
        x = Tensor.rand(4, 64, 56, 56, device="npu")
        out = fused(x)
        assert out.shape == (4, 128, 56, 56)

    def test_fused_invalid_input(self):
        fused = FusedConvBnRelu(3, 64, kernel_size=3)
        x = Tensor.rand(3, 224, 224, device="npu")
        with pytest.raises(ValueError, match="expects 4D input"):
            fused(x)

    def test_fused_wrong_channels(self):
        fused = FusedConvBnRelu(3, 64, kernel_size=3)
        x = Tensor.rand(1, 16, 224, 224, device="npu")
        with pytest.raises(ValueError, match="Expected 3 input channels"):
            fused(x)

    def test_fused_metrics(self):
        fused = FusedConvBnRelu(3, 64, kernel_size=3)
        x = Tensor.rand(1, 3, 32, 32, device="npu")
        fused(x)
        assert fused.metrics.flops > 0


class TestFlashAttention:
    """Tests for FlashAttention."""

    def test_flash_self_attention(self):
        fa = FlashAttention(embed_dim=512, num_heads=8)
        x = Tensor.rand(2, 1024, 512, device="npu")
        out = fa(x)
        assert out.shape == (2, 1024, 512)

    def test_flash_cross_attention(self):
        fa = FlashAttention(embed_dim=256, num_heads=4)
        q = Tensor.rand(1, 128, 256, device="npu")
        k = Tensor.rand(1, 64, 256, device="npu")
        v = Tensor.rand(1, 64, 256, device="npu")
        out = fa(q, k, v)
        assert out.shape == (1, 128, 256)

    def test_flash_invalid_embed_dim(self):
        with pytest.raises(ValueError, match="divisible by num_heads"):
            FlashAttention(embed_dim=100, num_heads=3)

    def test_flash_memory_efficiency(self):
        """Flash attention should use O(N) memory, not O(N^2)."""
        fa = FlashAttention(embed_dim=256, num_heads=4)
        x = Tensor.rand(1, 512, 256, device="npu")
        fa(x)
        # Memory should be proportional to output size, not seq_len^2
        expected_bytes = 1 * 512 * 256 * 4  # N * seq * embed * sizeof(float32)
        assert fa.metrics.memory_bytes == expected_bytes

    def test_flash_on_gpu(self):
        fa = FlashAttention(embed_dim=128, num_heads=4, device="gpu")
        x = Tensor.rand(1, 64, 128, device="gpu")
        out = fa(x)
        assert out.device.type == "gpu"


class TestListOps:
    """Tests for list_ops function."""

    def test_returns_all_ops(self):
        ops = list_ops()
        assert "Conv2d" in ops
        assert "Linear" in ops
        assert "MultiHeadAttention" in ops
        assert "LayerNorm" in ops
        assert "FusedConvBnRelu" in ops
        assert "FlashAttention" in ops

    def test_returns_list(self):
        ops = list_ops()
        assert isinstance(ops, list)
        assert len(ops) == 6
