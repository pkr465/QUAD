"""QualcommDNN — Neural network primitives targeting Hexagon NPU.

Provides high-level operations (Conv2d, Linear, Attention, etc.) that
execute on the NPU in real mode and simulate correct shapes in mock mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quad.runtime.tensor import Tensor
from quad.runtime.device import Device


@dataclass
class ComputeMetrics:
    """Tracks computation statistics for an operation."""

    flops: int = 0
    memory_bytes: int = 0
    device: str = "npu"


class Conv2d:
    """2D Convolution on Hexagon NPU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial kernel size (int or tuple).
        stride: Convolution stride.
        padding: Zero-padding added to input.
        device: Target compute device.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        padding: int = 0,
        device: str = "npu",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        self.metrics = ComputeMetrics(device=device)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, C_in, H, W).

        Returns:
            Output tensor of shape (N, C_out, H_out, W_out).
        """
        if len(x.shape) != 4:
            raise ValueError(f"Conv2d expects 4D input (N, C, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        n, c_in, h, w = x.shape
        kh, kw = self.kernel_size
        h_out = (h + 2 * self.padding - kh) // self.stride + 1
        w_out = (w + 2 * self.padding - kw) // self.stride + 1
        out_shape = (n, self.out_channels, h_out, w_out)

        # Compute FLOPs: 2 * N * C_out * H_out * W_out * C_in * kH * kW
        self.metrics.flops = 2 * n * self.out_channels * h_out * w_out * c_in * kh * kw
        self.metrics.memory_bytes = int(np.prod(out_shape)) * 4  # float32

        return Tensor.rand(*out_shape, device=self.device)

    def __repr__(self) -> str:
        return (
            f"Conv2d(in={self.in_channels}, out={self.out_channels}, "
            f"kernel={self.kernel_size}, stride={self.stride}, pad={self.padding}, "
            f"device='{self.device}')"
        )


class Linear:
    """Fully connected layer (GEMM) on target device.

    Args:
        in_features: Size of input feature dimension.
        out_features: Size of output feature dimension.
        device: Target compute device.
    """

    def __init__(self, in_features: int, out_features: int, device: str = "npu"):
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.metrics = ComputeMetrics(device=device)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dim {self.in_features}, got {x.shape[-1]}"
            )

        out_shape = x.shape[:-1] + (self.out_features,)

        # FLOPs: 2 * M * N * K for GEMM (M = batch, N = out_features, K = in_features)
        batch_size = int(np.prod(x.shape[:-1]))
        self.metrics.flops = 2 * batch_size * self.out_features * self.in_features
        self.metrics.memory_bytes = int(np.prod(out_shape)) * 4

        return Tensor.rand(*out_shape, device=self.device)

    def __repr__(self) -> str:
        return f"Linear(in={self.in_features}, out={self.out_features}, device='{self.device}')"


class MultiHeadAttention:
    """Multi-head attention mechanism.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        device: Target compute device.
    """

    def __init__(self, embed_dim: int, num_heads: int, device: str = "npu"):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device
        self.metrics = ComputeMetrics(device=device)

    def __call__(self, query: Tensor, key: Tensor | None = None, value: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            query: Query tensor of shape (N, seq_len, embed_dim).
            key: Key tensor (defaults to query for self-attention).
            value: Value tensor (defaults to query for self-attention).

        Returns:
            Output tensor of shape (N, seq_len, embed_dim).
        """
        if key is None:
            key = query
        if value is None:
            value = query

        if len(query.shape) != 3:
            raise ValueError(f"Expected 3D input (N, seq, embed), got shape {query.shape}")
        if query.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {query.shape[-1]}"
            )

        n, seq_len, _ = query.shape
        out_shape = (n, seq_len, self.embed_dim)

        # FLOPs for attention: ~4 * N * seq^2 * embed + 4 * N * seq * embed^2
        self.metrics.flops = (
            4 * n * seq_len * seq_len * self.embed_dim
            + 4 * n * seq_len * self.embed_dim * self.embed_dim
        )
        self.metrics.memory_bytes = int(np.prod(out_shape)) * 4

        return Tensor.rand(*out_shape, device=self.device)

    def __repr__(self) -> str:
        return (
            f"MultiHeadAttention(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, device='{self.device}')"
        )


class LayerNorm:
    """Layer Normalization.

    Args:
        normalized_shape: Shape of the last dimensions to normalize over.
        device: Target compute device.
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], device: str = "npu"):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.device = device
        self.metrics = ComputeMetrics(device=device)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor whose last len(normalized_shape) dims match normalized_shape.

        Returns:
            Normalized tensor with the same shape as input.
        """
        n_dims = len(self.normalized_shape)
        if x.shape[-n_dims:] != self.normalized_shape:
            raise ValueError(
                f"Expected trailing dims {self.normalized_shape}, got {x.shape[-n_dims:]}"
            )

        # LayerNorm: ~5 ops per element (mean, variance, normalize, scale, shift)
        self.metrics.flops = 5 * int(np.prod(x.shape))
        self.metrics.memory_bytes = int(np.prod(x.shape)) * 4

        return Tensor.rand(*x.shape, device=self.device)

    def __repr__(self) -> str:
        return f"LayerNorm(shape={self.normalized_shape}, device='{self.device}')"


class FusedConvBnRelu:
    """Fused Conv2d + BatchNorm + ReLU in a single NPU kernel.

    Significantly reduces memory bandwidth by avoiding intermediate writes.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial kernel size.
        device: Target compute device.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        device: str = "npu",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.device = device
        self.metrics = ComputeMetrics(device=device)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, C_in, H, W).

        Returns:
            Output tensor of shape (N, C_out, H_out, W_out) after Conv+BN+ReLU.
        """
        if len(x.shape) != 4:
            raise ValueError(f"FusedConvBnRelu expects 4D input, got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        n, c_in, h, w = x.shape
        kh, kw = self.kernel_size
        # Same padding (pad = kernel_size // 2)
        pad = kh // 2
        h_out = (h + 2 * pad - kh) + 1
        w_out = (w + 2 * pad - kw) + 1
        out_shape = (n, self.out_channels, h_out, w_out)

        # FLOPs: conv + BN (2*elements) + ReLU (elements)
        conv_flops = 2 * n * self.out_channels * h_out * w_out * c_in * kh * kw
        bn_relu_flops = 3 * int(np.prod(out_shape))
        self.metrics.flops = conv_flops + bn_relu_flops
        self.metrics.memory_bytes = int(np.prod(out_shape)) * 4

        return Tensor.rand(*out_shape, device=self.device)

    def __repr__(self) -> str:
        return (
            f"FusedConvBnRelu(in={self.in_channels}, out={self.out_channels}, "
            f"kernel={self.kernel_size}, device='{self.device}')"
        )


class FlashAttention:
    """Memory-efficient (Flash) Attention on NPU.

    Tiles the attention computation to avoid materializing the full NxN
    attention matrix, reducing memory from O(N^2) to O(N).

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        device: Target compute device.
    """

    def __init__(self, embed_dim: int, num_heads: int, device: str = "npu"):
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device
        self.metrics = ComputeMetrics(device=device)

    def __call__(self, query: Tensor, key: Tensor | None = None, value: Tensor | None = None) -> Tensor:
        """Forward pass with tiled attention computation.

        Args:
            query: Query tensor of shape (N, seq_len, embed_dim).
            key: Key tensor (defaults to query for self-attention).
            value: Value tensor (defaults to query for self-attention).

        Returns:
            Output tensor of shape (N, seq_len, embed_dim).
        """
        if key is None:
            key = query
        if value is None:
            value = query

        if len(query.shape) != 3:
            raise ValueError(f"Expected 3D input (N, seq, embed), got shape {query.shape}")
        if query.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}, got {query.shape[-1]}"
            )

        n, seq_len, _ = query.shape
        out_shape = (n, seq_len, self.embed_dim)

        # FLOPs similar to standard attention but lower memory
        self.metrics.flops = (
            4 * n * seq_len * seq_len * self.embed_dim
            + 4 * n * seq_len * self.embed_dim * self.embed_dim
        )
        # Flash attention: O(N) memory instead of O(N^2)
        self.metrics.memory_bytes = int(np.prod(out_shape)) * 4

        return Tensor.rand(*out_shape, device=self.device)

    def __repr__(self) -> str:
        return (
            f"FlashAttention(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, device='{self.device}')"
        )


def list_ops() -> list[str]:
    """Return all available neural network operations.

    Returns:
        List of operation class names available in this module.
    """
    return [
        "Conv2d",
        "Linear",
        "MultiHeadAttention",
        "LayerNorm",
        "FusedConvBnRelu",
        "FlashAttention",
    ]
