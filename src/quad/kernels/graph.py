"""QUAD Graphs — captured execution for sub-100us replay overhead.

A Graph captures a sequence of operations into a replayable unit. On
replay, the entire graph is dispatched in one call with minimal host-side
overhead — useful when the same op sequence runs many times (e.g. each
frame of a video pipeline).

Usage:
    with Graph.capture() as g:
        y = model_a(x)
        z = model_b(y)

    # Replay with sub-100us overhead (in real mode)
    for frame in frames:
        x.copy_from(frame)
        g.replay()
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GraphNode:
    """A single operation captured in a QUAD Graph.

    Attributes:
        name: Human-readable name for the operation.
        callable: The function/kernel that was captured.
        args: Positional arguments to the callable.
        kwargs: Keyword arguments to the callable.
        result: The result produced when this node executes.
    """

    name: str
    callable: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    result: Any = None


class Graph:
    """Captured execution graph for minimal-overhead replay.

    Captures a sequence of operations during a recording phase, then
    replays them efficiently. In mock mode, replay re-executes the
    operations sequentially. In real mode, the captured graph is
    submitted to hardware as a single command buffer.

    Attributes:
        nodes: List of captured GraphNode operations.
        is_captured: Whether capture has completed.
        num_nodes: Number of operations in the graph.
    """

    # Thread-local storage for the active capture context
    _local = threading.local()

    def __init__(self):
        self.nodes: list[GraphNode] = []
        self.is_captured: bool = False

    @property
    def num_nodes(self) -> int:
        """Number of captured operations in this graph."""
        return len(self.nodes)

    @classmethod
    @contextmanager
    def capture(cls):
        """Context manager that captures operations into a graph.

        During the capture phase, all operations recorded via
        `Graph.record_op()` are stored in the graph instead of
        being executed. After capture, `replay()` re-executes them.

        Yields:
            The Graph instance being captured into.

        Usage:
            with Graph.capture() as g:
                y = some_operation(x)
                z = another_operation(y)

            g.replay()  # re-execute captured operations
        """
        graph = cls()
        cls._set_active(graph)
        try:
            yield graph
        finally:
            cls._clear_active()
            graph.is_captured = True

    @classmethod
    def _set_active(cls, graph: Graph) -> None:
        """Set the currently active capture graph (thread-local)."""
        cls._local.active_graph = graph

    @classmethod
    def _clear_active(cls) -> None:
        """Clear the active capture graph."""
        cls._local.active_graph = None

    @classmethod
    def get_active(cls) -> Graph | None:
        """Get the currently active capture graph, or None if not capturing."""
        return getattr(cls._local, "active_graph", None)

    @classmethod
    def is_capturing(cls) -> bool:
        """Return True if currently in a capture context."""
        return cls.get_active() is not None

    def add_node(self, name: str, func: Callable, args: tuple = (), kwargs: dict | None = None) -> Any:
        """Record an operation during capture and execute it.

        During capture, the operation is both recorded and executed
        (so that dependent operations get real values to work with).

        Args:
            name: Name for the operation node.
            func: The callable to record.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            The result of executing the function.
        """
        if kwargs is None:
            kwargs = {}

        # Execute the function to get a real result
        result = func(*args, **kwargs)

        node = GraphNode(
            name=name,
            callable=func,
            args=args,
            kwargs=kwargs,
            result=result,
        )
        self.nodes.append(node)
        return result

    def replay(self) -> None:
        """Replay all captured operations in order.

        In mock mode, this re-executes each operation sequentially.
        In real mode, this would submit the entire graph as a single
        hardware command buffer for minimal dispatch overhead.

        Raises:
            RuntimeError: If the graph has not been captured yet.
        """
        if not self.is_captured:
            raise RuntimeError("Cannot replay a graph that has not been captured. "
                               "Use 'with Graph.capture() as g:' first.")

        for node in self.nodes:
            node.result = node.callable(*node.args, **node.kwargs)

    def reset(self) -> None:
        """Clear all captured nodes and reset the graph."""
        self.nodes.clear()
        self.is_captured = False

    def __repr__(self) -> str:
        status = "captured" if self.is_captured else "empty"
        return f"Graph(nodes={self.num_nodes}, status='{status}')"


def record_op(name: str, func: Callable, *args, **kwargs) -> Any:
    """Record an operation if a graph capture is active, otherwise execute directly.

    This is the integration point for QUAD operations. Call this function
    around any operation that should be capturable into a Graph.

    Args:
        name: Operation name.
        func: The function to execute/record.
        *args: Arguments to the function.
        **kwargs: Keyword arguments to the function.

    Returns:
        The result of the function execution.
    """
    active_graph = Graph.get_active()
    if active_graph is not None:
        return active_graph.add_node(name, func, args, kwargs)
    else:
        return func(*args, **kwargs)
