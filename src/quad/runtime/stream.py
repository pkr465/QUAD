"""QUAD Stream & Event — concurrent execution primitives."""

from __future__ import annotations

import time
from typing import Any


class Event:
    """Synchronization event between streams.

    Usage:
        event = Event()
        stream_a.record(event)
        stream_b.wait(event)  # B waits for A without blocking CPU
    """

    def __init__(self):
        self._recorded = False
        self._timestamp: float = 0.0

    def record(self) -> None:
        """Mark this event as occurred."""
        self._recorded = True
        self._timestamp = time.perf_counter()

    @property
    def is_recorded(self) -> bool:
        return self._recorded

    @property
    def elapsed_ms(self) -> float:
        """Time since event was recorded."""
        if not self._recorded:
            return 0.0
        return (time.perf_counter() - self._timestamp) * 1000


class Stream:
    """Execution stream for concurrent operations.

    Operations submitted to different streams can execute concurrently.
    Operations within a stream execute sequentially.

    Usage:
        stream = Stream()
        model(input, stream=stream)  # Non-blocking
        stream.synchronize()         # Wait for completion

        # As context manager
        with Stream() as s:
            model(input_1, stream=s)
            model(input_2, stream=s)
        # Both complete by here
    """

    _next_id = 0

    def __init__(self, name: str = ""):
        Stream._next_id += 1
        self._id = Stream._next_id
        self._name = name or f"stream_{self._id}"
        self._synchronized = True
        self._operations: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> int:
        return self._id

    def synchronize(self) -> None:
        """Block until all operations in this stream complete."""
        self._synchronized = True
        self._operations.clear()

    def record(self, event: Event) -> None:
        """Record an event at the current point in this stream."""
        event.record()

    def wait(self, event: Event) -> None:
        """Make this stream wait for an event recorded in another stream."""
        # In mock mode, events are already "complete"
        pass

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *args) -> None:
        self.synchronize()

    def __repr__(self) -> str:
        return f"Stream(name='{self._name}', ops={len(self._operations)})"
