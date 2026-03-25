"""Background async event loop for discrete-time adapters.

Multiple audio-native adapters need to bridge synchronous adapter methods
(``connect``, ``run_tick``, ``disconnect``) with an asynchronous provider
layer.  This module provides a thin helper that manages a dedicated
``asyncio`` event loop running in a daemon thread and exposes a blocking
``run_coroutine`` method for scheduling work on it.
"""

import asyncio
import threading
import time
from typing import Any, Optional, TypeVar

T = TypeVar("T")


class BackgroundAsyncLoop:
    """Manages a background thread running an asyncio event loop.

    Usage::

        bg = BackgroundAsyncLoop()
        bg.start()

        result = bg.run_coroutine(some_async_fn(), timeout=30.0)

        bg.stop()
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """The underlying event loop, or ``None`` if not started."""
        return self._loop

    @property
    def is_running(self) -> bool:
        """``True`` when the background loop is alive and usable."""
        return self._loop is not None

    def start(self) -> None:
        """Start the background thread and event loop.

        This is a no-op if the loop is already running.
        """
        if self._loop is not None:
            return

        def _run_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()

        while self._loop is None:
            time.sleep(0.01)

    def stop(self) -> None:
        """Stop the background event loop and join the thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self._loop = None
            self._thread = None

    def run_coroutine(self, coro: Any, timeout: float = 30.0) -> T:
        """Schedule *coro* on the background loop and block until it completes.

        Args:
            coro: An awaitable to execute on the background loop.
            timeout: Maximum seconds to wait for the result.

        Returns:
            The value returned by *coro*.

        Raises:
            RuntimeError: If the background loop is not running.
            Exception: Any exception raised by *coro*.
        """
        if self._loop is None:
            raise RuntimeError(
                "BackgroundAsyncLoop is not running. Call start() first."
            )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)
