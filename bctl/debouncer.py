import asyncio
from asyncio import Task
from collections.abc import Awaitable, Callable
from typing import Any


# note only the last execution gets invoked
class Debouncer:
    def __init__(self, delay: float) -> None:
        self.delay: float = delay
        self.task: Task[None] | None = None

    async def __call__[**P](
        self, func: Callable[P, Awaitable[Any]], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = asyncio.create_task(self._debounced_call(func, *args, **kwargs))

    async def _debounced_call[**P](
        self, func: Callable[P, Awaitable[Any]], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        try:
            await asyncio.sleep(self.delay)
            await func(*args, **kwargs)
        except asyncio.CancelledError:
            pass  # ignore cancelled calls  # ignore cancelled calls
