import anyio
from functools import partial

import anyio.to_thread

async def run_sync(fn, *args, **kws):
    """
    Run a synchronous function `fn` in a separate thread to avoid blocking the asyncio event loop.
    `*args` and `**kws` are passed to the function.
    """
    return await anyio.to_thread.run_sync(partial(fn, *args, **kws))
