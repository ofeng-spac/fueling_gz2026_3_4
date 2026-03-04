import anyio
import numpy as np

from ..task import run_sync
from ..minima.load_model import initialize_matcher


class MinimaMatcherService:
    def __init__(self, weight_path: str, max_parallel=4):
        self.model = initialize_matcher(ckpt_path=weight_path)

        self.send_stream, self.receive_stream = anyio.create_memory_object_stream()

        self.max_parallel = max_parallel
        self.sem_limit = anyio.Semaphore(self.max_parallel)


    async def process_item(self, item):
        event, left, right, result = (
            item['event'], item['left'], item['right'], item['result']
        )

        match_res = await run_sync(self.model, left, right)

        result['match_res'] = match_res
        event.set()

    async def loop_process_items(self):
        async with anyio.create_task_group() as tg:
            async with self.receive_stream:
                async for item in self.receive_stream:
                    async with self.sem_limit:
                        tg.start_soon(self.process_item, item)


    async def match(self, left: np.ndarray, right: np.ndarray) -> dict:
        event = anyio.Event()
        item = {
            'event': event,
            'left': left,
            'right': right,
            'result': {},
        }
        await self.send_stream.send(item)
        await event.wait()

        match_res = item['result']['match_res']
        return match_res
