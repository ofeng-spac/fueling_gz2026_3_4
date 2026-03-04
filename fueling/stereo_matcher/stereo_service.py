import anyio
import numpy as np

from ..task import run_sync
from .stereo import inference_stereo
from .unimatch.UniMatchStereo import UniMatchStereo
from .raft.RAFTStereo import RAFTStereoInference
from .defom.DefomStereo import DEFOMStereoInference
from .bridgedepth.BridgeDepthStereo import BridgeDepthStereo

MODEL_MAP = {
    "unimatch": UniMatchStereo,
    "raft": RAFTStereoInference,
    "defom": DEFOMStereoInference,
    "bridgedepth": BridgeDepthStereo,
}


class StereoMatcherService:
    def __init__(self, model: str, weight_path: str, max_parallel=4):
        model_cls = MODEL_MAP[model]
        self.model = model_cls(weight_path=weight_path)

        self.send_stream, self.receive_stream = anyio.create_memory_object_stream()

        self.max_parallel = max_parallel
        self.sem_limit = anyio.Semaphore(self.max_parallel)


    async def process_item(self, item):
        # Correct unpacking order: ensure `result` is the dict passed in `item`.
        event, left, right, pred_mode, bidir_verify_th, result = (
            item['event'], item['left'], item['right'], item['pred_mode'], item['bidir_verify_th'], item['result']
        )
        disp = await run_sync(inference_stereo,
            stereo_matcher=self.model,
            left_img=left,
            right_img=right,
            pred_mode=pred_mode,
            bidir_verify_th=bidir_verify_th
        )

        result['disp'] = disp
        event.set()

    async def loop_process_items(self):
        async with anyio.create_task_group() as tg:
            async with self.receive_stream:
                async for item in self.receive_stream:
                    async with self.sem_limit:
                        tg.start_soon(self.process_item, item)


    async def infer(self, left: np.ndarray, right: np.ndarray, pred_mode: str, bidir_verify_th: int) -> np.ndarray:
        event = anyio.Event()
        item = {
            'event': event,
            'left': left,
            'right': right,
            'pred_mode': pred_mode,
            'bidir_verify_th': bidir_verify_th,
            'result': {},
        }
        await self.send_stream.send(item)
        await event.wait()

        disp = item['result']['disp']
        # if isinstance(disp, dict):
        #     disp = disp.get('disparity_left', disp.get('disparity'))
        return disp