import os
import cv2
import time
import json
import anyio
import _jsonnet
from anyio.abc import TaskGroup
from pathlib import Path
from loguru import logger
from stereo_matcher.stereo_service import StereoMatcherService
logger.add("demo.log", rotation="10 MB", level="INFO")
from pathlib import Path
import cv2
import anyio 
IMG_ROOT = Path("../working_data")  
left_right_pairs = [
    (cv2.cvtColor(cv2.imread(str(l), cv2.IMREAD_ANYDEPTH), cv2.COLOR_GRAY2RGB),
     cv2.cvtColor(cv2.imread(str(r), cv2.IMREAD_ANYDEPTH), cv2.COLOR_GRAY2RGB))
    for l in sorted(IMG_ROOT.rglob("captured_left_ir.png"))
    for r in [l.parent / "captured_right_ir.png"]
    if r.exists()
]
logger.info(f"共载入 {len(left_right_pairs)} 对左右图")
async def run_fuel(arm_id: int, stereo_matcher: StereoMatcherService, tg: TaskGroup):

    idx = 0
    while True:                              
        left_ir, right_ir = left_right_pairs[idx % len(left_right_pairs)]
        t0 = time.time()
        images = {"left_ir": left_ir, "right_ir": right_ir}
        t1 = time.time()
        logger.info(f"[ARM{arm_id}] disk-read  cost {(t1-t0)*1000:.0f} ms")

        tg.start_soon(do_infer, stereo_matcher, images, idx, arm_id)
        idx += 1
        await anyio.sleep(0)                     # 20 Hz 投喂，可改 0


async def do_infer(matcher, images, idx, arm_id):
    t0 = time.time()
    disp = await matcher.infer(images["left_ir"], images["right_ir"], pred_mode="left", bidir_verify_th=0)
    t1 = time.time()
    logger.info(
        f"[ARM{arm_id}] infer #{idx:03d}  finished  cost {(t1 - t0) * 1000:.0f} ms"
    )

async def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    arm_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('arm')])
    config_files = [os.path.join(data_dir, arm_dir, "config.jsonnet") for arm_dir in arm_dirs]
    config_files = [f for f in config_files if os.path.isfile(f)][:2]  # 暂时2台

    proj_dir = Path(__file__).resolve().parent.parent

    model_name = "unimatch"  # 或从 config 里读
    model_path = json.loads(_jsonnet.evaluate_file(config_files[0]))['stereo_matcher'][model_name]['model_path']
    weight_path = f"{proj_dir}{model_path}"

    stereo_matcher = StereoMatcherService(model_name, weight_path)

    if not config_files:
        logger.error("没有找到有效的配置文件")
        return 1
    logger.info(f"Found config files: {config_files}")
    configs = []
    for i, config_file in enumerate(config_files):
        config = json.loads(_jsonnet.evaluate_file(config_file))
        configs.append(config)


    # 延迟错开启动（秒）
    delays = [0.0, 0.0]  # 第0台立即启动，第1台延迟1.0秒

    async def run_with_delay(i, delay):
        if delay > 0:
            logger.info(f"机械臂 {i} 延迟 {delay}s 启动")
            await anyio.sleep(delay)
        await run_fuel(i, stereo_matcher, tg)

    async with anyio.create_task_group() as tg:
        tg.start_soon(stereo_matcher.loop_process_items)
        for i, delay in enumerate(delays):
            logger.info(f"Starting task {i} for {config_files[i]}")
            tg.start_soon(run_with_delay, i, delay)

if __name__ == '__main__':
    anyio.run(main, backend='asyncio')
