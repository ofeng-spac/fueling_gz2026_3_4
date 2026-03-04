import asyncio

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fueling.robot_control.robot_connection import AsyncRobotRequestConnection
from loguru import logger
async def main():
    conn = AsyncRobotRequestConnection()
    await conn.connect("192.168.0.105", 40011)
    pose = await conn.get_target_tcp_pose()
    print(pose)
    await conn.disconnect()

asyncio.run(main())