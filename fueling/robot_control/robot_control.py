import time
import anyio
import numpy as np
import re
from loguru import logger
from ..error import RobotControlError, RecvTimeoutError
from typing import Tuple, Optional, Literal

# 导入自定义模块
from .robot_connection import AsyncRobotConnection, RobotClient

from ..pose_transformation import transform_1x6_to_4x4, transform_4x4_to_1x6

MoveType = Literal['movel', 'movej']

def rotation_angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    """用旋转矩阵求相对旋转角：angle = acos((trace(R_rel)-1)/2)"""
    R_rel = R1.T @ R2
    tr = np.trace(R_rel)
    # 数值稳定性处理
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle  # 返回弧度

def calc_pose_diff(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    x1, y1, z1 = p1[:3, 3]
    x2, y2, z2 = p2[:3, 3]
    pos_diff = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    R1 = p1[:3, :3]
    R2 = p2[:3, :3]
    rot_diff = rotation_angle_between(R1, R2)
    return pos_diff, rot_diff

class AsyncRobotClient:
    def __init__(self, addr: str, req_port: int, ctrl_port: int, type: MoveType, movel_params: dict, movej_params: dict, pos_tol: float, rot_tol: float, check_interval: float = 0.3):
        self.req_conn = AsyncRobotConnection(addr, req_port)
        self.ctrl_conn = AsyncRobotConnection(addr, ctrl_port)

        self.move_params = {
            'movel': movel_params,
            'movej': movej_params,
        }
        self.type: MoveType = type
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.check_interval = check_interval

    async def connect(self):
        await self.req_conn.connect()
        await self.ctrl_conn.connect()


    async def get_target_tcp_pose(self):
        await self.req_conn.send(b"req 1 get_target_tcp_pose()\n")
        response = await self.req_conn.recv()
        if response is None:
            raise RecvTimeoutError(self.req_conn.timeout, f'{self.req_conn.addr}:{self.req_conn.port}')

        pose_str = response.decode('utf-8')

        numbers = [float(i) for i in re.findall(r'-?\d+\.\d+', pose_str)]
        if len(numbers) < 6:
            raise RobotControlError("Failed to parse data, data is incomplete: " + pose_str)

        return transform_1x6_to_4x4(numbers)

    async def get_target_tcp_speed(self):
        await self.req_conn.send(b"req 1 get_actual_tcp_speed()\n")
        response = await self.req_conn.recv()
        if response is None:
            raise RecvTimeoutError(self.req_conn.timeout, f'{self.req_conn.addr}:{self.req_conn.port}')

        speed_str = response.decode('utf-8')

        numbers = [float(i) for i in re.findall(r'-?\d+\.\d+', speed_str)]
        if len(numbers) < 6:
            raise RobotControlError("Failed to parse data, data is incomplete: " + speed_str)

        return numbers

    async def move(self, poses: np.ndarray, type: Optional[MoveType] = None, params: Optional[dict] = None, wait_done: bool = True):
        if type is None:
            type = self.type
        if params is None:
            params = self.move_params[type]

        control_poses = transform_4x4_to_1x6(poses)
        pose_str = ", ".join(f"{v:.7f}" for v in control_poses)
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())

        script = f'''
def move_pose():
    {type}([{pose_str}], {params_str})
end
'''
        logger.info(f"发送指令:\n{script}")
        await self.ctrl_conn.send(script)
        if wait_done:
            await self.wait_for_done(poses)

    async def wait_for_done(self, target_position: np.ndarray):
        """等待机械臂到达指定位置"""
        while True:
            current_pose = await self.get_target_tcp_pose()
            current_speed = await self.get_target_tcp_speed()

            # logger.info(f"当前位置: {current_pose}, 目标位置: {target_position}")
            pos_diff, rot_diff = calc_pose_diff(current_pose, target_position)
            if pos_diff <= self.pos_tol and rot_diff <= self.rot_tol and all(s < 0.01 for s in current_speed):
                logger.debug("机械臂已到达目标位置")
                break
            # logger.debug("机械臂未到达目标位置...")
            await anyio.sleep(self.check_interval)

class SyncRobotClient:
    def __init__(self, addr: str, req_port: int, ctrl_port: int, type: MoveType, excel_sheet: str, movel_params: dict, movej_params: dict, pos_tol: float, rot_tol: float, check_interval: float = 0.3):
        self.req_conn = RobotClient(addr, req_port, excel_sheet)
        self.ctrl_conn = RobotClient(addr, ctrl_port, excel_sheet)

        self.move_params = {
            'movel': movel_params,
            'movej': movej_params,
        }
        self.type: MoveType = type
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.check_interval = check_interval

    def get_target_tcp_pose(self):
        self.req_conn.send(b"req 1 get_target_tcp_pose()\n")
        response = self.req_conn.recv()
        if response is None:
            raise RecvTimeoutError(self.req_conn.timeout, f'{self.req_conn.addr}:{self.req_conn.port}')

        pose_str = response.decode('utf-8')

        numbers = [float(i) for i in re.findall(r'-?\d+\.\d+', pose_str)]
        if len(numbers) < 6:
            raise RobotControlError("Failed to parse data, data is incomplete: " + pose_str)

        return transform_1x6_to_4x4(numbers)

    def get_target_tcp_speed(self):
        self.req_conn.send(b"req 1 get_actual_tcp_speed()\n")
        response = self.req_conn.recv()
        if response is None:
            raise RecvTimeoutError(self.req_conn.timeout, f'{self.req_conn.addr}:{self.req_conn.port}')

        speed_str = response.decode('utf-8')

        numbers = [float(i) for i in re.findall(r'-?\d+\.\d+', speed_str)]
        if len(numbers) < 6:
            raise RobotControlError("Failed to parse data, data is incomplete: " + speed_str)

        return numbers

    def move(self, poses: np.ndarray, type: Optional[MoveType] = None, params: Optional[dict] = None, wait_done: bool = True):
        if type is None:
            type = self.type
        if params is None:
            params = self.move_params[type]
        control_poses = transform_4x4_to_1x6(poses)
        pose_str = ", ".join(f"{v:.7f}" for v in control_poses)
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())

        script = f'''
def move_pose():
    {type}([{pose_str}], {params_str})
end
'''
        logger.info(f"发送指令:\n{script}")
        self.ctrl_conn.send(script)
        if wait_done:
            self.wait_for_done(poses)

    def wait_for_done(self, target_position: np.ndarray):
        """等待机械臂到达指定位置"""
        while True:
            current_pose = self.get_target_tcp_pose()
            current_speed = self.get_target_tcp_speed()

            # logger.info(f"当前位置: {current_pose}, 目标位置: {target_position}")
            pos_diff, rot_diff = calc_pose_diff(current_pose, target_position)
            if pos_diff <= self.pos_tol and rot_diff <= self.rot_tol and all(s < 0.01 for s in current_speed):
                logger.debug("机械臂已到达目标位置")
                break
            # logger.debug("机械臂未到达目标位置...")
            time.sleep(self.check_interval)

# drawing
def compute_transformed_fueling_pose(eye_hand_matrix: list, capture_pose: np.ndarray, fueling_pose: np.ndarray, T_Ca2_from_Ca1: np.ndarray):
    T_H_from_C = eye_hand_matrix
    T_C_from_H = np.linalg.inv(T_H_from_C)
    T_B_from_Ha = capture_pose
    T_B_from_Ha2 = capture_pose

    T_Ha_from_B = np.linalg.inv(T_B_from_Ha)
    T_B_from_Hb1 = fueling_pose
    T_Ha_from_Hb1 = T_Ha_from_B @ T_B_from_Hb1
    T_B_from_Hb2 = T_B_from_Ha2 @ T_H_from_C @ T_Ca2_from_Ca1 @ T_C_from_H @ T_Ha_from_Hb1
    return T_B_from_Hb2
