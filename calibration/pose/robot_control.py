import time
import numpy as np
import re
from loguru import logger
from error import RobotControlError, RecvTimeoutError
from typing import Tuple, Optional, Literal

# 导入自定义模块
from robot_connection import RobotClient


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


class SyncRobotClient:
    def __init__(self, addr: str, req_port: int, ctrl_port: int, excel_sheet: str):
        self.req_conn = RobotClient(addr, req_port, excel_sheet)
        self.ctrl_conn = RobotClient(addr, ctrl_port, excel_sheet)

    def get_target_tcp_pose(self):
        self.req_conn.send(b"req 1 get_target_tcp_pose()\n")
        response = self.req_conn.recv()
        if response is None:
            raise RecvTimeoutError(self.req_conn.timeout, f'{self.req_conn.addr}:{self.req_conn.port}')

        pose_str = response.decode('utf-8')

        numbers = [float(i) for i in re.findall(r'-?\d+\.\d+', pose_str)]
        if len(numbers) < 6:
            raise RobotControlError("Failed to parse data, data is incomplete: " + pose_str)

        return [numbers[0] * 1000, numbers[1] * 1000, numbers[2] * 1000, numbers[3], numbers[4], numbers[5]]

if __name__ == "__main__":
    robot = SyncRobotClient('192.168.0.108', 40011, 30001, './RobotStateMessage.xlsx:v2.6.0')
    pose = robot.get_target_tcp_pose()
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(" ".join(map(str, pose)) + "\n")
    print("Target TCP Pose:", pose)