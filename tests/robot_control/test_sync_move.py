import json, _jsonnet
import time
from loguru import logger
from robot_control.robot_connection import RobotClient, RobotException
from robot_control.robot_control import RobotControl

def exception_callback(robot_exception : RobotException):
        logger.info(f"time stamp: {robot_exception.time_stamp}")
        logger.info(f"source: {robot_exception.exception_source}")
        if robot_exception.exception_source == 10:
            logger.info(f"\tscrifrom .robot import *pt line: {robot_exception.script_line}")
            logger.info(f"\tscript line: {robot_exception.script_column}")
            logger.info(f"\tscript line: {robot_exception.description}")
        else:
            logger.info(f"\tcode: {robot_exception.code}")
            logger.info(f"\tsub-code: {robot_exception.subcode}")
            logger.info(f"\tlevel: {robot_exception.level}")
            logger.info(f"\tlevel: {robot_exception.data}")

def get_command(converted_poses, move_sequence_params=None):
        # control_conn.connect("192.168.0.105", 30001)
        converted_poses = [x * 0.001 for x in converted_poses[:3]] + converted_poses[3:]
        """异步执行一系列位姿的movel指令"""
        if isinstance(converted_poses[0], (int, float)) and len(converted_poses) == 6:
            converted_poses = [converted_poses]

        if move_sequence_params is None:
            move_sequence_params = {"a": 1, "v": 0.7, "t": 0, "r": 0}
        # print(move_sequence_params)
        a=move_sequence_params.get("a", 1)
        v=move_sequence_params.get("v", 0.7)
        t=move_sequence_params.get("t", 0)
        r=move_sequence_params.get("r", 0)

        # print(a,v,t,r)

        commands = []
        for pose in converted_poses:
            if len(pose) != 6:
                raise ValueError(f"位姿矩阵 {pose} 的长度必须为6")
            cmd = f"movel({pose},a={a},v={v},t={t},r={r})"
            commands.append(cmd)
        # print(commands)
        function_name = "move_pose"
        command_block = f"def {function_name}():\n"
        for cmd in commands:
            command_block += f"    {cmd}\n"
        command_block += "end\n"

        try:
            logger.info(f"发送指令:\n{command_block}")
            # control_conn.send(command_block.encode('utf-8'))
            return command_block # 避免过快发送
        except Exception as e:
            logger.error(f"发送指令失败: {e}")
            raise

async def main():
     robot_upper_pose = [-168.93, 576.71, 555.95, 0.388, -0.014, -3.079]
     move_sequence_params = {
      "a": 1,
      "r": 0.0,
      "t": 0.0,
      "v": 0.2
    }
     json_str = _jsonnet.evaluate_file("../../data/arm1/config.jsonnet")
     config = json.loads(json_str)
     robot_control = RobotControl(config, 1)
     await robot_control.connect()
     await robot_control.control_robot("move", robot_upper_pose, move_sequence_params=move_sequence_params)


if __name__ == "__main__":
    # import anyio
    # anyio.run(main)
    robot_control = RobotClient(exception_callback)
    robot_control.connect("192.168.0.105", 30001)
    robot_upper_pose = [-168.93, 576.71, 555.95, 0.388, -0.014, -3.079]
    move_sequence_params = {
        "a": 1.0,
        "r": 0.0,
        "t": 0.0,
        "v": 0.2
        }
    command = get_command(robot_upper_pose, move_sequence_params)
    # print(command)
    robot_control.send(command.encode('utf-8'))