from robot_control.robot_connection import *
import logging
import numpy as np
import asyncio

def robot_exception_cb(robot_exception : RobotException):
        print(f"time stamp: {robot_exception.time_stamp}")
        print(f"source: {robot_exception.exception_source}")
        if robot_exception.exception_source == 10:
            print(f"\tscript line: {robot_exception.script_line}")
            print(f"\tscript line: {robot_exception.script_column}")
            print(f"\tscript line: {robot_exception.description}")
        else:
            print(f"\tcode: {robot_exception.code}")
            print(f"\tsub-code: {robot_exception.subcode}")
            print(f"\tlevel: {robot_exception.level}")
            print(f"\tlevel: {robot_exception.data}")

async def get_robot_pose(robot, host, port):
    # When robot throw exception, this function will be called
    

    # robot = Robot('../../data/robot_arm/RobotStateMessage.xlsx', 'v2.6.0', robot_exception_cb)
    await robot.connect(host, port)

    # sample_count = samples
    np.set_printoptions(suppress=True)
    pose = None

    while pose == None:
        data = await robot.get_data()
        if data is None:
            logging.warning("Data is None")
            continue
        pose = [
            data.tcp_x * 1000,
            data.tcp_y * 1000,
            data.tcp_z * 1000,
            data.rot_x,
            data.rot_y,
            data.rot_z
        ]
        # sample_count -= 1
        # time.sleep(0.1)

    return pose

async def main():
    robot_client = AsyncRobotMessageConnection('../../data/robot_arm/RobotStateMessage.xlsx', 'v2.6.0', robot_exception_cb)
    # robot_client.connect('192.168.0.105', 30001)
    pose = await get_robot_pose(robot_client, '192.168.0.105', 30001)
    print(pose)

if __name__ == "__main__":
    anyio.run(main)
    


