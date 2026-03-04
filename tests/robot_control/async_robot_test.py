#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步机器人测试脚本
快速测试获取位姿和移动到fueling_pose的功能（异步版本）
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fueling.robot_control.robot_connection import AsyncRobotClient
from loguru import logger

# 配置
ROBOT_IP = "192.168.0.105"  # 请根据实际情况修改
ROBOT_PORT = 30001
EXCEL_PATH = project_root / 'data' / 'robot_arm' / 'RobotStateMessage.xlsx'
SHEET_NAME = 'v2.6.0'

# 目标位置（位置单位：毫米，角度单位：弧度）
CAPTURE_POSE = [
    -146.2112061594585,
    647.1673800803261,
    592.164279477909,
    0.4049901343787822,
    -0.007139323558163404,
    -3.113358424397259
]

INIT_POSE = [
    -153.83533321249192,
    464.05186997207875,
    419.515517516193,
    0.3533969543515678,
    -0.027601198738065366,
    -3.0810262589632034
]

async def test_get_pose(robot_client):
    """测试获取机器人位姿"""
    print("\n=== 测试获取机器人位姿 ===")
    try:
        pose = await robot_client.get_robot_pose()
        if pose:
            print(f"当前位姿:")
            print(f"  位置 (mm): X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f}")
            print(f"  姿态 (rad): RX={pose[3]:.4f}, RY={pose[4]:.4f}, RZ={pose[5]:.4f}")
            return pose
        else:
            print("❌ 获取位姿失败")
            return None
    except Exception as e:
        print(f"❌ 获取位姿异常: {e}")
        return None

async def test_move_to_target_pose(robot_client, target_pose, pose_name):
    """测试移动到目标位姿"""
    print(f"\n=== 测试移动到 {pose_name} ===")
    print(f"目标位置:")
    print(f"  位置 (mm): X={target_pose[0]:.2f}, Y={target_pose[1]:.2f}, Z={target_pose[2]:.2f}")
    print(f"  姿态 (rad): RX={target_pose[3]:.4f}, RY={target_pose[4]:.4f}, RZ={target_pose[5]:.4f}")

    try:
        # 使用直线运动
        await robot_client.control_robot(
            command="move",
            poses=target_pose,
            move_sequence_params={"a": 0.3, "v": 0.1, "t": 0, "r": 0}  # 较慢的速度确保安全
        )
        print("✓ 移动指令已发送")
        return True
    except Exception as e:
        print(f"❌ 移动失败: {e}")
        return False

async def main():
    """主函数"""
    print("异步机器人位姿获取和移动测试")
    print("=" * 40)
    print(f"机器人IP: {ROBOT_IP}")
    print(f"Excel配置: {EXCEL_PATH}")
    print(f"工作表: {SHEET_NAME}")

    # 初始化机器人客户端
    try:
        print("\n初始化异步机器人客户端...")
        robot_client = AsyncRobotClient(
            excel=str(EXCEL_PATH),
            sheet=SHEET_NAME
        )
        print("✓ 客户端初始化成功")
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return

    # 连接机器人
    try:
        print(f"\n连接机器人 {ROBOT_IP}:{ROBOT_PORT}...")
        await robot_client.connect(ROBOT_IP, ROBOT_PORT)
        print("✓ 机器人连接成功")
    except Exception as e:
        print(f"❌ 机器人连接失败: {e}")
        print("请检查:")
        print("1. 机器人IP地址是否正确")
        print("2. 机器人是否开机")
        print("3. 网络连接是否正常")
        return

    try:
        # 测试1: 获取当前位姿
        initial_pose = await test_get_pose(robot_client)
        if not initial_pose:
            print("❌ 无法获取机器人位姿，测试终止")
            return

        # 定义要测试的位姿列表
        poses_to_test = {
            "CAPTURE_POSE": CAPTURE_POSE,
            "INIT_POSE": INIT_POSE
        }

        for pose_name, target_pose in poses_to_test.items():
            response = input(f"\n是否进行移动到 {pose_name} 的测试？(y/N): ")
            if response.lower() not in ['y', 'yes']:
                print(f"跳过移动到 {pose_name} 的测试")
                continue
            success = await test_move_to_target_pose(robot_client, target_pose, pose_name)

            if success:
                print("\n等待运动完成...")
                await asyncio.sleep(5)  # 等待运动完成

                # 验证是否到达目标
                final_pose = await test_get_pose(robot_client)
                if final_pose:
                    # 计算位置误差
                    pos_error = [abs(final_pose[i] - target_pose[i]) for i in range(3)]
                    max_error = max(pos_error)
                    print(f"\n位置误差: {max_error:.2f} mm")
                    if max_error < 10.0:  # 10mm容差
                        print(f"✓ 成功到达 {pose_name} 位置")
                    else:
                        print(f"⚠️  到达 {pose_name} 位置误差较大")


        print("\n=== 测试完成 ===")

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生异常: {e}")
    finally:
        # 断开连接
        try:
            await robot_client.disconnect()
            print("机器人连接已断开")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())