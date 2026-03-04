#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人位姿获取和移动功能测试
主要测试:
1. 获取机器人当前位姿
2. 移动机器人到目标位置（fueling_pose）
"""

import sys
import os
import time
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fueling.robot_control.robot_connection import RobotClient, AsyncRobotClient
from loguru import logger

# 配置日志
logging.basicConfig(level=logging.INFO)
logger.add("robot_test.log", rotation="10 MB")

class RobotPoseAndMoveTest:
    """机器人位姿和移动测试类"""

    def __init__(self):
        # 机器人配置
        self.robot_ip = "192.168.1.105"  # 请根据实际情况修改IP地址
        self.robot_port = 30001

        # Excel配置文件路径
        self.excel_path = str(project_root / 'data' / 'robot_arm' / 'RobotStateMessage.xlsx')
        self.sheet_name = 'v2.6.0'

        # 目标位置 - fueling_pose（位置单位：毫米，角度单位：弧度）
        self.target_pose = [
            -143.75941746241833,
            609.5533240608562,
            847.6109283703697,
            -0.05628666479505898,
            -0.006824646946384569,
            -3.111639076663332
        ]

        self.robot_client = None

    def setup_robot_client(self):
        """初始化机器人客户端"""
        try:
            logger.info("初始化机器人客户端...")
            self.robot_client = RobotClient(
                excel=self.excel_path,
                sheet=self.sheet_name,
                timeout=10.0
            )
            logger.info("机器人客户端初始化成功")
            return True
        except Exception as e:
            logger.error(f"初始化机器人客户端失败: {e}")
            return False

    def connect_robot(self):
        """连接机器人"""
        try:
            logger.info(f"连接机器人 {self.robot_ip}:{self.robot_port}...")
            self.robot_client.connect(self.robot_ip, self.robot_port)
            logger.info("机器人连接成功")
            return True
        except Exception as e:
            logger.error(f"连接机器人失败: {e}")
            return False

    def disconnect_robot(self):
        """断开机器人连接"""
        try:
            if self.robot_client:
                self.robot_client.disconnect()
                logger.info("机器人连接已断开")
        except Exception as e:
            logger.error(f"断开机器人连接时出错: {e}")

    def test_get_current_pose(self):
        """测试获取机器人当前位姿"""
        logger.info("=== 测试获取机器人当前位姿 ===")

        try:
            # 获取当前位姿
            current_pose = self.robot_client.get_robot_pose()

            if current_pose is not None:
                logger.info("成功获取机器人当前位姿:")
                logger.info(f"  位置 (mm): X={current_pose[0]:.2f}, Y={current_pose[1]:.2f}, Z={current_pose[2]:.2f}")
                logger.info(f"  姿态 (rad): RX={current_pose[3]:.4f}, RY={current_pose[4]:.4f}, RZ={current_pose[5]:.4f}")
                return current_pose
            else:
                logger.error("获取机器人位姿失败")
                return None

        except Exception as e:
            logger.error(f"获取机器人位姿时发生异常: {e}")
            return None

    def test_move_to_target_pose(self, use_joint_move=False):
        """测试移动机器人到目标位置

        Args:
            use_joint_move (bool): True使用关节运动，False使用直线运动
        """
        move_type = "关节运动" if use_joint_move else "直线运动"
        logger.info(f"=== 测试{move_type}到目标位置 ===")

        try:
            logger.info("目标位置 (fueling_pose):")
            logger.info(f"  位置 (mm): X={self.target_pose[0]:.2f}, Y={self.target_pose[1]:.2f}, Z={self.target_pose[2]:.2f}")
            logger.info(f"  姿态 (rad): RX={self.target_pose[3]:.4f}, RY={self.target_pose[4]:.4f}, RZ={self.target_pose[5]:.4f}")

            # 选择运动类型
            command = "joint" if use_joint_move else "move"

            # 设置运动参数
            if use_joint_move:
                params = {"a": 0.8, "v": 0.3, "t": 0, "r": 0}  # 关节运动参数
                logger.info(f"关节运动参数: 加速度={params['a']}, 速度={params['v']}")
            else:
                params = {"a": 0.5, "v": 0.2, "t": 0, "r": 0}  # 直线运动参数
                logger.info(f"直线运动参数: 加速度={params['a']}, 速度={params['v']}")

            # 执行运动
            logger.info(f"开始执行{move_type}...")

            if use_joint_move:
                self.robot_client.control_robot(
                    command=command,
                    poses=self.target_pose,
                    joint_sequence_params=params
                )
            else:
                self.robot_client.control_robot(
                    command=command,
                    poses=self.target_pose,
                    move_sequence_params=params
                )

            logger.info(f"{move_type}指令已发送")

            # 等待运动完成
            logger.info("等待运动完成...")
            time.sleep(3)  # 给机器人一些时间开始运动

            # 检查是否到达目标位置
            self.verify_target_reached()

            return True

        except Exception as e:
            logger.error(f"执行{move_type}时发生异常: {e}")
            return False

    def verify_target_reached(self, tolerance=5.0):
        """验证是否到达目标位置

        Args:
            tolerance (float): 位置误差容忍度（毫米）
        """
        logger.info("=== 验证是否到达目标位置 ===")

        try:
            # 等待机器人稳定
            time.sleep(2)

            # 获取当前位姿
            current_pose = self.robot_client.get_robot_pose()

            if current_pose is None:
                logger.error("无法获取当前位姿进行验证")
                return False

            # 计算位置误差
            pos_error = [
                abs(current_pose[i] - self.target_pose[i]) for i in range(3)
            ]
            max_pos_error = max(pos_error)

            # 计算姿态误差（弧度）
            rot_error = [
                abs(current_pose[i] - self.target_pose[i]) for i in range(3, 6)
            ]
            max_rot_error = max(rot_error)

            logger.info("位置验证结果:")
            logger.info(f"  当前位置 (mm): X={current_pose[0]:.2f}, Y={current_pose[1]:.2f}, Z={current_pose[2]:.2f}")
            logger.info(f"  目标位置 (mm): X={self.target_pose[0]:.2f}, Y={self.target_pose[1]:.2f}, Z={self.target_pose[2]:.2f}")
            logger.info(f"  位置误差 (mm): X={pos_error[0]:.2f}, Y={pos_error[1]:.2f}, Z={pos_error[2]:.2f}")
            logger.info(f"  最大位置误差: {max_pos_error:.2f} mm")
            logger.info(f"  最大姿态误差: {max_rot_error:.4f} rad")

            if max_pos_error <= tolerance:
                logger.info(f"✓ 成功到达目标位置（误差 {max_pos_error:.2f} mm <= {tolerance} mm）")
                return True
            else:
                logger.warning(f"✗ 未能精确到达目标位置（误差 {max_pos_error:.2f} mm > {tolerance} mm）")
                return False

        except Exception as e:
            logger.error(f"验证目标位置时发生异常: {e}")
            return False

    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始机器人位姿和移动功能测试")
        logger.info("=" * 50)

        # 初始化和连接
        if not self.setup_robot_client():
            logger.error("测试终止：无法初始化机器人客户端")
            return False

        if not self.connect_robot():
            logger.error("测试终止：无法连接机器人")
            return False

        try:
            # 测试1: 获取当前位姿
            logger.info("\n" + "=" * 30)
            initial_pose = self.test_get_current_pose()
            if initial_pose is None:
                logger.error("测试失败：无法获取机器人位姿")
                return False

            # 测试2: 直线运动到目标位置
            logger.info("\n" + "=" * 30)
            if not self.test_move_to_target_pose(use_joint_move=False):
                logger.error("测试失败：直线运动失败")
                return False

            # 等待一段时间
            time.sleep(2)

            # 测试3: 关节运动回到初始位置
            logger.info("\n" + "=" * 30)
            logger.info("=== 测试关节运动回到初始位置 ===")
            if not self.test_move_to_initial_pose(initial_pose):
                logger.error("测试失败：无法回到初始位置")
                return False

            logger.info("\n" + "=" * 50)
            logger.info("✓ 所有测试完成！")
            return True

        except KeyboardInterrupt:
            logger.info("\n测试被用户中断")
            return False
        except Exception as e:
            logger.error(f"测试过程中发生异常: {e}")
            return False
        finally:
            self.disconnect_robot()

    def test_move_to_initial_pose(self, initial_pose):
        """测试移动回初始位置"""
        try:
            logger.info("移动回初始位置...")
            self.robot_client.control_robot(
                command="joint",
                poses=initial_pose,
                joint_sequence_params={"a": 0.8, "v": 0.3, "t": 0, "r": 0}
            )

            time.sleep(3)
            logger.info("已发送回到初始位置的指令")
            return True

        except Exception as e:
            logger.error(f"移动回初始位置时发生异常: {e}")
            return False


def main():
    """主函数"""
    print("机器人位姿获取和移动功能测试")
    print("注意：请确保机器人处于安全状态，并且周围没有障碍物")
    print("按 Ctrl+C 可随时停止测试")

    # 询问用户是否继续
    try:
        response = input("\n是否继续测试？(y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("测试已取消")
            return
    except KeyboardInterrupt:
        print("\n测试已取消")
        return

    # 创建测试实例
    test = RobotPoseAndMoveTest()

    # 运行测试
    success = test.run_all_tests()

    if success:
        print("\n🎉 测试成功完成！")
    else:
        print("\n❌ 测试失败")


if __name__ == "__main__":
    main()