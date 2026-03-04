#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 RobotControl 类的 get_tcp_pose 方法
"""

import asyncio
import sys
import os
from pathlib import Path
import _jsonnet
import json

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from robot_control.robot_control import RobotControl

async def test_get_tcp_pose():
    """测试获取TCP位姿功能"""
    
    print("开始加载机器人配置...")
    # Load config from file
    base_dir = Path(__file__).parent.parent # This should point to /home/vision/projects/fueling
    config_file_path = base_dir / '..'/ "data" / "arm1" / "config.jsonnet" # Assuming this path for a sample config

    if not config_file_path.exists():
        print(f"错误: 配置文件未找到: {config_file_path}")
        print("请确保 /home/vision/projects/fueling/data/arm1/config.jsonnet 存在，或者修改测试脚本中的路径。")
        return

    try:
        config_str = open(config_file_path).read()
        config = json.loads(_jsonnet.evaluate_snippet(str(config_file_path), config_str))
        print(f"成功加载配置文件: {config_file_path}")
    except Exception as e:
        print(f"加载配置文件时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    robot_id = 0 # Assuming robot_id is not directly in the config file for this test, or can be hardcoded.
                 # If it needs to be dynamic, we'd need to adjust. For now, keep it simple.
    
    try:
        # 创建 RobotControl 实例
        robot_control = RobotControl(config, robot_id)
        print(f"创建 RobotControl 实例成功: {robot_id}")
        
        # 连接机器人
        print("正在连接机器人...")
        await robot_control.connect()
        print("机器人连接成功")
        
        # 测试获取TCP位姿
        print("正在获取TCP位P姿...")
        tcp_pose = await robot_control.get_tcp_pose()
        
        if tcp_pose is not None:
            print(f"成功获取TCP位姿: {tcp_pose}")
            print(f"位姿类型: {type(tcp_pose)}")
            print(f"位姿长度: {len(tcp_pose) if hasattr(tcp_pose, '__len__') else 'N/A'}")
            
            # 验证位姿格式
            if isinstance(tcp_pose, list) and len(tcp_pose) == 6:
                print("✓ 位姿格式正确 (6个浮点数的列表)")
                for i, value in enumerate(tcp_pose):
                    print(f"  位姿[{i}]: {value}")
            else:
                print("✗ 位姿格式不正确")
        else:
            print("✗ 获取TCP位姿失败，返回None")
            
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理连接
        try:
            if 'robot_control' in locals():
                if hasattr(robot_control, 'control_conn') and robot_control.control_conn:
                    await robot_control.control_conn.disconnect()
                if hasattr(robot_control, 'req_client') and robot_control.req_client:
                    await robot_control.req_client.disconnect()
                print("机器人连接已断开")
        except Exception as e:
            print(f"断开连接时发生错误: {e}")

if __name__ == "__main__":
    print("开始测试 get_tcp_pose 方法...")
    print("="*50)
    
    # 运行测试
    asyncio.run(test_get_tcp_pose())
    
    print("="*50)
    print("测试完成")