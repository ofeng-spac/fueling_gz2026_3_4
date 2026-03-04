import sys
import os
import cv2
import numpy as np
from pathlib import Path

# 设置项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import asyncio
from loguru import logger
from fueling.minima.minima_service import MinimaMatcherService

# 配置日志
logger.remove()
logger.add(sys.stderr, level="DEBUG")

async def test_minima_matching():
    """
    测试MINIMA匹配功能的独立脚本
    """
    logger.info("开始测试MINIMA匹配功能...")
    
    # 设置项目目录
    proj_dir = Path(__file__).resolve().parent.parent
    
    # 初始化MINIMA服务
    minima_model_path = 'fueling/minima/weights/minima_v2_0_1.pth'
    minima_weight_path = str(proj_dir / minima_model_path)
    
    logger.info(f"加载MINIMA模型: {minima_weight_path}")
    
    try:
        minima_matcher_service = MinimaMatcherService("/home/vision/projects/fueling_gz/data/minima_weights/minima_roma.pth")
        # 启动后台处理循环
        asyncio.create_task(minima_matcher_service.loop_process_items())
        logger.success("MINIMA服务初始化成功")
    except Exception as e:
        logger.error(f"初始化MINIMA服务失败: {e}")
        return
    
    # 测试图像路径
    test_dir = proj_dir / "fueling" / "initialization" / "orbbec_output" / "Flood_light" / "arm2" / "pot4"
    
    # 初始图像路径
    initial_left_path = test_dir / "captured_left_ir.png"
    initial_right_path = test_dir / "captured_right_ir.png"
    
    # 模拟当前图像路径（使用同样的图像进行测试）
    current_left_path = test_dir / "captured_left_ir.png"
    current_right_path = test_dir / "captured_right_ir.png"
    
    logger.info(f"测试目录: {test_dir}")
    logger.info(f"初始左图路径: {initial_left_path}")
    logger.info(f"初始右图路径: {initial_right_path}")
    logger.info(f"当前左图路径: {current_left_path}")
    logger.info(f"当前右图路径: {current_right_path}")
    
    # 检查文件是否存在
    for path in [initial_left_path, initial_right_path, current_left_path, current_right_path]:
        if not path.exists():
            logger.error(f"文件不存在: {path}")
        else:
            logger.success(f"文件存在: {path}")
    
    # 读取图像
    try:
        initial_left = cv2.imread(str(initial_left_path), cv2.IMREAD_GRAYSCALE)
        initial_right = cv2.imread(str(initial_right_path), cv2.IMREAD_GRAYSCALE)
        current_left = cv2.imread(str(current_left_path), cv2.IMREAD_GRAYSCALE)
        current_right = cv2.imread(str(current_right_path), cv2.IMREAD_GRAYSCALE)
        
        # 检查图像是否成功读取
        if initial_left is None:
            raise ValueError(f"无法读取初始左图: {initial_left_path}")
        if initial_right is None:
            raise ValueError(f"无法读取初始右图: {initial_right_path}")
        if current_left is None:
            raise ValueError(f"无法读取当前左图: {current_left_path}")
        if current_right is None:
            raise ValueError(f"无法读取当前右图: {current_right_path}")
        
        logger.success("所有图像成功加载")
        
        # 显示图像信息
        logger.info(f"初始左图形状: {initial_left.shape}, 数据类型: {initial_left.dtype}")
        logger.info(f"初始右图形状: {initial_right.shape}, 数据类型: {initial_right.dtype}")
        logger.info(f"当前左图形状: {current_left.shape}, 数据类型: {current_left.dtype}")
        logger.info(f"当前右图形状: {current_right.shape}, 数据类型: {current_right.dtype}")
        
        # 检查图像值范围
        logger.info(f"初始左图值范围: [{initial_left.min()}, {initial_left.max()}]")
        logger.info(f"初始右图值范围: [{initial_right.min()}, {initial_right.max()}]")
        logger.info(f"当前左图值范围: [{current_left.min()}, {current_left.max()}]")
        logger.info(f"当前右图值范围: [{current_right.min()}, {current_right.max()}]")
        
    except Exception as e:
        logger.error(f"图像读取失败: {e}")
        return
    
    # 测试匹配功能
    logger.info("\n开始测试左图匹配...")
    try:
        match_res_left = await minima_matcher_service.match(initial_left, current_left)
        logger.success("左图匹配成功")
        
        # 输出匹配结果信息
        mkpts0 = match_res_left['mkpts0']
        mkpts1 = match_res_left['mkpts1']
        mconf = match_res_left['mconf']
        
        logger.info(f"匹配点数量: {len(mkpts0)}")
        logger.info(f"匹配点形状: mkpts0={mkpts0.shape}, mkpts1={mkpts1.shape}")
        logger.info(f"置信度形状: {mconf.shape}")
        logger.info(f"置信度范围: [{mconf.min():.3f}, {mconf.max():.3f}]")
        
        # 如果有匹配点，显示一些示例
        if len(mkpts0) > 0:
            logger.info(f"前5个匹配点:")
            for i in range(min(5, len(mkpts0))):
                logger.info(f"  {i}: mkpts0={mkpts0[i]}, mkpts1={mkpts1[i]}, conf={mconf[i]:.3f}")
        
    except Exception as e:
        logger.error(f"左图匹配失败: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n开始测试右图匹配...")
    try:
        match_res_right = await minima_matcher_service.match(initial_right, current_right)
        logger.success("右图匹配成功")
        
        # 输出匹配结果信息
        mkpts0 = match_res_right['mkpts0']
        mkpts1 = match_res_right['mkpts1']
        mconf = match_res_right['mconf']
        
        logger.info(f"匹配点数量: {len(mkpts0)}")
        logger.info(f"匹配点形状: mkpts0={mkpts0.shape}, mkpts1={mkpts1.shape}")
        logger.info(f"置信度形状: {mconf.shape}")
        logger.info(f"置信度范围: [{mconf.min():.3f}, {mconf.max():.3f}]")
        
        # 如果有匹配点，显示一些示例
        if len(mkpts0) > 0:
            logger.info(f"前5个匹配点:")
            for i in range(min(5, len(mkpts0))):
                logger.info(f"  {i}: mkpts0={mkpts0[i]}, mkpts1={mkpts1[i]}, conf={mconf[i]:.3f}")
        
    except Exception as e:
        logger.error(f"右图匹配失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试不同的图像对
    logger.info("\n测试不同的图像对...")
    try:
        # 使用左右图像进行交叉匹配（应该失败）
        logger.info("测试交叉匹配（左图vs右图）...")
        match_res_cross = await minima_matcher_service.match(initial_left, initial_right)
        logger.info(f"交叉匹配点数量: {len(match_res_cross['mkpts0'])}")
    except Exception as e:
        logger.error(f"交叉匹配失败（预期行为）: {e}")
    
    logger.info("\nMINIMA匹配测试完成！")

if __name__ == '__main__':
    # 运行异步测试
    try:
        asyncio.run(test_minima_matching())
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试运行时错误: {e}")
        import traceback
        traceback.print_exc()