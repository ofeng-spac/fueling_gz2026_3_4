import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import time
import json
import anyio
import _jsonnet
import numpy as np
import open3d as o3d
from pathlib import Path
from loguru import logger
from datetime import datetime
from fueling.task import run_sync
from fueling.obcamera import AsyncOrbbecCamera
from fueling.robot_control import AsyncRobotClient
from fueling.minima.minima_service import MinimaMatcherService
from fueling.stereo_matcher.stereo_service import StereoMatcherService
from fueling.drawing import visualize_matches
from fueling.pointcloud_processor import depth_to_point_cloud
from fueling.minima.geometry import project_pointcloud_to_image_float


def create_bbox_image(image, bbox, color=(0, 255, 0), thickness=2):
    """在图像上绘制边框"""
    if bbox is None:
        return image
    
    if len(image.shape) == 2:
        image_with_bbox = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_with_bbox = image.copy()
    
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image_with_bbox, (x1, y1), (x2, y2), color, thickness)
    
    info = f"BOX: [{x1},{y1},{x2},{y2}]"
    cv2.putText(image_with_bbox, info, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, color, 1)
    
    return image_with_bbox


def compute_bbox_from_matches(mkpts, padding=0, img_shape=None):
    """根据匹配点计算边框"""
    if len(mkpts) == 0:
        return None
    
    min_x, min_y = np.min(mkpts, axis=0)
    max_x, max_y = np.max(mkpts, axis=0)
    
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    
    if img_shape is not None:
        if len(img_shape) > 2:
            img_shape = img_shape[:2]
        height, width = img_shape
        max_x = min(width, max_x + padding)
        max_y = min(height, max_y + padding)
    else:
        max_x = max_x + padding
        max_y = max_y + padding
    
    return (min_x, min_y, max_x, max_y)


def crop_image_by_bbox(image, bbox):
    """根据边框裁剪图像"""
    if bbox is None:
        return None
    
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]


def filter_matches_by_bbox(mkpts0, mkpts1, bbox):
    """过滤匹配点，只保留mkpts0在bbox内的点"""
    if bbox is None:
        return mkpts0, mkpts1
    
    x1, y1, x2, y2 = bbox
    mask = (mkpts0[:, 0] >= x1) & (mkpts0[:, 0] <= x2) & \
           (mkpts0[:, 1] >= y1) & (mkpts0[:, 1] <= y2)
    
    return mkpts0[mask], mkpts1[mask]


def adjust_intrinsic_for_crop(K, bbox):
    """
    调整内参矩阵以适应裁剪后的图像
    
    Args:
        K: 原始内参矩阵 (3x3)
        bbox: 裁剪边框 (x1, y1, x2, y2)
        
    Returns:
        K_adjusted: 调整后的内参矩阵
    """
    if bbox is None:
        return K
    
    x1, y1, _, _ = bbox
    
    K_adjusted = K.copy()
    # 调整主点坐标：减去裁剪的偏移量
    K_adjusted[0, 2] = K[0, 2] - x1  # cx
    K_adjusted[1, 2] = K[1, 2] - y1  # cy
    
    return K_adjusted


def crop_images_to_same_size(image_left, bbox_left, image_right, bbox_right):
    """
    裁剪左右图像到相同大小，用于立体匹配
    
    Args:
        image_left: 左图
        bbox_left: 左图边框 (x1, y1, x2, y2)
        image_right: 右图
        bbox_right: 右图边框 (x1, y1, x2, y2)
        
    Returns:
        cropped_left: 裁剪后的左图
        cropped_right: 裁剪后的右图
        final_bbox_left: 调整后的左图边框
        final_bbox_right: 调整后的右图边框
    """
    if bbox_left is None or bbox_right is None:
        return None, None, None, None
    
    # 计算两个边框的尺寸
    x1_left, y1_left, x2_left, y2_left = bbox_left
    x1_right, y1_right, x2_right, y2_right = bbox_right
    
    width_left = x2_left - x1_left
    height_left = y2_left - y1_left
    width_right = x2_right - x1_right
    height_right = y2_right - y1_right
    
    # 计算最大宽度和最大高度
    max_width = max(width_left, width_right)
    max_height = max(height_left, height_right)
    
    logger.info(f"左图原始尺寸: {width_left}x{height_left}")
    logger.info(f"右图原始尺寸: {width_right}x{height_right}")
    logger.info(f"统一尺寸: {max_width}x{max_height}")
    
    # 调整左图边框（保持中心位置不变，调整到最大尺寸）
    center_x_left = (x1_left + x2_left) / 2
    center_y_left = (y1_left + y2_left) / 2
    
    final_x1_left = int(max(0, center_x_left - max_width / 2))
    final_y1_left = int(max(0, center_y_left - max_height / 2))
    final_x2_left = int(min(image_left.shape[1], final_x1_left + max_width))
    final_y2_left = int(min(image_left.shape[0], final_y1_left + max_height))
    
    # 如果宽度或高度不足，调整起始位置
    if final_x2_left - final_x1_left < max_width:
        final_x1_left = max(0, final_x2_left - max_width)
    if final_y2_left - final_y1_left < max_height:
        final_y1_left = max(0, final_y2_left - max_height)
    
    final_bbox_left = (final_x1_left, final_y1_left, final_x2_left, final_y2_left)
    
    # 调整右图边框（保持中心位置不变，调整到相同尺寸）
    center_x_right = (x1_right + x2_right) / 2
    center_y_right = (y1_right + y2_right) / 2
    
    final_x1_right = int(max(0, center_x_right - max_width / 2))
    final_y1_right = int(max(0, center_y_right - max_height / 2))
    final_x2_right = int(min(image_right.shape[1], final_x1_right + max_width))
    final_y2_right = int(min(image_right.shape[0], final_y1_right + max_height))
    
    # 如果宽度或高度不足，调整起始位置
    if final_x2_right - final_x1_right < max_width:
        final_x1_right = max(0, final_x2_right - max_width)
    if final_y2_right - final_y1_right < max_height:
        final_y1_right = max(0, final_y2_right - max_height)
    
    final_bbox_right = (final_x1_right, final_y1_right, final_x2_right, final_y2_right)
    
    # 确保最终尺寸完全一致
    final_width_left = final_x2_left - final_x1_left
    final_height_left = final_y2_left - final_y1_left
    final_width_right = final_x2_right - final_x1_right
    final_height_right = final_y2_right - final_y1_right
    
    # 如果尺寸不一致，使用最小尺寸
    min_width = min(final_width_left, final_width_right)
    min_height = min(final_height_left, final_height_right)
    
    # 调整左图边框到最小尺寸
    final_bbox_left = (
        final_x1_left,
        final_y1_left,
        final_x1_left + min_width,
        final_y1_left + min_height
    )
    
    # 调整右图边框到最小尺寸
    final_bbox_right = (
        final_x1_right,
        final_y1_right,
        final_x1_right + min_width,
        final_y1_right + min_height
    )
    
    # 裁剪图像
    cropped_left = crop_image_by_bbox(image_left, final_bbox_left)
    cropped_right = crop_image_by_bbox(image_right, final_bbox_right)
    
    # 确保尺寸完全一致（如果由于边界限制导致不一致，使用resize）
    if cropped_left is not None and cropped_right is not None:
        h_left, w_left = cropped_left.shape[:2]
        h_right, w_right = cropped_right.shape[:2]
        
        if h_left != h_right or w_left != w_right:
            # 使用最小尺寸
            min_h = min(h_left, h_right)
            min_w = min(w_left, w_right)
            
            if h_left != min_h or w_left != min_w:
                cropped_left = cv2.resize(cropped_left, (min_w, min_h))
                logger.info(f"左图调整后尺寸: {min_w}x{min_h}")
            
            if h_right != min_h or w_right != min_w:
                cropped_right = cv2.resize(cropped_right, (min_w, min_h))
                logger.info(f"右图调整后尺寸: {min_w}x{min_h}")
    
    return cropped_left, cropped_right, final_bbox_left, final_bbox_right


def transform_points_to_roi_coordinates(points, bbox):
    """将点坐标从完整图像坐标系转换到ROI坐标系"""
    if bbox is None:
        return points
    
    x1, y1, _, _ = bbox
    transformed = points.copy()
    transformed[:, 0] = points[:, 0] - x1
    transformed[:, 1] = points[:, 1] - y1
    
    return transformed


async def test_roi_based_pipeline(
    arm_id: int,
    stereo_matcher: StereoMatcherService,
    robot_client: AsyncRobotClient,
    config: dict,
    initial_images: dict,
    minima_matcher_service: MinimaMatcherService
):
    """
    测试基于ROI的完整流程
    
    流程：
    1. 计算初始投影边框
    2. MINIMA匹配（完整图像）
    3. 计算当前图像ROI边框
    4. 裁剪图像到ROI区域
    5. 立体匹配（ROI图像）
    6. 点云生成（ROI区域）
    7. 点云配准（ROI点云）
    """
    # ========== 配置参数提取 ==========
    capture_pose = np.array(config['robot']['capture_pose'])
    exposure = config['camera']['exposure']
    gain = config['camera']['gain']
    init_pose = np.array(config['robot']['init_pose'])
    K = config['robot']['RS_camera']['K']
    baseline = config['robot']['RS_camera']['stereo_baseline']
    pre_mode = config['stereo_matcher']['pred_mode']
    bidir_verify_th = config['stereo_matcher']['bidir_verify_th']
    debug_mode = config['robot'].get('debug_mode', True)
    
    # ========== 创建输出目录 ==========
    if debug_mode:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_working_dir = Path(f"../working_data/test_roi_pipeline_{timestamp}")
        arm_dir = base_working_dir / f'arm_{arm_id + 1}'
        
        # 创建需要的目录
        images_output_directory = arm_dir / 'images'
        matches_output_directory = arm_dir / 'matches'
        roi_output_directory = arm_dir / 'roi'
        pointcloud_output_directory = arm_dir / 'pointclouds'
        stereo_input_directory = arm_dir / 'stereo_input'
        
        for dir_path in [images_output_directory, matches_output_directory,
                        roi_output_directory, pointcloud_output_directory,
                        stereo_input_directory]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Task {arm_id}: 输出目录创建在 {arm_dir}")
    else:
        logger.info(f"Task {arm_id}: Debug模式已禁用")
        return
    
    # ========== 步骤1: 计算初始投影边框 ==========
    logger.info(f"Task {arm_id}: 步骤1 - 计算初始投影边框...")
    
    try:
        # 加载源点云
        data_dir = Path(__file__).resolve().parent.parent / 'data'
        target_pot = config['robot']['target_pot']
        source_down_path = data_dir / f'arm{arm_id + 1}' / target_pot / 'source_model.pcd'
        
        if not source_down_path.exists():
            logger.warning(f"Task {arm_id}: 源点云文件不存在: {source_down_path}")
            # 使用启发式方法估算初始边框
            height, width = initial_images['left_ir'].shape
            bbox1_left = (width//4, height//4, width*3//4, height*3//4)
            bbox1_right = bbox1_left
        else:
            source_pc = o3d.io.read_point_cloud(str(source_down_path))
            if not source_pc.has_points():
                logger.warning(f"Task {arm_id}: 源点云为空，使用默认边框")
                height, width = initial_images['left_ir'].shape
                bbox1_left = (width//4, height//4, width*3//4, height*3//4)
                bbox1_right = bbox1_left
            else:
                # 计算初始投影边框
                K_mat = np.array(K).reshape(3, 3)
                
                # 计算左图初始边框
                point_arr1_left, indices1_left, bbox1_left = project_pointcloud_to_image_float(
                    source_pc, initial_images['left_ir'], K_mat, None, None
                )
                
                # 计算右图初始边框
                T_left_to_right = np.eye(4)
                T_left_to_right[0, 3] = -baseline * 1000  # 转换为毫米
                
                point_arr1_right, indices1_right, bbox1_right = project_pointcloud_to_image_float(
                    source_pc, initial_images['right_ir'], K_mat, None, transform=T_left_to_right
                )
                
                logger.info(f"Task {arm_id}: 左图初始边框: {bbox1_left}")
                logger.info(f"Task {arm_id}: 右图初始边框: {bbox1_right}")
                
    except Exception as e:
        logger.error(f"Task {arm_id}: 计算初始投影边框失败: {e}")
        # 使用默认边框
        height, width = initial_images['left_ir'].shape
        bbox1_left = (width//4, height//4, width*3//4, height*3//4)
        bbox1_right = bbox1_left
    
    # 保存带有初始边框的图像
    initial_left_with_bbox = create_bbox_image(initial_images['left_ir'], bbox1_left)
    initial_right_with_bbox = create_bbox_image(initial_images['right_ir'], bbox1_right)
    
    cv2.imwrite(str(images_output_directory / 'initial_left_with_bbox.png'), initial_left_with_bbox)
    cv2.imwrite(str(images_output_directory / 'initial_right_with_bbox.png'), initial_right_with_bbox)
    
    # ========== 步骤2: 捕获当前图像 ==========
    logger.info(f"Task {arm_id}: 步骤2 - 捕获当前图像...")
    
    try:
        obcamera = AsyncOrbbecCamera(
            camera_serial=config['camera']['camera_serial'],
            pipeline_params={'enable_streams': [{'type': 'IR'}]}
        )
        from fueling.obcamera.obcamera import get_camera_params
        images = await obcamera.capture_stereo_ir(exposure, gain, 'flood')
        
        logger.info(f"Task {arm_id}: 图像捕获成功")
        
        # 保存原始捕获的图像
        captured_left_path = images_output_directory / 'captured_left_ir.png'
        captured_right_path = images_output_directory / 'captured_right_ir.png'
        cv2.imwrite(str(captured_left_path), images['left_ir'])
        cv2.imwrite(str(captured_right_path), images['right_ir'])
        
    except Exception as e:
        logger.error(f"Task {arm_id}: 图像捕获失败: {e}")
        return
    
    # ========== 步骤3: MINIMA匹配（完整图像） ==========
    logger.info(f"Task {arm_id}: 步骤3 - MINIMA匹配（完整图像）...")
    
    # 左图匹配（完整初始左图 vs 完整当前左图）
    match_res_left = await minima_matcher_service.match(initial_images['left_ir'], images['left_ir'])
    mkpts0_full_left, mkpts1_full_left, mconf_left = match_res_left['mkpts0'], match_res_left['mkpts1'], match_res_left['mconf']
    logger.info(f"Task {arm_id}: 左图原始匹配点数量: {len(mkpts0_full_left)}")
    
    # 右图匹配（完整初始右图 vs 完整当前右图）
    match_res_right = await minima_matcher_service.match(initial_images['right_ir'], images['right_ir'])
    mkpts0_full_right, mkpts1_full_right, mconf_right = match_res_right['mkpts0'], match_res_right['mkpts1'], match_res_right['mconf']
    logger.info(f"Task {arm_id}: 右图原始匹配点数量: {len(mkpts0_full_right)}")
    
    # ========== 步骤4: 过滤匹配点并计算当前图像ROI边框 ==========
    logger.info(f"Task {arm_id}: 步骤4 - 过滤匹配点并计算当前图像ROI边框...")
    
    # 过滤左图匹配点（只保留初始图像上在初始边框内的点）
    mkpts0_filtered_left, mkpts1_filtered_left = filter_matches_by_bbox(
        mkpts0_full_left, mkpts1_full_left, bbox1_left
    )
    logger.info(f"Task {arm_id}: 左图过滤后匹配点数量: {len(mkpts0_filtered_left)}")
    
    # 过滤右图匹配点
    mkpts0_filtered_right, mkpts1_filtered_right = filter_matches_by_bbox(
        mkpts0_full_right, mkpts1_full_right, bbox1_right
    )
    logger.info(f"Task {arm_id}: 右图过滤后匹配点数量: {len(mkpts0_filtered_right)}")
    
    # 根据过滤后的匹配点计算当前图像中的ROI边框
    bbox2_left = compute_bbox_from_matches(
        mkpts1_filtered_left,
        padding=0,
        img_shape=images['left_ir'].shape
    )
    
    bbox2_right = compute_bbox_from_matches(
        mkpts1_filtered_right,
        padding=0,
        img_shape=images['right_ir'].shape
    )
    
    logger.info(f"Task {arm_id}: 左图当前ROI边框: {bbox2_left}")
    logger.info(f"Task {arm_id}: 右图当前ROI边框: {bbox2_right}")
    
    # ========== 步骤5: 调整ROI边框到相同大小并裁剪图像 ==========
    logger.info(f"Task {arm_id}: 步骤5 - 调整ROI边框到相同大小并裁剪图像...")
    
    cropped_current_left, cropped_current_right, final_bbox_left, final_bbox_right = crop_images_to_same_size(
        images['left_ir'], bbox2_left,
        images['right_ir'], bbox2_right
    )
    
    if cropped_current_left is None or cropped_current_right is None:
        logger.error(f"Task {arm_id}: 裁剪图像失败")
        return
    
    logger.info(f"Task {arm_id}: 调整后左图ROI边框: {final_bbox_left}")
    logger.info(f"Task {arm_id}: 调整后右图ROI边框: {final_bbox_right}")
    logger.info(f"Task {arm_id}: 左图ROI尺寸: {cropped_current_left.shape}")
    logger.info(f"Task {arm_id}: 右图ROI尺寸: {cropped_current_right.shape}")
    
    # 保存ROI图像
    roi_left_path = roi_output_directory / 'roi_left.png'
    roi_right_path = roi_output_directory / 'roi_right.png'
    cv2.imwrite(str(roi_left_path), cropped_current_left)
    cv2.imwrite(str(roi_right_path), cropped_current_right)
    
    # 保存带有ROI边框的图像
    current_left_with_roi = create_bbox_image(images['left_ir'], final_bbox_left, color=(255, 0, 0))
    current_right_with_roi = create_bbox_image(images['right_ir'], final_bbox_right, color=(255, 0, 0))
    
    cv2.imwrite(str(images_output_directory / 'current_left_with_roi.png'), current_left_with_roi)
    cv2.imwrite(str(images_output_directory / 'current_right_with_roi.png'), current_right_with_roi)
    
    # ========== 步骤6: 调整内参矩阵 ==========
    logger.info(f"Task {arm_id}: 步骤6 - 调整内参矩阵...")
    
    # 原始内参矩阵
    K_mat = np.array(K).reshape(3, 3)
    logger.info(f"Task {arm_id}: 原始内参矩阵:\n{K_mat}")
    
    # 调整内参以适应裁剪
    K_left_adjusted = adjust_intrinsic_for_crop(K_mat, final_bbox_left)
    K_right_adjusted = adjust_intrinsic_for_crop(K_mat, final_bbox_right)
    
    logger.info(f"Task {arm_id}: 调整后左图内参矩阵:\n{K_left_adjusted}")
    logger.info(f"Task {arm_id}: 调整后右图内参矩阵:\n{K_right_adjusted}")
    
    # ========== 步骤7: 立体匹配（ROI图像） ==========
    logger.info(f"Task {arm_id}: 步骤7 - 立体匹配（ROI图像）...")
    
    inference_start_time = time.time()
    
    # 使用ROI图像进行立体匹配
    result = await stereo_matcher.infer(
        cropped_current_left, 
        cropped_current_right, 
        pred_mode=pre_mode, 
        bidir_verify_th=bidir_verify_th
    )
    
    inference_end_time = time.time()
    logger.info(f"Task {arm_id}: 立体匹配完成，耗时: {inference_end_time - inference_start_time:.2f}秒")
    if debug_mode:
        from fueling.stereo_matcher import save_disparity_map
        await run_sync(save_disparity_map, result=result, output_dir=images_output_directory)
        logger.info(f"Task {arm_id}: Saved disparity map to: {images_output_directory}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping disparity map saving")
    # 获取视差图
    if 'disparity_verified' in result:
        disparity_map = result['disparity_verified']
    else:
        disparity_map = result['disparity_left']
    
    # 处理无效视差
    disparity_map = np.where(disparity_map <= 0, 0.1, disparity_map)
    
    # ========== 步骤8: 生成深度图和点云 ==========
    logger.info(f"Task {arm_id}: 步骤8 - 生成深度图和点云...")
    
    # 生成深度图
    depth_map = (K_left_adjusted[0, 0] * baseline) / disparity_map
    depth_map_uint16 = np.uint16(depth_map * 1000)  # 转换为毫米
    
    # 保存深度图
    depth_map_path = roi_output_directory / 'depth_map.png'
    cv2.imwrite(str(depth_map_path), depth_map_uint16)
    logger.info(f"Task {arm_id}: 深度图已保存到 {depth_map_path}")
    
    # 生成点云（使用调整后的内参）
    pcd = await run_sync(
        depth_to_point_cloud,
        depth_map=np.array(depth_map_uint16),
        camera_intrinsics=K_left_adjusted.flatten().tolist(),
        max_distance=600,
    )
    
    # 保存点云
    if pcd is not None:
        pcd_path = pointcloud_output_directory / 'roi_pointcloud.pcd'
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        logger.info(f"Task {arm_id}: ROI点云已保存到 {pcd_path}")
        logger.info(f"Task {arm_id}: 点云点数: {len(pcd.points)}")
    
    # ========== 步骤9: 转换MINIMA匹配点到ROI坐标系 ==========
    logger.info(f"Task {arm_id}: 步骤9 - 转换MINIMA匹配点到ROI坐标系...")
    
    # 转换左图匹配点
    mkpts1_roi_left = transform_points_to_roi_coordinates(mkpts1_filtered_left, final_bbox_left)
    
    # 转换右图匹配点
    mkpts1_roi_right = transform_points_to_roi_coordinates(mkpts1_filtered_right, final_bbox_right)
    
    logger.info(f"Task {arm_id}: 左图匹配点已转换到ROI坐标系")
    logger.info(f"Task {arm_id}: 右图匹配点已转换到ROI坐标系")
    
        # ========== 步骤10: 总结输出 ==========
    logger.info(f"Task {arm_id}: 步骤10 - 总结输出...")
    
    # 创建一个可以JSON序列化的报告
    report = {
        "arm_id": arm_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "initial_bbox": {
            "left": list(map(int, bbox1_left)) if bbox1_left is not None else None,
            "right": list(map(int, bbox1_right)) if bbox1_right is not None else None
        },
        "current_roi_bbox": {
            "left": list(map(int, final_bbox_left)) if final_bbox_left is not None else None,
            "right": list(map(int, final_bbox_right)) if final_bbox_right is not None else None
        },
        "match_statistics": {
            "left_original_matches": int(len(mkpts0_full_left)),
            "left_filtered_matches": int(len(mkpts0_filtered_left)),
            "right_original_matches": int(len(mkpts0_full_right)),
            "right_filtered_matches": int(len(mkpts0_filtered_right))
        },
        "image_sizes": {
            "initial_left": list(initial_images['left_ir'].shape),
            "initial_right": list(initial_images['right_ir'].shape),
            "current_left": list(images['left_ir'].shape),
            "current_right": list(images['right_ir'].shape),
            "roi_left": list(cropped_current_left.shape) if cropped_current_left is not None else None,
            "roi_right": list(cropped_current_right.shape) if cropped_current_right is not None else None
        },
        "intrinsic_matrices": {
            "original": K_mat.tolist(),
            "left_adjusted": K_left_adjusted.tolist(),
            "right_adjusted": K_right_adjusted.tolist()
        }
    }
    
    report_path = arm_dir / 'test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Task {arm_id}: 测试报告已保存到 {report_path}")
    logger.info(f"Task {arm_id}: ROI-based pipeline测试完成")
    
    return {
        "cropped_left": cropped_current_left,
        "cropped_right": cropped_current_right,
        "final_bbox_left": final_bbox_left,
        "final_bbox_right": final_bbox_right,
        "K_left_adjusted": K_left_adjusted,
        "K_right_adjusted": K_right_adjusted,
        "pcd": pcd,
        "mkpts1_roi_left": mkpts1_roi_left,
        "mkpts1_roi_right": mkpts1_roi_right,
        "mkpts0_filtered_left": mkpts0_filtered_left,
        "mkpts0_filtered_right": mkpts0_filtered_right
    }


async def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    
    # 查找arm目录
    arm_dirs = sorted([d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('arm')])
    
    config_files = [os.path.join(data_dir, arm_dir, "config.jsonnet") 
                   for arm_dir in arm_dirs]
    config_files = [f for f in config_files if os.path.isfile(f)][:1]  # 只测试第一个
    
    if not config_files:
        logger.error("没有找到有效的配置文件")
        return 1
    
    logger.info(f"找到配置文件: {config_files[0]}")
    
    proj_dir = Path(__file__).resolve().parent.parent
    default_config = f"{data_dir}/default_config.jsonnet"
    default_set = json.loads(_jsonnet.evaluate_file(default_config))
    
    # 初始化立体匹配服务
    max_parallel = default_set['stereo_matcher']['max_parallel']
    model_name = default_set['stereo_matcher']['method']
    model_path = default_set['stereo_matcher'][model_name]['model_path']
    weight_path = f"{proj_dir}{model_path}"
    
    stereo_matcher = StereoMatcherService(model_name, weight_path, max_parallel)
    
    # 初始化MINIMA服务
    minima_config = default_set.get('minima', {})
    minima_model_path = minima_config.get('model_path', '')
    minima_weight_path = f"{proj_dir}{minima_model_path}"
    
    logger.info(f"加载MINIMA模型: {minima_weight_path}")
    minima_matcher_service = MinimaMatcherService(minima_weight_path)
    
    import asyncio
    asyncio.create_task(minima_matcher_service.loop_process_items())
    asyncio.create_task(stereo_matcher.loop_process_items())
    
    # 加载配置文件
    config_file = config_files[0]
    config = json.loads(_jsonnet.evaluate_file(config_file))
    
    # 加载机器人位姿
    target_pot = config['robot']['target_pot']
    robot_pose_path = os.path.join(data_dir, f'arm1', target_pot, 'robot_pose.json')
    if os.path.exists(robot_pose_path):
        with open(robot_pose_path, 'r') as f:
            robot_poses = json.load(f)
            config['robot']['init_pose'] = robot_poses['init_pose']
            config['robot']['capture_pose'] = robot_poses['capture_pose']
            config['robot']['fueling_pose'] = robot_poses['fueling_pose']
            logger.info(f"已加载机器人位姿从 {robot_pose_path}")
    else:
        logger.error(f"机器人位姿文件未找到: {robot_pose_path}")
        return 1
    
    # 初始化机器人客户端
    robot_client = AsyncRobotClient(
        addr=config["robot"]["ip"],
        req_port=config["robot"]["req_port"],
        ctrl_port=config["robot"]["control_port"],
        type=config["robot"]["move_command"],
        movel_params=config["robot"]["movel_params"],
        movej_params=config["robot"]["movej_params"],
        pos_tol=config['robot']['pos_tol'],
        rot_tol=config['robot']['rot_tol'],
        check_interval=config['robot']['check_interval']
    )
    await robot_client.connect()
    
    # 加载初始图像
    left_ir_path = config['minima']['left_ir_path']
    right_ir_path = config['minima']['right_ir_path']
    
    # 转换为绝对路径
    if not os.path.isabs(left_ir_path):
        left_ir_path = os.path.join(proj_dir, left_ir_path)
    if not os.path.isabs(right_ir_path):
        right_ir_path = os.path.join(proj_dir, right_ir_path)
    
    # 检查文件是否存在
    if not os.path.exists(left_ir_path):
        # 尝试其他可能的文件名
        left_dir = os.path.dirname(left_ir_path)
        if os.path.exists(left_dir):
            possible_files = [f for f in os.listdir(left_dir) 
                            if 'left' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if possible_files:
                left_ir_path = os.path.join(left_dir, possible_files[0])
    
    if not os.path.exists(right_ir_path):
        right_dir = os.path.dirname(right_ir_path)
        if os.path.exists(right_dir):
            possible_files = [f for f in os.listdir(right_dir) 
                            if 'right' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if possible_files:
                right_ir_path = os.path.join(right_dir, possible_files[0])
    
    initial_images = {
        'left_ir': cv2.imread(left_ir_path, cv2.IMREAD_GRAYSCALE),
        'right_ir': cv2.imread(right_ir_path, cv2.IMREAD_GRAYSCALE)
    }
    
    if initial_images['left_ir'] is None or initial_images['right_ir'] is None:
        logger.error(f"初始图像无法加载: {left_ir_path}, {right_ir_path}")
        return 1
    
    logger.info(f"已加载初始图像: {left_ir_path}, {right_ir_path}")
    
    # 运行测试
    result = await test_roi_based_pipeline(
        0,  # arm_id
        stereo_matcher,
        robot_client,
        config,
        initial_images,
        minima_matcher_service
    )
    

    
    logger.info("测试完成")
    
    return result


if __name__ == '__main__':
    start_time = time.time()
    try:
        anyio.run(main, backend='asyncio')
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        end_time = time.time()
        print(f"测试总用时: {end_time - start_time:.2f} 秒")