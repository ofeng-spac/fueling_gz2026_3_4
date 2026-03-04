import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
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
from fueling.pose_transformation import get_upper_pose
from fueling.drawing import display_point_cloud, visualize_sparse_pointclouds, display_point_cloud_with_axes
from fueling.pointcloud_processor import PointCloudRegistration, depth_to_point_cloud, preprocess_pointcloud
from fueling.pointcloud_processor.depth_to_point_cloud import filter_outliers_by_distance, filter_outliers_by_superansac
from fueling.stereo_matcher import save_disparity_map
from fueling.robot_control import AsyncRobotClient, compute_transformed_fueling_pose
from fueling.stereo_matcher.stereo_service import StereoMatcherService
from fueling.minima.minima_service import MinimaMatcherService
from fueling.drawing import visualize_matches

def fill_roi_to_full_size(roi_image, roi_bbox, full_shape):
    """
    将ROI图像填充回原始大小

    Args:
        roi_image: ROI区域的图像/视差图
        roi_bbox: ROI边框 (x1, y1, x2, y2)
        full_shape: 原始图像的形状 (height, width)

    Returns:
        full_image: 填充后的完整图像
    """
    x1, y1, x2, y2 = map(int, roi_bbox)
    height, width = full_shape

    # 创建全零矩阵
    full_image = np.zeros(full_shape, dtype=roi_image.dtype)

    # 确保ROI区域在图像范围内
    h_roi, w_roi = roi_image.shape[:2]
    y_end = min(y1 + h_roi, height)
    x_end = min(x1 + w_roi, width)

    # 填充ROI区域
    full_image[y1:y_end, x1:x_end] = roi_image[:y_end-y1, :x_end-x1]

    return full_image
# ========== ROI相关辅助函数 ==========

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


async def run_fuel(arm_id: int, stereo_matcher: StereoMatcherService,
                   robot_client: AsyncRobotClient, pc_registration: PointCloudRegistration, config: dict,
                   source_pc: o3d.geometry.PointCloud, initial_images: dict, minima_matcher_service: MinimaMatcherService):
    """
    Main function to run the end-to-end camera capture and inference pipeline.
    Now with ROI-based stereo matching.

    Args:
        arm_id: Unique identifier for this task (0-3)
    """
    # ========== 配置参数提取 ==========
    capture_pose = np.array(config['robot']['capture_pose'])
    exposure = config['camera']['exposure']
    gain = config['camera']['gain']
    eye_hand_matrix = config['robot']['eye_hand_matrix']["T"]
    fueling_pose = np.array(config['robot']['fueling_pose'])
    cut_box = config['point_cloud']['cut_box']
    voxel_size = config['point_cloud']['voxel_size']
    radius = config['point_cloud']['radius']
    min_neighbors = config['point_cloud']['min_neighbors']
    remove_outliers = config['point_cloud']['remove_outliers']
    init_pose = np.array(config['robot']['init_pose'])
    K = config['robot']['RS_camera']['K']
    baseline = config['robot']['RS_camera']['stereo_baseline']
    pre_mode = config['stereo_matcher']['pred_mode']
    bidir_verify_th = config['stereo_matcher']['bidir_verify_th']
    debug_mode = config['robot']['debug_mode']

    def get_stereo_transform_from_config(config):
        """
        从配置中提取立体变换矩阵
        假设：相机已经过立体校正，只有X方向平移
        """
        baseline = config['robot']['RS_camera']['stereo_baseline']  # 米
        baseline_mm = baseline * 1000  # 毫米

        # 构建4x4变换矩阵
        # 假设右相机在左相机的正X方向
        T_left_to_right = np.eye(4)
        T_left_to_right[0, 3] = -baseline_mm  # 将点从左相机坐标系变换到右相机坐标系

        return T_left_to_right

    T_left_to_right = get_stereo_transform_from_config(config)
    logger.info(f"Task {arm_id}: Stereo transform T_left_to_right:\n{T_left_to_right}")

    # ========== 统一目录创建逻辑 ==========
    if debug_mode:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_working_dir = Path(f"../working_data/{timestamp}")
        arm_dir = base_working_dir / f'arm_{arm_id + 1}'

        # 创建所有需要的目录
        images_output_directory = arm_dir / 'ir_images'
        disparity_output_directory = arm_dir / 'disparities'
        output_depth_directory = arm_dir / 'depth'
        point_clouds_directory = arm_dir / 'point_clouds'
        config_output_directory = arm_dir / 'config'
        log_dir = arm_dir / 'logs'
        minima_output_directory = arm_dir / 'minima'
        roi_output_directory = arm_dir / 'roi'

        # 一次性创建所有目录
        for dir_path in [images_output_directory, disparity_output_directory,
                        output_depth_directory, point_clouds_directory,
                        config_output_directory, log_dir, minima_output_directory,
                        roi_output_directory]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 添加日志文件
        logger.add(log_dir / 'test.log', level='INFO', enqueue=True)  # type: ignore (no-untyped-call)
    else:
        # 非debug模式使用简化的路径
        images_output_directory = f'captured_images/task_{arm_id}'
        disparity_output_directory = f'disparity_output/task_{arm_id}'
        output_depth_directory = f"depth_output/task_{arm_id}"
        point_clouds_directory = f'pointclouds_output/task_{arm_id}'
        minima_output_directory = f'minima_output/task_{arm_id}'
        roi_output_directory = f'roi_output/task_{arm_id}'

    # ========== 步骤1: 计算初始投影边框 ==========
    logger.info(f"Task {arm_id}: 步骤1 - 计算初始投影边框...")
    K_mat = np.array(config['robot']['RS_camera']['K']).reshape(3, 3)
    from fueling.minima.geometry import project_pointcloud_to_image_float

    # 计算初始投影边框
    if debug_mode:
        projected_img_path1_left = os.path.join(minima_output_directory, "projected_source_left.png")
        projected_img_path1_right = os.path.join(minima_output_directory, "projected_source_right.png")
        # 先计算左图
        point_arr1_left, indices1_left, bbox1_left = project_pointcloud_to_image_float(source_pc, initial_images['left_ir'], K_mat, projected_img_path1_left, None)
        # 计算右图时传入左图的边框，确保大小一致
        point_arr1_right, indices1_right, bbox1_right = project_pointcloud_to_image_float(source_pc, initial_images['right_ir'], K_mat, projected_img_path1_right, transform=T_left_to_right, other_bbox=bbox1_left)
        # 如果需要，可以再次计算左图以确保完全一致
        if bbox1_right is not None:
            point_arr1_left, indices1_left, bbox1_left = project_pointcloud_to_image_float(source_pc, initial_images['left_ir'], K_mat, projected_img_path1_left, None, other_bbox=bbox1_right)
    else:
        # 非debug模式也使用相同的逻辑
        point_arr1_left, indices1_left, bbox1_left = project_pointcloud_to_image_float(source_pc, initial_images['left_ir'], K_mat, None, None)
        point_arr1_right, indices1_right, bbox1_right = project_pointcloud_to_image_float(source_pc, initial_images['right_ir'], K_mat, None, transform=T_left_to_right, other_bbox=bbox1_left)
        if bbox1_right is not None:
            point_arr1_left, indices1_left, bbox1_left = project_pointcloud_to_image_float(source_pc, initial_images['left_ir'], K_mat, None, None, other_bbox=bbox1_right)

    # 记录边框信息
    if bbox1_left is not None and bbox1_right is not None:
        left_width = bbox1_left[2] - bbox1_left[0]
        left_height = bbox1_left[3] - bbox1_left[1]
        right_width = bbox1_right[2] - bbox1_right[0]
        right_height = bbox1_right[3] - bbox1_right[1]
        logger.info(f"Task {arm_id}: 源点云左图像投影边框: [{bbox1_left[0]:.1f}, {bbox1_left[1]:.1f}, {bbox1_left[2]:.1f}, {bbox1_left[3]:.1f}] 大小: {left_width:.1f}x{left_height:.1f}")
        logger.info(f"Task {arm_id}: 源点云右图像投影边框: [{bbox1_right[0]:.1f}, {bbox1_right[1]:.1f}, {bbox1_right[2]:.1f}, {bbox1_right[3]:.1f}] 大小: {right_width:.1f}x{right_height:.1f}")

    # 保存带有初始边框的图像
    if debug_mode:
        initial_left_with_bbox = create_bbox_image(initial_images['left_ir'], bbox1_left)
        initial_right_with_bbox = create_bbox_image(initial_images['right_ir'], bbox1_right)
        cv2.imwrite(str(images_output_directory / 'initial_left_with_bbox.png'), initial_left_with_bbox)
        cv2.imwrite(str(images_output_directory / 'initial_right_with_bbox.png'), initial_right_with_bbox)

    # ========== 移动到捕获位置 ==========
    await robot_client.move(capture_pose)
    logger.info(f"Task {arm_id}: Moved to capture position.")

    total_start_time = time.time()

    # ========== 初始化相机 ==========
    logger.info(f"Task {arm_id}: Initializing camera...")
    try:
        obcamera = AsyncOrbbecCamera(
                camera_serial=config['camera']['camera_serial'],
                pipeline_params={'enable_streams': [{'type': 'IR'}]}
            )
        logger.info(f"Task {arm_id}: Camera initialized successfully.")
    except Exception as e:
        logger.error(f"Task {arm_id}: Error: Failed to initialize camera: {e}")
        return

    # ========== 保存配置（debug模式） ==========
    if debug_mode:
        output_config_path = f'{config_output_directory}/config{arm_id}.json'
        with open(output_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Task {arm_id}: Updated config and saved to {output_config_path}")
    else:
        logger.info(f"Task {arm_id}: Config updated in memory, debug mode disabled so not saving to file")

    # ========== 捕获图像 ==========
    logger.info(f"Task {arm_id}: Capturing IR image pair...")
    try:
        images = await obcamera.capture_stereo_ir(exposure, gain, 'flood')
        logger.info(f"Task {arm_id}: Image pair captured.")
    except Exception as e:
        logger.error(f"Task {arm_id}: Error: Failed to capture images: {e}")
        return

    # ========== 保存图像（debug模式） ==========
    if debug_mode:
        captured_left_path = f'{images_output_directory}/captured_left_ir.png'
        captured_right_path = f'{images_output_directory}/captured_right_ir.png'
        cv2.imwrite(str(captured_left_path), images['left_ir'])
        cv2.imwrite(str(captured_right_path), images['right_ir'])
        logger.info(f"Task {arm_id}: Saved captured images for verification to: {images_output_directory}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping image saving")

    # ========== 步骤2: MINIMA匹配（完整图像） ==========
    logger.info(f"Task {arm_id}: 步骤2 - MINIMA匹配（完整图像）...")

    # 左图匹配（完整初始左图 vs 完整当前左图）
    match_res_left = await minima_matcher_service.match(initial_images['left_ir'], images['left_ir'])
    mkpts0_full_left, mkpts1_full_left, mconf_left = match_res_left['mkpts0'], match_res_left['mkpts1'], match_res_left['mconf']
    logger.info(f"Task {arm_id}: 左图原始匹配点数量: {len(mkpts0_full_left)}")

    # 右图匹配（完整初始右图 vs 完整当前右图）
    match_res_right = await minima_matcher_service.match(initial_images['right_ir'], images['right_ir'])
    mkpts0_full_right, mkpts1_full_right, mconf_right = match_res_right['mkpts0'], match_res_right['mkpts1'], match_res_right['mconf']
    logger.info(f"Task {arm_id}: 右图原始匹配点数量: {len(mkpts0_full_right)}")

    # ========== 步骤3: 过滤匹配点并计算当前图像ROI边框 ==========
    logger.info(f"Task {arm_id}: 步骤3 - 过滤匹配点并计算当前图像ROI边框...")

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

    # ========== 步骤4: 调整ROI边框到相同大小并裁剪图像 ==========
    logger.info(f"Task {arm_id}: 步骤4 - 调整ROI边框到相同大小并裁剪图像...")

    cropped_current_left, cropped_current_right, final_bbox_left, final_bbox_right = crop_images_to_same_size(
        images['left_ir'], bbox2_left,
        images['right_ir'], bbox2_right
    )

    if cropped_current_left is None or cropped_current_right is None:
        logger.error(f"Task {arm_id}: 裁剪图像失败，使用全图")
        cropped_current_left = images['left_ir']
        cropped_current_right = images['right_ir']
        final_bbox_left = (0, 0, images['left_ir'].shape[1], images['left_ir'].shape[0])
        final_bbox_right = (0, 0, images['right_ir'].shape[1], images['right_ir'].shape[0])

    logger.info(f"Task {arm_id}: 调整后左图ROI边框: {final_bbox_left}")
    logger.info(f"Task {arm_id}: 调整后右图ROI边框: {final_bbox_right}")
    logger.info(f"Task {arm_id}: 左图ROI尺寸: {cropped_current_left.shape}")
    logger.info(f"Task {arm_id}: 右图ROI尺寸: {cropped_current_right.shape}")

    # 保存ROI图像和边框图像
    if debug_mode:
        roi_left_path = roi_output_directory / 'roi_left.png'
        roi_right_path = roi_output_directory / 'roi_right.png'
        cv2.imwrite(str(roi_left_path), cropped_current_left)
        cv2.imwrite(str(roi_right_path), cropped_current_right)

        current_left_with_roi = create_bbox_image(images['left_ir'], final_bbox_left, color=(255, 0, 0))
        current_right_with_roi = create_bbox_image(images['right_ir'], final_bbox_right, color=(255, 0, 0))
        cv2.imwrite(str(images_output_directory / 'current_left_with_roi.png'), current_left_with_roi)
        cv2.imwrite(str(images_output_directory / 'current_right_with_roi.png'), current_right_with_roi)

    # ========== 步骤5: 调整内参矩阵 ==========
    logger.info(f"Task {arm_id}: 步骤5 - 调整内参矩阵...")

    # 调整内参以适应裁剪
    K_left_adjusted = adjust_intrinsic_for_crop(K_mat, final_bbox_left)
    K_right_adjusted = adjust_intrinsic_for_crop(K_mat, final_bbox_right)

    logger.info(f"Task {arm_id}: 原始内参矩阵:\n{K_mat}")
    logger.info(f"Task {arm_id}: 调整后左图内参矩阵:\n{K_left_adjusted}")
    logger.info(f"Task {arm_id}: 调整后右图内参矩阵:\n{K_right_adjusted}")

    # ========== 立体匹配（ROI图像） ==========
    logger.info(f"Task {arm_id}: 运行ROI区域立体匹配...")
    inference_start_time = time.time()

    result = await stereo_matcher.infer(
        cropped_current_left,
        cropped_current_right,
        pred_mode=pre_mode,
        bidir_verify_th=bidir_verify_th
    )

    if debug_mode:
        await run_sync(save_disparity_map, result=result, output_dir=disparity_output_directory)
        logger.info(f"Task {arm_id}: Saved disparity map to: {disparity_output_directory}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping disparity map saving")

    total_end_time = time.time()
    logger.info(f"Task {arm_id}: Inference completed in {total_end_time - inference_start_time:.2f} seconds.")

    # ========== 深度图生成 ==========
    if 'disparity_verified' in result:
        logger.info(f"Task {arm_id}: Using 'disparity_verified' for depth calculation")
        disparity_roi = result['disparity_verified']
    else:
        logger.warning(f"Task {arm_id}: 'disparity_verified' not found in result, using 'disparity_left' instead")
        disparity_roi = result['disparity_left']

    # 处理ROI区域视差图
    disparity_roi = np.where(disparity_roi <= 0, 0.1, disparity_roi)

    # 将ROI视差图填充回原始图像大小
    height, width = images['left_ir'].shape[:2]
    full_disparity = np.zeros((height, width), dtype=disparity_roi.dtype)

    # 获取ROI边框坐标
    x1, y1, x2, y2 = map(int, final_bbox_left)
    h_roi, w_roi = disparity_roi.shape

    # 确保ROI区域在图像范围内
    y_end = min(y1 + h_roi, height)
    x_end = min(x1 + w_roi, width)
    h_fill = y_end - y1
    w_fill = x_end - x1

    # 填充ROI区域
    full_disparity[y1:y_end, x1:x_end] = disparity_roi[:h_fill, :w_fill]

    # 创建一个掩码，标记有效视差区域
    valid_mask = full_disparity > 0.1  # 视差大于0.1的为有效区域

    # 使用原始内参计算深度图
    fx = K_mat[0, 0]  # 使用K_mat获取fx，确保使用正确的内参矩阵
    baseline_m = baseline  # 基线已经是以米为单位（从配置读取）

    # 计算深度图（单位：米）
    depth_map = np.zeros_like(full_disparity, dtype=np.float32)
    depth_map[valid_mask] = (fx * baseline_m) / full_disparity[valid_mask]

    # 深度范围检查
    min_depth = np.min(depth_map[valid_mask]) if np.any(valid_mask) else 0
    max_depth = np.max(depth_map[valid_mask]) if np.any(valid_mask) else 0
    logger.info(f"Task {arm_id}: 计算后的深度范围: {min_depth:.3f} - {max_depth:.3f} 米")

    # 将深度转换为毫米，并保存为 16 位
    depth_map_mm = depth_map * 1000  # 转换为毫米
    depth_map_mm_clipped = np.clip(depth_map_mm, 0, 65535)  # 限制在16位范围内
    depth_map_uint16 = np.uint16(depth_map_mm_clipped)

    if debug_mode:
        output_path = os.path.join(output_depth_directory, f"stereo_depth.png")
        cv2.imwrite(output_path, depth_map_uint16)
        logger.info(f"Task {arm_id}: Saved depth map to {output_path}")

        # 保存视差图用于调试
        disparity_output_path = os.path.join(output_depth_directory, f"full_disparity.png")
        # 归一化显示，将0显示为黑色
        disp_normalized = cv2.normalize(full_disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_normalized = np.uint8(disp_normalized)
        cv2.imwrite(disparity_output_path, disp_normalized)
        logger.info(f"Task {arm_id}: Saved full disparity map to {disparity_output_path}")

        # 保存有效区域掩码
        mask_output_path = os.path.join(output_depth_directory, f"valid_mask.png")
        cv2.imwrite(mask_output_path, np.uint8(valid_mask * 255))
        logger.info(f"Task {arm_id}: Saved valid mask to {mask_output_path}")

        # 打印统计信息
        valid_pixels = np.sum(valid_mask)
        total_pixels = height * width
        logger.info(f"Task {arm_id}: 有效视差点数: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels*100:.1f}%)")
        logger.info(f"Task {arm_id}: 视差范围: {np.min(full_disparity[valid_mask]):.2f} - {np.max(full_disparity[valid_mask]):.2f}")
        logger.info(f"Task {arm_id}: 深度范围: {np.min(depth_map[valid_mask]):.2f} - {np.max(depth_map[valid_mask]):.2f} 米")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping depth map saving")

    # ========== 点云生成（使用原始内参） ==========
    # 先检查是否有有效深度点
    valid_pixel_count = np.sum(valid_mask)
    if valid_pixel_count < 100:  # 如果有效点太少
        logger.error(f"Task {arm_id}: 有效深度点太少 ({valid_pixel_count})，跳过点云生成")

        # 调试：保存深度图以供分析
        if debug_mode:
            debug_depth_path = os.path.join(output_depth_directory, f"debug_depth_numpy.npy")
            np.save(debug_depth_path, depth_map_uint16)
            logger.info(f"Task {arm_id}: 保存深度图到 {debug_depth_path} 用于调试")

        return

    # 打印深度图统计信息
    logger.info(f"Task {arm_id}: 深度图统计 - 最小值: {np.min(depth_map_uint16[depth_map_uint16>0]) if np.any(depth_map_uint16>0) else 0}, 最大值: {np.max(depth_map_uint16)}")

    # 生成点云
    pcd = await run_sync(depth_to_point_cloud,
        depth_map=depth_map_uint16,
        camera_intrinsics=K,  # 使用3x3内参矩阵
        max_distance=600,  # 增加最大距离限制
        min_distance=0,    # 添加最小距离限制
    )
    # 检查点云是否为空
    if pcd is None:
        logger.error(f"Task {arm_id}: 深度转点云返回None")
        return

    if len(pcd.points) == 0:
        logger.error(f"Task {arm_id}: 生成的点云为空，检查深度图输入")

        # 调试：检查深度图统计
        non_zero_depth = depth_map_uint16[depth_map_uint16 > 0]
        if len(non_zero_depth) > 0:
            logger.info(f"Task {arm_id}: 非零深度点数量: {len(non_zero_depth)}")
            logger.info(f"Task {arm_id}: 非零深度范围: {np.min(non_zero_depth)} - {np.max(non_zero_depth)}")
        else:
            logger.info(f"Task {arm_id}: 深度图全为零")

        return

    logger.info(f"Task {arm_id}: 生成点云点数: {len(pcd.points)}")
    # 检查点云是否为空
    if pcd is None or len(pcd.points) == 0:
        logger.error(f"Task {arm_id}: 生成的点云为空")
        return

    logger.info(f"Task {arm_id}: 生成点云点数: {len(pcd.points)}")

    if debug_mode:
        try:
            display_point_cloud_with_axes(pcd, None)
        except Exception as e:
            logger.warning(f"Task {arm_id}: 无法显示点云: {e}")

        target_path = f"{point_clouds_directory}/original_target.pcd"
        o3d.io.write_point_cloud(target_path, pcd)
        logger.info(f"Task {arm_id}: Saved point cloud to {target_path}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping original point cloud saving")

    # ========== MINIMA稀疏点云匹配部分 ==========
    current_pc1 = pcd

    current_pc2 = pcd

    #对稀疏当前点云进行预处理
    current_pc1 = await run_sync(preprocess_pointcloud,
        eye_hand_matrix=eye_hand_matrix,
        source_pcd=current_pc1,
        dimensions=cut_box,
        capture_pose=capture_pose,
        fueling_pose=fueling_pose,
        voxel_size=0.001,
        remove_outliers=False,
        radius=0.01,
        min_neighbors=5,
        downsample=False,
        is_crop=False
    )
    current_pc2 = await run_sync(preprocess_pointcloud,
        eye_hand_matrix=eye_hand_matrix,
        source_pcd=current_pc2,
        dimensions=cut_box,
        capture_pose=capture_pose,
        fueling_pose=fueling_pose,
        voxel_size=0.001,
        remove_outliers=False,
        radius=0.01,
        min_neighbors=5,
        downsample=False,
        is_crop=False
    )

    # ========== 当前点云在当前图像上的投影（使用原始内参） ==========
    if debug_mode:
        projected_img_path2_left = os.path.join(minima_output_directory, "projected_target_left.png")
        projected_img_path2_right = os.path.join(minima_output_directory, "projected_target_right.png")
        point_arr2_left, indices2_left, bbox2_left = project_pointcloud_to_image_float(current_pc1, images['left_ir'], K_mat, projected_img_path2_left, None)
        point_arr2_right, indices2_right, bbox2_right = project_pointcloud_to_image_float(current_pc1, images['right_ir'], K_mat, projected_img_path2_right, transform=T_left_to_right)
    else:
        point_arr2_left, indices2_left, bbox2_left = project_pointcloud_to_image_float(current_pc1, images['left_ir'], K_mat, None, None)
        point_arr2_right, indices2_right, bbox2_right = project_pointcloud_to_image_float(current_pc1, images['right_ir'], K_mat, None, None)

    # ========== 转换MINIMA匹配点到ROI坐标系 ==========
    logger.info(f"Task {arm_id}: 转换MINIMA匹配点到ROI坐标系...")

    # 转换左图匹配点
    mkpts1_roi_left = transform_points_to_roi_coordinates(mkpts1_filtered_left, final_bbox_left)

    # 转换右图匹配点
    mkpts1_roi_right = transform_points_to_roi_coordinates(mkpts1_filtered_right, final_bbox_right)

    # 转换当前点云投影点到ROI坐标系
    point_arr2_left_roi = transform_points_to_roi_coordinates(point_arr2_left, final_bbox_left)
    point_arr2_right_roi = transform_points_to_roi_coordinates(point_arr2_right, final_bbox_right)

    logger.info(f"Task {arm_id}: 匹配点已转换到ROI坐标系")

    # ========== 使用转换后的点进行双向匹配 ==========
    from fueling.minima.matching import compute_bidirectional_intersection_kdtree

    # 注意：这里使用ROI坐标系下的点
    intersection_results_kdtree_left = compute_bidirectional_intersection_kdtree(
        mkpts0_filtered_left, mkpts1_roi_left, point_arr1_left, point_arr2_left_roi, tolerance=1.0
    )
    logger.info(f"Task {arm_id}: intersection_results_kdtree (Left): {intersection_results_kdtree_left}")

    both_idx_left = intersection_results_kdtree_left["indices"]
    mkpts0_filtered_left_final = mkpts0_filtered_left[both_idx_left]
    mkpts1_filtered_left_final = mkpts1_roi_left[both_idx_left]
    mconf_filtered_left = mconf_left[both_idx_left] if len(mconf_left) > 0 else np.array([])
    logger.info(f"Task {arm_id}: [最终使用] 双向都在投影内的匹配对数量 (Left): {len(both_idx_left)}")

    if debug_mode and len(both_idx_left) > 0:
        # 注意：这里需要将ROI坐标转换回完整图像坐标进行可视化
        mkpts1_full_coords = mkpts1_filtered_left_final.copy()
        mkpts1_full_coords[:, 0] += final_bbox_left[0]
        mkpts1_full_coords[:, 1] += final_bbox_left[1]
        visualize_matches(initial_images['left_ir'], images['left_ir'],
                         mkpts0_filtered_left_final, mkpts1_full_coords,
                         str(minima_output_directory), "matches_left_roi")

    intersection_results_kdtree_right = compute_bidirectional_intersection_kdtree(
        mkpts0_filtered_right, mkpts1_roi_right, point_arr1_right, point_arr2_right_roi, tolerance=1.0
    )
    logger.info(f"Task {arm_id}: intersection_results_kdtree (Right): {intersection_results_kdtree_right}")

    both_idx_right = intersection_results_kdtree_right["indices"]
    mkpts0_filtered_right_final = mkpts0_filtered_right[both_idx_right]
    mkpts1_filtered_right_final = mkpts1_roi_right[both_idx_right]
    mconf_filtered_right = mconf_right[both_idx_right] if len(mconf_right) > 0 else np.array([])

    if debug_mode and len(both_idx_right) > 0:
        # 注意：这里需要将ROI坐标转换回完整图像坐标进行可视化
        mkpts1_full_coords = mkpts1_filtered_right_final.copy()
        mkpts1_full_coords[:, 0] += final_bbox_right[0]
        mkpts1_full_coords[:, 1] += final_bbox_right[1]
        visualize_matches(initial_images['right_ir'], images['right_ir'],
                         mkpts0_filtered_right_final, mkpts1_full_coords,
                         str(minima_output_directory), "matches_right_roi")

    logger.info(f"Task {arm_id}: [最终使用] 双向都在投影内的匹配对数量 (Right): {len(both_idx_right)}")

    # ========== 创建稀疏点云 ==========
    from fueling.minima.pointcloud import create_sparse_pointclouds_from_bidirectional_matches_float

    sparse_pc1_left, sparse_pc2_left = create_sparse_pointclouds_from_bidirectional_matches_float(
        source_pc, current_pc1,
        point_arr1_left, point_arr2_left_roi,
        indices1_left, indices2_left,
        mkpts0_filtered_left_final, mkpts1_filtered_left_final
    )
    logger.info(f"Task {arm_id}: 稀疏点云对数量 (Left): {len(sparse_pc1_left.points)}")

    sparse_pc1_right, sparse_pc2_right = create_sparse_pointclouds_from_bidirectional_matches_float(
        source_pc, current_pc1,
        point_arr1_right, point_arr2_right_roi,
        indices1_right, indices2_right,
        mkpts0_filtered_right_final, mkpts1_filtered_right_final
    )
    logger.info(f"Task {arm_id}: 稀疏点云对数量 (Right): {len(sparse_pc1_right.points)}")

    # 合并左右两边的稀疏点云
    sparse_pc1 = sparse_pc1_left + sparse_pc1_right
    sparse_pc2 = sparse_pc2_left + sparse_pc2_right

    if debug_mode:
        o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc1_left.pcd"), sparse_pc1_left)
        o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc2_left.pcd"), sparse_pc2_left)
        o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc1_right.pcd"), sparse_pc1_right)
        o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc2_right.pcd"), sparse_pc2_right)
        o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc1_combined.pcd"), sparse_pc1)
        o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc2_combined.pcd"), sparse_pc2)

    logger.info(f"Task {arm_id}: 合并后的稀疏点云对数量: {len(sparse_pc1.points)}")

    if debug_mode and len(sparse_pc1.points) > 0 and len(sparse_pc2.points) > 0:
        # 可视化稀疏点云对
        visualize_sparse_pointclouds(sparse_pc1, sparse_pc2, "Sparse Point Clouds Before Registration")

    # 使用SuperANSAC进行离群点剔除
    if len(sparse_pc1.points) > 10 and len(sparse_pc2.points) > 10:
        source_sparse_filtered, target_sparse_filtered, _, T_superansac = filter_outliers_by_superansac(
            sparse_pc1, sparse_pc2,
        )
        logger.info(f"Task {arm_id}: SuperANSAC Inliers: {len(source_sparse_filtered.points)}")
        if debug_mode and len(source_sparse_filtered.points) > 0:
            visualize_sparse_pointclouds(source_sparse_filtered, target_sparse_filtered, "Sparse Point Clouds After SuperANSAC")
    else:
        source_sparse_filtered, target_sparse_filtered = sparse_pc1, sparse_pc2
        T_superansac = np.identity(4)
        logger.info(f"Task {arm_id}: 点云数量不足，跳过SuperANSAC")

    # ========== 稀疏点云配准 ==========
    try:
        # 使用过滤后的稀疏点云进行配准
        sparse_registration = PointCloudRegistration(
            source_pc=source_sparse_filtered,
            method="filterreg",
            voxel_size=0.001,
            remove_outliers=False
        )
        sparse_T_Ca2_from_Ca1 = await run_sync(sparse_registration.compute_registration, target_pc=target_sparse_filtered)
        logger.info(f"Task {arm_id}: ICP Registration completed.{sparse_T_Ca2_from_Ca1}")
    except Exception as e:
        logger.error(f"Task {arm_id}: ICP Registration failed: {e}")
        return

    # 可视化稀疏配准后的点云
    if debug_mode and len(source_sparse_filtered.points) > 0:
        display_point_cloud(source_sparse_filtered, target_sparse_filtered, title="配准前sparse_source和sparse_target点云")
        transformed_sparse_source = source_sparse_filtered.transform(sparse_T_Ca2_from_Ca1)
        display_point_cloud(transformed_sparse_source, target_sparse_filtered, title="配准后sparse_source和sparse_target点云")

        logger.info(f"可视化初始稀疏点云配准结果")

        # 保存变换矩阵和点云
        np.savetxt(f"{point_clouds_directory}/T_Ca2_from_Ca1.txt", sparse_T_Ca2_from_Ca1)
        o3d.io.write_point_cloud(f"{point_clouds_directory}/source_pc_transformed.pcd", transformed_sparse_source)

    # ========== 计算加油位姿 ==========
    robot_fueling_pose = compute_transformed_fueling_pose(eye_hand_matrix, capture_pose, fueling_pose, sparse_T_Ca2_from_Ca1)

    # 保存加油位姿
    if debug_mode:
        output_config_path = f'{config_output_directory}/robot_fueling_pose.json'
        with open(output_config_path, 'w') as f:
            json.dump(robot_fueling_pose.tolist(), f, indent=4)
        logger.info(f"Task {arm_id}: Updated config and saved to {output_config_path}")
    else:
        logger.info(f"Task {arm_id}: Config updated in memory, debug mode disabled so not saving to file")

    # ========== 保存ROI测试报告 ==========
    if debug_mode:
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

        report_path = arm_dir / 'roi_test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        logger.info(f"Task {arm_id}: ROI测试报告已保存到 {report_path}")

    # ========== 执行加油动作 ==========
    upper_pose = get_upper_pose(robot_fueling_pose, offset=100)

    # 注释掉实际移动代码，仅用于测试
    # await robot_client.move(upper_pose)
    # await robot_client.move(robot_fueling_pose)
    # await robot_client.move(upper_pose)
    # await robot_client.move(init_pose)

    print(f"Task {arm_id}: Total time from capture to inference end: {total_end_time - total_start_time:.2f} seconds.")
    logger.info(f"Task {arm_id}: ROI-based pipeline测试完成")


async def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    arm_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('arm')])
    config_files = [os.path.join(data_dir, arm_dir, "config.jsonnet") for arm_dir in arm_dirs]
    config_files = [f for f in config_files if os.path.isfile(f)][0:1] # 暂时1台

    proj_dir = Path(__file__).resolve().parent.parent

    default_config = f"{data_dir}/default_config.jsonnet"
    default_set = json.loads(_jsonnet.evaluate_file(default_config))
    max_parallel = default_set['stereo_matcher']['max_parallel']
    model_name = default_set['stereo_matcher']['method']
    model_path = default_set['stereo_matcher'][model_name]['model_path']
    weight_path = f"{proj_dir}{model_path}"

    stereo_matcher = StereoMatcherService(model_name, weight_path, max_parallel)

    # 初始化MINIMA服务
    minima_config = default_set.get('minima', {})
    minima_model_path = minima_config['model_path']
    minima_weight_path = f"{proj_dir}{minima_model_path}"
    logger.info(f"加载MINIMA模型: {minima_weight_path}")

    minima_matcher_service = MinimaMatcherService(minima_weight_path)
    import asyncio
    asyncio.create_task(minima_matcher_service.loop_process_items())

    if not config_files:
        logger.error("没有找到有效的配置文件")
        return 1
    logger.info(f"Found config files: {config_files}")

    robot_clients = []
    pc_registrations = []
    configs = []
    source_pcs = []
    delays = []
    initial_images_list = []

    for i, config_file in enumerate(config_files):
        config = json.loads(_jsonnet.evaluate_file(config_file))

        # Load robot poses from robot_pose.json
        target_pot = config['robot']['target_pot']
        robot_pose_path = os.path.join(data_dir, f'arm{i + 1}', target_pot, 'robot_pose.json')
        if os.path.exists(robot_pose_path):
            with open(robot_pose_path, 'r') as f:
                robot_poses = json.load(f)
                config['robot']['init_pose'] = robot_poses['init_pose']
                config['robot']['capture_pose'] = robot_poses['capture_pose']
                config['robot']['fueling_pose'] = robot_poses['fueling_pose']
                logger.info(f"Loaded robot poses from {robot_pose_path}")
        else:
            logger.error(f"Robot pose file not found at {robot_pose_path}, using placeholder poses.")

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
        robot_clients.append(robot_client)

        target_pot = config['robot']['target_pot']
        source_down_path = os.path.join(data_dir, f'arm{i + 1}', target_pot, 'source_model.pcd')
        source_pc = o3d.io.read_point_cloud(source_down_path)
        if not source_pc.has_points():
            logger.error(f"Failed to read source point cloud from {source_down_path} or point cloud is empty.")
            continue
        pc_registration = PointCloudRegistration(
            source_pc=source_pc,
            method="filterreg",
            voxel_size=config['point_cloud']['voxel_size'],
            remove_outliers=config['point_cloud']['remove_outliers'],
            radius=config['point_cloud']['radius'],
            min_neighbors=config['point_cloud']['min_neighbors']
        )
        pc_registrations.append(pc_registration)
        configs.append(config)
        source_pcs.append(source_pc)
        delays.append(config['robot']['delay'])

        # 加载初始图像 - 从配置文件中读取路径
        left_ir_path = config['minima']['left_ir_path']
        right_ir_path = config['minima']['right_ir_path']

        # 如果路径是相对路径，转换为绝对路径
        if not os.path.isabs(left_ir_path):
            left_ir_path = os.path.join(proj_dir, left_ir_path)
        if not os.path.isabs(right_ir_path):
            right_ir_path = os.path.join(proj_dir, right_ir_path)

        # 检查文件是否存在，如果不存在尝试其他可能的文件名
        if not os.path.exists(left_ir_path):
            left_dir = os.path.dirname(left_ir_path)
            if os.path.exists(left_dir):
                possible_files = [f for f in os.listdir(left_dir) if 'left' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if possible_files:
                    left_ir_path = os.path.join(left_dir, possible_files[0])
                else:
                    raise FileNotFoundError(f"Left IR image not found at {left_ir_path} and no alternative found in {left_dir}")
            else:
                raise FileNotFoundError(f"Left IR image directory not found: {left_dir}")

        if not os.path.exists(right_ir_path):
            right_dir = os.path.dirname(right_ir_path)
            if os.path.exists(right_dir):
                possible_files = [f for f in os.listdir(right_dir) if 'right' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if possible_files:
                    right_ir_path = os.path.join(right_dir, possible_files[0])
                else:
                    raise FileNotFoundError(f"Right IR image not found at {right_ir_path} and no alternative found in {right_dir}")
            else:
                raise FileNotFoundError(f"Right IR image directory not found: {right_dir}")

        initial_images = {
            'left_ir': cv2.imread(left_ir_path, cv2.IMREAD_GRAYSCALE),
            'right_ir': cv2.imread(right_ir_path, cv2.IMREAD_GRAYSCALE)
        }
        if initial_images['left_ir'] is None or initial_images['right_ir'] is None:
            raise FileNotFoundError(f"Initial IR images not found or cannot be read: {left_ir_path}, {right_ir_path}")

        logger.info(f"Loaded initial images for arm{i+1}: {left_ir_path}, {right_ir_path}")
        initial_images_list.append(initial_images)

    async def run_with_delay(i, delay):
        if delay > 0:
            logger.info(f"机械臂 {i} 延迟 {delay}s 启动")
            await anyio.sleep(delay)
        await run_fuel(i, stereo_matcher, robot_clients[i], pc_registrations[i], configs[i], source_pcs[i], initial_images_list[i], minima_matcher_service)

    async with anyio.create_task_group() as tg:
        tg.start_soon(stereo_matcher.loop_process_items)
        for i, delay in enumerate(delays):
            logger.info(f"Starting task {i} for {config_files[i]}")
            tg.start_soon(run_with_delay, i, delay)


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
        print(f"All tasks completed in {end_time - start_time:.2f} seconds.")