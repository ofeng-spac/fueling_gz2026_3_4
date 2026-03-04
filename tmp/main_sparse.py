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
from registration_evaluator import RegistrationEvaluator, quick_evaluate

async def run_fuel(arm_id: int, stereo_matcher: StereoMatcherService,
                   robot_client: AsyncRobotClient, pc_registration: PointCloudRegistration, config: dict,
                   source_pc: o3d.geometry.PointCloud, initial_images: dict, minima_matcher_service: MinimaMatcherService):
    """
    Main function to run the end-to-end camera capture and inference pipeline.

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

        # 一次性创建所有目录
        for dir_path in [images_output_directory, disparity_output_directory,
                        output_depth_directory, point_clouds_directory,
                        config_output_directory, log_dir, minima_output_directory]:
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

    # ========== 初始图像投影计算 ==========
    # 使用源点云在初始图像上的投影
    K_mat = np.array(config['robot']['RS_camera']['K']).reshape(3, 3)
    from fueling.minima.geometry import project_pointcloud_to_image_float

    # ========== 初始图像投影计算 ==========
    # 使用源点云在初始图像上的投影
    K_mat = np.array(config['robot']['RS_camera']['K']).reshape(3, 3)
    from fueling.minima.geometry import project_pointcloud_to_image_float

    # 修改后：先计算左图，然后计算右图时传入左图的边框
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

    # ========== 立体匹配 ==========
    logger.info(f"Task {arm_id}: Running inference on captured images...")
    inference_start_time = time.time()

    result = await stereo_matcher.infer(images['left_ir'], images['right_ir'], pred_mode=pre_mode, bidir_verify_th=bidir_verify_th)
    if debug_mode:
        await run_sync(save_disparity_map, result=result, output_dir=disparity_output_directory)  # type: ignore (no-untyped-call)
        logger.info(f"Task {arm_id}: Saved disparity map to: {disparity_output_directory}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping disparity map saving")

    total_end_time = time.time()
    logger.info(f"Task {arm_id}: Inference completed in {total_end_time - inference_start_time:.2f} seconds.")

    # ========== 深度图生成 ==========
    if 'disparity_verified' in result:
        disparity_map = result['disparity_verified']
    else:
        disparity_map = result['disparity_left']

    disparity_map = np.where(disparity_map <= 0, 0.1, disparity_map)
    depth_map = (K[0] * baseline) / disparity_map
    depth_map_uint16 = np.uint16(depth_map * 1000)  # 将深度转换为毫米，并保存为 16 位

    if debug_mode:
        output_path = os.path.join(output_depth_directory, f"stereo_depth.png")
        cv2.imwrite(output_path, depth_map_uint16)  # type: ignore (no-untyped-call)
        logger.info(f"Task {arm_id}: Saved depth map to {output_path}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping depth map saving")

    # ========== 点云生成 ==========
    pcd = await run_sync(depth_to_point_cloud,  # type: ignore (no-untyped-call)
        depth_map=np.array(depth_map_uint16),
        camera_intrinsics=K,
        max_distance=600,
    )

    if debug_mode:
        display_point_cloud_with_axes(pcd, None)
        target_path = f"{point_clouds_directory}/original_target.pcd"
        o3d.io.write_point_cloud(target_path, pcd)
        logger.info(f"Task {arm_id}: Saved point cloud to {target_path}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping original point cloud saving")

    # ========== MINIMA稀疏点云匹配部分 ==========
    current_pc1 = pcd
    current_pc2 = pcd

    # 对稀疏当前点云进行预处理
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
        downsample=False
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
        downsample=False
    )

    # ========== 当前点云在当前图像上的投影 ==========
    if debug_mode:
        projected_img_path2_left = os.path.join(minima_output_directory, "projected_target_left.png")
        projected_img_path2_right = os.path.join(minima_output_directory, "projected_target_right.png")
        point_arr2_left, indices2_left, bbox2_left = project_pointcloud_to_image_float(current_pc1, images['left_ir'], K_mat, projected_img_path2_left, None)
        point_arr2_right, indices2_right, bbox2_right = project_pointcloud_to_image_float(current_pc1, images['right_ir'], K_mat, projected_img_path2_right, transform=T_left_to_right)
    else:
        point_arr2_left, indices2_left, bbox2_left = project_pointcloud_to_image_float(current_pc1, images['left_ir'], K_mat, None, None)
        point_arr2_right, indices2_right, bbox2_right = project_pointcloud_to_image_float(current_pc1, images['right_ir'], K_mat, None, None)
    match_res_left = await minima_matcher_service.match(initial_images['left_ir'], images['left_ir'])
    match_res_right = await minima_matcher_service.match(initial_images['right_ir'], images['right_ir'])
    import asyncio
    mkpts0_orig_left, mkpts1_orig_left, mconf_orig_left = match_res_left['mkpts0'], match_res_left['mkpts1'], match_res_left['mconf']
    logger.info(f"Task {arm_id}: MINIMA原始匹配点数量 (Left): {len(mkpts0_orig_left)}")
    asyncio.create_task(minima_matcher_service.loop_process_items())
    mkpts0_orig_right, mkpts1_orig_right, mconf_orig_right = match_res_right['mkpts0'], match_res_right['mkpts1'], match_res_right['mconf']
    logger.info(f"Task {arm_id}: MINIMA原始匹配点数量 (Right): {len(mkpts0_orig_right)}")
    asyncio.create_task(minima_matcher_service.loop_process_items())
    from fueling.minima.matching import compute_bidirectional_intersection_kdtree

    intersection_results_kdtree_left = compute_bidirectional_intersection_kdtree(
            mkpts0_orig_left, mkpts1_orig_left, point_arr1_left, point_arr2_left, tolerance=1.0
        )
    logger.info(f"Task {arm_id}: intersection_results_kdtree (Left): {intersection_results_kdtree_left}")

    both_idx_left = intersection_results_kdtree_left["indices"]
    mkpts0_filtered_left = mkpts0_orig_left[both_idx_left]
    mkpts1_filtered_left = mkpts1_orig_left[both_idx_left]
    mconf_filtered_left = mconf_orig_left[both_idx_left]
    logger.info(f"Task {arm_id}: [最终使用] 双向都在投影内的匹配对数量 (Left): {len(both_idx_left)}")

    if debug_mode and len(both_idx_left) > 0:
        visualize_matches(initial_images['left_ir'], images['left_ir'], mkpts0_filtered_left, mkpts1_filtered_left, str(minima_output_directory), "matches_left")

    intersection_results_kdtree_right = compute_bidirectional_intersection_kdtree(
            mkpts0_orig_right, mkpts1_orig_right, point_arr1_right, point_arr2_right, tolerance=1.0
        )
    logger.info(f"Task {arm_id}: intersection_results_kdtree (Right): {intersection_results_kdtree_right}")

    both_idx_right = intersection_results_kdtree_right["indices"]
    mkpts0_filtered_right = mkpts0_orig_right[both_idx_right]
    mkpts1_filtered_right = mkpts1_orig_right[both_idx_right]
    if debug_mode and len(both_idx_right) > 0:
        visualize_matches(initial_images['right_ir'], images['right_ir'], mkpts0_filtered_right, mkpts1_filtered_right, str(minima_output_directory), "matches_right")

    mconf_filtered_right = mconf_orig_right[both_idx_right]
    logger.info(f"Task {arm_id}: [最终使用] 双向都在投影内的匹配对数量 (Right): {len(both_idx_right)}")

    from fueling.minima.pointcloud import  create_sparse_pointclouds_from_bidirectional_matches_float
    sparse_pc1_left, sparse_pc2_left = create_sparse_pointclouds_from_bidirectional_matches_float(
        source_pc, current_pc1,
        point_arr1_left, point_arr2_left,
        indices1_left, indices2_left,
        mkpts0_filtered_left, mkpts1_filtered_left
    )
    logger.info(f"Task {arm_id}: 稀疏点云对数量 (Left): {len(sparse_pc1_left.points)}")

    sparse_pc1_right, sparse_pc2_right = create_sparse_pointclouds_from_bidirectional_matches_float(
        source_pc, current_pc1,
        point_arr1_right, point_arr2_right,
        indices1_right, indices2_right,
        mkpts0_filtered_right, mkpts1_filtered_right
    )
    logger.info(f"Task {arm_id}: 稀疏点云对数量 (Right): {len(sparse_pc1_right.points)}")

    # 合并左右两边的稀疏点云
    sparse_pc1 = sparse_pc1_left + sparse_pc1_right
    sparse_pc2 = sparse_pc2_left + sparse_pc2_right
    o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc1_left.pcd"), sparse_pc1_left)
    o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc2_left.pcd"), sparse_pc2_left)
    o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc1_right.pcd"), sparse_pc1_right)
    o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc2_right.pcd"), sparse_pc2_right)
    o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc1_combined.pcd"), sparse_pc1)
    o3d.io.write_point_cloud(os.path.join(minima_output_directory, "sparse_pc2_combined.pcd"), sparse_pc2)
    logger.info(f"Task {arm_id}: 合并后的稀疏点云对数量: {len(sparse_pc1.points)}")

    if debug_mode:
        # 可视化稀疏点云对
        visualize_sparse_pointclouds(sparse_pc1, sparse_pc2, "Sparse Point Clouds Before Registration")

    # 使用SuperANSAC进行离群点剔除
    if len(sparse_pc1.points) > 10 and len(sparse_pc2.points) > 10:
        source_sparse_filtered, target_sparse_filtered, _, T_superansac = filter_outliers_by_superansac(
            sparse_pc1, sparse_pc2,
            # distance_threshold=0.01,
            # confidence=0.99,
            # iterations=1000
        )
        logger.info(f"Task {arm_id}: SuperANSAC Inliers: {len(source_sparse_filtered.points)}")
        if debug_mode:
            visualize_sparse_pointclouds(source_sparse_filtered, target_sparse_filtered, "Sparse Point Clouds After SuperANSAC")
    else:
        source_sparse_filtered, target_sparse_filtered = sparse_pc1, sparse_pc2
        T_superansac = np.identity(4)
        logger.info(f"Task {arm_id}: 点云数量不足，跳过SuperANSAC")




    # try: #稀疏点云配准
    #     sparse_T_Ca2_from_Ca1 = await run_sync(pc_registration.compute_registration, target_pc = target_sparse_filtered)
    #     logger.info(f"Task {arm_id}: Sparse Registration completed.{sparse_T_Ca2_from_Ca1}")
    # except Exception as e:
    #     logger.error(f"Task {arm_id}: Sparse Registration failed: {e}")
    #     return

    try: #稀疏点云配准
        pc_registration = PointCloudRegistration(
            source_pc=source_sparse_filtered,
            method="filterreg",
            voxel_size=0.001,
            remove_outliers=False
        )
        sparse_T_Ca2_from_Ca1 = await run_sync(pc_registration.compute_registration, target_pc=target_sparse_filtered)
        logger.info(f"Task {arm_id}: ICP Registration completed.{sparse_T_Ca2_from_Ca1}")
        if debug_mode:
            # 准备对应点数据
            source_pts = np.asarray(source_sparse_filtered.points)
            target_pts = np.asarray(target_sparse_filtered.points)

            # 评估方案2 (有对应关系)
            eval_metrics = quick_evaluate(
                source_pcd=source_sparse_filtered,
                target_pcd=target_sparse_filtered,
                transformation=sparse_T_Ca2_from_Ca1,
                correspondences=(source_pts, target_pts),  # 稀疏对应点
                method_name="Sparse_MINIMA",
                debug_mode=True,
                output_dir=arm_dir
            )
            logger.info(f"配准评估结果: {eval_metrics}")

    except Exception as e:
        logger.error(f"Task {arm_id}: ICP Registration failed: {e}")
        return

    # 可视化稀疏配准后的点云
    if debug_mode:
        display_point_cloud(source_sparse_filtered, target_sparse_filtered, title="配准前sparse_source和sparse_target点云")
        transformed_sparse_source = source_sparse_filtered.transform(sparse_T_Ca2_from_Ca1) # type: ignore (source is not None here)
        display_point_cloud(transformed_sparse_source, target_sparse_filtered, title="配准后sparse_source和sparse_target点云")

        logger.info(f"可视化初始稀疏点云配准结果")

        # 保存变换矩阵和点云
        np.savetxt(f"{point_clouds_directory}/T_Ca2_from_Ca1.txt", sparse_T_Ca2_from_Ca1)
        o3d.io.write_point_cloud(f"{point_clouds_directory}/source_pc_transformed.pcd", transformed_sparse_source)

    robot_fueling_pose = compute_transformed_fueling_pose(eye_hand_matrix, capture_pose, fueling_pose, sparse_T_Ca2_from_Ca1)

    # result.json
    if debug_mode:
        output_config_path = f'{config_output_directory}/robot_fueling_pose.json'
        with open(output_config_path, 'w') as f:
            json.dump(robot_fueling_pose.tolist(), f, indent=4)
        logger.info(f"Task {arm_id}: Updated config and saved to {output_config_path}")
    else:
        logger.info(f"Task {arm_id}: Config updated in memory, debug mode disabled so not saving to file")

    upper_pose = get_upper_pose(robot_fueling_pose, offset=100)

    await robot_client.move(upper_pose)
    await robot_client.move(robot_fueling_pose)
    await robot_client.move(upper_pose)
    await robot_client.move(init_pose)

    print(f"Task {arm_id}: Total time from capture to inference end: {total_end_time - total_start_time:.2f} seconds.")


async def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    arm_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('arm')])
    config_files = [os.path.join(data_dir, arm_dir, "config.jsonnet") for arm_dir in arm_dirs]
    config_files = [f for f in config_files if os.path.isfile(f)][0:1] # 暂时2台

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
    anyio.run(main, backend='asyncio')
    end_time = time.time()
    print(f"All tasks completed in {end_time - start_time:.2f} seconds.")