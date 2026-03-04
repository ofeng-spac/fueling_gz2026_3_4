import os
import sys
import cv2
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import time
import torch
import json
import anyio
import _jsonnet
import numpy as np
import open3d as o3d
import asyncio
from pathlib import Path
from loguru import logger
from datetime import datetime
from fueling.task import run_sync
from fueling.obcamera import AsyncOrbbecCamera
from fueling.pose_transformation import get_upper_pose
from fueling.drawing import display_point_cloud_with_axes, display_point_cloud
from fueling.pointcloud_processor import PointCloudRegistration, depth_to_point_cloud, preprocess_pointcloud
from fueling.stereo_matcher import save_disparity_map
from fueling.robot_control import AsyncRobotClient, compute_transformed_fueling_pose
from fueling.stereo_matcher.stereo_service import StereoMatcherService
from fueling.test_cam_pose_reg_result import compute_registration_error, compare_images_pixel_perfect, calculate_target_robot_pose

logger.add("demo.log", rotation="10 MB", level="INFO")  # type: ignore (no-untyped-call)

async def run_fuel(arm_id: int, stereo_matcher: StereoMatcherService,
                   robot_client: AsyncRobotClient, pc_registration: PointCloudRegistration, config: dict,
                   source_pc: o3d.geometry.PointCloud):
    """
    Main function to run the end-to-end camera capture and inference pipeline.

    Args:
        task_id: Unique identifier for this task (0-3)
    """

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
    debug_mode = config['robot']['debug_mode']
    pre_mode = config['stereo_matcher']['pred_mode']
    bidir_verify_th = config['stereo_matcher']['bidir_verify_th']
    strategy = config['robot']['strategy']

    await robot_client.move(capture_pose)

    logger.info(f"Task {arm_id}: Moved to capture position.")

    total_start_time = time.time()
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


    if debug_mode:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_working_dir = Path(f"../working_data/{timestamp}")
        base_working_dir.mkdir(parents=True, exist_ok=True)
        arm_dir = base_working_dir / f'arm_{arm_id + 1}'

        images_output_directory = arm_dir / 'ir_images'
        disparity_output_directory = arm_dir / 'disparities'
        output_depth_directory = arm_dir / 'depth'
        point_clouds_directory = arm_dir / 'point_clouds'
        config_output_directory = arm_dir / 'config'
        log_dir = arm_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / 'test.log', level='INFO', enqueue=True)  # type: ignore (no-untyped-call)
    else:
        images_output_directory = f'captured_images/task_{arm_id}'
        disparity_output_directory = f'disparity_output/task_{arm_id}'
        output_depth_directory = f"depth_output/task_{arm_id}"
        point_clouds_directory = f'pointclouds_output/task_{arm_id}'

    logger.info(f"Task {arm_id}: Capturing IR image pair...")
    try:
        images = await obcamera.capture_stereo_ir(exposure, gain, 'flood')
        logger.info(f"Task {arm_id}: Image pair captured.")
    except Exception as e:
        logger.error(f"Task {arm_id}: Error: Failed to capture images: {e}")
        return

    if debug_mode:
        images_output_dir = Path(images_output_directory)
        images_output_dir.mkdir(parents=True, exist_ok=True)

        disparity_output_dir = Path(disparity_output_directory)
        disparity_output_dir.mkdir(parents=True, exist_ok=True)

        output_depth_dir = Path(output_depth_directory)
        output_depth_dir.mkdir(parents=True, exist_ok=True)

        point_clouds_dir = Path(point_clouds_directory)
        point_clouds_dir.mkdir(parents=True, exist_ok=True)

        config_output_dir = Path(config_output_directory)
        config_output_dir.mkdir(parents=True, exist_ok=True)

    if debug_mode:
        output_config_path = f'{config_output_dir}/config{arm_id}.json'
        with open(output_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Task {arm_id}: Updated config and saved to {output_config_path}")
    else:
        logger.info(f"Task {arm_id}: Config updated in memory, debug mode disabled so not saving to file")

    if debug_mode:
        captured_left_path = f'{images_output_dir}/captured_left_ir.png'
        captured_right_path = f'{images_output_dir}/captured_right_ir.png'
        cv2.imwrite(str(captured_left_path), images['left_ir'])
        cv2.imwrite(str(captured_right_path), images['right_ir'])
        logger.info(f"Task {arm_id}: Saved captured images for verification to: {images_output_dir}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping image saving")

    logger.info(f"Task {arm_id}: Running inference on captured images...")
    inference_start_time = time.time()

    result = await stereo_matcher.infer(images['left_ir'], images['right_ir'], pred_mode = pre_mode, bidir_verify_th = bidir_verify_th)
    if debug_mode:
        await run_sync(save_disparity_map, result = result, output_dir=disparity_output_dir) # type: ignore (no-untyped-call)
        logger.info(f"Task {arm_id}: Saved disparity map to: {disparity_output_dir}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping disparity map saving")
    total_end_time = time.time()
    logger.info(f"Task {arm_id}: Inference completed in {total_end_time - inference_start_time:.2f} seconds.")

    if 'disparity_verified' in result:
        disparity_map = result['disparity_verified']
    else:
        disparity_map = result['disparity_left']

    disparity_map = np.where(disparity_map <= 0, 0.1, disparity_map)

    depth_map = (K[0] * baseline) / disparity_map
    depth_map_uint16 = np.uint16(depth_map * 1000)  # 将深度转换为毫米，并保存为 16 位
    if debug_mode:
        output_path = os.path.join(output_depth_dir, f"stereo_depth.png")
        cv2.imwrite(output_path, depth_map_uint16) # type: ignore (no-untyped-call)
        logger.info(f"Task {arm_id}: Saved depth map to {output_path}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping depth map saving")

    pcd = await run_sync(depth_to_point_cloud, # type: ignore (no-untyped-call)
        depth_map=np.array(depth_map_uint16),
        camera_intrinsics=K,
        max_distance=600,
    )

    if debug_mode:
        display_point_cloud_with_axes(pcd, None)
        target_path = f"{point_clouds_dir}/original_target.pcd"
        o3d.io.write_point_cloud(target_path, pcd)
        logger.info(f"Task {arm_id}: Saved point cloud to {target_path}")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping original point cloud saving")

    # preprocess_pointcloud, cut + subsampling
    target = await run_sync(preprocess_pointcloud, # type: ignore (no-untyped-call)
        eye_hand_matrix=eye_hand_matrix,
        source_pcd=pcd,
        dimensions=cut_box,
        capture_pose=capture_pose,
        fueling_pose=fueling_pose,
        voxel_size=voxel_size,
        remove_outliers=remove_outliers,
        radius=radius,
        min_neighbors=min_neighbors
    )
    if debug_mode:
        display_point_cloud(target, None)

    if debug_mode:
        if target is not None and len(target.points) > 0:
            o3d.io.write_point_cloud(f"{point_clouds_dir}/target_preprocess.pcd", target)
            logger.info(f"Task {arm_id}: Saved cut point cloud to {point_clouds_dir}/target_preprocess.pcd")
        else:
            logger.warning(f"Task {arm_id}: Cut point cloud is empty, skipping save")
    else:
        logger.info(f"Task {arm_id}: Debug mode disabled, skipping cut point cloud saving")

    if target is None or len(target.points) == 0:
        logger.error(f"Task {arm_id}: Cut point cloud is empty or None, cannot proceed with registration")
        return
    try:
        # rename param to 4x4 transformation matrix, change naming convention
        T_Ca2_from_Ca1 = await run_sync(pc_registration.compute_registration, target_pc = target)
        logger.info(f"Task {arm_id}: Registration completed.{T_Ca2_from_Ca1}")
    except Exception as e:
        logger.error(f"Task {arm_id}: Registration failed: {e}")
        return

    # 可视化配准后的点云 (如果不是自动运行)
    if debug_mode:
        logger.info(f"可视化初始source和target点云")
        transformed_source = source_pc.transform(T_Ca2_from_Ca1) # type: ignore (source is not None here)
        display_point_cloud(transformed_source, target)

    mean_error, inlier_rmse, fitness = compute_registration_error(source_pc, target, T_Ca2_from_Ca1, max_correspondence_distance=5.0)
    logger.info(f"Task {arm_id}: registration errors - Mean Error: {mean_error}, Inlier RMSE: {inlier_rmse}, Fitness: {fitness}")

    robot_fueling_pose = compute_transformed_fueling_pose(eye_hand_matrix, capture_pose, fueling_pose, T_Ca2_from_Ca1)

    # result.json
    if debug_mode:
        output_config_path = f'{config_output_dir}/robot_fueling_pose.json'
        with open(output_config_path, 'w') as f:
            json.dump(robot_fueling_pose.tolist(), f, indent=4)
        logger.info(f"Task {arm_id}: Updated config and saved to {output_config_path}")
    else:
        logger.info(f"Task {arm_id}: Config updated in memory, debug mode disabled so not saving to file")

    upper_pose = get_upper_pose(robot_fueling_pose, offset=100)

    if strategy == 'method2':
        new_cap_pose = calculate_target_robot_pose(capture_pose, eye_hand_matrix, T_Ca2_from_Ca1)
        logger.info(f"Task {arm_id}: Calculated new capture pose for method2: {new_cap_pose}")
        await robot_client.move(new_cap_pose)
        try:
            new_images = await obcamera.capture_stereo_ir(exposure, gain, is_flood_on=False)
            logger.info(f"Task {arm_id}: Image pair captured.")
        except Exception as e:
            logger.error(f"Task {arm_id}: Error: Failed to capture images: {e}")
            return
        logger.info(f"Task {arm_id}: Captured stereo images for method2")
        template_img = cv2.imread('/home/vision/projects/fueling/fueling/initialization/orbbec_output/Laser_light/arm1/captured_left_ir.png')
        compare_images_pixel_perfect(template_img, new_images['left_ir'], debug_mode)
        logger.info(f"Task {arm_id}: Moved to new capture position for method2")
        result = await stereo_matcher.infer(new_images['left_ir'], new_images['right_ir'], pred_mode = pre_mode, bidir_verify_th = bidir_verify_th)
        if 'disparity_verified' in result:
            disparity_map = result['disparity_verified']
        else:
            disparity_map = result['disparity_left']

        disparity_map = np.where(disparity_map <= 0, 0.1, disparity_map)
        depth_map = (K[0] * baseline) / disparity_map
        depth_map_uint16 = np.uint16(depth_map * 1000)
        pcd = await run_sync(depth_to_point_cloud, # type: ignore (no-untyped-call)
            depth_map=np.array(depth_map_uint16),
            camera_intrinsics=K,
            max_distance=600,
        )
        target = await run_sync(preprocess_pointcloud, # type: ignore (no-untyped-call)
            eye_hand_matrix=eye_hand_matrix,
            source_pcd=pcd,
            dimensions=cut_box,
            capture_pose=new_cap_pose,
            fueling_pose=robot_fueling_pose,
            voxel_size=voxel_size,
            remove_outliers=remove_outliers,
            radius=radius,
            min_neighbors=min_neighbors
        )
        if debug_mode:
            display_point_cloud(target, None)
        mean_error, inlier_rmse, fitness = compute_registration_error(source_pc, target, None, max_correspondence_distance=5.0)
        logger.info(f"Task {arm_id}: Method2 registration errors - Mean Error: {mean_error}, Inlier RMSE: {inlier_rmse}, Fitness: {fitness}")

    await robot_client.move(upper_pose)
    await robot_client.move(robot_fueling_pose)
    await robot_client.move(upper_pose)
    await robot_client.move(init_pose)

    print(f"Task {arm_id}: Total time from capture to inference end: {total_end_time - total_start_time:.2f} seconds.")

async def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base_dir, '../..', 'data')

    arm_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('arm')])
    config_files = [os.path.join(data_dir, arm_dir, "config.jsonnet") for arm_dir in arm_dirs]
    config_files = [f for f in config_files if os.path.isfile(f)][:1]  # 暂时2台

    proj_dir = Path(__file__).resolve().parent.parent.parent
    # model_name = 'raft'
    # model_path = json.loads(_jsonnet.evaluate_file(config_files[0]))['stereo_matcher'][model_name]['model_path']
    # weight_path = f"{proj_dir}{model_path}"
    default_config = f"{data_dir}/default_config.jsonnet"
    default_set = json.loads(_jsonnet.evaluate_file(default_config))
    max_parallel = default_set['stereo_matcher']['max_parallel']
    model_name = default_set['stereo_matcher']['method']
    model_path = default_set['stereo_matcher'][model_name]['model_path']
    weight_path = f"{proj_dir}{model_path}"

    stereo_matcher = StereoMatcherService(model_name, weight_path, max_parallel)

    if not config_files:
        logger.error("没有找到有效的配置文件")
        return 1
    logger.info(f"Found config files: {config_files}")

    robot_clients = []
    pc_registrations = []
    configs = []
    source_pcs = []
    delays = []

    for i, config_file in enumerate(config_files):
        config = json.loads(_jsonnet.evaluate_file(config_file))

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

        source_down_path = os.path.join(data_dir, f'arm{i + 1}', 'calib', 'source_model.pcd')
        source_pc = o3d.io.read_point_cloud(source_down_path)
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


    async def run_with_delay(i, delay):
        if delay > 0:
            logger.info(f"机械臂 {i} 延迟 {delay}s 启动")
            await anyio.sleep(delay)
        await run_fuel(i, stereo_matcher, robot_clients[i], pc_registrations[i], configs[i], source_pcs[i])

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
