import os
from pathlib import Path
import cv2
from obcamera.sync_obcamera import OrbbecCamera
from pointcloud_processor.depth_to_point_cloud import depth_to_point_cloud
from robot_control.robot_connection import RobotClient, RobotException
from stereo_matcher_old.unimatch.UniMatchStereo import UniMatchStereo
import time
import numpy as np
from loguru import logger
from stereo_matcher.io_ import save_disparity_map
from pointcloud_processor.cut_point_cloud import cut_point_cloud
import open3d as o3d
import _jsonnet, json
from datetime import datetime

def display_point_cloud_with_axes(pcd1, pcd2=None):
    # 创建坐标轴
    axis_length = 200  # 坐标轴的长度
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=[0, 0, 0])

    # 设置 pcd1 和 pcd2 的颜色
    if pcd1 is not None:
        pcd1.paint_uniform_color([0, 0, 0])
    if pcd2 is not None:
        pcd2.paint_uniform_color([0, 1, 0])

    geometries = [pcd1, coordinate_frame]
    if pcd2 is not None:
        geometries.append(pcd2)

    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud with XYZ Axes")


# converted_poses:1x6 list or array, 单位:mm
def move_sequence(control_conn, converted_poses, move_sequence_params=None):
        converted_poses = [x * 0.001 for x in converted_poses[:3]] + converted_poses[3:]
        """异步执行一系列位姿的movel指令"""
        if isinstance(converted_poses[0], (int, float)) and len(converted_poses) == 6:
            converted_poses = [converted_poses]

        if move_sequence_params is None:
            move_sequence_params = {"a": 1, "v": 0.7, "t": 0, "r": 0}

        a=move_sequence_params.get("a", 1)
        v=move_sequence_params.get("v", 0.7)
        t=move_sequence_params.get("t", 0)
        r=move_sequence_params.get("r", 0)

        commands = []
        for pose in converted_poses:
            if len(pose) != 6:
                raise ValueError(f"位姿矩阵 {pose} 的长度必须为6")
            cmd = f"movel({pose},a={a},v={v},t={t},r={r})"
            commands.append(cmd)

        function_name = "move_pose"
        command_block = f"def {function_name}():\n"
        for cmd in commands:
            command_block += f"    {cmd}\n"
        command_block += "end\n"

        try:
            logger.info(f"发送指令:\n{command_block}")
            control_conn.send(command_block.encode('utf-8'))
            time.sleep(0.1)  # 避免过快发送
        except Exception as e:
            logger.error(f"发送指令失败: {e}")
            raise

# config: 配置文件
# poses: list of 1x6 list or array, 单位:mm
def pipeline(poses, config, show = False):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    move_sequence_params = config["robot"]["move_sequence_params"]
    ob_camera = config.get("device", "arm1")
    ROBOT_IP = config["robot"]["ip"]  #机械臂IP地址
    CONTROL_PORT = config["robot"]["control_port"]
    FL_matrix = np.array(config['robot']['eye_hand_matrix']['T'])

    intrinsics = config['robot']["RS_camera"]["K"]
    baseline = config['robot']['RS_camera']['stereo_baseline']
    device_id = int(ob_camera.replace("arm", ""))-1
    obcamera = OrbbecCamera(
                id = device_id,  # 使用任务ID作为相机ID
                pipeline_params = {'enable_streams': [{'type': 'IR'}]},
            )
    inferencer = UniMatchStereo(
            weight_path='../../data/stereo_weights/unimatch/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth',
            max_size=1024,
            padding_factor=32,
            upsample_factor=4,
            num_scales=2,
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=3,
        )
    # 移动机械臂
    robot_control = RobotClient('/home/vision/projects/fueling/data/robot_arm/RobotStateMessage.xlsx', 'v2.6.0')
    robot_control.connect(ROBOT_IP, CONTROL_PORT)
    move_sequence(robot_control, poses, move_sequence_params)
    time.sleep(2)  # 等待机械臂移动完成
    # 拍照
    images = obcamera.capture_stereo_ir(config['camera']['exposure'], config['camera']['gain'], is_flood_on=False)
    captured_left_path = Path(f"{current_time}/images_output/{config['device']}/captured_left_ir.png")
    captured_left_path.parent.mkdir(parents=True, exist_ok=True)
    captured_right_path = Path(f"{current_time}/images_output/{config['device']}/captured_right_ir.png")
    captured_right_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(captured_left_path), images['left_ir'])
    cv2.imwrite(str(captured_right_path), images['right_ir'])
    logger.info(f"Captured images saved to {captured_left_path} and {captured_right_path}")
    # 计算视差图
    # pred_mode='left', bidir_verify_th=0: 不进行双向验证
    # pred_mode='bidir', bidir_verify_th>0: 进行双向验证
    result = inferencer.inference_stereo(
                    images['left_ir'], images['right_ir'],
                    pred_mode='left', bidir_verify_th=0
                )
    disparity_map_path = Path(f"{current_time}/output_disparity/{config['device']}")
    save_disparity_map(result, output_dir=disparity_map_path)
    logger.info(f"Disparity map saved to {disparity_map_path}")
    # 计算深度图
    depth_output_dir = f'{current_time}/output_depth/{config["device"]}'
    os.makedirs(depth_output_dir, exist_ok=True)
    if 'disparity_verified' in result:
        disparity_map = result['disparity_verified']
    else:
        disparity_map = result['disparity_left']
    disparity_map = np.where(disparity_map <= 0, 0.1, disparity_map)
    depth_map = (intrinsics[0] * baseline) / disparity_map
    depth_map_uint16 = np.uint16(depth_map * 1000)
    output_path = os.path.join(depth_output_dir, f"stereo_depth.png")
    cv2.imwrite(output_path, depth_map_uint16)
    logger.info(f"模具深度图片地址: {depth_output_dir}")
    # 计算点云
    pcd = depth_to_point_cloud(
            depth_map=depth_map_uint16,
            camera_intrinsics=intrinsics,
            max_distance=600,
        )
    if show:
        display_point_cloud_with_axes(pcd)
    pcd_output_dir = f'{current_time}/output_pcd/{config["device"]}'
    os.makedirs(pcd_output_dir, exist_ok=True)
    pcd_path = os.path.join(pcd_output_dir, f"point_cloud.pcd")
    o3d.io.write_point_cloud(pcd_path, pcd)
    logger.info(f"模具点云地址: {pcd_path}")
    # 裁剪点云
    dimensions = config["point_cloud"]["cut_box"]
    fueling_pose = config["robot"]["fueling_pose"]
    pcd_cutted = cut_point_cloud(
        cut="box",
        eye_hand_matrix=FL_matrix,
        source_pcd=pcd,
        dimensions=dimensions,
        capture_pose=poses,
        fueling_pose=fueling_pose,
    )
    if show:
        display_point_cloud_with_axes(pcd_cutted)
    pcd_cutted_path = os.path.join(pcd_output_dir, f"point_cloud_cutted.pcd")
    o3d.io.write_point_cloud(pcd_cutted_path, pcd_cutted)
    logger.info(f"裁剪后模具点云地址: {pcd_cutted_path}")

if __name__ == "__main__":
    json_str = _jsonnet.evaluate_file('../../data/arm1/config.jsonnet')
    config = json.loads(json_str)
    test_poses = [-118.57,
        611.9,
        636.2,
        0.302,
        -0.032,
        3.137]
    pipeline(test_poses, config, show=True)