import json
import numpy as np
import open3d as o3d
import os
import time
from loguru import logger
from pathlib import Path
from fueling.obcamera import SyncOrbbecCamera
from fueling.pointcloud_processor import depth_to_point_cloud, preprocess_pointcloud
from fueling.robot_control import SyncRobotClient
from fueling.stereo_matcher import save_disparity_map, inference_stereo, RAFTStereoInference, UniMatchStereo, BridgeDepthStereo
from fueling.pose_transformation import get_upper_pose, transform_1x6_to_4x4
import cv2
import glob
import numpy as np
from fueling.drawing import display_point_cloud_with_axes
from fueling.minima.geometry import project_pointcloud_to_image_float

def init_model(light, config, robot_pose_config, camera, cut):
    # 在init_model函数开头附近添加
    target_pot = config["robot"]["target_pot"]
    # 确保所有必要的目录都存在
    os.makedirs(f"orbbec_output/Laser_light/{config['device']}/{target_pot}", exist_ok=True)
    os.makedirs(f"orbbec_output/Flood_light/{config['device']}/{target_pot}", exist_ok=True)
    os.makedirs(f"output_disparity/orbbec/Laser_light/{config['device']}/{target_pot}", exist_ok=True)
    os.makedirs(f"output_disparity/orbbec/Flood_light/{config['device']}/{target_pot}", exist_ok=True)
    os.makedirs(f"output_depth/{config['device']}/{target_pot}", exist_ok=True)
    os.makedirs(f"../../data/{config['device']}/{target_pot}", exist_ok=True)
    #机械臂参数
    ROBOT_IP = config["robot"]["ip"]  #机械臂IP地址
    CONTROL_PORT = config["robot"]["control_port"] #机械臂通信端口号
    REQ_PORT = config["robot"]["req_port"]
    COMMAND = config["robot"]["move_command"] #运动方式
    joint_sequence_params = config["robot"]["movej_params"]
    move_sequence_params = config["robot"]["movel_params"]
    FL_matrix = config['robot']['eye_hand_matrix']['T']
    intrinsics = config['robot']["RS_camera"]["K"]
    baseline = config['robot']['RS_camera']['stereo_baseline']
    check_interval = config['robot']['check_interval']
    pos_tol = config['robot']['pos_tol']
    rot_tol = config['robot']['rot_tol']
    proj_dir = Path(__file__).resolve().parent.parent.parent
    # model_path = config['stereo_matcher']['unimatch']['model_path']
    # model_path = config['stereo_matcher']['raft']['model_path']
    # model_path = config['stereo_matcher']['defom']['model_path']
    model_path = config['stereo_matcher']['unimatch']['model_path']
    weight_path = f"{proj_dir}{model_path}"

    exposure = config['camera']['exposure']
    gain = config['camera']['gain']
    type = config['robot']['move_command']

    target_pot = config["robot"]["target_pot"]  # 从配置中获取目标壶

    robot_control = SyncRobotClient(ROBOT_IP, REQ_PORT, CONTROL_PORT, type, str(proj_dir / 'data/robot_arm/RobotStateMessage.xlsx') + ':v2.6.0',
                                    move_sequence_params, joint_sequence_params, pos_tol, rot_tol, check_interval)

    input(f"请确认机械臂已移动到初始位置，然后按回车键继续...\n")
    init_pose = robot_control.get_target_tcp_pose() #初始位置
    logger.info(f"机械臂初始位姿: {init_pose}")
    input(f"请确认机械臂已移动到相机拍摄位置，然后按回车键继续...\n")
    capture_pose = robot_control.get_target_tcp_pose() #拍照位置
    logger.info(f"机械臂拍照位姿: {capture_pose}")
    data_pose = json.loads(Path(robot_pose_config).read_text(encoding='utf-8'))
    data_pose['init_pose'] = init_pose.tolist()
    data_pose['capture_pose'] = capture_pose.tolist()

    logger.info(f"\n正在初始化 {config['device']} 的位置，请人工拉取机械臂到模具加注口")
    # 等待用户确认机械臂移动到位
    input(f"请确认机械臂已移动到位，然后按回车键继续...\n")
    fueling_pose = robot_control.get_target_tcp_pose() # 加注位置
    logger.info(f"机械臂加注位姿: {fueling_pose}")
    data_pose['fueling_pose'] = fueling_pose.tolist()


    with open(robot_pose_config, "w", encoding="utf-8") as file:
        json.dump(data_pose, file, indent=4, ensure_ascii=False)
    logger.info(f"成功将更新后的数据写回配置文件: {robot_pose_config}")

    robot_upper_pose = get_upper_pose(fueling_pose, offset = 100)
    robot_control.move(robot_upper_pose)

    robot_control.move(capture_pose)
    # breakpoint()
    #_____________________________________________________________________________
    # 拍照和选项逻辑
    def show_and_choose_images(light_type):
        """显示拍摄的左右图像并提供选项"""
        while True:
            # 确保OpenCV窗口被正确关闭（如果之前存在）
            cv2.destroyAllWindows()

            # 获取最新的左右图像文件
            light_folder = "Laser_light" if light_type == "laser" else "Flood_light"
            device_path = os.path.join("orbbec_output", light_folder, f"{config['device']}")
            # 修改设备路径
            target_pot = config["robot"]["target_pot"]
            device_path = os.path.join("orbbec_output", light_folder, f"{config['device']}", target_pot)
            # 查找最新的左右图像
            left_images = glob.glob(f"{device_path}/*left*.png")
            right_images = glob.glob(f"{device_path}/*right*.png")

            if not left_images or not right_images:
                logger.error(f"未找到左右图像文件，路径: {device_path}")
                logger.error(f"左图像文件: {left_images}")
                logger.error(f"右图像文件: {right_images}")
                return False

            # 获取最新的文件
            latest_left = max(left_images, key=os.path.getctime)
            latest_right = max(right_images, key=os.path.getctime)

            # 显示图像
            try:
                left_img = cv2.imread(latest_left)
                right_img = cv2.imread(latest_right)

                if left_img is None or right_img is None:
                    logger.error("无法读取图像文件")
                    return False

                # 调整图像大小以便显示
                scale = 0.5
                left_img = cv2.resize(left_img, None, fx=scale, fy=scale)
                right_img = cv2.resize(right_img, None, fx=scale, fy=scale)

                # 获取图像尺寸
                height, width = left_img.shape[:2]

                # 将左右图像拼接成一个图像
                combined_img = np.hstack((left_img, right_img))

                # 在图像上添加文字标注
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(combined_img, "Left Camera", (10, 30), font, 1, (0, 255, 0), 2)
                cv2.putText(combined_img, "Right Camera", (width + 10, 30), font, 1, (0, 255, 0), 2)

                # 显示拼接后的图像
                cv2.imshow("Left & Right Cameras", combined_img)

                # 调整窗口位置和大小，确保完全显示
                cv2.moveWindow("Left & Right Cameras", 100, 100)
                cv2.resizeWindow("Left & Right Cameras", width * 2, height)

                logger.info("显示当前拍摄的左右图像")
                # logger.info("按 's' 保存并继续，按 'r' 重新拍照，按 'q' 退出")

                # 设置窗口焦点并等待按键
                cv2.setWindowProperty("Left & Right Cameras", cv2.WND_PROP_TOPMOST, 1)

                # 让OpenCV处理窗口事件，确保图像正确显示
                cv2.waitKey(100)
                return True
               
            except Exception as e:
                logger.error(f"显示图像时出错: {e}")
                return False

    # 定义变量来保存图像
    captured_images = None

    if camera == "OB":
        logger.info("正在使用奥比中光相机获取左右红外图像")
        # 获取设备ID
        # 从配置文件中读取OB_camera配置项，并提取设备ID数字部分
        ob_camera = config.get("device", "arm1")
        device_id = int(ob_camera.replace("arm", ""))-1
        # print(f"使用的OB相机ID: {device_id}")
        # breakpoint()
        obcamera = SyncOrbbecCamera(
                camera_serial=config['camera']['camera_serial'],
                pipeline_params={'enable_streams': [{'type': 'IR'}]}
            )
        inferencer = UniMatchStereo(weight_path=weight_path)

        # inferencer = RAFTStereoInference(
        #     restore_ckpt='../../data/stereo_weights/RAFT_Stereo/raftstereo-middlebury.pth',
        # )
        # inferencer = BridgeDepthStereo(
        #     checkpoint_path='/home/vision/projects/fueling/data/stereo_weights/BridgeDepth/bridge_middlebury.pth',
        # )
        logger.info(f"当前设备: {config['device']}")

        if light == "laser":
            logger.info("光选择为散斑")
            try:
                logger.info("机械臂移动完成，正在使用奥比中光相机获取左右红外图像")
                try:
                    images = obcamera.capture_stereo_ir(exposure, gain, light_mode='flood')
                    target_pot = config["robot"]["target_pot"]
                    captured_left_path = Path(f"orbbec_output/Laser_light/{config['device']}/{target_pot}/captured_left_ir.png")
                    captured_right_path = Path(f"orbbec_output/Laser_light/{config['device']}/{target_pot}/captured_right_ir.png")
                    cv2.imwrite(str(captured_left_path), images['left_ir'])
                    cv2.imwrite(str(captured_right_path), images['right_ir'])
                    captured_images = images  # 保存图像用于后续投影
                    logger.debug("OB get image successfully.")
                except Exception as e:
                    logger.error(f"启动OB相机时出错，请检查相机是否被占用: {e}")
                    return

                # 显示图像并提供选项
                if not show_and_choose_images("laser"):
                    logger.info("用户取消处理")
                    return

                logger.info("正在计算视差图")
                # 使用新的参数传递方式
                result = inference_stereo(inferencer, images['left_ir'], images['right_ir'], pred_mode='left', bidir_verify_th=0)
                # run_file(f"unimatch/run/gmstereo_infer_laser_device{device_id}.sh", [])
                logger.debug("get disparity map successfully.")
            except Exception as e:
                logger.error(f"计算视差图时出错: {e}")
                return
        elif light == "flood":
            logger.info("光选择为泛光")
            try:
                logger.info("机械臂移动完成，正在使用奥比中光相机获取左右泛光图像")
                try:
                    images = obcamera.capture_stereo_ir(exposure, gain, light_mode='flood')
                    captured_left_path = Path(f"orbbec_output/Flood_light/{config['device']}/{target_pot}/captured_left_ir.png")
                    captured_right_path = Path(f"orbbec_output/Flood_light/{config['device']}/{target_pot}/captured_right_ir.png")
                    cv2.imwrite(str(captured_left_path), images['left_ir'])
                    cv2.imwrite(str(captured_right_path), images['right_ir'])
                    captured_images = images  # 保存图像用于后续投影
                    logger.debug("OB get image successfully.")
                except Exception as e:
                    logger.error(f"启动OB相机时出错，请检查相机是否被占用: {e}")
                    return

                # 显示图像并提供选项
                if not show_and_choose_images("flood"):
                    logger.info("用户取消处理")
                    return

                logger.info("正在计算视差图")
                result = inference_stereo(inferencer, images['left_ir'], images['right_ir'], pred_mode='left', bidir_verify_th=0)

                logger.debug("get disparity map successfully.")
            except Exception as e:
                logger.error(f"计算视差图时出错: {e}")
                return

        logger.info("正在计算深度图")
        # 设置视差图路径
        light_type = "Laser_light" if light == "laser" else "Flood_light"
        target_pot = config["robot"]["target_pot"]
        disparity_map_path = Path(f"output_disparity/orbbec/{light_type}/{config['device']}/{target_pot}")


        save_disparity_map(result, output_dir=disparity_map_path)
        logger.info("模具视差图地址: %s", disparity_map_path)
        # 根据设备ID设置深度图输出目录
        depth_output_dir = f'output_depth/{config["device"]}/{target_pot}'
        os.makedirs(depth_output_dir, exist_ok=True)
        if 'disparity_verified' in result:
            disparity_map = result['disparity_verified']
        else:
            disparity_map = result['disparity_left']
        disparity_map = np.where(disparity_map <= 0, 0.1, disparity_map)
        depth_map = (intrinsics[0] * baseline) / disparity_map
        depth_map_uint16 = np.uint16(depth_map * 1000)
        output_path = os.path.join(depth_output_dir, f"stereo_depth.png")
        cv2.imwrite(output_path, depth_map_uint16) # type: ignore
        logger.info("模具深度图片地址: %s", depth_output_dir)

        # 生成pcd
        logger.info("正在生成点云图像")
        # 根据设备ID命名点云文件
        pcd = depth_to_point_cloud(
            depth_map=np.array(depth_map_uint16),
            camera_intrinsics=intrinsics,
            max_distance=600,
        )
        o3d.io.write_point_cloud("/home/vision/projects/fueling/tmp/MINIMA/all.pcd", pcd)
        if not pcd.has_points():
            logger.error("生成的点云图像为空，请检查深度图像或拍摄环境")
            return
        display_point_cloud_with_axes(pcd)

    if cut == "box":
        # logger.info(f"开始裁剪点云，裁剪方式: {cut}")
        dimensions = config["point_cloud"]["cut_box"]
        voxel_size = config['point_cloud']['voxel_size']
        radius = config['point_cloud']['radius']
        min_neighbors = config['point_cloud']['min_neighbors']
        remove_outliers = config['point_cloud']['remove_outliers']

        logger.info("voxel_size, radius, min_neighbors参数为:", voxel_size, radius, min_neighbors)

        source = preprocess_pointcloud(
            eye_hand_matrix=FL_matrix,
            source_pcd=pcd,
            dimensions=dimensions,
            capture_pose=capture_pose,
            fueling_pose=fueling_pose,
            voxel_size=5,
            remove_outliers=remove_outliers,
            radius=2,
            min_neighbors=8,
            downsample=False
        )
        if source is None or len(source.points) == 0:
            logger.warning("保存点云数为0，请重新选取区域")
            return
        logger.info(f"裁剪后点云数量为: {len(source.points)}")

        display_point_cloud_with_axes(source)


        # 方式1、处理完成后，回到最初的robot_pos_ini位置
        logger.info("所有处理完成，正在回到初始位置")
        logger.info(f"正在移动机械臂回到初始位置: {init_pose}")
        robot_control.move(init_pose)
        logger.info("所有机械臂已回到初始位置")
        target_pot = config["robot"]["target_pot"]
        file_path = f"../../data/{config['device']}/{target_pot}/source_model.pcd"

        o3d.io.write_point_cloud(file_path, source)
        logger.info(f"{config['device']}的点云已保存到:{file_path}, 模具点云数量为:{len(source.points)}")


        o3d.visualization.draw_geometries([source]) # type: ignore

    logger.info(f"\n初始化 {config['device']}位置结束")

    # return source