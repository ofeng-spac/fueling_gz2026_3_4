import json
import numpy as np
import open3d as o3d
import os
import time
from loguru import logger
from pathlib import Path
import cv2
import _jsonnet
import glob

# 引入原有的依赖
from fueling.obcamera import SyncOrbbecCamera
from fueling.pointcloud_processor import depth_to_point_cloud, preprocess_pointcloud
from fueling.robot_control import SyncRobotClient
from fueling.stereo_matcher import save_disparity_map, inference_stereo, RAFTStereoInference, UniMatchStereo, BridgeDepthStereo
from fueling.pose_transformation import get_upper_pose, transform_1x6_to_4x4

class FuelingModelBackend:
    def __init__(self, config_path: str, light_type: str = "flood", camera_type: str = "OB", cut_type: str = "box"):
        """
        初始化后端类
        :param config_path: 配置文件路径 (.jsonnet)
        :param light_type: "laser" 或 "flood"
        :param camera_type: "OB" 等
        :param cut_type: "box" 等
        """
        self.config_path = config_path
        self.light_type = light_type
        self.camera_type = camera_type
        self.cut_type = cut_type

        # 状态存储
        self.config = None
        self.robot_client = None
        self.camera_client = None
        self.inferencer = None

        # 位姿数据
        self.init_pose = None
        self.capture_pose = None
        self.fueling_pose = None

        # 图像数据
        self.captured_images = None
        self.latest_pcd = None
        self.latest_source_pcd = None

        # 加载配置
        self._load_config()

    def _load_config(self):
        try:
            json_str = _jsonnet.evaluate_file(self.config_path)
            self.config = json.loads(json_str)
            logger.info(f"成功加载配置文件: {self.config_path}")

            self.target_pot = self.config["robot"]["target_pot"]
            self.device_name = self.config['device']

            # 路径相关的配置
            self.robot_pose_config_path = str(Path(self.config_path).parent / self.target_pot / "robot_pose.json")

            # 确保目录存在
            self._prepare_directories()

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise e

    def _prepare_directories(self):
        """创建所有必要的输出目录"""
        dirs = [
            Path(self.robot_pose_config_path).parent,
            Path(f"orbbec_output/Laser_light/{self.device_name}/{self.target_pot}"),
            Path(f"orbbec_output/Flood_light/{self.device_name}/{self.target_pot}"),
            Path(f"output_disparity/orbbec/Laser_light/{self.device_name}/{self.target_pot}"),
            Path(f"output_disparity/orbbec/Flood_light/{self.device_name}/{self.target_pot}"),
            Path(f"output_depth/{self.device_name}/{self.target_pot}"),
            Path(f"../../data/{self.device_name}/{self.target_pot}")
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def connect_robot(self):
        """连接机械臂"""
        try:
            robot_cfg = self.config["robot"]
            proj_dir = Path(__file__).resolve().parent.parent.parent
            self.robot_client = SyncRobotClient(
                robot_cfg["ip"],
                robot_cfg["req_port"],
                robot_cfg["control_port"],
                robot_cfg["move_command"],
                str(proj_dir / 'data/robot_arm/RobotStateMessage.xlsx') + ':v2.6.0',
                robot_cfg["movel_params"],
                robot_cfg["movej_params"],
                robot_cfg["pos_tol"],
                robot_cfg["rot_tol"],
                robot_cfg["check_interval"]
            )
            logger.info("机械臂连接成功")
            return True
        except Exception as e:
            logger.error(f"机械臂连接失败: {e}")
            return False

    def connect_camera(self):
        """连接相机"""
        if self.camera_type == "OB":
            try:
                self.camera_client = SyncOrbbecCamera(
                    camera_serial=self.config['camera']['camera_serial'],
                    pipeline_params={'enable_streams': [{'type': 'IR'}]}
                )
                logger.info("相机连接成功")
                return True
            except Exception as e:
                logger.error(f"相机连接失败: {e}")
                return False
        return False

    def load_ai_model(self):
        """加载 AI 模型"""
        try:
            proj_dir = Path(__file__).resolve().parent.parent.parent
            model_rel_path = self.config['stereo_matcher']['unimatch']['model_path']
            weight_path = f"{proj_dir}{model_rel_path}"

            # 这里可以根据配置扩展支持其他模型
            self.inferencer = UniMatchStereo(weight_path=weight_path)
            logger.info(f"AI 模型加载成功: {weight_path}")
            return True
        except Exception as e:
            logger.error(f"AI 模型加载失败: {e}")
            return False

    # ---Step 1, 2, 3: 记录位姿---

    def get_current_robot_pose(self):
        """获取当前机械臂位姿"""
        if not self.robot_client:
            logger.error("机械臂未连接")
            return None
        return self.robot_client.get_target_tcp_pose()

    def record_init_pose(self):
        """记录初始位姿"""
        pose = self.get_current_robot_pose()
        if pose is not None:
            self.init_pose = pose
            logger.info(f"已记录初始位姿: {self.init_pose}")
        return pose

    def record_capture_pose(self):
        """记录拍照位姿"""
        pose = self.get_current_robot_pose()
        if pose is not None:
            self.capture_pose = pose
            logger.info(f"已记录拍照位姿: {self.capture_pose}")
        return pose

    def record_fueling_pose(self):
        """记录加注位姿"""
        pose = self.get_current_robot_pose()
        if pose is not None:
            self.fueling_pose = pose
            logger.info(f"已记录加注位姿: {self.fueling_pose}")
        return pose

    def save_poses_to_file(self):
        """将记录的位姿保存到 json 文件"""
        if self.init_pose is None or self.capture_pose is None or self.fueling_pose is None:
            logger.error("位姿记录不完整，无法保存")
            return False

        try:
            # 读取旧文件以保留其他可能的字段，如果文件不存在则为空字典
            if os.path.exists(self.robot_pose_config_path):
                data_pose = json.loads(Path(self.robot_pose_config_path).read_text(encoding='utf-8'))
            else:
                data_pose = {}

            data_pose['init_pose'] = self.init_pose.tolist()
            data_pose['capture_pose'] = self.capture_pose.tolist()
            data_pose['fueling_pose'] = self.fueling_pose.tolist()

            with open(self.robot_pose_config_path, "w", encoding="utf-8") as file:
                json.dump(data_pose, file, indent=4, ensure_ascii=False)

            logger.info(f"成功保存位姿数据到: {self.robot_pose_config_path}")
            return True
        except Exception as e:
            logger.error(f"保存位姿数据失败: {e}")
            return False

    # ---Step 4: 移动到拍照位---

    def move_to_capture_pose(self):
        """先移动到上方，再移动到拍照位"""
        if self.robot_client is None or self.capture_pose is None or self.fueling_pose is None:
            logger.error("缺少必要条件（机械臂未连接或位姿未记录）")
            return False

        try:
            # 模仿原逻辑：先去上方点
            robot_upper_pose = get_upper_pose(self.fueling_pose, offset=100)
            self.robot_client.move(robot_upper_pose)

            # 再去拍照点
            self.robot_client.move(self.capture_pose)
            logger.info("已移动到拍照位置")
            return True
        except Exception as e:
            logger.error(f"移动过程出错: {e}")
            return False

    # ---Step 5: 拍照与预览---

    def get_latest_frame(self, exposure=None, gain=None):
        """
        获取最新的一帧图像用于实时预览 (不保存文件)
        :param exposure: 曝光值 (int)，如果为None则使用配置文件值
        :param gain: 增益值 (int)，如果为None则使用配置文件值
        :return: images (dict)
        """
        if not self.camera_client:
            return None

        # 如果未指定参数，使用默认配置
        if exposure is None:
            exposure = self.config['camera']['exposure']
        if gain is None:
            gain = self.config['camera']['gain']

        try:
            # 获取图像 (light_mode固定为flood，保持与原逻辑一致)
            images = self.camera_client.capture_stereo_ir(exposure, gain, light_mode='flood')
            return images
        except Exception as e:
            # 预览时的错误不一定要打印全部堆栈，避免刷屏
            # logger.warning(f"获取预览帧失败: {e}")
            return None

    def capture_image(self):
        """
        拍摄图像
        :return: (left_img_path, right_img_path, images_dict)
        """
        if not self.camera_client:
            logger.error("相机未连接")
            return None, None, None

        logger.info(f"正在使用 {self.camera_type} 相机获取左右图像 (模式: {self.light_type})")

        exposure = self.config['camera']['exposure']
        gain = self.config['camera']['gain']

        try:
            # 这里的 light_mode='flood' 似乎在原代码中固定了，即使外部选 laser。原代码逻辑里 laser 模式下也传了 flood？
            # 仔细看原代码：
            # laser 分支: obcamera.capture_stereo_ir(exposure, gain, light_mode='flood') <- 这里可能是笔误或者是通用参数
            # flood 分支: obcamera.capture_stereo_ir(exposure, gain, light_mode='flood')
            # 我们暂时保持一致，或者根据 light_type 传参（如果 capture_stereo_ir 支持的话）
            # 假设 capture_stereo_ir 主要控制是否开关激光，但这在 capture_stereo_ir 内部实现。
            # 这里先照搬原代码调用方式。

            images = self.camera_client.capture_stereo_ir(exposure, gain, light_mode='flood')

            folder_name = "Laser_light" if self.light_type == "laser" else "Flood_light"
            save_dir = Path(f"orbbec_output/{folder_name}/{self.device_name}/{self.target_pot}")

            left_path = save_dir / "captured_left_ir.png"
            right_path = save_dir / "captured_right_ir.png"

            cv2.imwrite(str(left_path), images['left_ir'])
            cv2.imwrite(str(right_path), images['right_ir'])

            self.captured_images = images
            logger.info(f"图像采集成功，已保存至 {save_dir}")
            return str(left_path), str(right_path), images

        except Exception as e:
            logger.error(f"拍照失败: {e}")
            return None, None, None

    # ---Step 6: 计算视差与深度---

    def compute_stereo_and_depth(self):
        """执行推理，计算视差和深度图"""
        if self.captured_images is None:
            logger.error("没有已采集的图像")
            return False

        if self.inferencer is None:
            logger.error("推理模型未加载")
            return False

        try:
            logger.info("开始计算视差图...")
            result = inference_stereo(
                self.inferencer,
                self.captured_images['left_ir'],
                self.captured_images['right_ir'],
                pred_mode='left',
                bidir_verify_th=0
            )

            folder_name = "Laser_light" if self.light_type == "laser" else "Flood_light"
            disparity_output_dir = Path(f"output_disparity/orbbec/{folder_name}/{self.device_name}/{self.target_pot}")
            save_disparity_map(result, output_dir=disparity_output_dir)
            logger.info(f"视差图已保存: {disparity_output_dir}")

            # 计算深度
            logger.info("开始计算深度图...")
            intrinsics = self.config['robot']["RS_camera"]["K"]
            baseline = self.config['robot']['RS_camera']['stereo_baseline']

            if 'disparity_verified' in result:
                disparity_map = result['disparity_verified']
            else:
                disparity_map = result['disparity_left']

            disparity_map = np.where(disparity_map <= 0, 0.1, disparity_map)
            depth_map = (intrinsics[0] * baseline) / disparity_map
            depth_map_uint16 = np.uint16(depth_map * 1000)

            depth_output_dir = Path(f"output_depth/{self.device_name}/{self.target_pot}")
            depth_path = depth_output_dir / "stereo_depth.png"
            cv2.imwrite(str(depth_path), depth_map_uint16)
            logger.info(f"深度图已保存: {depth_path}")

            # 暂存数据用于点云生成
            self.latest_depth_map = depth_map_uint16
            self.intrinsics = intrinsics

            return str(depth_path)

        except Exception as e:
            logger.error(f"计算视差/深度失败: {e}")
            return False

    # ---Step 7: 生成点云与处理---

    def generate_point_cloud(self):
        """从深度图生成点云"""
        if self.latest_depth_map is None:
            logger.error("没有可用的深度图数据")
            return None

        try:
            logger.info("正在生成点云...")
            pcd = depth_to_point_cloud(
                depth_map=np.array(self.latest_depth_map),
                camera_intrinsics=self.intrinsics,
                max_distance=600,
            )
            # 临时保存一份全量点云
            tmp_path = "/home/vision/projects/fueling/tmp/MINIMA/all.pcd"
            Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(tmp_path, pcd)

            self.latest_pcd = pcd
            if not pcd.has_points():
                logger.error("生成的点云为空")
                return None
            return pcd
        except Exception as e:
            logger.error(f"生成点云失败: {e}")
            return None

    def process_and_save_pointcloud(self):
        """裁剪并保存最终点云"""
        if self.latest_pcd is None:
            if hasattr(self, 'latest_depth_map') and self.latest_depth_map is not None:
                logger.info("未检测到原始点云，正在尝试从最新深度图生成...")
                if self.generate_point_cloud() is None:
                    logger.error("自动生成点云失败")
                    return None
            else:
                logger.error("没有原始点云 (latest_pcd) 且无可用深度图 (latest_depth_map)")
                return None

        if self.cut_type != "box":
            logger.warning(f"目前仅实现了 'box' 裁剪模式，当前为: {self.cut_type}")
            # 如果不是 box，可能直接返回原始点云或处理其他逻辑
            # return self.latest_pcd

        try:
            logger.info(f"开始裁剪点云 ({self.cut_type})...")

            FL_matrix = self.config['robot']['eye_hand_matrix']['T']
            dimensions = self.config["point_cloud"]["cut_box"]
            # 读取配置参数
            # voxel_size = self.config['point_cloud']['voxel_size']  # 原代码中此处打印了参数但传入函数时用了硬编码
            remove_outliers = self.config['point_cloud']['remove_outliers']

            source = preprocess_pointcloud(
                eye_hand_matrix=FL_matrix,
                source_pcd=self.latest_pcd,
                dimensions=dimensions,
                capture_pose=self.capture_pose,
                fueling_pose=self.fueling_pose,
                voxel_size=5, # 原代码硬编码
                remove_outliers=remove_outliers,
                radius=2,     # 原代码硬编码
                min_neighbors=8, # 原代码硬编码
                downsample=False
            )

            if source is None or len(source.points) == 0:
                logger.warning("裁剪后点云为空")
                return None

            self.latest_source_pcd = source
            logger.info(f"裁剪完成，点数: {len(source.points)}")

            # 保存最终结果
            file_path = f"../../data/{self.device_name}/{self.target_pot}/source_model.pcd"
            o3d.io.write_point_cloud(file_path, source)
            logger.info(f"最终模型已保存: {file_path}")

            return source

        except Exception as e:
            logger.error(f"处理点云失败: {e}")
            return None

    # ---Step 8: 复位---

    def return_to_init(self):
        """机械臂回到初始位置"""
        if self.robot_client and self.init_pose is not None:
            try:
                logger.info("正在返回初始位置...")
                self.robot_client.move(self.init_pose)
                logger.info("已回到初始位置")
                return True
            except Exception as e:
                logger.error(f"复位失败: {e}")
                return False
        return False
