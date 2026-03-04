import os
from pathlib import Path
import time
import os
import cv2
import pickle
from stereo_matcher.drawing import vis_disparity
from loguru import logger
import numpy as np
import open3d as o3d


class FuelingLogger:
    def __init__(self, working_dir, debug=True):
        self.working_dir = Path(working_dir)
        self.debug = debug

    def start(self):
        folder = Path(time.strftime("%Y-%m-%d_%H-%M-%S"))
        all_dir = []
        self.data_dir = self.working_dir / folder
        os.makedirs(self.data_dir, exist_ok=True)
        self.arm_1_dir = self.data_dir / "arm_1"
        self.arm_2_dir = self.data_dir / "arm_2"
        self.arm_3_dir = self.data_dir / "arm_3"
        self.arm_4_dir = self.data_dir / "arm_4"
        all_dir.append(self.arm_1_dir)
        all_dir.append(self.arm_2_dir)
        all_dir.append(self.arm_3_dir)
        all_dir.append(self.arm_4_dir)
        os.makedirs(self.arm_1_dir, exist_ok=True)
        os.makedirs(self.arm_2_dir, exist_ok=True)
        os.makedirs(self.arm_3_dir, exist_ok=True)
        os.makedirs(self.arm_4_dir, exist_ok=True)
        for dir in [self.arm_1_dir, self.arm_2_dir, self.arm_3_dir, self.arm_4_dir]:
            os.makedirs(dir / "config", exist_ok=True)
            os.makedirs(dir / "depthmap", exist_ok=True)
            os.makedirs(dir / "disparity", exist_ok=True)
            os.makedirs(dir / "point_clouds", exist_ok=True)
            os.makedirs(dir / "ir_images", exist_ok=True)
        return all_dir
            # os.makedirs(dir / "template_pcd", exist_ok=True)


    def save_frame(self, images,arm_id):
        arm_dir = getattr(self, f"arm_{arm_id}_dir", None)
        if arm_dir is None:
            logger.error(f"Invalid arm_id: {arm_id}")
            return
        images_dir = arm_dir / "ir_images"
        os.makedirs(images_dir, exist_ok=True)
        captured_left_path =  images_dir/ 'captured_left_ir.png'
        captured_right_path = images_dir/ 'captured_right_ir.png'
        cv2.imwrite(str(captured_left_path), images['left_ir'])
        cv2.imwrite(str(captured_right_path), images['right_ir'])
        logger.info(f"Task {arm_id}: Saved captured images for verification to: {images_dir}")


    def save_disparity_map(self, result, arm_id):
        # Compute the correct arm directory based on arm_id
        arm_dir = getattr(self, f"arm_{arm_id}_dir", None)
        if arm_dir is None:
            logger.error(f"Invalid arm_id: {arm_id}")
            return

        disparities_dir = arm_dir / "disparities"
        os.makedirs(disparities_dir, exist_ok=True)

        if 'disparity_left' in result:
            disp = result['disparity_left']  # 获取左视差图
            disp_vis = vis_disparity(disp)
            cv2.imwrite(f"{disparities_dir}/left_disp.png", disp_vis)
            with open(f"{disparities_dir}/left_disp.pkl", 'wb') as fp:
                pickle.dump(disp, fp)
            logger.info(f"已保存左视差图: {disparities_dir}/left_disp.png")

        if 'disparity_right' in result:
            disp = result['disparity_right']  # 获取右视差图
            disp_vis = vis_disparity(disp)
            cv2.imwrite(f"{disparities_dir}/right_disp.png", disp_vis)
            with open(f"{disparities_dir}/right_disp.pkl", 'wb') as fp:
                pickle.dump(disp, fp)
            logger.info(f"已保存右视差图: {disparities_dir}/right_disp.png")

        if 'disparity_verified' in result:
            disp = result['disparity_verified']  # 获取双向验证后的视差图
            disp_vis = vis_disparity(disp)
            cv2.imwrite(f"{disparities_dir}/left_disp_verified.png", disp_vis)
            with open(f"{disparities_dir}/left_disp_verified.pkl", 'wb') as fp:
                pickle.dump(disp, fp)
            logger.info(f"已保存双向验证后的视差图: {disparities_dir}/left_disp_verified.png")


    def save_depth(self,depth_map,arm_id):
        arm_dir = getattr(self, f"arm_{arm_id}_dir", None)
        if arm_dir is None:
            logger.error(f"Invalid arm_id: {arm_id}")
            return

        depthmap_dir = arm_dir / "depthmap"
        os.makedirs(depthmap_dir, exist_ok=True)
        depth_map_uint16 = np.uint16(depth_map * 1000)  # 将深度转换为毫米，并保存为 16 位
        depthmap_path = depthmap_dir / "depth.png"  # 文件名固定为 depth.png
        cv2.imwrite(str(depthmap_path), depth_map_uint16)  # 需要转 str，因为 OpenCV 不支持 Path 对象
        logger.info(f"Task {arm_id}: Saved depth map to {depthmap_path}")


    def save_point_cloud(self, pcd, arm_id, pcd_type="target", is_cropped=False, is_downsampled=False):
        """
        保存点云文件
        Args:
            pcd: 点云对象
            arm_id: 机械臂ID
            pcd_type: 点云类型，"source" 或 "target"
            is_cropped: 是否经过裁剪
            is_downsampled: 是否经过下采样
        """
        arm_dir = getattr(self, f"arm_{arm_id}_dir", None)
        if arm_dir is None:
            logger.error(f"Invalid arm_id: {arm_id}")
            return

        pointcloud_dir = arm_dir / "point_clouds"
        os.makedirs(pointcloud_dir, exist_ok=True)

        # 构建文件名
        filename_parts = [pcd_type]
        if is_cropped:
            filename_parts.append("cropped")
        if is_downsampled:
            filename_parts.append("downsampled")

        filename = "_".join(filename_parts) + ".pcd"
        filepath = pointcloud_dir / filename

        # 保存点云
        success = o3d.io.write_point_cloud(str(filepath), pcd)
        if success:
            logger.info(f"Task {arm_id}: Saved {pcd_type} point cloud to {filepath}")
        else:
            logger.error(f"Task {arm_id}: Failed to save {pcd_type} point cloud to {filepath}")

        #cut和match的可视化
    def visualize_cropped_point_cloud(cropped_pcd, pose_matrix, dimensions,
                                window_name="Point Cloud Cropping Visualization"):
        """
        可视化裁剪的点云和长方形包围盒
        Args:
            cropped_pcd: 裁剪后的点云 (o3d.geometry.PointCloud)
            pose_matrix: 4x4位姿矩阵，定义包围盒的位置和方向
            dimensions: [width, height, depth] 包围盒的尺寸
            window_name: 可视化窗口名称
        """
        logger.info("开始可视化裁剪的点云")

        cropped_pcd_vis = cropped_pcd.copy()
        cropped_pcd_vis.paint_uniform_color([0.2, 0.8, 0.2])   # 绿色

        # 创建包围盒
        pose_matrix_mm = pose_matrix.copy()
        center_point = pose_matrix_mm[:3, 3]
        rotation_matrix = pose_matrix_mm[:3, :3]

        width, height, depth = dimensions
        bounding_box = o3d.geometry.OrientedBoundingBox(
            center=center_point, R=rotation_matrix, extent=[width, height, depth]
        )
        bounding_box.color = [1, 0, 0]  # 红色包围盒
        # 创建中心点标记
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
        center_sphere.translate(center_point)
        center_sphere.paint_uniform_color([1, 0, 0])  # 红色中心点
        # 创建坐标轴
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=50, origin=center_point
        )
        # 创建线框包围盒（更清晰的边界显示）
        box_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(bounding_box)
        box_lines.paint_uniform_color([1, 0, 0])  # 红色线框

        # 收集所有要显示的几何体
        geometries = [
            cropped_pcd_vis,      # 裁剪后的点云（绿色）
            box_lines,            # 包围盒线框（红色）
            center_sphere,        # 中心点（红色球）
            coordinate_frame      # 坐标轴
        ]

        # 打印统计信息
        logger.info(f"裁剪后点云点数: {len(cropped_pcd.points)}")
        logger.info(f"包围盒中心: [{center_point[0]:.2f}, {center_point[1]:.2f}, {center_point[2]:.2f}]")
        logger.info(f"包围盒尺寸: [{width:.2f}, {height:.2f}, {depth:.2f}]")

        # 显示可视化
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1200,
            height=800,
            left=50,
            top=50
        )

    #可视化配准结果 点云输入都是预处理后的点云
    def visualize_registration(self, source_cpd, target_cpd, transformation,arm_id):
        logger.info(f"机械臂 {arm_id} 可视化配准后的点云")
        transformed_source = source_cpd.transform(transformation)
        transformed_source.paint_uniform_color([0, 0, 1])  # 蓝色
        target_cpd.paint_uniform_color([0, 1, 0])  # 绿色
        o3d.visualization.draw_geometries([transformed_source, target_cpd], window_name=f"机械臂 {arm_id} - 配准后点云")


