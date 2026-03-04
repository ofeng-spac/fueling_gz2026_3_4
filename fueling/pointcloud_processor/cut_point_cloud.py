import numpy as np
import open3d as o3d
import numpy.linalg as npl
from loguru import logger
from ..pose_transformation import transform_1x6_to_4x4
import numpy as np
import time

#去噪
def radius_outlier_removal(point_cloud:o3d.geometry.PointCloud, radius:float=0.05, min_neighbors:int=10):
    _, ind = point_cloud.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    filtered_point_cloud = point_cloud.select_by_index(ind)
    return filtered_point_cloud

def remove_and_downsample(point_cloud:o3d.geometry.PointCloud, voxel_size:float=1, remove_outliers:bool=True, radius:float = 0.5, min_neighbors:int=10):
    """
    对点云进行预处理：去噪和降采样
    """
    # 去除无效点
    point_cloud.remove_non_finite_points()
    logger.debug(f"After remove_non_finite_points: {np.asarray(point_cloud.points).shape}")

    start_time = time.time()
    # 去噪
    if remove_outliers:
        point_cloud = radius_outlier_removal(point_cloud, radius=radius, min_neighbors=min_neighbors)
    # o3d.io.write_point_cloud("denoised_point_cloud.pcd", point_cloud)
    end_time = time.time()
    logger.debug(f"去噪耗时: {end_time - start_time:.2f} 秒")

    # 降采样
    logger.debug(f"After 去噪: {np.asarray(point_cloud.points).shape}")
    point_cloud_down = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return point_cloud_down

def preprocess_pointcloud(eye_hand_matrix: list, source_pcd: o3d.geometry.PointCloud, dimensions: tuple, capture_pose: np.ndarray, 
                          fueling_pose: np.ndarray, voxel_size:float=1.15, remove_outliers:bool=True, radius:float=1.5, 
                          min_neighbors:int=5, downsample:bool=True, is_crop:bool=True):
    """
    预处理点云
    """
    T_B_from_Ha = capture_pose
    T_Ha_from_B = npl.inv(T_B_from_Ha)
    T_B_from_Hb = fueling_pose
    T_FL_from_camL = np.array(eye_hand_matrix)
    T_cam_from_FL = npl.inv(T_FL_from_camL)
    T_cam = transform_1x6_to_4x4([0, 0, 0, 0, 0, 0])

    # 计算变换矩阵
    T_Ha_from_Hb = T_Ha_from_B @ T_B_from_Hb
    X_Ca = T_cam_from_FL @ T_Ha_from_Hb @ T_cam
    center_point = X_Ca[:3, 3]
    rotation_matrix = X_Ca[:3, :3]

    # === 创建包围盒 ===
    width, height, depth = dimensions
    bounding_box = o3d.geometry.OrientedBoundingBox(
        center=center_point,
        R=rotation_matrix,
        extent=[width, height, depth]
    )

    # === 创建中心点标记（可视化辅助） ===
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
    center_sphere.translate(center_point)
    center_sphere.paint_uniform_color([1, 0, 0])

    # === 执行裁剪 ===
    logger.info("裁剪box区域内的点云")
    if is_crop:  
        source_pcd = source_pcd.crop(bounding_box)
        if len(source_pcd.points) == 0:
            logger.warning("裁剪后没有点云数据，可能是包围盒参数不正确。")
            return None
    
    if downsample:
        source_pcd = remove_and_downsample(source_pcd, voxel_size=voxel_size, remove_outliers=remove_outliers, radius=radius, min_neighbors=min_neighbors)
    return source_pcd



#显示点云xyz尺寸范围
def display_pcd_xyz_dimensions(source_path: str):
    source_pcd = o3d.io.read_point_cloud(source_path)
    points = np.asarray(source_pcd.points)

    # 获取点云在 x, y, z 平面的最大最小值
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    # 计算 x, y, z 尺寸
    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min

    print(f"Point Cloud XYZ Dimensions:")
    print(f"X range: {x_min} to {x_max}, Size: {x_size}")
    print(f"Y range: {y_min} to {y_max}, Size: {y_size}")
    print(f"Z range: {z_min} to {z_max}, Size: {z_size}")

