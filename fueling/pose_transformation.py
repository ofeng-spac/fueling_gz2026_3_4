import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import math
from loguru import logger

def calculate_transform_a_to_b(pose_a: list, pose_b: list):

    # pose_A, pose_B: 机械臂A和B的1x6位姿向量，格式为 [x, y, z, rx, ry, rz]

    transform_a = transform_1x6_to_4x4(pose_a)
    transform_b = transform_1x6_to_4x4(pose_b)
    transform_a_to_b = np.dot(np.linalg.inv(transform_a), transform_b)
    print("matrix_a_to_b:\n", transform_a_to_b)
    return transform_a_to_b

def calculate_transform_a_to_c(transform_a_to_b: np.ndarray, transform_b_to_c: np.ndarray) -> np.ndarray:
    """
    通过A到B的变换矩阵和B到C的变换矩阵，计算A到C的变换矩阵。
    transform_A_to_B, transform_B_to_C: 4x4的变换矩阵。
    返回：4x4的变换矩阵，从A到C。
    """
    transform_a_to_c = np.dot(transform_a_to_b, transform_b_to_c)
    print("matrix_a_to_c:\n", transform_a_to_c)
    return transform_a_to_c


def create_transformation_matrix(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:

    # Combine a 3x3 rotation matrix and a 1x3 translation vector into a 4x4 transformation matrix.

    # 检查输入的矩阵形状
    if rotation_matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")
    if translation_vector.shape != (3,):
        raise ValueError("Translation vector must be 1x3")

    # 创建4x4的单位矩阵
    transformation_matrix = np.eye(4)

    # 将3x3的旋转矩阵放入4x4矩阵的左上角
    transformation_matrix[:3, :3] = rotation_matrix

    # 将1x3的平移向量放入4x4矩阵的第4列
    transformation_matrix[:3, 3] = translation_vector

    # 设置输出格式为浮点数
    np.set_printoptions(suppress=True, floatmode='fixed', precision=6)

    # print("Transformation Matrix:\n", transformation_matrix)

    return transformation_matrix

#计算点云平均移动距离______________________________________________________
def transform_absolute_distance(transform_matrix: np.ndarray):
    # 提取平移向量部分
    t = transform_matrix[:3, 3]
    # 计算平移向量的欧式距离
    distance = np.linalg.norm(t)
    print("xyz方向移动距离为: ",distance,"mm")
    return distance

def get_upper_pose(original_pose: np.ndarray, offset: float = 50):

    if original_pose.shape != (4, 4):
        raise ValueError("original_pose 必须是 4×4 齐次矩阵")
    # 提取旋转矩阵
    rotation_matrix = original_pose[:3, :3]
    upper_pos = np.eye(4)
    upper_pos[:3, :3] = rotation_matrix
    r = Rotation.from_matrix(rotation_matrix)
    rx, ry, rz = r.as_euler('xyz', degrees=False)
    R = np.array([
            [math.cos(ry)*math.cos(rz),
             math.cos(rz)*math.sin(rx)*math.sin(ry) - math.cos(rx)*math.sin(rz),
             math.cos(rx)*math.cos(rz)*math.sin(ry) + math.sin(rx)*math.sin(rz)],
            [math.cos(ry)*math.sin(rz),
             math.cos(rx)*math.cos(rz) + math.sin(rx)*math.sin(ry)*math.sin(rz),
             -math.cos(rz)*math.sin(rx) + math.cos(rx)*math.sin(ry)*math.sin(rz)],
            [-math.sin(ry),
             math.cos(ry)*math.sin(rx),
             math.cos(rx)*math.cos(ry)]
        ])
    z_axis = R[:, 2]

    # 计算新位置
    new_pos = original_pose[:3, 3] - z_axis * offset   # shape (3,)

    # 写回平移部分，旋转部分保持不动
    upper_pos[:3, 3] = new_pos

    return upper_pos

def transform_4x4_to_1x6(transformation_matrix: np.ndarray, order: str = 'xyz') -> list:
    if transformation_matrix.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4")

    # 提取平移向量
    x, y, z = transformation_matrix[:3, 3]

    # 提取旋转矩阵
    rotation_matrix = transformation_matrix[:3, :3]

    # 将旋转矩阵转换为欧拉角（roll, pitch, yaw）
    r = Rotation.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler(order, degrees=False)

    # 合并平移和旋转信息为1x6的机械臂位姿矩阵
    return [x / 1000, y / 1000, z / 1000, roll, pitch, yaw]

def transform_1x6_to_4x4(matrix_1x6: list, order: str = 'xyz') -> np.ndarray:
    matrix_1x6 = [matrix_1x6[0] * 1000, matrix_1x6[1] * 1000, matrix_1x6[2] * 1000, matrix_1x6[3], matrix_1x6[4], matrix_1x6[5]]
    # 提取平移和旋转信息
    translation = np.array(matrix_1x6[:3])  # X, Y, Z 平移
    rx, ry, rz = matrix_1x6[3:]  # Rx, Ry, Rz 旋转
    rot = Rotation.from_euler(order, [rx, ry, rz]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = translation
    return T

def transform_point_cloud(pcd: o3d.geometry.PointCloud, transform_matrix: np.ndarray):

    # 使用 4x4 矩阵变换点云

    points = np.asarray(pcd.points)

    # 将点云转换为齐次坐标 [x, y, z, 1]
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # 使用 4x4 矩阵变换点云
    transformed_points_homogeneous = np.dot(transform_matrix, points_homogeneous.T).T

    # 将齐次坐标还原为 [x, y, z]
    transformed_points = transformed_points_homogeneous[:, :3]

    # 创建新的点云对象并更新点云坐标
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)

    # 保留原始点云的颜色（如果有）
    if pcd.has_colors():
        transformed_pcd.colors = pcd.colors

    return transformed_pcd

def transform_deg_to_rad(arr: list) -> list:
    rz_rad = math.radians(arr[3])
    ry_rad = math.radians(arr[4])
    rx_rad = math.radians(arr[5])
    matrix_rad = [arr[0], arr[1], arr[2], rz_rad, ry_rad, rx_rad]
    return matrix_rad

def transform_rad_to_deg(arr: list) -> list:
    rz_deg = math.degrees(arr[3])
    ry_deg = math.degrees(arr[4])
    rx_deg = math.degrees(arr[5])
    matrix_deg = [arr[0], arr[1], arr[2], rz_deg, ry_deg, rx_deg]
    return matrix_deg