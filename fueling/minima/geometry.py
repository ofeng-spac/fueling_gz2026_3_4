# flb2_utils/geometry.py
import numpy as np
import cv2
from typing import Tuple, Optional

def draw_projection_points(image: np.ndarray, points: np.ndarray, 
                          color: Tuple[int, int, int] = (0, 255, 0), 
                          sampling_rate: int = 10) -> None:
    """在图像上绘制投影点"""
    for px, py in points[::sampling_rate]:
        cv2.circle(image, (px, py), 1, color, -1)

def create_visualization_image(ir_img: np.ndarray) -> np.ndarray:
    """根据图像类型创建可视化图像"""
    if len(ir_img.shape) == 2 and ir_img.dtype == np.uint16:
        ir_vis = cv2.convertScaleAbs(ir_img, alpha=255.0 / 4095)
        ir_vis = cv2.cvtColor(ir_vis, cv2.COLOR_GRAY2BGR)
    else:
        ir_vis = ir_img.copy()
    return ir_vis



def project_pointcloud_to_image_float(pcd, 
                                    ir_img: np.ndarray, 
                                    K: np.ndarray,
                                    out_path: Optional[str] = None,
                                    transform: Optional[np.ndarray] = None,
                                    other_bbox: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    将点云投影到红外图像上并保存可视化结果
    返回浮点数坐标、对应的原始点云索引和矩形边框mask
    
    Args:
        pcd: 点云（在左相机坐标系中）
        ir_img: 红外图像
        K: 相机内参矩阵
        out_path: 输出图像路径（可选）
        transform: 从世界/左相机坐标系到当前相机坐标系的变换矩阵（4x4）
                   如果是左图像，通常为None或单位矩阵
                   如果是右图像，需要提供从左到右的变换
        other_bbox: 另一个图像的边框 [x_min, y_min, x_max, y_max]，用于统一边框大小
    
    Returns:
        uv_visible: 可见点的图像坐标（浮点数）
        original_indices: 对应原始点云中的索引
        bbox_mask: 矩形边框mask [x_min, y_min, x_max, y_max]，如果没有可见点则返回None
    """
    if pcd.is_empty():
        raise ValueError("点云为空")
    
    pts = np.asarray(pcd.points)
    
    # 如果提供了变换矩阵，将点云变换到当前相机坐标系
    if transform is not None:
        # 将点转换为齐次坐标
        pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
        # 应用变换
        pts_transformed = (transform @ pts_hom.T).T
        pts = pts_transformed[:, :3]
    
    # 将毫米转换为米
    pts_m = pts / 1000.0
    
    if ir_img is None:
        raise ValueError("图像为空")
    
    # 投影到图像平面
    uv_hom = (K @ pts_m.T).T
    uv_float = uv_hom[:, :2] / uv_hom[:, 2:3]  # 保持浮点数

    img_height, img_width = ir_img.shape[:2]
    mask = (uv_float[:, 0] >= 0) & (uv_float[:, 0] < img_width) & \
           (uv_float[:, 1] >= 0) & (uv_float[:, 1] < img_height) & \
           (uv_hom[:, 2] > 0)
    
    uv_visible = uv_float[mask]  # 保持浮点数
    original_indices = np.where(mask)[0]  # 保存原始索引
    
    # 计算矩形边框
    bbox_mask = None
    if len(uv_visible) > 0:
        x_min = np.min(uv_visible[:, 0])
        y_min = np.min(uv_visible[:, 1])
        x_max = np.max(uv_visible[:, 0])
        y_max = np.max(uv_visible[:, 1])
        
        # 如果有另一个边框，统一大小
        if other_bbox is not None:
            # 计算当前边框的中心和大小
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            # 计算另一个边框的大小
            other_width = other_bbox[2] - other_bbox[0]
            other_height = other_bbox[3] - other_bbox[1]
            
            # 取最大的宽度和高度
            unified_width = max(width, other_width)
            unified_height = max(height, other_height)
            
            # 保持中心不变，使用统一的大小
            x_min = center_x - unified_width / 2
            y_min = center_y - unified_height / 2
            x_max = center_x + unified_width / 2
            y_max = center_y + unified_height / 2
            
            # 确保边框在图像范围内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width - 1, x_max)
            y_max = min(img_height - 1, y_max)
            
            # 再次确保包含所有点
            x_min = min(x_min, np.min(uv_visible[:, 0]))
            y_min = min(y_min, np.min(uv_visible[:, 1]))
            x_max = max(x_max, np.max(uv_visible[:, 0]))
            y_max = max(y_max, np.max(uv_visible[:, 1]))
        
        bbox_mask = np.array([x_min, y_min, x_max, y_max])
    
    if out_path:
        ir_vis = create_visualization_image(ir_img)
        # 可视化时使用四舍五入的整数坐标
        uv_int = np.round(uv_visible).astype(int)
        draw_projection_points(ir_vis, uv_int)
        
        # 绘制矩形边框
        if bbox_mask is not None:
            x_min_int = int(np.floor(bbox_mask[0]))
            y_min_int = int(np.floor(bbox_mask[1]))
            x_max_int = int(np.ceil(bbox_mask[2]))
            y_max_int = int(np.ceil(bbox_mask[3]))
            
            # 绘制矩形
            cv2.rectangle(ir_vis, 
                         (x_min_int, y_min_int), 
                         (x_max_int, y_max_int), 
                         (0, 0, 255),  # 红色边框
                         2)
            
            # 添加边框信息文本
            bbox_text = f"BBox: [{x_min_int}, {y_min_int}, {x_max_int}, {y_max_int}]"
            cv2.putText(ir_vis, bbox_text, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 0, 255), 2)
            
            # 如果使用了统一大小，添加标记
            if other_bbox is not None:
                unified_text = "Unified Size"
                cv2.putText(ir_vis, unified_text, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (255, 0, 0), 2)  # 蓝色文本
        
        cv2.imwrite(out_path, ir_vis)
        print(f"已保存投影结果：{out_path}  （共投影 {len(uv_visible)} 个点）")

    return uv_visible, original_indices, bbox_mask

# def project_pointcloud_to_image_float(pcd, 
#                                     ir_img: np.ndarray, 
#                                     K: np.ndarray,
#                                     out_path: Optional[str] = None,
#                                     transform: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     将点云投影到红外图像上并保存可视化结果
#     返回浮点数坐标和对应的原始点云索引
    
#     Args:
#         pcd: 点云（在左相机坐标系中）
#         ir_img: 红外图像
#         K: 相机内参矩阵
#         out_path: 输出图像路径（可选）
#         transform: 从世界/左相机坐标系到当前相机坐标系的变换矩阵（4x4）
#                    如果是左图像，通常为None或单位矩阵
#                    如果是右图像，需要提供从左到右的变换
    
#     Returns:
#         uv_visible: 可见点的图像坐标（浮点数）
#         original_indices: 对应原始点云中的索引
#     """
#     if pcd.is_empty():
#         raise ValueError("点云为空")
    
#     pts = np.asarray(pcd.points)
    
#     # 如果提供了变换矩阵，将点云变换到当前相机坐标系
#     if transform is not None:
#         # 将点转换为齐次坐标
#         pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
#         # 应用变换
#         pts_transformed = (transform @ pts_hom.T).T
#         pts = pts_transformed[:, :3]
    
#     # 将毫米转换为米
#     pts_m = pts / 1000.0
    
#     if ir_img is None:
#         raise ValueError("图像为空")
    
#     # 投影到图像平面
#     uv_hom = (K @ pts_m.T).T
#     uv_float = uv_hom[:, :2] / uv_hom[:, 2:3]  # 保持浮点数

#     img_height, img_width = ir_img.shape[:2]
#     mask = (uv_float[:, 0] >= 0) & (uv_float[:, 0] < img_width) & \
#            (uv_float[:, 1] >= 0) & (uv_float[:, 1] < img_height) & \
#            (uv_hom[:, 2] > 0)
    
#     uv_visible = uv_float[mask]  # 保持浮点数
#     original_indices = np.where(mask)[0]  # 保存原始索引

#     if out_path:
#         ir_vis = create_visualization_image(ir_img)
#         # 可视化时使用四舍五入的整数坐标
#         uv_int = np.round(uv_visible).astype(int)
#         draw_projection_points(ir_vis, uv_int)
#         cv2.imwrite(out_path, ir_vis)
#         print(f"已保存投影结果：{out_path}  （共投影 {len(uv_visible)} 个点）")

#     return uv_visible, original_indices
# def project_pointcloud_to_image_float(pcd, 
#                                     ir_img: np.ndarray, 
#                                     K: np.ndarray,
#                                     out_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     将点云投影到红外图像上并保存可视化结果
#     返回浮点数坐标和对应的原始点云索引
#     """

#     if pcd.is_empty():
#         raise ValueError("点云为空")
    
#     pts = np.asarray(pcd.points)
#     pts_m = pts / 1000.0
#     if ir_img is None:
#         raise ValueError("图像为空")
    
#     uv_hom = (K @ pts_m.T).T
#     uv_float = uv_hom[:, :2] / uv_hom[:, 2:3]  # 保持浮点数

#     img_height, img_width = ir_img.shape[:2]
#     mask = (uv_float[:, 0] >= 0) & (uv_float[:, 0] < img_width) & \
#            (uv_float[:, 1] >= 0) & (uv_float[:, 1] < img_height) & \
#            (uv_hom[:, 2] > 0)
    
#     uv_visible = uv_float[mask]  # 保持浮点数
#     original_indices = np.where(mask)[0]  # 保存原始索引

#     if out_path:
#         ir_vis = create_visualization_image(ir_img)
#         # 可视化时使用四舍五入的整数坐标
#         uv_int = np.round(uv_visible).astype(int)
#         draw_projection_points(ir_vis, uv_int)
#         cv2.imwrite(out_path, ir_vis)
#         print(f"已保存投影结果：{out_path}  （共投影 {len(uv_visible)} 个点）")

#     return uv_visible, original_indices  # 返回浮点数坐标和对应的原始索引

def essential_matrix_to_transform(E, K, mkpts0, mkpts1):
    """
    从本质矩阵恢复旋转和平移变换
    """
    _, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
    
    print("恢复的旋转矩阵 R:")
    print(R)
    
    print("恢复的平移向量 t (归一化):")
    print(t)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    print("完整的4x4变换矩阵 T:")
    print(T)
    
    points_3d = triangulate_points(K, R, t, mkpts0, mkpts1) if len(mkpts0) > 0 else None
    
    return R, t, T, points_3d

def triangulate_points(K, R, t, mkpts0, mkpts1):
    """三角化得到3D点"""
    P0 = K @ np.eye(3, 4)
    P1 = K @ np.hstack((R, t))
    
    points_4d = cv2.triangulatePoints(P0, P1, mkpts0.T, mkpts1.T)
    points_3d = points_4d[:3] / points_4d[3]
    
    print(f"三角化得到 {points_3d.shape[1]} 个3D点")
    print(f"3D点深度范围: {points_3d[2].min():.3f} - {points_3d[2].max():.3f}")
    
    return points_3d

def rotation_matrix_to_euler_angles(R):
    """将旋转矩阵转换为欧拉角"""
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
        
    return np.array([x, y, z]) * 180 / np.pi