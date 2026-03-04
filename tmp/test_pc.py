import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import os

def test_depth_to_pointcloud():
    # 1. 加载深度图
    depth_path = "../working_data/20251221230151/arm_1/depth/stereo_depth.png"
    depth_map_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if depth_map_uint16 is None:
        print("无法加载深度图")
        return
    
    print(f"深度图形状: {depth_map_uint16.shape}")
    print(f"深度图数据类型: {depth_map_uint16.dtype}")
    print(f"深度图最小值: {np.min(depth_map_uint16)}")
    print(f"深度图最大值: {np.max(depth_map_uint16)}")
    print(f"深度图非零点数: {np.sum(depth_map_uint16 > 0)}")
    
    # 2. 相机内参（从你的配置中获取）
    K = [958.7590451, 959.7424983, 627.92186167, 408.1675004]  # fx, fy, cx, cy
    
    # 3. 转换为米
    depth_map_meters = depth_map_uint16.astype(np.float32) / 1000.0
    
    # 4. 创建点云
    pcd = o3d.geometry.PointCloud()
    
    # 获取图像尺寸
    height, width = depth_map_meters.shape
    
    # 生成网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 筛选有效点
    valid_mask = depth_map_meters > 0
    
    if np.sum(valid_mask) == 0:
        print("没有有效深度点")
        return
    
    u_valid = u[valid_mask].flatten()
    v_valid = v[valid_mask].flatten()
    z_valid = depth_map_meters[valid_mask].flatten()
    
    # 反投影到3D
    fx, fy, cx, cy = K
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy
    
    # 创建点云
    points = np.stack([x_valid, y_valid, z_valid], axis=-1)
    
    print(f"生成点云点数: {len(points)}")
    
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 5. 保存点云
    output_path = "test_pointcloud.pcd"
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到: {output_path}")
    
    # 6. 可视化（可选）
    try:
        o3d.visualization.draw_geometries([pcd])
    except:
        print("无法可视化点云")

if __name__ == "__main__":
    test_depth_to_pointcloud()