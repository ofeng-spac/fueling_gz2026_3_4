# flb2_utils/pointcloud.py
import open3d as o3d
import numpy as np
import os.path as osp
from scipy.spatial import KDTree


def extract_sparse_pointcloud_from_dense_float(pcd_dense, projected_points_2d, projected_indices, mkpts_2d):
    """
    从稠密点云中提取与2D匹配点对应的3D点（仅几何信息）
    【新函数】使用浮点数坐标和原始索引映射
    """
    print(f"从稠密点云提取稀疏点云（浮点数精度）...")
    
    pcd_sparse = o3d.geometry.PointCloud()
    points_3d = []
    
    dense_points = np.asarray(pcd_dense.points)
    
    # 使用浮点数构建KDTree
    kdtree = KDTree(projected_points_2d)  # 已经是浮点数
    
    matched_count = 0
    
    for mkpt in mkpts_2d:
        distance, idx_in_projected = kdtree.query(mkpt)
        
        if distance < 2.0 and idx_in_projected < len(projected_indices):
            # 使用映射找到原始点云中的正确索引
            original_idx = projected_indices[idx_in_projected]
            point_3d = dense_points[original_idx]
            points_3d.append(point_3d)
            matched_count += 1
    
    print(f"成功匹配 {matched_count}/{len(mkpts_2d)} 个点")
    
    if len(points_3d) > 0:
        pcd_sparse.points = o3d.utility.Vector3dVector(np.array(points_3d))
        return pcd_sparse
    else:
        print(f"警告：无法从稠密点云中找到对应的3D点")
        return None


def create_sparse_pointclouds_from_bidirectional_matches_float(pcd1, pcd2, point_arr1, point_arr2,
                                                             indices1, indices2,
                                                             mkpts0_filtered, mkpts1_filtered):
    """
    为双向交集的匹配点创建稀疏点云
    【新函数】使用浮点数精度和索引映射
    """
    print("\n=== 为双向交集点创建稀疏点云（浮点数精度）===")
    sparse_pcd0 = extract_sparse_pointcloud_from_dense_float(
        pcd_dense=pcd1,
        projected_points_2d=point_arr1,
        projected_indices=indices1,
        mkpts_2d=mkpts0_filtered,
    )
    sparse_pcd1 = extract_sparse_pointcloud_from_dense_float(
        pcd_dense=pcd2,
        projected_points_2d=point_arr2,
        projected_indices=indices2,
        mkpts_2d=mkpts1_filtered,
    )
    return sparse_pcd0, sparse_pcd1

def save_point_cloud(pcd, save_path):
    """保存点云到文件"""
    if pcd is not None and not pcd.is_empty():
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"点云已保存到: {save_path}")
    else:
        print(f"警告: 无法保存点云，因为它为空或为 None。路径: {save_path}")