# flb2_utils/matching.py
import numpy as np
import cv2
import os.path as osp
from typing import Tuple, Optional, Dict, Any
from .visualization import visualize_points_on_image, visualize_method_differences
from .validation import validate_intersection_methods
import open3d as o3d
import numpy as np
import copy

def pointcloud_intersection(source_pc, target_pc, distance_threshold=0.001):
    """
    计算两个点云的交集（两个点云中都存在的点）
    
    Args:
        source_pc: 源点云
        target_pc: 目标点云
        distance_threshold: 距离阈值，小于此距离被认为是同一个点
    
    Returns:
        source_intersection: 源点云中的交集点
        target_intersection: 目标点云中的交集点
    """
    # 使用KDTree进行最近邻搜索
    target_kdtree = o3d.geometry.KDTreeFlann(target_pc)
    
    source_points = np.asarray(source_pc.points)
    target_points = np.asarray(target_pc.points)
    
    source_intersection_indices = []
    target_intersection_indices = []
    
    # 对于源点云中的每个点，在目标点云中寻找最近邻
    for i, source_point in enumerate(source_points):
        [k, idx, _] = target_kdtree.search_knn_vector_3d(source_point, 1)
        if k > 0:
            nearest_target_point = target_points[idx[0]]
            distance = np.linalg.norm(source_point - nearest_target_point)
            if distance < distance_threshold:
                source_intersection_indices.append(i)
                target_intersection_indices.append(idx[0])
    
    # 创建交集点云
    source_intersection = o3d.geometry.PointCloud()
    source_intersection.points = o3d.utility.Vector3dVector(source_points[source_intersection_indices])
    
    target_intersection = o3d.geometry.PointCloud()
    target_intersection.points = o3d.utility.Vector3dVector(target_points[target_intersection_indices])
    
    return source_intersection, target_intersection

def pointcloud_union(source_pc, target_pc, distance_threshold=0.001):
    """
    计算两个点云的并集（合并两个点云，去除重复点）
    
    Args:
        source_pc: 源点云
        target_pc: 目标点云
        distance_threshold: 距离阈值，用于去除重复点
    
    Returns:
        source_union: 并集点云（从源点云角度）
        target_union: 并集点云（从目标点云角度）
    """
    # 合并两个点云
    combined_pc = source_pc + target_pc
    
    # 使用体素下采样去除重复点
    union_pc = combined_pc.voxel_down_sample(voxel_size=distance_threshold)
    
    # 返回两个相同的点云（因为是并集）
    return copy.deepcopy(union_pc), copy.deepcopy(union_pc)

def compute_bidirectional_intersection_set(mkpts0: np.ndarray,  # array([[1267.07336426,  155.49673462], [ 654.90405273,  354.30285645],  [ 810.72351074,  372.70922852], ...,  [1034.81481934,  417.12960815], [ 720.74072266,  490.277771  ], [1037.48022461,  252.29510498]])
                                     mkpts1: np.ndarray,
                                     point_arr1: np.ndarray,
                                     point_arr2: np.ndarray,
                                     ) -> Dict[str, Any]:
    """
    计算双向交集，找出在两个视图中都落在点云投影区域内的匹配点。
    返回左右都落在投影点集里的匹配对索引、mkpts0、mkpts1
    """

    # 将匹配点坐标转换为整数
    mkpts0_int = np.round(mkpts0).astype(int)
    mkpts1_int = np.round(mkpts1).astype(int)

    # 创建投影点集的集合用于快速查找
    set1 = set(map(tuple, point_arr1))
    set2 = set(map(tuple, point_arr2))

    # 创建掩码：只有当两个视图中的点都在各自的投影点集中时才保留
    mask = np.array([
        tuple(pt0) in set1 and tuple(pt1) in set2
        for pt0, pt1 in zip(mkpts0_int, mkpts1_int)
    ])

    # 获取满足条件的索引
    idx = np.where(mask)[0]


    print(f"双向交集点数: {len(idx)}")

    # 返回结果
    results = {
        "length" : len(idx),
        "indices": idx,
        "points0": mkpts0[idx],
        "points1": mkpts1[idx]
    }
    
    return results

from scipy.spatial import KDTree

def compute_bidirectional_intersection_kdtree(mkpts0: np.ndarray, 
                                     mkpts1: np.ndarray,
                                     point_arr1: np.ndarray,
                                     point_arr2: np.ndarray,
                                     tolerance: float = 1.0) -> Dict[str, Any]:
    """
    计算双向交集，使用KDTree进行精确的最近邻搜索
    """
    
    # 为两个投影点集构建KDTree
    tree1 = KDTree(point_arr1)
    tree2 = KDTree(point_arr2)
    
    # 查找每个匹配点在各自投影点集中的最近邻距离
    dist1, idx1 = tree1.query(mkpts0, distance_upper_bound=tolerance)
    dist2, idx2 = tree2.query(mkpts1, distance_upper_bound=tolerance)
    
    # 创建掩码：距离在容差范围内且在有效范围内
    mask = (dist1 <= tolerance) & (dist2 <= tolerance) & (idx1 < len(point_arr1)) & (idx2 < len(point_arr2))
    
    # 获取满足条件的索引
    idx = np.where(mask)[0]


    print(f"双向交集点数: {len(idx)}")

    return {
        "length" : len(idx),
        "indices": idx,
        "points0": mkpts0[idx],
        "points1": mkpts1[idx]
    }
def compute_bidirectional_intersection_grid(mkpts0: np.ndarray, 
                                          mkpts1: np.ndarray,
                                          point_arr1: np.ndarray,
                                          point_arr2: np.ndarray,
                                          img0: np.ndarray,
                                          img1: np.ndarray,
                                          figures_dir: str,
                                          method: str,
                                          save_figs: bool = True,
                                          grid_size: float = 1.0,
                                          tolerance: float = 1.0) -> Dict[str, Any]:
    """
    使用网格分桶和哈希表加速查找
    """
    
    def create_grid_lookup(points, grid_size):
        """创建网格查找表"""
        grid_dict = {}
        for i, pt in enumerate(points):
            # 将坐标按网格大小分桶
            grid_key = (int(pt[0] // grid_size), int(pt[1] // grid_size))
            if grid_key not in grid_dict:
                grid_dict[grid_key] = []
            grid_dict[grid_key].append(i)
        return grid_dict
    
    def is_point_in_grid(point, grid_dict, points_array, grid_size, tolerance):
        """检查点是否在网格中的某个点附近"""
        grid_key = (int(point[0] // grid_size), int(point[1] // grid_size))
        
        # 检查当前网格和相邻网格
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_key = (grid_key[0] + dx, grid_key[1] + dy)
                if check_key in grid_dict:
                    for idx in grid_dict[check_key]:
                        if np.linalg.norm(point - points_array[idx]) <= tolerance:
                            return True
        return False
    
    # 将点数组转换为浮点型
    points_array1 = point_arr1.astype(float)
    points_array2 = point_arr2.astype(float)
    
    # 创建网格查找表
    grid_dict1 = create_grid_lookup(point_arr1, grid_size)
    grid_dict2 = create_grid_lookup(point_arr2, grid_size)
    
    # 创建掩码
    mask = []
    for pt0, pt1 in zip(mkpts0, mkpts1):
        in_set1 = is_point_in_grid(pt0, grid_dict1, points_array1, grid_size, tolerance)
        in_set2 = is_point_in_grid(pt1, grid_dict2, points_array2, grid_size, tolerance)
        mask.append(in_set1 and in_set2)
    
    mask = np.array(mask)
    idx = np.where(mask)[0]
    
    if save_figs and len(idx) > 0 and img0 is not None:
        vis_path = osp.join(figures_dir, f"both_views_in_projection_{method}.png")
        visualize_points_on_image(mkpts0[idx], vis_path, img0)

    print(f"双向交集点数: {len(idx)}")
    
    return {
        "length" : len(idx),
        "indices": idx,
        "points0": mkpts0[idx],
        "points1": mkpts1[idx]
    }
from sklearn.neighbors import BallTree

def compute_bidirectional_intersection_balltree(mkpts0: np.ndarray, 
                                              mkpts1: np.ndarray,
                                              point_arr1: np.ndarray,
                                              point_arr2: np.ndarray,
                                              img0: np.ndarray,
                                              img1: np.ndarray,
                                              figures_dir: str,
                                              method: str,
                                              save_figs: bool = True,
                                              tolerance: float = 1.0) -> Dict[str, Any]:
    """
    使用BallTree进行范围搜索
    """
    
    # 构建BallTree
    tree1 = BallTree(point_arr1.astype(float))
    tree2 = BallTree(point_arr2.astype(float))
    
    # 查找在容差范围内的点
    indices1 = tree1.query_radius(mkpts0, r=tolerance)
    indices2 = tree2.query_radius(mkpts1, r=tolerance)
    
    # 创建掩码：两个视图中都有至少一个近邻点
    mask = np.array([len(idx1) > 0 and len(idx2) > 0 
                    for idx1, idx2 in zip(indices1, indices2)])
    
    idx = np.where(mask)[0]
    
    if save_figs and len(idx) > 0 and img0 is not None:
        vis_path = osp.join(figures_dir, f"both_views_in_projection_{method}.png")
        visualize_points_on_image(mkpts0[idx], vis_path, img0)

    print(f"双向交集点数: {len(idx)}")
    
    return {
        "length" : len(idx),
        "indices": idx,
        "points0": mkpts0[idx],
        "points1": mkpts1[idx]
    }
try:
    import annoy
except ImportError:
    print("Annoy库未安装，请先安装: pip install annoy")

def compute_bidirectional_intersection_annoy(mkpts0: np.ndarray, 
                                           mkpts1: np.ndarray,
                                           point_arr1: np.ndarray,
                                           point_arr2: np.ndarray,
                                           img0: np.ndarray,
                                           img1: np.ndarray,
                                           figures_dir: str,
                                           method: str,
                                           save_figs: bool = True,
                                           tolerance: float = 1.0,
                                           n_trees: int = 10) -> Dict[str, Any]:
    """
    使用Annoy进行近似最近邻搜索（适合大数据集）
    """
    
    def build_annoy_index(points, n_trees):
        """构建Annoy索引"""
        index = annoy.AnnoyIndex(2, 'euclidean')
        for i, point in enumerate(points.astype(float)):
            index.add_item(i, point)
        index.build(n_trees)
        return index
    
    # 构建Annoy索引
    index1 = build_annoy_index(point_arr1, n_trees)
    index2 = build_annoy_index(point_arr2, n_trees)
    
    # 查找最近邻
    mask = []
    for pt0, pt1 in zip(mkpts0, mkpts1):
        idx1, dist1 = index1.get_nns_by_vector(pt0, 1, include_distances=True)
        idx2, dist2 = index2.get_nns_by_vector(pt1, 1, include_distances=True)
        
        in_set1 = (len(dist1) > 0 and dist1[0] <= tolerance)
        in_set2 = (len(dist2) > 0 and dist2[0] <= tolerance)
        mask.append(in_set1 and in_set2)
    
    mask = np.array(mask)
    idx = np.where(mask)[0]
    
    if save_figs and len(idx) > 0 and img0 is not None:
        vis_path = osp.join(figures_dir, f"both_views_in_projection_{method}.png")
        visualize_points_on_image(mkpts0[idx], vis_path, img0)

    print(f"双向交集点数: {len(idx)}")
    
    return {
        "length" : len(idx),
        "indices": idx,
        "points0": mkpts0[idx],
        "points1": mkpts1[idx]
    }