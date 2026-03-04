import os
from typing import Optional, Dict, Any

import numpy as np

from fueling.task import run_sync
from fueling.minima.geometry import project_pointcloud_to_image_float
from fueling.minima.matching import compute_bidirectional_intersection_kdtree
from fueling.minima.pointcloud import create_sparse_pointclouds_from_bidirectional_matches_float
from fueling.pointcloud_processor.depth_to_point_cloud import filter_outliers_by_superansac
from fueling.pointcloud_processor import preprocess_pointcloud
from fueling.minima.visualization import visualize_points_on_image, visualize_projection_matches, visualize_sparse_pointclouds


async def run_minima_pipeline(
    minima_service,
    source_pc,
    current_pc,
    initial_images: Dict[str, np.ndarray],
    current_images: Dict[str, np.ndarray],
    K_mat: np.ndarray,
    eye_hand_matrix,
    capture_pose,
    fueling_pose,
    cut_box,
    sparse_voxel_size: float = 0.001,
    debug: bool = False,
    output_dir: Optional[str] = None,
):
    """运行 MINIMA 稀疏匹配流水线（过程式函数）。

    步骤：
    1. 将 source 投影到初始图像
    2. 对 current_pc 做预处理并投影到当前图像
    3. 通过 minima_service 的队列调用 MINIMA，得到左右匹配
    4. 双向投影过滤
    5. 构建左右稀疏点云并合并
    6. 使用 SuperRANSAC 过滤离群点

    返回：包含稀疏点云、匹配点和 SuperRANSAC 结果的字典
    """

    # 投影 source 到初始图像
    point_arr1_left, indices1_left = project_pointcloud_to_image_float(source_pc, initial_images.get('left_ir'), K_mat)
    point_arr1_right, indices1_right = project_pointcloud_to_image_float(source_pc, initial_images.get('right_ir'), K_mat)

    # 对 current 点云做预处理（稀疏化）
    current_pc_sparse = await run_sync(
        preprocess_pointcloud,
        eye_hand_matrix=eye_hand_matrix,
        source_pcd=current_pc,
        dimensions=cut_box,
        capture_pose=capture_pose,
        fueling_pose=fueling_pose,
        voxel_size=sparse_voxel_size,
        remove_outliers=False,
        radius=0.01,
        min_neighbors=5,
    )

    # 投影 current 稀疏点云到当前图像
    point_arr2_left, indices2_left = project_pointcloud_to_image_float(current_pc_sparse, current_images.get('left_ir'), K_mat)
    point_arr2_right, indices2_right = project_pointcloud_to_image_float(current_pc_sparse, current_images.get('right_ir'), K_mat)

    # 调用 MINIMA（通过 service，内部做队列+worker，线程安全）
    match_res_left = await minima_service.match(initial_images.get('left_ir'), current_images.get('left_ir'))
    match_res_right = await minima_service.match(initial_images.get('right_ir'), current_images.get('right_ir'))

    mkpts0_orig_left = match_res_left.get('mkpts0', np.empty((0, 2)))
    mkpts1_orig_left = match_res_left.get('mkpts1', np.empty((0, 2)))
    mconf_orig_left = match_res_left.get('mconf', np.empty((0,)))

    mkpts0_orig_right = match_res_right.get('mkpts0', np.empty((0, 2)))
    mkpts1_orig_right = match_res_right.get('mkpts1', np.empty((0, 2)))
    mconf_orig_right = match_res_right.get('mconf', np.empty((0,)))

    # 双向投影过滤（KDTree）
    intersection_results_kdtree_left = compute_bidirectional_intersection_kdtree(
        mkpts0_orig_left, mkpts1_orig_left, point_arr1_left, point_arr2_left, tolerance=1.0
    )
    intersection_results_kdtree_right = compute_bidirectional_intersection_kdtree(
        mkpts0_orig_right, mkpts1_orig_right, point_arr1_right, point_arr2_right, tolerance=1.0
    )

    both_idx_left = intersection_results_kdtree_left.get('indices', np.array([], dtype=int))
    both_idx_right = intersection_results_kdtree_right.get('indices', np.array([], dtype=int))

    # 过滤匹配对
    mkpts0_filtered_left = mkpts0_orig_left[both_idx_left] if len(both_idx_left) > 0 else np.empty((0, 2))
    mkpts1_filtered_left = mkpts1_orig_left[both_idx_left] if len(both_idx_left) > 0 else np.empty((0, 2))
    mconf_filtered_left = mconf_orig_left[both_idx_left] if len(both_idx_left) > 0 else np.empty((0,))

    mkpts0_filtered_right = mkpts0_orig_right[both_idx_right] if len(both_idx_right) > 0 else np.empty((0, 2))
    mkpts1_filtered_right = mkpts1_orig_right[both_idx_right] if len(both_idx_right) > 0 else np.empty((0, 2))
    mconf_filtered_right = mconf_orig_right[both_idx_right] if len(both_idx_right) > 0 else np.empty((0,))

    # 构建稀疏点云
    sparse_source_pc_left, sparse_current_pc_left = create_sparse_pointclouds_from_bidirectional_matches_float(
        pcd1=source_pc,
        pcd2=current_pc_sparse,
        point_arr1=point_arr1_left,
        point_arr2=point_arr2_left,
        indices1=indices1_left,
        indices2=indices2_left,
        mkpts0_filtered=mkpts0_filtered_left,
        mkpts1_filtered=mkpts1_filtered_left,
    )

    sparse_source_pc_right, sparse_current_pc_right = create_sparse_pointclouds_from_bidirectional_matches_float(
        pcd1=source_pc,
        pcd2=current_pc_sparse,
        point_arr1=point_arr1_right,
        point_arr2=point_arr2_right,
        indices1=indices1_right,
        indices2=indices2_right,
        mkpts0_filtered=mkpts0_filtered_right,
        mkpts1_filtered=mkpts1_filtered_right,
    )

    # 合并左右
    sparse_source_pc = sparse_source_pc_left + sparse_source_pc_right
    sparse_current_pc = sparse_current_pc_left + sparse_current_pc_right

    # 合并匹配信息
    try:
        mkpts0_filtered = np.vstack((mkpts0_filtered_left, mkpts0_filtered_right))
    except Exception:
        mkpts0_filtered = np.empty((0, 2))
    try:
        mkpts1_filtered = np.vstack((mkpts1_filtered_left, mkpts1_filtered_right))
    except Exception:
        mkpts1_filtered = np.empty((0, 2))
    try:
        mconf_filtered = np.concatenate((mconf_filtered_left, mconf_filtered_right))
    except Exception:
        mconf_filtered = np.empty((0,))

    # 离群点过滤（SuperRANSAC）
    sparse_source_pc, sparse_current_pc, valid_indices, T_superansac = filter_outliers_by_superansac(
        sparse_source_pc, sparse_current_pc
    )

    # 根据有效索引更新匹配点
    if len(valid_indices) > 0 and mkpts0_filtered.shape[0] > 0:
        mkpts0_filtered = mkpts0_filtered[valid_indices]
        mkpts1_filtered = mkpts1_filtered[valid_indices]
        mconf_filtered = mconf_filtered[valid_indices]

    # 可视化与保存（可选）
    if debug and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        visualize_points_on_image(mkpts0_orig_left, os.path.join(output_dir, 'minima_left_matches.png'), background_img=initial_images.get('left_ir'))
        visualize_points_on_image(mkpts1_orig_left, os.path.join(output_dir, 'minima_right_matches.png'), background_img=current_images.get('left_ir'))
        visualize_projection_matches(initial_images.get('left_ir'), current_images.get('left_ir'), mkpts0_orig_left, mkpts1_orig_left, os.path.join(output_dir, 'minima_matchlines.png'))
        visualize_projection_matches(initial_images.get('left_ir'), current_images.get('left_ir'), mkpts0_filtered, mkpts1_filtered, os.path.join(output_dir, 'minima_final_inliers_matchlines.png'))
        visualize_sparse_pointclouds(sparse_source_pc, sparse_current_pc, title=f"MINIMA Sparse ({len(sparse_source_pc.points)} pts)")

    return {
        'sparse_source_pc': sparse_source_pc,
        'sparse_current_pc': sparse_current_pc,
        'mkpts0_filtered': mkpts0_filtered,
        'mkpts1_filtered': mkpts1_filtered,
        'mconf_filtered': mconf_filtered,
        'valid_indices': valid_indices,
        'T_superansac': T_superansac,
        'intersection_left': both_idx_left,
        'intersection_right': both_idx_right,
    }
