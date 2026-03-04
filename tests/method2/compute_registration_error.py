from typing import Optional
import numpy as np
import open3d as o3d

def compute_registration_error(source_pc:o3d.geometry.PointCloud, target_pc:o3d.geometry.PointCloud, transformation:Optional[np.ndarray] = None, max_correspondence_distance: float = 5.0):
    # 1. 平均距离误差 (Mean Error):所有源点到目标点的平均距离，这个值容易受少数离群点影响而变大。
    if transformation is None:
        transformation = np.eye(4)
    source_pc_transformed = source_pc.transform(transformation)
    distances = source_pc_transformed.compute_point_cloud_distance(target_pc)
    mean_error = np.mean(distances)

    # 2. 内点均方根误差 (Inlier RMSE):只计算距离小于 {inlier_threshold} mm 的'好'点对的误差，更能反映主体部分的配准精度。这个值通常会'好看'很多。
    # 3. 适应度 (Fitness):目标点云中，能找到对应源点云（在阈值范围内）的点的比例。值越高，说明匹配得越好。
    # 这只会考虑距离在 max_correspondence_distance 以内的“好”的匹配点
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_pc, target_pc, max_correspondence_distance, transformation)
    inlier_rmse = evaluation.inlier_rmse
    fitness = evaluation.fitness

    return mean_error, inlier_rmse, fitness

