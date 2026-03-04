import numpy as np
import open3d as o3d
import time
import time
import numpy as np

def run_superansac_rigid_transform(correspondences, config=None):
    """
    使用SuperRANSAC计算刚体变换矩阵
    
    Parameters:
    -----------
    correspondences : numpy.ndarray
        对应点对，形状为 (N, 6)，每行 [x1, y1, z1, x2, y2, z2]
    config : pysuperansac.RANSACSettings, optional
        RANSAC配置
    
    Returns:
    --------
    T : numpy.ndarray
        4x4变换矩阵
    inliers : numpy.ndarray
        内点索引
    """
    import pysuperansac
    
    if config is None:
        config = pysuperansac.RANSACSettings()
        config.inlier_threshold = 0.5
        config.min_iterations = 1000
        config.max_iterations = 1000
        config.confidence = 0.999
        config.sampler = pysuperansac.SamplerType.PROSAC
        config.scoring = pysuperansac.ScoringType.MAGSAC
        config.local_optimization = pysuperansac.LocalOptimizationType.NestedRANSAC
        config.final_optimization = pysuperansac.LocalOptimizationType.LSQ
    
    # 处理坐标平移（避免负坐标）
    min_coordinates = np.min(correspondences, axis=0)
    T1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], 
                  [-min_coordinates[0], -min_coordinates[1], -min_coordinates[2], 1]])
    T2inv = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], 
                     [min_coordinates[3], min_coordinates[4], min_coordinates[5], 1]])
    transformed_correspondences = correspondences - min_coordinates
    
    # 运行SuperRANSAC
    tic = time.perf_counter()
    T, inliers, score, iterations = pysuperansac.estimateRigidTransform(
        np.ascontiguousarray(transformed_correspondences), 
        np.max(transformed_correspondences, axis=0),
        config=config)
    toc = time.perf_counter()
    elapsed_time = toc - tic
    
    if T is not None:
        # 应用逆平移变换
        T = T1 @ T @ T2inv
        T = T.T
    
    print(f'{len(inliers)} inliers found by SupeRANSAC in {elapsed_time:0.3f} seconds')
    return T, inliers
def filter_outliers_by_superansac(sparse_source_pc, sparse_current_pc, config=None):
    """
    使用SuperRANSAC过滤离群点并计算刚体变换
    
    Args:
        sparse_source_pc: 源稀疏点云
        sparse_current_pc: 目标稀疏点云  
        config: SuperRANSAC配置参数
    
    Returns:
        filtered_source_pc: 过滤后的源点云
        filtered_current_pc: 过滤后的目标点云
        valid_indices: 有效点索引
        T: 变换矩阵（可选）
    """
    source_points = np.asarray(sparse_source_pc.points)
    current_points = np.asarray(sparse_current_pc.points)
    
    # 检查点云是否为空
    if len(source_points) == 0 or len(current_points) == 0:
        print("警告：点云为空，无法进行SuperRANSAC过滤")
        return sparse_source_pc, sparse_current_pc, np.arange(len(source_points)), None
    
    # 构建对应点对
    correspondences = np.hstack([source_points, current_points])
    
    print(f"SuperRANSAC处理前点对数量: {len(correspondences)}")
    
    # 运行SuperRANSAC
    T, inliers = run_superansac_rigid_transform(correspondences, config)
    
    if T is not None and inliers is not None:
        # 使用内点索引过滤点云
        valid_indices = inliers
        
        # 创建过滤后的点云
        filtered_source_pc = sparse_source_pc.select_by_index(valid_indices)
        filtered_current_pc = sparse_current_pc.select_by_index(valid_indices)
        
        print(f"SuperRANSAC找到的内点数量: {len(valid_indices)}")
        
        return filtered_source_pc, filtered_current_pc, valid_indices, T
    else:
        print("SuperRANSAC失败，使用原始点云")
        # 如果SuperRANSAC失败，返回原始点云
        valid_indices = np.arange(len(source_points))
        return sparse_source_pc, sparse_current_pc, valid_indices, None
    
def filter_outliers_by_distance(sparse_source_pc, sparse_current_pc, std_threshold=2.0):
    """
    基于点对距离的统计特性过滤离群点
    
    Args:
        sparse_source_pc: 源稀疏点云
        sparse_current_pc: 目标稀疏点云  
        std_threshold: 标准差阈值，越大过滤越严格
    """
    source_points = np.asarray(sparse_source_pc.points)
    current_points = np.asarray(sparse_current_pc.points)
    
    # 计算每个点对的距离
    distances = np.linalg.norm(source_points - current_points, axis=1)
    
    # 计算距离的统计特性
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # 设置过滤阈值
    threshold = mean_dist + std_threshold * std_dist
    
    # 过滤离群点
    valid_indices = distances < threshold
    
    # 创建过滤后的点云
    filtered_source_pc = sparse_source_pc.select_by_index(np.where(valid_indices)[0])
    filtered_current_pc = sparse_current_pc.select_by_index(np.where(valid_indices)[0])
    
    print(f"过滤前点对数量: {len(distances)}")
    print(f"过滤后点对数量: {np.sum(valid_indices)}")
    print(f"距离统计 - 均值: {mean_dist:.4f}, 标准差: {std_dist:.4f}, 阈值: {threshold:.4f}")
    
    return filtered_source_pc, filtered_current_pc, valid_indices

def filter_by_local_consistency(sparse_source_pc, sparse_current_pc, consistency_threshold=0.05):
    """
    基于局部几何一致性的过滤，对整体偏移不敏感
    """
    source_points = np.asarray(sparse_source_pc.points)
    current_points = np.asarray(sparse_current_pc.points)
    
    if len(source_points) < 4:
        return sparse_source_pc, sparse_current_pc, np.ones(len(source_points), dtype=bool)
    
    # 计算每个点的局部邻域关系
    valid_indices = []
    
    for i in range(len(source_points)):
        # 计算当前点到其他所有点的距离（源点云）
        source_dists = np.linalg.norm(source_points - source_points[i], axis=1)
        
        # 计算当前点到其他所有点的距离（当前点云）
        current_dists = np.linalg.norm(current_points - current_points[i], axis=1)
        
        # 找到最近的几个邻居（排除自己）
        k = min(5, len(source_points)-1)
        nearest_indices = np.argsort(source_dists)[1:k+1]  # 跳过自己
        
        # 检查局部距离的一致性
        source_local_dists = source_dists[nearest_indices]
        current_local_dists = current_dists[nearest_indices]
        
        # 计算相对距离的一致性
        dist_ratios = current_local_dists / (source_local_dists + 1e-8)
        consistency = np.std(dist_ratios)  # 标准差越小，一致性越好
        
        if consistency < consistency_threshold:
            valid_indices.append(i)
    
    valid_indices = np.array(valid_indices)
    
    filtered_source_pc = sparse_source_pc.select_by_index(valid_indices)
    filtered_current_pc = sparse_current_pc.select_by_index(valid_indices)
    
    print(f"局部一致性过滤: {len(source_points)} -> {len(valid_indices)} 个点")
    
    return filtered_source_pc, filtered_current_pc, valid_indices

def depth_to_point_cloud(depth_map: np.ndarray, camera_intrinsics: list, max_distance=600, min_distance=0):
    
    if depth_map is None:
        raise ValueError(f"无法读取深度图")

    # 获取摄像机内参
    intrinsics = np.array(camera_intrinsics).reshape(3, 3)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 获取深度图的高和宽
    height, width = depth_map.shape

    # 创建像素网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # 计算归一化坐标
    x_normalized = (u - cx) / fx
    y_normalized = (v - cy) / fy

    # 计算三维点坐标
    X = x_normalized * depth_map
    Y = y_normalized * depth_map
    Z = depth_map

    # 将点云整理为 (N x 3) 的形式
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    # 根据距离过滤点云
    point_cloud_all = points[(points[:, 2] <= max_distance) & (points[:, 2] >= min_distance)]
    
    # 过滤掉Z值为0的点（无效点）
    point_cloud_all = point_cloud_all[point_cloud_all[:, 2] > 0]

    # 将 NumPy 数组转为 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_all)

    return point_cloud

