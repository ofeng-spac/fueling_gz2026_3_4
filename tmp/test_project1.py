import numpy as np
import cv2
import pickle
import os

def compare_depth_calculations():
    """
    对比新旧版本的深度计算
    """
    print("=== 深度计算对比 ===")
    
    # 找到最新的工作目录（新版本）
    working_dirs = sorted(os.listdir("../working_data"))
    latest_dir = working_dirs[-1]
    print(f"使用最新目录: {latest_dir}")
    
    # 加载配置
    import _jsonnet
    config = json.loads(_jsonnet.evaluate_file("../data/arm1/config.jsonnet"))
    
    # 获取参数
    K = config['robot']['RS_camera']['K']  # 原始内参（列表）
    baseline = config['robot']['RS_camera']['stereo_baseline']
    K_mat = np.array(K).reshape(3, 3)  # 矩阵形式
    
    print(f"K (列表): {K}")
    print(f"K_mat[0,0]: {K_mat[0, 0]}")
    print(f"baseline: {baseline} 米")
    
    # 加载新版本的视差图（ROI计算）
    result_path = f"../working_data/{latest_dir}/arm_1/disparities/left_disp.pkl"
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    
    disparity_roi = result
    
    # 新版本计算深度（ROI方式）
    disparity_roi_valid = np.where(disparity_roi <= 0, 0.1, disparity_roi)
    fx_new = K_mat[0, 0]
    depth_new = (fx_new * baseline) / disparity_roi_valid
    
    # 旧版本计算深度（假设有旧版本的视差图）
    # 如果旧版本保存了视差图，可以加载对比
    # 否则，我们需要模拟旧版本：用全图计算视差
    
    print(f"\n新版本（ROI）统计:")
    print(f"  视差图尺寸: {disparity_roi.shape}")
    print(f"  视差范围: {np.min(disparity_roi):.2f} - {np.max(disparity_roi):.2f}")
    print(f"  深度范围: {np.min(depth_new):.3f} - {np.max(depth_new):.3f} 米")
    print(f"  平均深度: {np.mean(depth_new):.3f} 米")
    
    # 检查K[0]和K_mat[0,0]是否相同
    print(f"\n内参对比:")
    print(f"  K[0] (列表第一个元素): {K[0]}")
    print(f"  K_mat[0,0] (矩阵元素): {K_mat[0, 0]}")
    print(f"  是否相同: {abs(K[0] - K_mat[0, 0]) < 0.001}")
    
    # 测试不同的计算方式
    print(f"\n=== 测试不同的计算方式 ===")
    
    # 方式1：使用K[0]（旧版本方式）
    fx1 = K[0]
    depth1 = (fx1 * baseline) / disparity_roi_valid
    print(f"方式1 (K[0]={fx1}): 平均深度 = {np.mean(depth1):.3f} 米")
    
    # 方式2：使用K_mat[0,0]（新版本方式）
    fx2 = K_mat[0, 0]
    depth2 = (fx2 * baseline) / disparity_roi_valid
    print(f"方式2 (K_mat[0,0]={fx2}): 平均深度 = {np.mean(depth2):.3f} 米")
    import open3d as o3d
    # 方式3：尝试缩放视差（基于源点云）
    source_pc = o3d.io.read_point_cloud("../data/arm1/pot3/source_model.pcd")
    source_points = np.asarray(source_pc.points)
    source_avg_depth_m = np.mean(source_points[:, 2]) / 1000.0
    
    expected_disparity = fx2 * baseline / source_avg_depth_m
    actual_average_disparity = np.mean(disparity_roi_valid)
    scale_factor = expected_disparity / actual_average_disparity
    
    disparity_scaled = disparity_roi_valid * scale_factor
    depth3 = (fx2 * baseline) / disparity_scaled
    print(f"方式3 (缩放因子={scale_factor:.2f}): 平均深度 = {np.mean(depth3):.3f} 米")
    print(f"  源点云平均深度: {source_avg_depth_m:.3f} 米")
    
    return depth1, depth2, depth3

if __name__ == '__main__':
    import json
    depth1, depth2, depth3 = compare_depth_calculations()