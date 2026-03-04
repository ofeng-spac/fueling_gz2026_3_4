import numpy as np
import open3d as o3d

def load_point_cloud(file_path):
    # 支持 .ply, .pcd, .xyz 等格式
    return o3d.io.read_point_cloud(file_path)

def load_transformation_matrix(file_path):
    # 假设为4x4的txt或npy文件
    if file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        return np.loadtxt(file_path)

def compute_registration_error(source_pc, target_pc, transformation, max_correspondence_distance):
    # 1. 平均距离误差 (Mean Error) - 这是您之前使用的指标
    source_pc_transformed = source_pc.transform(transformation)
    distances = source_pc_transformed.compute_point_cloud_distance(target_pc)
    mean_error = np.mean(distances)

    # 2. 内点均方根误差 (Inlier RMSE) 和 适应度 (Fitness) - 新增的指标
    #    这只会考虑距离在 max_correspondence_distance 以内的“好”的匹配点
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_pc, target_pc, max_correspondence_distance, transformation)
    inlier_rmse = evaluation.inlier_rmse
    fitness = evaluation.fitness

    return mean_error, inlier_rmse, fitness


if __name__ == "__main__":
    # 文件路径
    source_path = "data_pcd/pot_1_source_0_cut_point_cloud_box.pcd"
    target_path = "data_pcd/pot_1_target_device0_cut_point_cloud_box.pcd"
    transformation_path = "transformation.npy"

    # --- 新增参数 ---
    # 设置一个距离阈值（单位与点云单位相同，例如mm），用于区分内点和外点
    # 只有小于这个距离的对应点对才被认为是内点（inliers）
    inlier_threshold = 4.5  # 您可以根据数据的实际情况调整这个值

    # 读取点云和变换矩阵
    source_pc = load_point_cloud(source_path)
    target_pc = load_point_cloud(target_path)
    transformation = load_transformation_matrix(transformation_path)

    # 计算多种误差
    mean_err, rmse_err, fit_rate = compute_registration_error(source_pc, target_pc, transformation, inlier_threshold)

    print("--- 配准精度评估 ---")
    print(f"平均配准距离 (Mean Error): {mean_err:.4f} mm")
    print(f"内点均方根误差 (Inlier RMSE): {rmse_err:.4f} mm")
    print(f"适应度 (Fitness): {fit_rate:.4f}")
    print("\n--- 指标解释 ---")
    print("1. 平均配准距离: 所有源点到目标点的平均距离，这个值容易受少数离群点影响而变大。")
    print(f"2. 内点均方根误差: 只计算距离小于 {inlier_threshold} mm 的'好'点对的误差，更能反映主体部分的配准精度。这个值通常会'好看'很多。")
    print("3. 适应度: 目标点云中，能找到对应源点云（在阈值范围内）的点的比例。值越高，说明匹配得越好。")