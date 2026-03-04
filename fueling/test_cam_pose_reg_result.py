import numpy as np
from typing import Optional
import numpy as np
import open3d as o3d
import cv2
import numpy as np
from loguru import logger


def compare_images_pixel_perfect(img1, img2, show=True):
    """
    逐像素比较两张图像是否完全相同（支持灰度和彩色）。
    """
    # 1. 检查尺寸
    if img1.shape != img2.shape:
        logger.error(f"尺寸不同：img1={img1.shape}, img2={img2.shape}")
        return

    # 2. 转浮点计算MSE
    err = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    psnr = 10 * np.log10((255 ** 2) / err) if err != 0 else float('inf')

    # 3. 输出结果
    if err == 0:
        logger.info("✅ 两张图片在像素级完全相同。")
        return
    else:
        logger.warning(f"⚠️ 两张图片不同。MSE={err:.4f}, PSNR={psnr:.2f} dB")

    # 4. 差异图
    diff = cv2.absdiff(img1, img2)
    cv2.imwrite("difference_image.png", diff)
    logger.info("差异图已保存为 'difference_image.png'")

    # 5. 转为三通道以便拼接和叠加
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    else:
        img1_color, img2_color, diff_color = img1, img2, diff

    # 6. 拼接可视化
    vis = np.hstack((img1_color, img2_color, diff_color))
    cv2.putText(vis, "Image 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, "Image 2", (img1.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, "Difference", (img1.shape[1]*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite("comparison_visual.png", vis)
    logger.info("拼接图已保存为 'comparison_visual.png'")

    # 7. 叠加效果
    overlay = cv2.addWeighted(img1_color, 0.5, img2_color, 0.5, 0)
    cv2.imwrite("overlay_image.png", overlay)
    logger.info("叠加图已保存为 'overlay_image.png'")

    # 8. 显示（可选）
    if show:
        cv2.imshow("Comparison", vis)
        cv2.imshow("Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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


def calculate_target_robot_pose(T_B_from_H_init, eye_hand_matrix, T_Ca2_from_Ca1):
    """
    根据初始位姿、手眼矩阵和点云配准矩阵，计算机器人需要移动到的新拍照位姿。

    :param initial_robot_pose_1x6: 初始拍照时的机器人TCP位姿 (1x6 list or array).
    :param eye_hand_matrix: 手眼标定矩阵 (4x4 numpy array or list of lists).
    :param cloud_point_transformation_matrix: 点云配准计算出的变换矩阵 (4x4 numpy array or list of lists).
    :return: 机器人需要移动到的新拍照位姿 (4x4 numpy array).
    """
    # --- 1. 将所有输入转换为 NumPy 4x4 矩阵 ---
    T_H_from_C = np.array(eye_hand_matrix)
    T_C_from_H = np.linalg.inv(T_H_from_C)  # 计算相机到手部的逆矩阵

    # --- 2. 开始推导计算 ---
    T_B_from_C_init = np.dot(T_B_from_H_init, T_H_from_C)
    T_B_from_C_final = np.dot(T_B_from_C_init, T_Ca2_from_Ca1)
    T_B_from_H_final = np.dot(T_B_from_C_final, T_C_from_H)
    return T_B_from_H_final
