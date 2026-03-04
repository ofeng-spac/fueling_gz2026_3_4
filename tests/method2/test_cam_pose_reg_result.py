import numpy as np

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
