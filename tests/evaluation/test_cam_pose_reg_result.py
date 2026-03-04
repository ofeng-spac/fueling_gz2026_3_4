import yaml
import json
import numpy as np
from utils.transform_matrix import transform_1x6_to_4x4, transform_4x4_to_1x6

def calculate_target_robot_pose(initial_robot_pose_1x6, eye_hand_matrix, cloud_point_transformation_matrix):
    """
    根据初始位姿、手眼矩阵和点云配准矩阵，计算机器人需要移动到的新拍照位姿。

    :param initial_robot_pose_1x6: 初始拍照时的机器人TCP位姿 (1x6 list or array).
    :param eye_hand_matrix: 手眼标定矩阵 (4x4 numpy array or list of lists).
    :param cloud_point_transformation_matrix: 点云配准计算出的变换矩阵 (4x4 numpy array or list of lists).
    :return: 机器人需要移动到的新拍照位姿 (4x4 numpy array).
    """
    # --- 1. 将所有输入转换为 NumPy 4x4 矩阵 ---
    T_B_from_H_init = transform_1x6_to_4x4(initial_robot_pose_1x6)
    T_H_from_C = np.array(eye_hand_matrix)
    T_C_from_H = np.linalg.inv(T_H_from_C)  # 计算相机到手部的逆矩阵

    # --- 2. 开始推导计算 ---
    T_B_from_C_init = np.dot(T_B_from_H_init, T_H_from_C)
    T_B_from_C_final = np.dot(T_B_from_C_init, cloud_point_transformation_matrix)
    T_B_from_H_final = np.dot(T_B_from_C_final, T_C_from_H)
    return T_B_from_H_final

def load_yaml_config(file_path):
    """加载并解析 YAML 配置文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_json_config(file_path):
    """加载并解析 JSON 文件"""
    with open(file_path, 'r') as file:
        return json.load(file)

def main():
    # --- 1. 定义所有文件路径 ---
    config_path = 'config/robot_pot_init_1.yml'
    eye_hand_matrix_path = 'eye_hand_matrix/eye_hand_matrix_1.json'
    # 这是您通过点云配准计算出的变换矩阵
    pcd_transform_path = 'transformation.npy'

    # --- 2. 加载所有需要的配置和矩阵 ---
    try:
        # 从 YAML 中加载初始拍照的机器人位姿
        yaml_config = load_yaml_config(config_path)
        robot_pos_ini_1x6 = yaml_config['pots']['A']['robots']['1']['source_pos_ini']

        # 从 JSON 中加载手眼矩阵
        json_config = load_json_config(eye_hand_matrix_path)
        T_H_from_C_list = json_config['eye_hand_matrix']['T']

        # 加载点云配准的结果矩阵
        cloud_point_transformation_matrix = np.load(pcd_transform_path)

    except (FileNotFoundError, KeyError) as e:
        print(f"错误：加载文件失败，请检查路径和文件内容。 {e}")
        return

    # --- 3. 调用核心函数计算最终位姿 ---
    T_B_from_H_final = calculate_target_robot_pose(
        initial_robot_pose_1x6=robot_pos_ini_1x6,
        eye_hand_matrix=T_H_from_C_list,
        cloud_point_transformation_matrix=cloud_point_transformation_matrix
    )

    # --- 4. 将最终的4x4矩阵转换为机器人能识别的1x6格式 ---
    final_robot_pose_1x6 = transform_4x4_to_1x6(T_B_from_H_final)

    # --- 5. 打印最终结果 ---
    print("\n--- 计算结果 ---")
    print("机器人需要移动到的新拍照位姿 (4x4 矩阵):")
    print(T_B_from_H_final)
    print("\n机器人需要移动到的新拍照位姿 (1x6 格式，可直接用于机器人控制):")
    print(final_robot_pose_1x6)

if __name__ == "__main__":
    main()