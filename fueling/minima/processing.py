# flb2_utils/processing.py
import cv2
import numpy as np
import torch
import os.path as osp
import matplotlib.cm as cm
from typing import Tuple, Optional, Dict, Any
import open3d as o3d

from .visualization import save_ransac_matching_figures, H_transform
from .validation import verify_essential_matrix
from .geometry import essential_matrix_to_transform


def eval_relapose(mkpts0: np.ndarray, mkpts1: np.ndarray, K: np.ndarray,
                    color: np.ndarray, img0: np.ndarray, img1: np.ndarray,
                    intersection_results: Dict[str, Any], save_figs: bool,
                    figures_dir: Optional[str] = None, method: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    根据给定的匹配点评估相对位姿，计算本质矩阵和变换。
    """
    inliers_E, R, t, T = compute_essential_matrix_and_transform(
        mkpts0, mkpts1, K, color, img0, img1,
        intersection_results, save_figs, figures_dir, method
    )

    return inliers_E, R, t, T

def load_and_preprocess_images(im0_path, im1_path):
    """加载并预处理图像"""
    img0_color = cv2.imread(im0_path)
    img1_color = cv2.imread(im1_path)
    img0_color = cv2.cvtColor(img0_color, cv2.COLOR_BGR2RGB)
    img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
    return img0_color, img1_color

def prepare_color_mapping(mconf):
    """准备颜色映射"""
    if len(mconf) > 0:
        conf_min = mconf.min()
        conf_max = mconf.max()
        mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)
    return cm.jet(mconf)

def compute_essential_matrix_and_transform(mkpts0, mkpts1, K, color, img0, img1, 
                                         intersection_results, save_figs, figures_dir, method):
    """计算本质矩阵和变换矩阵"""
    inliers_E, R, t, T = None, None, None, None
    
    if len(mkpts0) >= 4:
        print(f"\n[RANSAC] 参与本质矩阵计算的点数: {len(mkpts0)}")
        ret_E, inliers_E = cv2.findEssentialMat(
            mkpts0, mkpts1, cameraMatrix=K,
            method=cv2.RANSAC, prob=0.999, threshold=0.01
        )
        print("ret_E:", ret_E)
        
        if ret_E is not None:
            rank, S, det = verify_essential_matrix(ret_E)
            R, t, T, points_3d = essential_matrix_to_transform(ret_E, K, mkpts0, mkpts1)
            
            if save_figs and figures_dir is not None:
                save_transformation_results(
                    R, t, T, inliers_E, intersection_results, figures_dir, method
                )
    
    print(f"Number of inliers: {inliers_E.sum() if inliers_E is not None else 0}")
    
    if save_figs:
        save_ransac_results(img0, img1, mkpts0, mkpts1, inliers_E, color, figures_dir, method, ret_E)
    
    return inliers_E, R, t, T

def save_transformation_results(R, t, T, inliers, intersection_results, figures_dir, method):
    """保存变换矩阵结果"""
    transform_file = osp.join(figures_dir, f"transform_matrix_{method}.txt")
    with open(transform_file, 'w') as f:
        f.write("旋转矩阵 R:\n")
        f.write(str(R) + "\n\n")
        f.write("平移向量 t:\n")
        f.write(str(t) + "\n\n")
        f.write("变换矩阵 T:\n")
        f.write(str(T) + "\n\n")
        f.write(f"内点数量: {inliers.sum() if inliers is not None else 0}\n")
        f.write(f"双向交集点数: {len(intersection_results['indices'])}\n")
    print(f"变换矩阵已保存到: {transform_file}")

def save_ransac_results(img0, img1, mkpts0, mkpts1, inliers_E, color, figures_dir, method, ret_E):
    """保存RANSAC结果"""
    img0_name_E = f"fig1_E_{osp.basename(method).split('.')[0]}"
    img1_name_E = f"fig2_E_{osp.basename(method).split('.')[0]}"
    path_before = osp.join(figures_dir, f"{img0_name_E}_{img1_name_E}_before_ransac_E_{method}.jpg")
    path_after = osp.join(figures_dir, f"{img0_name_E}_{img1_name_E}_after_ransac_E_{method}.jpg")

    save_ransac_matching_figures(
        path_before, path_after, img0, img1, mkpts0, mkpts1, inliers_E, color
    )

    if ret_E is not None:
        img0_bgr = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        im0_tensor = torch.tensor(img0_bgr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.
        ret_E_tensor = torch.tensor(ret_E, dtype=torch.float32).unsqueeze(0)
        im0_tensor = H_transform(im0_tensor, ret_E_tensor)
        im0 = im0_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
        fig_path = osp.join(figures_dir, f"{img0_name_E}_after_homography_{method}.jpg")
        cv2.imwrite(fig_path, im0)
