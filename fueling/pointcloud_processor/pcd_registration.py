from probreg import filterreg
from .run_time import *
from loguru import logger
import open3d as o3d
from .cut_point_cloud import remove_and_downsample
from ..pose_transformation import create_transformation_matrix

import numpy as np

def similarity_transform_3d_umeyama(src, dst, with_scaling=False, compute_rmse=True):
    """
    Umeyama 方法计算相似变换（可含缩放）。
    src, dst: (N,3)
    with_scaling: True -> 估计 scale s；False -> 固定 s=1（等同刚性）
    返回:
        T (4x4)
    """
    assert src.shape == dst.shape
    N = src.shape[0]
    assert N >= 3
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # 方差（src）
    var_src = np.sum(src_centered**2) / N

    # 协方差矩阵
    cov = (dst_centered.T @ src_centered) / N

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    if with_scaling:
        scale = np.trace(np.diag(D) @ S) / var_src
    else:
        scale = 1.0
    t = dst_mean - scale * R @ src_mean

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    if compute_rmse:
        src_h = np.hstack([src, np.ones((N,1))])
        src_trans = (T @ src_h.T).T[:, :3]
        rmse = np.sqrt(np.mean(np.sum((src_trans - dst)**2, axis=1)))
        return T, rmse
    return T
class PointCloudRegistration:
    def __init__(self, source_pc:o3d.geometry.PointCloud, method:str="filterreg", 
                 voxel_size:float=1, remove_outliers:bool=True, radius:float=0.5, min_neighbors:int=10):
        """
        初始化点云配准类
        Args:
            source_pc: 源点云 (Open3D PointCloud)
            method: 配准方法 ("filterreg", "pcd", "gmmtree", "l2dist_regs")
            voxel_size: 降采样体素大小
            remove_outliers: 是否去除离群点
            radius: 离群点检测半径
            min_neighbors: 最小邻居数
        """
        self.method = method
        self.voxel_size = voxel_size
        self.remove_outliers = remove_outliers
        self.radius = radius
        self.min_neighbors = min_neighbors
        # self.source_processed = source_pc
        
        # 保存原始源点云
        # self.source_original = copy.deepcopy(source_pc)
        logger.info(f"原始模板点云个数: {len(source_pc.points)}")
        # 预处理源点云
        self.source_processed = remove_and_downsample(
            source_pc, 
            voxel_size=self.voxel_size,
            remove_outliers=self.remove_outliers,
            radius=self.radius,
            min_neighbors=self.min_neighbors
        )
        logger.info(f"预处理后模板点云个数: {len(self.source_processed.points)}")
    @timeit
    def compute_registration(self, target_pc):
    
        logger.info(f"目标点云个数: {len(target_pc.points)}")
        # target_pc = remove_and_downsample(
        #     target_pc, 
        #     voxel_size=self.voxel_size,
        #     remove_outliers=self.remove_outliers,
        #     radius=self.radius,
        #     min_neighbors=self.min_neighbors
        # )
        
        # 执行配准
        if self.method == "filterreg":
            objective_type = 'pt2pt'
            result = filterreg.registration_filterreg(
                self.source_processed, 
                target_pc,
                objective_type=objective_type,
                sigma2=None,
                update_sigma2=True
            )
            if result is None:
                logger.error("FilterReg registration failed.")
                return None  # 或者 raise Exception(...)
            param, _, _ = result
            rot = param.rot
            t = param.t
            transformation_matrix = create_transformation_matrix(rot, t)
            return transformation_matrix
        elif self.method == "icp":
            threshold = 0.02
            trans_init = np.identity(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                self.source_processed, target_pc, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            return reg_p2p.transformation

        else:
            raise ValueError(f"不支持的配准方法: {self.method}")            