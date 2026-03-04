"""
点云配准评估模块
支持有对应关系和无对应关系两种配准方式的评估
"""

import numpy as np
import open3d as o3d
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import json
import matplotlib.pyplot as plt
from loguru import logger
import cv2


class RegistrationEvaluator:
    """点云配准评估器"""

    def __init__(self, debug_mode: bool = True, output_dir: Optional[Path] = None):
        """
        Args:
            debug_mode: 是否输出调试信息和可视化
            output_dir: 评估结果输出目录
        """
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        if output_dir:
            self.eval_dir = output_dir / 'evaluation'
            self.eval_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self,
                 source_pcd: o3d.geometry.PointCloud,
                 target_pcd: o3d.geometry.PointCloud,
                 transformation: np.ndarray,
                 correspondences: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 method_name: str = "Unknown",
                 gt_transformation: Optional[np.ndarray] = None) -> Dict:
        """
        统一的配准评估接口

        Args:
            source_pcd: 源点云
            target_pcd: 目标点云
            transformation: 4x4变换矩阵
            correspondences: 对应点对 (source_points, target_points), 可选
            method_name: 方法名称 (用于报告)
            gt_transformation: ground truth变换矩阵 (如果有)

        Returns:
            metrics: 评估指标字典
        """
        logger.info(f"开始评估配准结果 - 方法: {method_name}")

        # 应用变换
        source_transformed = o3d.geometry.PointCloud(source_pcd)
        source_transformed.transform(transformation)

        metrics = {
            'method_name': method_name,
            'transformation': transformation.tolist(),
        }

        # 根据是否有对应关系选择评估方法
        if correspondences is not None:
            # 有对应关系：使用精确的对应点误差评估
            logger.info("使用对应点误差评估 (有对应关系)")
            corr_metrics = self._evaluate_with_correspondences(
                source_pcd, target_pcd, transformation, correspondences
            )
            metrics.update(corr_metrics)
        else:
            # 无对应关系：使用密集点云评估
            logger.info("使用密集点云评估 (无对应关系)")
            dense_metrics = self._evaluate_dense_registration(
                source_transformed, target_pcd
            )
            metrics.update(dense_metrics)

        # 如果有ground truth，评估变换精度
        if gt_transformation is not None:
            gt_metrics = self._evaluate_transformation_accuracy(
                transformation, gt_transformation
            )
            metrics.update(gt_metrics)

        # 生成评估报告
        if self.debug_mode and self.output_dir:
            self._generate_report(metrics, source_transformed, target_pcd, correspondences)

        return metrics

    def _evaluate_with_correspondences(self,
                                       source_pcd: o3d.geometry.PointCloud,
                                       target_pcd: o3d.geometry.PointCloud,
                                       transformation: np.ndarray,
                                       correspondences: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """评估有对应关系的配准"""
        source_pts, target_pts = correspondences

        # 确保是numpy数组
        if isinstance(source_pts, o3d.geometry.PointCloud):
            source_pts = np.asarray(source_pts.points)
        if isinstance(target_pts, o3d.geometry.PointCloud):
            target_pts = np.asarray(target_pts.points)

        # 应用变换到源点
        source_pts_homo = np.hstack([source_pts, np.ones((len(source_pts), 1))])
        source_pts_transformed = (transformation @ source_pts_homo.T).T[:, :3]

        # 计算对应点误差
        errors = np.linalg.norm(source_pts_transformed - target_pts, axis=1)

        metrics = {
            'num_correspondences': len(errors),
            'mean_correspondence_error': float(np.mean(errors)),
            'median_correspondence_error': float(np.median(errors)),
            'std_correspondence_error': float(np.std(errors)),
            'min_correspondence_error': float(np.min(errors)),
            'max_correspondence_error': float(np.max(errors)),
            'rmse_correspondence': float(np.sqrt(np.mean(errors ** 2))),
            'errors_percentiles': {
                '25th': float(np.percentile(errors, 25)),
                '50th': float(np.percentile(errors, 50)),
                '75th': float(np.percentile(errors, 75)),
                '90th': float(np.percentile(errors, 90)),
                '95th': float(np.percentile(errors, 95)),
            }
        }

        # 计算不同阈值下的成功率
        for threshold in [1.0, 2.0, 3.0, 5.0]:
            success_rate = np.sum(errors < threshold) / len(errors)
            metrics[f'success_rate_{threshold}mm'] = float(success_rate)

        # 保存误差分布用于可视化
        self._error_distribution = errors

        logger.info(f"对应点数量: {len(errors)}")
        logger.info(f"平均误差: {metrics['mean_correspondence_error']:.3f} mm")
        logger.info(f"中位数误差: {metrics['median_correspondence_error']:.3f} mm")
        logger.info(f"RMSE: {metrics['rmse_correspondence']:.3f} mm")

        return metrics

    def _evaluate_dense_registration(self,
                                     source_transformed: o3d.geometry.PointCloud,
                                     target_pcd: o3d.geometry.PointCloud) -> Dict:
        """评估密集点云配准 (无对应关系)"""

        source_pts = np.asarray(source_transformed.points)
        target_pts = np.asarray(target_pcd.points)

        # 1. Chamfer Distance (双向最近邻距离)
        chamfer_dist = self._compute_chamfer_distance(source_pts, target_pts)

        # 2. Hausdorff Distance
        hausdorff_dist = self._compute_hausdorff_distance(source_pts, target_pts)

        # 3. Overlap Ratio (不同阈值)
        overlap_metrics = {}
        for threshold in [1.0, 2.0, 3.0, 5.0]:
            overlap_ratio = self._compute_overlap_ratio(source_pts, target_pts, threshold)
            overlap_metrics[f'overlap_ratio_{threshold}mm'] = float(overlap_ratio)

        # 4. Fitness Score (Open3D ICP风格)
        fitness, inlier_rmse = self._compute_fitness_score(
            source_transformed, target_pcd, threshold=2.0
        )

        metrics = {
            'chamfer_distance': float(chamfer_dist),
            'hausdorff_distance': float(hausdorff_dist),
            'fitness_score': float(fitness),
            'inlier_rmse': float(inlier_rmse),
            'num_source_points': len(source_pts),
            'num_target_points': len(target_pts),
            **overlap_metrics
        }

        logger.info(f"Chamfer Distance: {chamfer_dist:.3f} mm")
        logger.info(f"Hausdorff Distance: {hausdorff_dist:.3f} mm")
        logger.info(f"Fitness Score: {fitness:.3f}")
        logger.info(f"Inlier RMSE: {inlier_rmse:.3f} mm")

        return metrics

    def _compute_chamfer_distance(self, source_pts: np.ndarray, target_pts: np.ndarray) -> float:
        """计算Chamfer距离 (双向最近邻平均距离)"""
        # 构建KD树
        target_tree = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pts))
        )
        source_tree = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pts))
        )

        # Source to Target
        s2t_distances = []
        for pt in source_pts:
            k, idx, dist = target_tree.search_knn_vector_3d(pt, 1)
            s2t_distances.append(np.sqrt(dist[0]))

        # Target to Source
        t2s_distances = []
        for pt in target_pts:
            k, idx, dist = source_tree.search_knn_vector_3d(pt, 1)
            t2s_distances.append(np.sqrt(dist[0]))

        # Chamfer距离是双向平均
        chamfer = (np.mean(s2t_distances) + np.mean(t2s_distances)) / 2.0
        return chamfer

    def _compute_hausdorff_distance(self, source_pts: np.ndarray, target_pts: np.ndarray) -> float:
        """计算Hausdorff距离 (双向最大最近邻距离)"""
        target_tree = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pts))
        )
        source_tree = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pts))
        )

        # Source to Target max
        s2t_distances = []
        for pt in source_pts:
            k, idx, dist = target_tree.search_knn_vector_3d(pt, 1)
            s2t_distances.append(np.sqrt(dist[0]))

        # Target to Source max
        t2s_distances = []
        for pt in target_pts:
            k, idx, dist = source_tree.search_knn_vector_3d(pt, 1)
            t2s_distances.append(np.sqrt(dist[0]))

        hausdorff = max(np.max(s2t_distances), np.max(t2s_distances))
        return hausdorff

    def _compute_overlap_ratio(self, source_pts: np.ndarray, target_pts: np.ndarray, threshold: float) -> float:
        """计算重叠率 (在阈值内的点的比例)"""
        target_tree = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pts))
        )

        inlier_count = 0
        for pt in source_pts:
            k, idx, dist = target_tree.search_knn_vector_3d(pt, 1)
            if np.sqrt(dist[0]) < threshold:
                inlier_count += 1

        overlap_ratio = inlier_count / len(source_pts)
        return overlap_ratio

    def _compute_fitness_score(self, source_pcd: o3d.geometry.PointCloud,
                               target_pcd: o3d.geometry.PointCloud,
                               threshold: float = 2.0) -> Tuple[float, float]:
        """计算Fitness Score和Inlier RMSE"""
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source_pcd, target_pcd, threshold, np.eye(4)
        )
        return evaluation.fitness, evaluation.inlier_rmse

    def _evaluate_transformation_accuracy(self,
                                         pred_transform: np.ndarray,
                                         gt_transform: np.ndarray) -> Dict:
        """评估变换矩阵精度 (相对于ground truth)"""
        # 提取旋转和平移
        pred_R = pred_transform[:3, :3]
        pred_t = pred_transform[:3, 3]
        gt_R = gt_transform[:3, :3]
        gt_t = gt_transform[:3, 3]

        # 平移误差
        translation_error = np.linalg.norm(pred_t - gt_t)

        # 旋转误差 (角度)
        R_diff = pred_R @ gt_R.T
        trace = np.trace(R_diff)
        # 确保trace在有效范围内 [-1, 3]
        trace = np.clip(trace, -1, 3)
        rotation_error_rad = np.arccos((trace - 1) / 2)
        rotation_error_deg = np.degrees(rotation_error_rad)

        metrics = {
            'translation_error_mm': float(translation_error),
            'rotation_error_deg': float(rotation_error_deg),
            'rotation_error_rad': float(rotation_error_rad),
        }

        logger.info(f"Translation Error: {translation_error:.3f} mm")
        logger.info(f"Rotation Error: {rotation_error_deg:.3f} degrees")

        return metrics

    def _generate_report(self, metrics: Dict,
                        source_transformed: o3d.geometry.PointCloud,
                        target_pcd: o3d.geometry.PointCloud,
                        correspondences: Optional[Tuple] = None):
        """生成评估报告和可视化"""
        if not self.output_dir:
            return

        # 1. 保存JSON报告
        report_path = self.eval_dir / f"registration_evaluation_{metrics['method_name']}.json"
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"评估报告已保存: {report_path}")

        # 2. 可视化误差分布 (如果有对应关系)
        if hasattr(self, '_error_distribution'):
            self._plot_error_distribution(metrics['method_name'])

        # 3. 可视化配准结果
        self._visualize_registration_result(
            source_transformed, target_pcd, metrics['method_name']
        )

        # 4. 如果有对应关系，可视化对应点连线
        if correspondences is not None:
            self._visualize_correspondences(
                source_transformed, target_pcd, correspondences, metrics['method_name']
            )

    def _plot_error_distribution(self, method_name: str):
        """绘制误差分布直方图和箱型图"""
        errors = self._error_distribution

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 直方图
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}mm')
        axes[0].axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.2f}mm')
        axes[0].set_xlabel('Correspondence Error (mm)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Error Distribution - {method_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 箱型图
        axes[1].boxplot(errors, vert=True)
        axes[1].set_ylabel('Correspondence Error (mm)')
        axes[1].set_title(f'Error Box Plot - {method_name}')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.eval_dir / f"error_distribution_{method_name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"误差分布图已保存: {save_path}")

    def _visualize_registration_result(self,
                                       source_transformed: o3d.geometry.PointCloud,
                                       target_pcd: o3d.geometry.PointCloud,
                                       method_name: str):
        """可视化配准结果 (点云叠加)"""
        # 为点云上色
        source_colored = o3d.geometry.PointCloud(source_transformed)
        target_colored = o3d.geometry.PointCloud(target_pcd)

        source_colored.paint_uniform_color([1, 0, 0])  # 红色 - 源点云
        target_colored.paint_uniform_color([0, 1, 0])  # 绿色 - 目标点云

        # 保存可视化图像
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(source_colored)
        vis.add_geometry(target_colored)

        # 添加坐标轴
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        vis.poll_events()
        vis.update_renderer()

        img_path = self.eval_dir / f"registration_result_{method_name}.png"
        vis.capture_screen_image(str(img_path))
        vis.destroy_window()
        logger.info(f"配准结果可视化已保存: {img_path}")

    def _visualize_correspondences(self,
                                  source_transformed: o3d.geometry.PointCloud,
                                  target_pcd: o3d.geometry.PointCloud,
                                  correspondences: Tuple,
                                  method_name: str):
        """可视化对应点连线"""
        source_pts, target_pts = correspondences

        if isinstance(source_pts, o3d.geometry.PointCloud):
            source_pts = np.asarray(source_pts.points)
        if isinstance(target_pts, o3d.geometry.PointCloud):
            target_pts = np.asarray(target_pts.points)

        # 创建连线
        lines = [[i, i] for i in range(len(source_pts))]

        # 创建LineSet
        line_set = o3d.geometry.LineSet()

        # 合并点
        all_points = np.vstack([source_pts, target_pts])
        line_set.points = o3d.utility.Vector3dVector(all_points)

        # 创建连线索引
        lines = [[i, i + len(source_pts)] for i in range(len(source_pts))]
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # 根据误差着色
        if hasattr(self, '_error_distribution'):
            errors = self._error_distribution
            # 归一化误差到[0, 1]
            normalized_errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
            # 使用颜色映射 (蓝色=小误差, 红色=大误差)
            colors = np.zeros((len(errors), 3))
            colors[:, 0] = normalized_errors  # Red channel
            colors[:, 2] = 1 - normalized_errors  # Blue channel
            line_set.colors = o3d.utility.Vector3dVector(colors)

        # 可视化
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)

        # 添加点云
        source_colored = o3d.geometry.PointCloud()
        source_colored.points = o3d.utility.Vector3dVector(source_pts)
        source_colored.paint_uniform_color([1, 0, 0])

        target_colored = o3d.geometry.PointCloud()
        target_colored.points = o3d.utility.Vector3dVector(target_pts)
        target_colored.paint_uniform_color([0, 1, 0])

        vis.add_geometry(source_colored)
        vis.add_geometry(target_colored)
        vis.add_geometry(line_set)

        vis.poll_events()
        vis.update_renderer()

        img_path = self.eval_dir / f"correspondences_{method_name}.png"
        vis.capture_screen_image(str(img_path))
        vis.destroy_window()
        logger.info(f"对应点连线可视化已保存: {img_path}")

    def compare_methods(self, results_list: List[Dict], output_name: str = "methods_comparison"):
        """
        比较多个方法的评估结果

        Args:
            results_list: 多个evaluate()返回的结果列表
            output_name: 输出文件名
        """
        if not results_list:
            logger.warning("没有结果可比较")
            return

        # 创建比较表格
        comparison = {
            'methods': [],
            'metrics': {}
        }

        for result in results_list:
            comparison['methods'].append(result['method_name'])

        # 提取所有共有的指标
        common_metrics = set(results_list[0].keys())
        for result in results_list[1:]:
            common_metrics &= set(result.keys())

        # 排除非数值指标
        exclude_keys = ['method_name', 'transformation', 'errors_percentiles']

        for metric in common_metrics:
            if metric not in exclude_keys and isinstance(results_list[0].get(metric), (int, float)):
                comparison['metrics'][metric] = [result.get(metric, None) for result in results_list]

        # 保存比较结果
        if self.output_dir:
            comparison_path = self.eval_dir / f"{output_name}.json"
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"方法比较结果已保存: {comparison_path}")

            # 可视化比较
            self._plot_method_comparison(comparison, output_name)

        return comparison

    def _plot_method_comparison(self, comparison: Dict, output_name: str):
        """绘制方法比较图"""
        methods = comparison['methods']
        metrics = comparison['metrics']

        # 选择几个关键指标进行可视化
        key_metrics = []
        for metric_name in ['mean_correspondence_error', 'median_correspondence_error',
                           'rmse_correspondence', 'chamfer_distance', 'fitness_score']:
            if metric_name in metrics:
                key_metrics.append(metric_name)

        if not key_metrics:
            logger.warning("没有可比较的关键指标")
            return

        # 创建柱状图
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(methods))
        width = 0.8 / len(key_metrics)

        for i, metric in enumerate(key_metrics):
            values = metrics[metric]
            offset = (i - len(key_metrics)/2) * width + width/2
            ax.bar(x + offset, values, width, label=metric)

        ax.set_xlabel('Methods')
        ax.set_ylabel('Metric Value')
        ax.set_title('Registration Methods Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.eval_dir / f"{output_name}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"方法比较图已保存: {save_path}")


def quick_evaluate(source_pcd: o3d.geometry.PointCloud,
                   target_pcd: o3d.geometry.PointCloud,
                   transformation: np.ndarray,
                   correspondences: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   method_name: str = "Unknown",
                   debug_mode: bool = False,
                   output_dir: Optional[Path] = None) -> Dict:
    """
    快速评估接口 (便捷函数)

    Args:
        source_pcd: 源点云
        target_pcd: 目标点云
        transformation: 变换矩阵
        correspondences: 对应点对 (可选)
        method_name: 方法名称
        debug_mode: 是否输出调试信息
        output_dir: 输出目录

    Returns:
        评估指标字典
    """
    evaluator = RegistrationEvaluator(debug_mode=debug_mode, output_dir=output_dir)
    return evaluator.evaluate(source_pcd, target_pcd, transformation,
                             correspondences, method_name)
