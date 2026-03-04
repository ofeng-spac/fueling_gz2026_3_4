#!/usr/bin/env python3
"""
统计分析脚本 - 分析多次运行的点云配准评估结果

使用方法:
    python analyze_results.py                          # 分析所有evaluation文件
    python analyze_results.py --method Sparse_ROI      # 只分析特定方案
    python analyze_results.py --latest 10              # 只分析最近10次结果
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

class ResultAnalyzer:
    def __init__(self, working_data_dir: str):
        self.working_data = Path(working_data_dir)
        self.results_by_method = defaultdict(list)

    def collect_results(self, method_filter: Optional[str] = None, latest_n: Optional[int] = None):
        """收集所有evaluation结果"""
        print("正在收集评估结果...")

        # 查找所有evaluation JSON文件
        eval_pattern = "*/arm_1/evaluation/registration_evaluation_*.json"
        eval_files = list(self.working_data.glob(eval_pattern))

        # 按修改时间排序
        eval_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # 如果指定了latest_n，只取最新的N个
        if latest_n:
            eval_files = eval_files[:latest_n]

        print(f"找到 {len(eval_files)} 个评估文件")

        # 按方案分组
        for eval_file in eval_files:
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                method_name = data.get('method_name')
                if method_filter and method_name != method_filter:
                    continue

                # 添加文件信息
                data['_file'] = str(eval_file)
                data['_timestamp'] = eval_file.stat().st_mtime

                self.results_by_method[method_name].append(data)

            except Exception as e:
                print(f"读取文件 {eval_file} 时出错: {e}")

        # 按时间排序每个方案的结果
        for method in self.results_by_method:
            self.results_by_method[method].sort(key=lambda x: x['_timestamp'], reverse=True)

        print(f"\n收集完成! 各方案结果数量:")
        for method, results in self.results_by_method.items():
            print(f"  {method}: {len(results)} 个")

    def analyze_method(self, method_name: str, results: List[Dict]) -> Dict:
        """分析单个方案的统计信息"""
        if not results:
            return {}

        # 判断方案类型（Dense还是Sparse）
        is_dense = method_name == "FLB_Dense"

        stats = {
            "method_name": method_name,
            "num_runs": len(results),
            "timestamps": [
                datetime.fromtimestamp(r['_timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                for r in results
            ]
        }

        if is_dense:
            # Dense方案的指标
            metrics = {
                "chamfer_distance": [],
                "hausdorff_distance": [],
                "fitness_score": [],
                "inlier_rmse": [],
                "overlap_ratio_1.0mm": [],
                "overlap_ratio_2.0mm": [],
                "overlap_ratio_3.0mm": [],
                "overlap_ratio_5.0mm": []
            }

            for result in results:
                for key in metrics.keys():
                    if key in result:
                        metrics[key].append(result[key])

            # 计算统计量
            for key, values in metrics.items():
                if values:
                    stats[key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values)),
                        "values": values
                    }

        else:
            # Sparse方案的指标
            metrics = {
                "num_correspondences": [],
                "mean_correspondence_error": [],
                "median_correspondence_error": [],
                "std_correspondence_error": [],
                "rmse_correspondence": [],
                "max_correspondence_error": []
            }

            percentiles = {
                "25th": [],
                "50th": [],
                "75th": [],
                "90th": [],
                "95th": []
            }

            for result in results:
                for key in metrics.keys():
                    if key in result:
                        metrics[key].append(result[key])

                # 收集百分位数据
                if 'errors_percentiles' in result:
                    for pkey in percentiles.keys():
                        if pkey in result['errors_percentiles']:
                            percentiles[pkey].append(result['errors_percentiles'][pkey])

            # 计算统计量
            for key, values in metrics.items():
                if values:
                    stats[key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values)),
                        "values": values
                    }

            # 计算百分位统计
            stats["percentiles"] = {}
            for pkey, values in percentiles.items():
                if values:
                    stats["percentiles"][pkey] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "values": values
                    }

        return stats

    def print_analysis(self):
        """打印分析结果"""
        print("\n" + "="*100)
        print("点云配准方案统计分析报告")
        print("="*100)

        for method_name, results in self.results_by_method.items():
            stats = self.analyze_method(method_name, results)

            print(f"\n{'='*100}")
            print(f"方案: {method_name}")
            print(f"运行次数: {stats['num_runs']}")
            print(f"{'='*100}")

            if method_name == "FLB_Dense":
                self._print_dense_stats(stats)
            else:
                self._print_sparse_stats(stats)

        # 打印对比表格
        self._print_comparison_table()

    def _print_dense_stats(self, stats: Dict):
        """打印Dense方案统计"""
        print("\n核心指标:")
        print(f"{'指标':<30} {'平均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
        print("-" * 80)

        key_metrics = [
            ("chamfer_distance", "Chamfer距离 (mm)"),
            ("inlier_rmse", "Inlier RMSE (mm)"),
            ("fitness_score", "Fitness Score"),
            ("overlap_ratio_2.0mm", "重叠率@2mm"),
            ("overlap_ratio_5.0mm", "重叠率@5mm")
        ]

        for key, label in key_metrics:
            if key in stats:
                data = stats[key]
                print(f"{label:<30} {data['mean']:>11.4f} {data['std']:>11.4f} "
                      f"{data['min']:>11.4f} {data['max']:>11.4f}")

        # 稳定性分析
        if 'inlier_rmse' in stats:
            rmse_cv = stats['inlier_rmse']['std'] / stats['inlier_rmse']['mean']
            print(f"\n稳定性指标 (变异系数 CV):")
            print(f"  Inlier RMSE CV: {rmse_cv:.2%}")

    def _print_sparse_stats(self, stats: Dict):
        """打印Sparse方案统计"""
        print("\n核心指标:")
        print(f"{'指标':<35} {'平均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
        print("-" * 85)

        key_metrics = [
            ("num_correspondences", "对应点数量"),
            ("mean_correspondence_error", "平均对应误差 (mm)"),
            ("median_correspondence_error", "中位对应误差 (mm)"),
            ("rmse_correspondence", "RMSE (mm)"),
            ("std_correspondence_error", "误差标准差 (mm)"),
            ("max_correspondence_error", "最大误差 (mm)")
        ]

        for key, label in key_metrics:
            if key in stats:
                data = stats[key]
                print(f"{label:<35} {data['mean']:>11.4f} {data['std']:>11.4f} "
                      f"{data['min']:>11.4f} {data['max']:>11.4f}")

        # 百分位统计
        if 'percentiles' in stats:
            print("\n误差百分位:")
            print(f"{'百分位':<20} {'平均值':<12} {'标准差':<12}")
            print("-" * 45)
            for pkey in ['25th', '50th', '75th', '90th', '95th']:
                if pkey in stats['percentiles']:
                    data = stats['percentiles'][pkey]
                    print(f"{pkey:<20} {data['mean']:>11.4f} {data['std']:>11.4f}")

        # 稳定性分析
        if 'mean_correspondence_error' in stats:
            error_cv = stats['mean_correspondence_error']['std'] / stats['mean_correspondence_error']['mean']
            print(f"\n稳定性指标 (变异系数 CV):")
            print(f"  平均误差 CV: {error_cv:.2%}")

        if 'num_correspondences' in stats:
            corr_cv = stats['num_correspondences']['std'] / stats['num_correspondences']['mean']
            print(f"  对应点数量 CV: {corr_cv:.2%}")

    def _print_comparison_table(self):
        """打印方案对比表"""
        print("\n" + "="*100)
        print("方案对比表 (平均值)")
        print("="*100)

        # 提取所有方案的关键指标
        sparse_methods = {k: v for k, v in self.results_by_method.items() if k != "FLB_Dense"}
        dense_methods = {k: v for k, v in self.results_by_method.items() if k == "FLB_Dense"}

        if sparse_methods:
            print("\n稀疏方案对比:")
            print(f"{'方案':<20} {'运行次数':<10} {'平均误差(mm)':<15} {'RMSE(mm)':<12} "
                  f"{'对应点数':<12} {'稳定性(CV)':<12}")
            print("-" * 95)

            for method_name, results in sparse_methods.items():
                stats = self.analyze_method(method_name, results)
                mean_error = stats.get('mean_correspondence_error', {}).get('mean', 0)
                rmse = stats.get('rmse_correspondence', {}).get('mean', 0)
                num_corr = stats.get('num_correspondences', {}).get('mean', 0)
                cv = (stats.get('mean_correspondence_error', {}).get('std', 0) /
                     stats.get('mean_correspondence_error', {}).get('mean', 1))

                print(f"{method_name:<20} {stats['num_runs']:<10} {mean_error:<15.4f} {rmse:<12.4f} "
                      f"{num_corr:<12.0f} {cv:<12.2%}")

        if dense_methods:
            print("\n密集方案:")
            print(f"{'方案':<20} {'运行次数':<10} {'Chamfer(mm)':<15} {'RMSE(mm)':<12} "
                  f"{'Fitness':<12}")
            print("-" * 70)

            for method_name, results in dense_methods.items():
                stats = self.analyze_method(method_name, results)
                chamfer = stats.get('chamfer_distance', {}).get('mean', 0)
                rmse = stats.get('inlier_rmse', {}).get('mean', 0)
                fitness = stats.get('fitness_score', {}).get('mean', 0)

                print(f"{method_name:<20} {stats['num_runs']:<10} {chamfer:<15.4f} {rmse:<12.4f} "
                      f"{fitness:<12.4f}")

    def save_analysis(self, output_dir: str):
        """保存分析结果到JSON文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"analysis_report_{timestamp}.json"

        analysis_data = {}
        for method_name, results in self.results_by_method.items():
            analysis_data[method_name] = self.analyze_method(method_name, results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)

        print(f"\n📊 分析报告已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="分析多次运行的点云配准评估结果",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--working-data', type=str,
                       default='../working_data',
                       help='working_data目录路径 (默认: ../working_data)')
    parser.add_argument('--method', type=str, choices=['FLB_Dense', 'Sparse_MINIMA', 'Sparse_ROI', 'Sparse_Disparity'],
                       help='只分析指定方案')
    parser.add_argument('--latest', type=int,
                       help='只分析最近N次结果')
    parser.add_argument('--save', type=str,
                       help='保存分析报告到指定目录')

    args = parser.parse_args()

    # 获取working_data路径
    script_dir = Path(__file__).parent
    working_data_path = (script_dir / args.working_data).resolve()

    if not working_data_path.exists():
        print(f"错误: working_data目录不存在: {working_data_path}")
        sys.exit(1)

    # 创建分析器
    analyzer = ResultAnalyzer(str(working_data_path))

    # 收集结果
    analyzer.collect_results(method_filter=args.method, latest_n=args.latest)

    # 打印分析
    analyzer.print_analysis()

    # 保存报告
    if args.save:
        analyzer.save_analysis(args.save)
    else:
        # 默认保存到batch_test_results目录
        default_output = script_dir.parent / "batch_test_results"
        analyzer.save_analysis(str(default_output))


if __name__ == '__main__':
    main()
