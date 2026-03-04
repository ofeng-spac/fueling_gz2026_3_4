# 点云配准评估模块使用指南

## 概述

`registration_evaluator.py` 提供了统一的点云配准评估接口，支持：
- **有对应关系的配准**（方案2、3、4）：使用对应点误差评估
- **无对应关系的配准**（方案1）：使用Chamfer距离等密集点云评估

## 快速开始

### 1. 在各方案中导入评估模块

在主程序开头添加导入：

```python
from registration_evaluator import RegistrationEvaluator, quick_evaluate
```

### 2. 方案1 (main_flb.py) - 密集点云配准

在配准完成后添加评估代码：

```python
# 在 line ~209 配准完成后
T_Ca2_from_Ca1 = await run_sync(pc_registration.compute_registration, target_pc=target)
logger.info(f"Task {arm_id}: Registration completed.{T_Ca2_from_Ca1}")

# ========== 添加评估代码 ==========
if debug_mode:
    # 评估方案1 (无对应关系)
    eval_metrics = quick_evaluate(
        source_pcd=source_pc,
        target_pcd=target,
        transformation=T_Ca2_from_Ca1,
        correspondences=None,  # 方案1没有对应关系
        method_name="FLB_Dense",
        debug_mode=True,
        output_dir=arm_dir
    )
    logger.info(f"配准评估结果: {eval_metrics}")
```

### 3. 方案2 (main_sparse.py) - 稀疏特征匹配

在配准完成后添加评估代码：

```python
# 在 line ~379 配准完成后
sparse_T_Ca2_from_Ca1 = await run_sync(pc_registration.compute_registration,
                                        target_pc=target_sparse_filtered)
logger.info(f"Task {arm_id}: ICP Registration completed.{sparse_T_Ca2_from_Ca1}")

# ========== 添加评估代码 ==========
if debug_mode:
    # 准备对应点数据
    source_pts = np.asarray(source_sparse_filtered.points)
    target_pts = np.asarray(target_sparse_filtered.points)

    # 评估方案2 (有对应关系)
    eval_metrics = quick_evaluate(
        source_pcd=source_sparse_filtered,
        target_pcd=target_sparse_filtered,
        transformation=sparse_T_Ca2_from_Ca1,
        correspondences=(source_pts, target_pts),  # 稀疏对应点
        method_name="Sparse_MINIMA",
        debug_mode=True,
        output_dir=arm_dir
    )
    logger.info(f"配准评估结果: {eval_metrics}")
```

### 4. 方案3 (main_sparse2.py) - ROI优化稀疏匹配

在配准完成后添加评估代码：

```python
# 在 line ~592 配准完成后
sparse_T_Ca2_from_Ca1 = await run_sync(sparse_registration.compute_registration,
                                        target_pc=target_sparse_filtered)
logger.info(f"Task {arm_id}: ICP Registration completed.{sparse_T_Ca2_from_Ca1}")

# ========== 添加评估代码 ==========
if debug_mode:
    # 准备对应点数据
    source_pts = np.asarray(source_sparse_filtered.points)
    target_pts = np.asarray(target_sparse_filtered.points)

    # 评估方案3 (有对应关系 + ROI优化)
    eval_metrics = quick_evaluate(
        source_pcd=source_sparse_filtered,
        target_pcd=target_sparse_filtered,
        transformation=sparse_T_Ca2_from_Ca1,
        correspondences=(source_pts, target_pts),
        method_name="Sparse_ROI",
        debug_mode=True,
        output_dir=arm_dir
    )
    logger.info(f"配准评估结果: {eval_metrics}")
```

### 5. 方案4 (main_sparse3.py) - 视差图引导稀疏匹配

在配准完成后添加评估代码：

```python
# 在 line ~608 配准完成后
sparse_T_Ca2_from_Ca1 = await run_sync(sparse_registration.compute_registration,
                                        target_pc=target_sparse_filtered)
logger.info(f"Task {arm_id}: ICP Registration completed.{sparse_T_Ca2_from_Ca1}")

# ========== 添加评估代码 ==========
if debug_mode:
    # 准备对应点数据
    source_pts = np.asarray(source_sparse_filtered.points)
    target_pts = np.asarray(target_sparse_filtered.points)

    # 评估方案4 (视差图引导)
    eval_metrics = quick_evaluate(
        source_pcd=source_sparse_filtered,
        target_pcd=target_sparse_filtered,
        transformation=sparse_T_Ca2_from_Ca1,
        correspondences=(source_pts, target_pts),
        method_name="Sparse_Disparity",
        debug_mode=True,
        output_dir=arm_dir
    )
    logger.info(f"配准评估结果: {eval_metrics}")
```

## 高级用法

### 多方法比较

如果你想比较多个方法的性能：

```python
# 在主程序末尾或单独的比较脚本中
from registration_evaluator import RegistrationEvaluator

# 创建评估器
evaluator = RegistrationEvaluator(debug_mode=True, output_dir=Path("../evaluation_results"))

# 运行多个方法并收集结果
results = []

# 评估方案1
result1 = evaluator.evaluate(source1, target1, T1, None, "FLB_Dense")
results.append(result1)

# 评估方案2
result2 = evaluator.evaluate(source2, target2, T2, (src_pts2, tgt_pts2), "Sparse_MINIMA")
results.append(result2)

# 评估方案3
result3 = evaluator.evaluate(source3, target3, T3, (src_pts3, tgt_pts3), "Sparse_ROI")
results.append(result3)

# 评估方案4
result4 = evaluator.evaluate(source4, target4, T4, (src_pts4, tgt_pts4), "Sparse_Disparity")
results.append(result4)

# 生成比较报告
evaluator.compare_methods(results, "all_methods_comparison")
```

## 评估指标说明

### 有对应关系的评估（方案2、3、4）

- `mean_correspondence_error`: 平均对应点误差 (mm)
- `median_correspondence_error`: 中位数对应点误差 (mm)
- `rmse_correspondence`: 对应点RMSE (mm)
- `std_correspondence_error`: 对应点误差标准差 (mm)
- `success_rate_Xmm`: 在X毫米阈值内的成功率
- `errors_percentiles`: 误差百分位数 (25%, 50%, 75%, 90%, 95%)

### 无对应关系的评估（方案1）

- `chamfer_distance`: Chamfer距离 (mm) - 双向最近邻平均距离
- `hausdorff_distance`: Hausdorff距离 (mm) - 双向最大最近邻距离
- `fitness_score`: 配准适应度分数 (0-1)
- `inlier_rmse`: 内点RMSE (mm)
- `overlap_ratio_Xmm`: 在X毫米阈值内的重叠率

### 变换精度评估（如果有ground truth）

- `translation_error_mm`: 平移误差 (mm)
- `rotation_error_deg`: 旋转误差 (度)

## 输出文件

评估模块会在 `arm_X/evaluation/` 目录下生成以下文件：

```
arm_X/evaluation/
├── registration_evaluation_{method_name}.json    # JSON格式的详细评估报告
├── error_distribution_{method_name}.png         # 误差分布图 (有对应关系时)
├── registration_result_{method_name}.png        # 配准结果可视化
├── correspondences_{method_name}.png            # 对应点连线可视化 (有对应关系时)
└── all_methods_comparison.json                  # 多方法比较报告
└── all_methods_comparison.png                   # 多方法比较图
```

## 完整示例

参考 `example_evaluation.py` 查看完整的评估示例代码。

## 注意事项

1. **确保点云对齐**: 评估前确保源点云和目标点云已经在同一坐标系下
2. **对应关系格式**: 对应点可以是 numpy 数组或 Open3D PointCloud 对象
3. **调试模式**: 只在 `debug_mode=True` 时生成可视化和报告文件
4. **输出目录**: 建议使用与主程序相同的 `arm_dir` 作为输出目录

## 性能建议

- 对于大规模点云（>100k点），Chamfer距离计算可能较慢
- 可视化生成需要额外时间，生产环境可以关闭 `debug_mode`
- 使用 `quick_evaluate` 函数进行简单快速评估
