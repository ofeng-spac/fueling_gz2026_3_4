# 批量测试和统计分析工具使用指南

本文档介绍如何使用批量测试脚本自动运行多次点云配准测试，并分析结果。

## 工具概览

1. **batch_test.py** - 批量测试脚本，自动运行四种方案多次
2. **analyze_results.py** - 统计分析脚本，分析多次运行的结果

## 一、批量测试脚本 (batch_test.py)

### 功能特点

- ✅ 自动依次运行四种方案（FLB_Dense, Sparse_MINIMA, Sparse_ROI, Sparse_Disparity）
- ✅ 可配置每个方案的运行次数
- ✅ 可选择只测试部分方案
- ✅ 自动收集和保存测试结果
- ✅ 支持Ctrl+C中断，已完成的测试结果会被保存
- ✅ 实时显示测试进度和耗时

### 使用方法

#### 1. 测试所有方案，各运行5次（推荐）
```bash
cd /home/vision/projects/fueling_gz/tmp
python batch_test.py --runs 5
```

#### 2. 测试所有方案，各运行10次
```bash
python batch_test.py --runs 10
```

#### 3. 只测试特定方案
```bash
# 只测试Sparse_ROI方案5次
python batch_test.py --runs 5 --methods 3

# 只测试Sparse_MINIMA和Sparse_ROI，各5次
python batch_test.py --runs 5 --methods 2 3

# 测试所有稀疏方案
python batch_test.py --runs 5 --methods 2 3 4
```

**方案编号对照：**（根据 README_METHODS_COMPARISON.md）
- 1: FLB_Dense (方案1: 传统密集点云配准 - main_flb.py)
- 2: Sparse_MINIMA (方案2: 稀疏特征匹配 - main_sparse1.py) ⭐
- 3: Sparse_ROI (方案3: ROI优化稀疏匹配 - main_sparse2.py) ⭐⭐
- 4: Sparse_Disparity (方案4: 视差图引导稀疏匹配 - main_sparse3.py)

#### 4. 查看帮助
```bash
python batch_test.py --help
```

### 输出结果

测试结果会保存在：
```
/home/vision/projects/fueling_gz/batch_test_results/YYYYmmdd_HHMMSS/
├── batch_test_results.json    # 测试运行记录
```

### 测试过程

1. 脚本会按顺序运行每个方案
2. 每次运行之间会暂停5秒让系统稳定
3. 每个方案的输出会实时显示在终端
4. 运行完成后会显示统计摘要
5. 可以随时按Ctrl+C中断，已完成的测试会被保存

### 示例输出

```
================================================================================
批量测试开始
测试方案: Sparse_MINIMA, Sparse_ROI
每个方案运行次数: 5
总测试次数: 10
================================================================================

🚀 开始测试方案: 稀疏MINIMA方案

================================================================================
开始运行: 稀疏MINIMA方案 (Sparse_MINIMA)
第 1/5 次运行
脚本: /home/vision/projects/fueling_gz/tmp/main_sparse1.py
================================================================================

... (测试输出) ...

✅ 测试完成! 耗时: 45.23秒
评估结果: /home/vision/projects/fueling_gz/working_data/.../evaluation/...json

等待5秒后继续下一次测试...

... (重复) ...

================================================================================
批量测试结束
总耗时: 8.5分钟
完成测试数: 10
================================================================================

测试摘要
================================================================================

📊 稀疏MINIMA方案 (Sparse_MINIMA):
   总运行次数: 5
   成功: 5 | 失败: 0
   成功率: 100.0%
   平均耗时: 42.50秒

📊 稀疏ROI方案 (Sparse_ROI):
   总运行次数: 5
   成功: 5 | 失败: 0
   成功率: 100.0%
   平均耗时: 38.20秒
```

---

## 二、统计分析脚本 (analyze_results.py)

### 功能特点

- ✅ 自动收集所有evaluation结果
- ✅ 计算每个方案的统计指标（均值、标准差、最小值、最大值、中位数）
- ✅ 分析稳定性（变异系数CV）
- ✅ 生成方案对比表
- ✅ 保存详细分析报告

### 使用方法

#### 1. 分析所有结果
```bash
cd /home/vision/projects/fueling_gz/tmp
python analyze_results.py
```

#### 2. 只分析特定方案
```bash
# 只分析Sparse_ROI方案
python analyze_results.py --method Sparse_ROI

# 只分析Sparse_MINIMA方案
python analyze_results.py --method Sparse_MINIMA
```

#### 3. 只分析最近N次结果
```bash
# 只分析最近10次结果
python analyze_results.py --latest 10

# 只分析最近20次结果
python analyze_results.py --latest 20
```

#### 4. 组合使用
```bash
# 分析Sparse_ROI方案的最近5次结果
python analyze_results.py --method Sparse_ROI --latest 5
```

#### 5. 指定working_data路径
```bash
python analyze_results.py --working-data /path/to/working_data
```

### 输出结果

分析报告会保存在：
```
/home/vision/projects/fueling_gz/batch_test_results/
└── analysis_report_YYYYmmdd_HHMMSS.json
```

### 示例输出

```
====================================================================================================
点云配准方案统计分析报告
====================================================================================================

====================================================================================================
方案: Sparse_ROI
运行次数: 5
====================================================================================================

核心指标:
指标                                 平均值          标准差          最小值          最大值
-------------------------------------------------------------------------------------
对应点数量                              649.0000       15.8114       630.0000       670.0000
平均对应误差 (mm)                         0.2294        0.0123         0.2150         0.2450
中位对应误差 (mm)                         0.2112        0.0089         0.2010         0.2210
RMSE (mm)                              0.2559        0.0145         0.2390         0.2730
误差标准差 (mm)                           0.1134        0.0067         0.1050         0.1220
最大误差 (mm)                             0.5191        0.0234         0.4890         0.5490

误差百分位:
百分位              平均值          标准差
---------------------------------------------
25th                     0.1392        0.0078
50th                     0.2112        0.0089
75th                     0.3115        0.0156
90th                     0.3921        0.0189
95th                     0.4316        0.0223

稳定性指标 (变异系数 CV):
  平均误差 CV: 5.36%
  对应点数量 CV: 2.44%

====================================================================================================

方案对比表 (平均值)
====================================================================================================

稀疏方案对比:
方案                  运行次数    平均误差(mm)      RMSE(mm)    对应点数      稳定性(CV)
-----------------------------------------------------------------------------------------------
Sparse_MINIMA        5          0.2453          0.2729          929          4.88%
Sparse_ROI           5          0.2294          0.2559          649          5.36%
Sparse_Disparity     5          0.3050          0.3281          177          3.94%

📊 分析报告已保存: /home/vision/projects/fueling_gz/batch_test_results/analysis_report_20251230_120000.json
```

---

## 三、完整工作流程

### 推荐工作流

```bash
# 1. 进入tmp目录
cd /home/vision/projects/fueling_gz/tmp

# 2. 运行批量测试（建议先测试少量次数）
python batch_test.py --runs 3

# 3. 查看实时测试进度（终端会显示每次测试的输出）

# 4. 测试完成后，运行统计分析
python analyze_results.py --latest 12  # 分析刚才的12次测试（4个方案 × 3次）

# 5. 如果结果满意，可以运行更多次数获得更可靠的统计
python batch_test.py --runs 10

# 6. 再次分析所有结果
python analyze_results.py
```

### 针对特定方案的深度测试

如果你想重点测试某个方案（比如表现最好的Sparse_ROI）：

```bash
# 1. 对Sparse_ROI进行20次测试
python batch_test.py --runs 20 --methods 3

# 2. 分析Sparse_ROI的20次结果
python analyze_results.py --method Sparse_ROI --latest 20
```

### 对比测试

如果你想对比两个方案：

```bash
# 1. 测试Sparse_MINIMA和Sparse_ROI各10次
python batch_test.py --runs 10 --methods 2 3

# 2. 分析最近的20次结果
python analyze_results.py --latest 20
```

---

## 四、理解统计指标

### 稀疏方案指标

- **平均对应误差** (mean_correspondence_error): 所有匹配点对之间的平均距离误差，越小越好
- **RMSE**: 均方根误差，考虑了误差的平方，对大误差更敏感
- **对应点数量**: 成功匹配的特征点数量，越多表示匹配越可靠
- **标准差**: 误差的波动程度，越小越稳定
- **变异系数 (CV)**: 标准差/平均值，用于评估稳定性，越小越好
  - CV < 10%: 非常稳定
  - CV 10-20%: 稳定
  - CV > 20%: 不稳定

### 密集方案指标

- **Chamfer距离**: 两个点云之间的平均最近点距离
- **Inlier RMSE**: 内点的均方根误差
- **Fitness Score**: 匹配质量分数，0-1之间，越大越好
- **重叠率**: 在不同阈值下的点云重叠比例

---

## 五、常见问题

### Q1: 测试过程中可以中断吗？
A: 可以！按Ctrl+C即可中断，已完成的测试结果会被保存。

### Q2: 每次测试大约需要多长时间？
A: 根据当前的测试，每次大约30-60秒，具体取决于方案和硬件。

### Q3: 建议运行多少次测试？
A:
- 初步评估: 3-5次
- 可靠统计: 10次
- 深度分析: 20-30次

### Q4: 如何判断哪个方案最好？
A: 综合考虑：
1. **精度**: 平均误差和RMSE越小越好
2. **稳定性**: CV越小越好（说明每次运行结果一致）
3. **鲁棒性**: 对应点数量越多通常越可靠
4. **速度**: 平均耗时越短越好

### Q5: 分析脚本会分析所有历史数据吗？
A: 默认会分析working_data下的所有evaluation文件。可以用`--latest N`只分析最近N次。

### Q6: 可以并行运行多个方案吗？
A: 由于硬件资源限制（相机、机器人），目前只能串行运行。

---

## 六、下一步建议

基于之前的单次测试结果，建议：

1. **优先测试 Sparse_ROI**（表现最好）
   ```bash
   python batch_test.py --runs 10 --methods 3
   ```

2. **对比 Sparse_MINIMA 和 Sparse_ROI**
   ```bash
   python batch_test.py --runs 10 --methods 2 3
   python analyze_results.py --latest 20
   ```

3. **全面评估**（如果时间充足）
   ```bash
   python batch_test.py --runs 5
   python analyze_results.py
   ```

---

## 七、结果解读示例

假设运行了5次测试，分析结果显示：

```
Sparse_ROI:
  平均误差: 0.229mm (std: 0.012mm)
  CV: 5.36%
```

这说明：
- ✅ 平均精度非常高（0.229mm亚毫米级）
- ✅ 标准差很小（0.012mm），说明每次运行结果很接近
- ✅ CV只有5.36%，说明方案非常稳定可靠
- ✅ 可以放心使用这个方案

---

有任何问题欢迎随时咨询！
