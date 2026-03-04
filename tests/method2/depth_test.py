#!/usr/bin/env python3
import numpy as np
import cv2
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt 

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========= 配置参数 =========
PATH_DEPTH1 = r'1_depth.png'
PATH_DEPTH2 = r'2_depth.png'
OUT_DIFF_PNG = r'diff_visual.png'
OUT_REGION_ANALYSIS = r'region_analysis.png'

TOL = 1e-6

# ========= 全局变量用于交互式选择 =========
drawing = False
ix, iy = -1, -1
regions = []
current_region = []
img_display = None

# ========= 鼠标回调函数 =========
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, current_region, img_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_region = [x, y, 0, 0]
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_temp = img_display.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('深度差异图 - 框选区域 (按空格确认,按q退出)', img_temp)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        w, h = x2 - x1, y2 - y1
        
        if w > 10 and h > 10:
            regions.append([x1, y1, w, h])
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_display, f'Region{len(regions)}', (x1, y1-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('深度差异图 - 框选区域 (按空格确认,按q退出)', img_display)
            print(f'✅ 已选择区域 {len(regions)}: [{x1}, {y1}, {w}, {h}]')

# ========= 加载深度图 =========
def load_raw(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    im = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f'无法读取: {path}')
    return im.astype(np.float32)

# ========= 改进的可视化函数 =========
def create_better_visualization(diff: np.ndarray) -> np.ndarray:
    """创建可视化效果，相同深度图显示全蓝色"""
    
    # 检查是否几乎完全相同（在容差范围内）
    max_diff = np.nanmax(diff)
    
    if max_diff < TOL:  # 差异小于容差，认为是相同的
        print("✅ 两张深度图完全相同，显示全蓝色")
        # 创建全蓝色图像
        h, w = diff.shape
        blue_image = np.zeros((h, w, 3), dtype=np.uint8)
        blue_image[:, :, 0] = 255  # OpenCV是BGR格式，所以蓝色通道设为255
        return blue_image
    else:
        # 正常处理有差异的情况
        print(f"📊 检测到差异，最大差异: {max_diff:.6f}")
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        return colored

# ========= 区域分析函数 =========
def analyze_regions(diff: np.ndarray, regions: List[List[int]]) -> Dict:
    """分析用户框选区域的差异统计"""
    results = {}
    
    for i, region in enumerate(regions):
        x, y, w, h = region
        region_diff = diff[y:y+h, x:x+w]
        valid_mask = ~np.isnan(region_diff)
        
        if np.any(valid_mask):
            region_valid = region_diff[valid_mask]
            results[f'region_{i+1}'] = {
                'bbox': region,
                'max_diff': np.max(region_valid),
                'mean_diff': np.mean(region_valid),
                'median_diff': np.median(region_valid),
                'std_diff': np.std(region_valid),
                'pixel_count': len(region_valid),
                'diff_pixels': np.sum(region_valid > TOL),
                'diff_percentage': (np.sum(region_valid > TOL) / len(region_valid) * 100) if len(region_valid) > 0 else 0
            }
        else:
            results[f'region_{i+1}'] = {
                'bbox': region,
                'max_diff': 0.0,
                'mean_diff': 0.0,
                'median_diff': 0.0,
                'std_diff': 0.0,
                'pixel_count': 0,
                'diff_pixels': 0,
                'diff_percentage': 0.0,
                'note': '区域全为NaN或零差异'
            }
    
    if regions:
        mask = np.ones(diff.shape, dtype=bool)
        for region in regions:
            x, y, w, h = region
            mask[y:y+h, x:x+w] = False
        
        outside_diff = diff[mask]
        outside_valid = outside_diff[~np.isnan(outside_diff)]
        
        if len(outside_valid) > 0:
            results['outside_regions'] = {
                'max_diff': np.max(outside_valid),
                'mean_diff': np.mean(outside_valid),
                'median_diff': np.median(outside_valid),
                'std_diff': np.std(outside_valid),
                'pixel_count': len(outside_valid),
                'diff_pixels': np.sum(outside_valid > TOL),
                'diff_percentage': (np.sum(outside_valid > TOL) / len(outside_valid) * 100)
            }
        else:
            results['outside_regions'] = {
                'max_diff': 0.0,
                'mean_diff': 0.0,
                'median_diff': 0.0,
                'std_diff': 0.0,
                'pixel_count': 0,
                'diff_pixels': 0,
                'diff_percentage': 0.0,
                'note': '框外区域全为NaN或零差异'
            }
    
    return results

# ========= 检查深度图是否相同 =========
def check_if_identical(d1: np.ndarray, d2: np.ndarray, tol: float = 1e-6) -> bool:
    """检查两张深度图是否完全相同"""
    if d1.shape != d2.shape:
        return False
    
    diff = np.abs(d1 - d2)
    max_diff = np.nanmax(diff)
    
    # 如果最大差异小于容差，认为是相同的
    if max_diff < tol:
        return True
    
    # 额外检查：如果差异很小且分布均匀，也可能是相同的
    mean_diff = np.nanmean(diff)
    if mean_diff < tol * 0.1:  # 平均差异更严格
        return True
        
    return False

# ========= 深度直方图 =========
def show_histogram(region_data, region_idx):
    plt.figure()
    plt.hist(region_data, bins=50, color='blue', alpha=0.7)
    plt.title(f'区域{region_idx+1} 深度直方图')
    plt.xlabel('深度值')
    plt.ylabel('像素数量')
    plt.grid(True)
    plt.show()

# ========= 主程序 =========
def main():
    global img_display, regions
    
    try:
        # 检查文件路径
        same_file = os.path.abspath(PATH_DEPTH1) == os.path.abspath(PATH_DEPTH2)
        if same_file:
            print("📝 注意：比较的是同一个文件")
        
        # 加载深度图
        print("📁 加载深度图中...")
        d1 = load_raw(PATH_DEPTH1)
        d2 = load_raw(PATH_DEPTH2)
        
        if d1.shape != d2.shape:
            print(f'❌ 形状不同: {d1.shape} vs {d2.shape}')
            return
        
        # 检查是否相同
        identical = check_if_identical(d1, d2, TOL)
        
        # 计算差异
        diff = np.abs(d1 - d2)
        
        # 打印基本信息
        print(f"📊 深度图1范围: {np.nanmin(d1):.3f} ~ {np.nanmax(d1):.3f}")
        print(f"📊 深度图2范围: {np.nanmin(d2):.3f} ~ {np.nanmax(d2):.3f}")
        print(f"📊 差异范围: {np.nanmin(diff):.6f} ~ {np.nanmax(diff):.6f}")
        
        if identical:
            print("🎯 结论：两张深度图完全相同")
        else:
            print("🎯 结论：两张深度图存在差异")
        
        # 创建可视化
        img_display = create_better_visualization(diff)
        
        # 交互式区域选择（即使相同也允许框选分析）
        print("\n🎯 交互式区域选择模式")
        print("=" * 50)
        print("使用说明:")
        print("1. 在图像上按住鼠标左键拖动来框选区域")
        print("2. 松开鼠标左键确认当前区域") 
        print("3. 按空格键完成所有区域选择")
        print("4. 按 'q' 键退出选择")
        print("5. 按 'r' 键重置所有区域")
        print("=" * 50)
        
        cv2.namedWindow('深度差异图 - 框选区域 (按空格确认,按q退出)')
        cv2.setMouseCallback('深度差异图 - 框选区域 (按空格确认,按q退出)', draw_rectangle)
        
        while True:
            cv2.imshow('深度差异图 - 框选区域 (按空格确认,按q退出)', img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                break
            elif key == ord('q'):
                print("❌ 用户取消选择")
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                regions = []
                img_display = create_better_visualization(diff)
                print("🔄 已重置所有区域")
        
        cv2.destroyAllWindows()
        
        # 区域分析
        if regions:
            print("\n📊 正在分析区域...")
            results = analyze_regions(diff, regions)
            
            # 保存结果
            cv2.imwrite(OUT_REGION_ANALYSIS, img_display)
            cv2.imwrite(OUT_DIFF_PNG, img_display)
            
            # 输出结果
            print("\n" + "=" * 60)
            print("📈 深度差异区域分析报告")
            print("=" * 60)
            
            valid_diff = diff[~np.isnan(diff)] if len(diff[~np.isnan(diff)]) > 0 else np.array([0.0])
            max_diff = np.max(valid_diff)
            mean_diff = np.mean(valid_diff)
            diff_pixel_ratio = np.sum(valid_diff > TOL) / len(valid_diff) * 100 if len(valid_diff) > 0 else 0

            print(f"\n🌍 全局统计:")
            print(f"   最大差异: {max_diff:.6f} 单位")
            print(f"   平均差异: {mean_diff:.6f} 单位")
            print(f"   差异像素比例: {diff_pixel_ratio:.2f}%")

            # 全局深度直方图（像素深度分布）
            plt.figure()
            plt.hist(d1[~np.isnan(d1)], bins=50, color='green', alpha=0.7, label='深度图1')
            plt.hist(d2[~np.isnan(d2)], bins=50, color='blue', alpha=0.5, label='深度图2')
            plt.title('全局像素深度直方图')
            plt.xlabel('深度值')
            plt.ylabel('像素数量')
            plt.legend()
            plt.grid(True)
            plt.show()

            # 全局差异直方图
            plt.figure()
            plt.hist(valid_diff, bins=50, color='red', alpha=0.7)
            plt.title('全局深度差异直方图')
            plt.xlabel('差异值')
            plt.ylabel('像素数量')
            plt.grid(True)
            plt.show()

            # 自动结论输出（只用平均差异和比例判断）
            THRESH_MEAN = 2.0    # 平均差异阈值（你可以根据实际情况调整）
            THRESH_RATIO = 40.0  # 差异像素比例阈值（百分比）

            if mean_diff < THRESH_MEAN and diff_pixel_ratio < THRESH_RATIO:
                print("\n✅ 结论：两张深度图【基本一致】，大多数像素差异很小。")
            elif mean_diff < THRESH_MEAN * 5 and diff_pixel_ratio < THRESH_RATIO * 1.5:
                print("\n🟡 结论：两张深度图【大致一致】，有部分像素差异。")
            else:
                print("\n❌ 结论：两张深度图【存在明显差异】。")

        print(f"\n💾 分析完成")
        if identical:
            print("✅ 两张深度图完全相同，差异图显示为全蓝色")
        else:
            print("📊 存在差异，已生成差异分析报告")
        
    except Exception as e:
        print(f'❌ 错误: {e}')

if __name__ == '__main__':
    main()