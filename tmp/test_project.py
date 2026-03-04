import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Tuple
import json
from fueling.minima.geometry import project_pointcloud_to_image_float
def create_visualization_image(ir_img: np.ndarray) -> np.ndarray:
    """创建用于可视化的彩色图像"""
    if len(ir_img.shape) == 2:
        ir_vis = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
    else:
        ir_vis = ir_img.copy()
    return ir_vis

def draw_projection_points(image: np.ndarray, points: np.ndarray, color=(0, 255, 0), radius=1):
    """在图像上绘制投影点"""
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), radius, color, -1)

def test_pointcloud_projection():
    """
    测试点云投影到左图和右图
    """
    print("=== 点云投影测试开始 ===")
    
    # 加载配置
    config_path = "../data/arm1/config.jsonnet"
    import _jsonnet
    config = json.loads(_jsonnet.evaluate_file(config_path))
    
    # 获取参数
    K_mat = np.array(config['robot']['RS_camera']['K']).reshape(3, 3)
    baseline = config['robot']['RS_camera']['stereo_baseline']
    target_pot = config['robot']['target_pot']
    
    # 加载点云
    data_dir = "../data"
    source_down_path = os.path.join(data_dir, f'arm1', target_pot, 'source_model.pcd')
    source_pc = o3d.io.read_point_cloud(source_down_path)
    
    if not source_pc.has_points():
        print(f"错误: 无法加载点云文件: {source_down_path}")
        return
    
    print(f"加载点云: {len(source_pc.points)} 个点")
    
    # 加载初始图像
    proj_dir = Path(__file__).resolve().parent.parent
    left_ir_path = config['minima']['left_ir_path']
    right_ir_path = config['minima']['right_ir_path']
    
    if not os.path.isabs(left_ir_path):
        left_ir_path = os.path.join(proj_dir, left_ir_path)
    if not os.path.isabs(right_ir_path):
        right_ir_path = os.path.join(proj_dir, right_ir_path)
    
    initial_left = cv2.imread(left_ir_path, cv2.IMREAD_GRAYSCALE)
    initial_right = cv2.imread(right_ir_path, cv2.IMREAD_GRAYSCALE)
    
    if initial_left is None or initial_right is None:
        print(f"错误: 无法加载初始图像: {left_ir_path}, {right_ir_path}")
        return
    
    print(f"加载初始图像: 左图 {initial_left.shape}, 右图 {initial_right.shape}")
    
    # 创建测试输出目录
    test_dir = Path("test_projection")
    test_dir.mkdir(exist_ok=True)
    
    # 测试1: 将源点云投影到初始左图
    print("\n=== 测试1: 源点云投影到初始左图 ===")
    uv_left1, indices_left1, bbox_left1 = project_pointcloud_to_image_float(
        source_pc, initial_left, K_mat, 
        str(test_dir / "source_projected_left.png"), None
    )
    print(f"左图投影点数: {len(uv_left1)}")
    print(f"左图边框: {bbox_left1}")
    
    # 测试2: 将源点云投影到初始右图（使用变换）
    print("\n=== 测试2: 源点云投影到初始右图 ===")
    # 构造从左相机到右相机的变换矩阵
    T_left_to_right = np.eye(4)
    T_left_to_right[0, 3] = -baseline * 1000  # 转换为毫米
    
    uv_right1, indices_right1, bbox_right1 = project_pointcloud_to_image_float(
        source_pc, initial_right, K_mat,
        str(test_dir / "source_projected_right.png"), 
        transform=T_left_to_right,
        other_bbox=bbox_left1
    )
    print(f"右图投影点数: {len(uv_right1)}")
    print(f"右图边框: {bbox_right1}")
    
    # 测试3: 重新计算左图投影以统一边框
    print("\n=== 测试3: 重新计算左图投影（统一边框） ===")
    uv_left2, indices_left2, bbox_left2 = project_pointcloud_to_image_float(
        source_pc, initial_left, K_mat,
        str(test_dir / "source_projected_left_unified.png"), 
        None,
        other_bbox=bbox_right1
    )
    print(f"左图统一后边框: {bbox_left2}")
    
    # 测试4: 加载当前拍摄的图像并测试
    print("\n=== 测试4: 加载当前拍摄的图像 ===")
    # 假设当前图像在working_data中，使用最新的目录
    working_dirs = sorted(Path("../working_data").glob("*/"))
    if working_dirs:
        latest_dir = working_dirs[-1]
        current_left_path = latest_dir / "arm_1" / "ir_images" / "captured_left_ir.png"
        current_right_path = latest_dir / "arm_1" / "ir_images" / "captured_right_ir.png"
        
        if current_left_path.exists() and current_right_path.exists():
            current_left = cv2.imread(str(current_left_path), cv2.IMREAD_GRAYSCALE)
            current_right = cv2.imread(str(current_right_path), cv2.IMREAD_GRAYSCALE)
            
            print(f"加载当前图像: 左图 {current_left.shape}, 右图 {current_right.shape}")
            
            # 测试5: 当前点云投影到当前左图
            print("\n=== 测试5: 当前点云投影到当前左图 ===")
            # 这里需要先获得当前点云，我们可以使用之前生成的
            current_pc_path = latest_dir / "arm_1" / "point_clouds" / "original_target.pcd"
            if current_pc_path.exists():
                current_pc = o3d.io.read_point_cloud(str(current_pc_path))
                if current_pc.has_points():
                    uv_current_left, indices_current_left, bbox_current_left = project_pointcloud_to_image_float(
                        current_pc, current_left, K_mat,
                        str(test_dir / "current_projected_left.png"), None
                    )
                    print(f"当前点云左图投影点数: {len(uv_current_left)}")
                    print(f"当前点云左图边框: {bbox_current_left}")
                    
                    # 测试6: 当前点云投影到当前右图
                    print("\n=== 测试6: 当前点云投影到当前右图 ===")
                    uv_current_right, indices_current_right, bbox_current_right = project_pointcloud_to_image_float(
                        current_pc, current_right, K_mat,
                        str(test_dir / "current_projected_right.png"),
                        transform=T_left_to_right
                    )
                    print(f"当前点云右图投影点数: {len(uv_current_right)}")
                    print(f"当前点云右图边框: {bbox_current_right}")
                else:
                    print("警告: 当前点云文件为空")
            else:
                print("警告: 未找到当前点云文件")
        else:
            print("警告: 未找到当前拍摄的图像")
    
    # 测试7: 分析投影结果
    print("\n=== 测试7: 投影结果分析 ===")
    if bbox_left1 is not None and bbox_right1 is not None:
        left_width = bbox_left1[2] - bbox_left1[0]
        left_height = bbox_left1[3] - bbox_left1[1]
        right_width = bbox_right1[2] - bbox_right1[0]
        right_height = bbox_right1[3] - bbox_right1[1]
        
        print(f"源点云左图边框尺寸: {left_width:.1f} x {left_height:.1f}")
        print(f"源点云右图边框尺寸: {right_width:.1f} x {right_height:.1f}")
        print(f"尺寸差异: width={abs(left_width-right_width):.1f}, height={abs(left_height-right_height):.1f}")
    
    # 测试8: 创建对比图像
    print("\n=== 测试8: 创建对比图像 ===")
    if bbox_left1 is not None and bbox_right1 is not None:
        # 左图带边框
        left_with_bbox = create_visualization_image(initial_left)
        x1, y1, x2, y2 = map(int, bbox_left1)
        cv2.rectangle(left_with_bbox, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(test_dir / "left_with_bbox.png"), left_with_bbox)
        
        # 右图带边框
        right_with_bbox = create_visualization_image(initial_right)
        x1, y1, x2, y2 = map(int, bbox_right1)
        cv2.rectangle(right_with_bbox, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(test_dir / "right_with_bbox.png"), right_with_bbox)
        
        print("已保存带边框的图像")
    
    print("\n=== 测试完成 ===")
    print(f"所有结果保存在: {test_dir.absolute()}")
    
    return {
        'source_left': (uv_left1, bbox_left1),
        'source_right': (uv_right1, bbox_right1),
        'source_left_unified': (uv_left2, bbox_left2)
    }

def debug_transformation():
    """
    调试变换矩阵
    """
    print("\n=== 变换矩阵调试 ===")
    
    # 加载配置
    config_path = "../data/arm1/config.jsonnet"
    import _jsonnet
    config = json.loads(_jsonnet.evaluate_file(config_path))
    
    baseline = config['robot']['RS_camera']['stereo_baseline']
    
    # 构造变换矩阵
    T_left_to_right = np.eye(4)
    T_left_to_right[0, 3] = -baseline * 1000  # 转换为毫米
    
    print(f"基线: {baseline} 米")
    print(f"基线（毫米）: {baseline * 1000} mm")
    print(f"变换矩阵 T_left_to_right:")
    print(T_left_to_right)
    
    # 测试变换效果
    test_points = np.array([
        [0, 0, 1000],  # 距离1米，正前方
        [100, 0, 1000],  # 距离1米，X方向偏移100mm
        [0, 0, 2000],  # 距离2米，正前方
    ])
    
    print(f"\n测试点（左相机坐标系，毫米）:")
    for i, pt in enumerate(test_points):
        print(f"  点{i}: {pt}")
    
    # 变换到右相机坐标系
    pts_hom = np.hstack([test_points, np.ones((len(test_points), 1))])
    pts_right = (T_left_to_right @ pts_hom.T).T[:, :3]
    
    print(f"\n变换后点（右相机坐标系，毫米）:")
    for i, pt in enumerate(pts_right):
        print(f"  点{i}: {pt}")
        print(f"    X方向偏移: {pt[0] - test_points[i][0]} mm")

if __name__ == '__main__':
    # 运行投影测试
    test_pointcloud_projection()
    
    # 运行变换矩阵调试
    debug_transformation()