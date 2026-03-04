# flb2_utils/visualization.py
import open3d as o3d
import numpy as np
import cv2
import torch
from kornia.geometry.transform import warp_perspective

from .src.utils.plotting import make_matching_figure
from typing import Tuple, Optional

def display_point_cloud_with_axes(pcd1, pcd2=None, title1="Point Cloud with Axes", title2="Point Cloud with Axes") -> None:
    """显示带坐标轴的点云"""
    axis_length = 200
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=[0, 0, 0])

    if pcd1 is not None:
        pcd1.paint_uniform_color([1, 1, 0])  
    if pcd2 is not None:
        pcd2.paint_uniform_color([1, 0, 0])  

    geometries = [pcd1, coordinate_frame]
    if pcd2 is not None:
        geometries.append(pcd2)

    o3d.visualization.draw_geometries(geometries, window_name="Point Cloud with XYZ Axes")

def create_visualization_image(ir_img: np.ndarray) -> np.ndarray:
    """根据图像类型创建可视化图像"""
    if len(ir_img.shape) == 2 and ir_img.dtype == np.uint16:
        ir_vis = cv2.convertScaleAbs(ir_img, alpha=255.0 / 4095)
        ir_vis = cv2.cvtColor(ir_vis, cv2.COLOR_GRAY2BGR)
    else:
        ir_vis = ir_img.copy()
    return ir_vis

def draw_projection_points(image: np.ndarray, points: np.ndarray, 
                          color: Tuple[int, int, int] = (0, 255, 0), 
                          sampling_rate: int = 10) -> None:
    """在图像上绘制投影点"""
    for px, py in points[::sampling_rate]:
        cv2.circle(image, (px, py), 1, color, -1)

def visualize_points_on_image(points: np.ndarray, 
                            save_path: str, 
                            background_img: Optional[np.ndarray] = None,
                            color: Tuple[int, int, int] = (0, 0, 255),
                            point_size: int = 2) -> None:
    """在图像上可视化点集"""
    if background_img is None:
        background_img = cv2.imread("/home/vision/projects/flb/MINIMA/demo/four/projected_ir_left.png")
    
    if background_img is not None:
        vis = background_img.copy()
        for pt in points:
            cv2.circle(vis, tuple(np.round(pt).astype(int)), point_size, color, -1)
        cv2.imwrite(save_path, vis)
        print(f"可视化已保存 -> {save_path}")

def visualize_projection_matches(img0, img1, mkpts0, mkpts1, save_path, title="投影匹配点对"):
    """可视化双向交集中的匹配点对"""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    if h0 != h1:
        scale = h0 / h1
        new_w1 = int(w1 * scale)
        img1 = cv2.resize(img1, (new_w1, h0))
        w1 = new_w1
    
    canvas = np.zeros((h0, w0 + w1, 3), dtype=np.uint8)
    canvas[:, :w0] = img0
    canvas[:, w0:w0+w1] = img1
    
    for i, (pt0, pt1) in enumerate(zip(mkpts0, mkpts1)):
        np.random.seed(i)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        x0, y0 = int(round(pt0[0])), int(round(pt0[1]))
        cv2.circle(canvas, (x0, y0), 4, color, -1)
        cv2.putText(canvas, str(i), (x0+5, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        x1, y1 = int(round(pt1[0])) + w0, int(round(pt1[1]))
        cv2.circle(canvas, (x1, y1), 4, color, -1)
        cv2.putText(canvas, str(i), (x1+5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
    
    cv2.putText(canvas, f"{title} - 共 {len(mkpts0)} 对匹配点", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.line(canvas, (w0, 0), (w0, h0), (255, 255, 255), 2)
    
    cv2.imwrite(save_path, canvas)
    print(f"投影匹配点对可视化已保存: {save_path}")
    
    return canvas

def visualize_projection_matches_grid(img0, img1, mkpts0, mkpts1, save_path, 
                                    grid_size=5, title="投影匹配点对网格"):
    """以网格形式可视化投影匹配点对"""
    h, w = img0.shape[:2]
    
    max_matches = grid_size * grid_size
    if len(mkpts0) > max_matches:
        indices = np.random.choice(len(mkpts0), max_matches, replace=False)
        mkpts0 = mkpts0[indices]
        mkpts1 = mkpts1[indices]
        print(f"匹配点对过多，随机选择 {max_matches} 对进行显示")
    
    cell_h, cell_w = h // 2, w // 2
    
    rows = (len(mkpts0) + grid_size - 1) // grid_size
    canvas = np.zeros((rows * cell_h * 2, grid_size * cell_w, 3), dtype=np.uint8)
    
    for i, (pt0, pt1) in enumerate(zip(mkpts0, mkpts1)):
        row = i // grid_size
        col = i % grid_size
        
        x0, y0 = int(round(pt0[0])), int(round(pt0[1]))
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        
        x0_start, x0_end = max(0, x0-cell_w//2), min(w, x0+cell_w//2)
        y0_start, y0_end = max(0, y0-cell_h//2), min(h, y0+cell_h//2)
        x1_start, x1_end = max(0, x1-cell_w//2), min(w, x1+cell_w//2)
        y1_start, y1_end = max(0, y1-cell_h//2), min(h, y1+cell_h//2)
        
        patch0 = img0[y0_start:y0_end, x0_start:x0_end]
        patch1 = img1[y1_start:y1_end, x1_start:x1_end]
        
        patch0 = cv2.resize(patch0, (cell_w, cell_h))
        patch1 = cv2.resize(patch1, (cell_w, cell_h))
        
        center_x0, center_y0 = cell_w//2, cell_h//2
        center_x1, center_y1 = cell_w//2, cell_h//2
        
        cv2.circle(patch0, (center_x0, center_y0), 3, (0, 255, 0), -1)
        cv2.circle(patch1, (center_x1, center_y1), 3, (0, 255, 0), -1)
        
        canvas[row*cell_h*2:row*cell_h*2+cell_h, col*cell_w:col*cell_w+cell_w] = patch0
        canvas[row*cell_h*2+cell_h:row*cell_h*2+cell_h*2, col*cell_w:col*cell_w+cell_w] = patch1
        
        cv2.putText(canvas, str(i), (col*cell_w+5, row*cell_h*2+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(canvas, f"{title} - 显示 {len(mkpts0)} 对匹配点", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(save_path, canvas)
    print(f"投影匹配点对网格可视化已保存: {save_path}")
    
    return canvas

def save_ransac_matching_figures(path_before: str, path_after: str,
                                 img0: np.ndarray, img1: np.ndarray, 
                                 mkpts0: np.ndarray, mkpts1: np.ndarray, 
                                 inlier_mask: np.ndarray, color: np.ndarray):
    """保存 RANSAC 前后的匹配对比图"""
    text_before = [f'Matches (before RANSAC): {len(mkpts0)}']
    make_matching_figure(
        img0, img1, mkpts0, mkpts1,
        color, text=text_before, path=path_before, dpi=150, draw_lines=False
    )

    if inlier_mask is not None and inlier_mask.sum() > 0:
        inlier_mask = inlier_mask.astype(bool).squeeze()
        mkpts0_inliers = mkpts0[inlier_mask]
        mkpts1_inliers = mkpts1[inlier_mask]
        color_inliers = color[inlier_mask]
        text_after = [f'Matches (after RANSAC): {len(mkpts0_inliers)}']
        make_matching_figure(
            img0, img1, mkpts0_inliers, mkpts1_inliers,
            color_inliers, text=text_after, path=path_after, dpi=150, draw_lines=True
        )

def visualize_method_differences(idx1, idx2, mkpts0, img_bg, figures_dir, method):
    """可视化两种方法的差异"""
    import os.path as osp
    only_method1 = np.setdiff1d(idx1, idx2)
    if len(only_method1) > 0:
        vis_diff = img_bg.copy()
        for idx in only_method1:
            pt = mkpts0[idx]
            cv2.circle(vis_diff, tuple(np.round(pt).astype(int)), 3, (255, 0, 0), -1)
        cv2.imwrite(osp.join(figures_dir, f"only_in_method1_{method}.png"), vis_diff)
    
    only_method2 = np.setdiff1d(idx2, idx1)
    if len(only_method2) > 0:
        vis_diff = img_bg.copy()
        for idx in only_method2:
            pt = mkpts0[idx]
            cv2.circle(vis_diff, tuple(np.round(pt).astype(int)), 3, (0, 0, 255), -1)
        cv2.imwrite(osp.join(figures_dir, f"only_in_method2_{method}.png"), vis_diff)

def visualize_matches(img0, img1, mkpts0, mkpts1, figures_dir, method):
    """可视化匹配点对"""
    import os.path as osp
    print("\n=== 生成匹配点对可视化 ===")
    
    # 创建BGR版本的图像用于OpenCV可视化
    img0_bgr = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    
    # 水平拼接+连线可视化
    matches_vis_path = osp.join(figures_dir, f"projection_matches_{method}.jpg")
    visualize_projection_matches(
        img0_bgr, img1_bgr, mkpts0, mkpts1, 
        matches_vis_path, 
        title=f"Projection Matches ({method})"
    )
    
    # 网格形式可视化
    grid_vis_path = osp.join(figures_dir, f"projection_matches_grid_{method}.jpg")
    visualize_projection_matches_grid(
        img0_bgr, img1_bgr, mkpts0, mkpts1,
        grid_vis_path,
        title=f"Projection Matches Grid ({method})"
    )

def H_transform(img2_tensor, homography):
    """应用单应性变换"""
    image_shape = img2_tensor.shape[2:]
    img2_tensor = warp_perspective(img2_tensor, homography, image_shape, align_corners=True)
    return img2_tensor

def visualize_sparse_pointclouds(sparse_pcd0, sparse_pcd1, title=""):
    """可视化稀疏点云及其匹配关系，并为点云设置不同颜色"""
    if sparse_pcd0 is None or sparse_pcd1 is None:
        print("警告: 一个或两个稀疏点云为空，无法进行可视化。")
        return

    print("可视化稀疏点云...")

    # 创建副本以避免修改原始点云
    pcd0_vis = o3d.geometry.PointCloud(sparse_pcd0)
    pcd1_vis = o3d.geometry.PointCloud(sparse_pcd1)

    # 为点云设置不同的颜色
    pcd0_vis.paint_uniform_color([1, 0, 1])  # 紫色
    pcd1_vis.paint_uniform_color([0, 1, 0])  # 绿色
    
    # 将第二个点云平移以便观察
    pcd1_vis.translate([0, 0, 0]) # pcd1_vis.translate([0.3, 0, 0])
    
    lines = []
    # 假设两个点云中的点是一一对应的
    num_points = min(len(pcd0_vis.points), len(pcd1_vis.points))
    for i in range(num_points):
        lines.append([i, i + len(pcd0_vis.points)])
    
    line_set = o3d.geometry.LineSet()
    if lines:
        all_points = np.vstack([
            np.asarray(pcd0_vis.points), 
            np.asarray(pcd1_vis.points)
        ])
        line_set.points = o3d.utility.Vector3dVector(all_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in range(len(lines))])
        
    # 一起显示带连线的点云
    o3d.visualization.draw_geometries(
        [pcd0_vis, pcd1_vis, line_set],
        window_name=title
    )
