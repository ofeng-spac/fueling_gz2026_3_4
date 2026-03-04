import numpy as np
import cv2
import logging
logger = logging.getLogger(__name__)


def fill_roi_to_full_size(roi_image, roi_bbox, full_shape):
    """
    将ROI图像填充回原始大小
    
    Args:
        roi_image: ROI区域的图像/视差图
        roi_bbox: ROI边框 (x1, y1, x2, y2)
        full_shape: 原始图像的形状 (height, width)
        
    Returns:
        full_image: 填充后的完整图像
    """
    x1, y1, x2, y2 = map(int, roi_bbox)
    height, width = full_shape
    
    # 创建全零矩阵
    full_image = np.zeros(full_shape, dtype=roi_image.dtype)
    
    # 确保ROI区域在图像范围内
    h_roi, w_roi = roi_image.shape[:2]
    y_end = min(y1 + h_roi, height)
    x_end = min(x1 + w_roi, width)
    
    # 填充ROI区域
    full_image[y1:y_end, x1:x_end] = roi_image[:y_end-y1, :x_end-x1]
    
    return full_image
# ========== ROI相关辅助函数 ==========

def create_bbox_image(image, bbox, color=(0, 255, 0), thickness=2):
    """在图像上绘制边框"""
    if bbox is None:
        return image
    
    if len(image.shape) == 2:
        image_with_bbox = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_with_bbox = image.copy()
    
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image_with_bbox, (x1, y1), (x2, y2), color, thickness)
    
    info = f"BOX: [{x1},{y1},{x2},{y2}]"
    cv2.putText(image_with_bbox, info, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, color, 1)
    
    return image_with_bbox


def compute_bbox_from_matches(mkpts, padding=0, img_shape=None):
    """根据匹配点计算边框"""
    if len(mkpts) == 0:
        return None
    
    min_x, min_y = np.min(mkpts, axis=0)
    max_x, max_y = np.max(mkpts, axis=0)
    
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    
    if img_shape is not None:
        if len(img_shape) > 2:
            img_shape = img_shape[:2]
        height, width = img_shape
        max_x = min(width, max_x + padding)
        max_y = min(height, max_y + padding)
    else:
        max_x = max_x + padding
        max_y = max_y + padding
    
    return (min_x, min_y, max_x, max_y)


def crop_image_by_bbox(image, bbox):
    """根据边框裁剪图像"""
    if bbox is None:
        return None
    
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]


def filter_matches_by_bbox(mkpts0, mkpts1, bbox):
    """过滤匹配点，只保留mkpts0在bbox内的点"""
    if bbox is None:
        return mkpts0, mkpts1
    
    x1, y1, x2, y2 = bbox
    mask = (mkpts0[:, 0] >= x1) & (mkpts0[:, 0] <= x2) & \
           (mkpts0[:, 1] >= y1) & (mkpts0[:, 1] <= y2)
    
    return mkpts0[mask], mkpts1[mask]


def adjust_intrinsic_for_crop(K, bbox):
    """
    调整内参矩阵以适应裁剪后的图像
    
    Args:
        K: 原始内参矩阵 (3x3)
        bbox: 裁剪边框 (x1, y1, x2, y2)
        
    Returns:
        K_adjusted: 调整后的内参矩阵
    """
    if bbox is None:
        return K
    
    x1, y1, _, _ = bbox
    
    K_adjusted = K.copy()
    # 调整主点坐标：减去裁剪的偏移量
    K_adjusted[0, 2] = K[0, 2] - x1  # cx
    K_adjusted[1, 2] = K[1, 2] - y1  # cy
    
    return K_adjusted


def crop_images_to_same_size(image_left, bbox_left, image_right, bbox_right):
    """
    裁剪左右图像到相同大小，用于立体匹配
    
    Args:
        image_left: 左图
        bbox_left: 左图边框 (x1, y1, x2, y2)
        image_right: 右图
        bbox_right: 右图边框 (x1, y1, x2, y2)
        
    Returns:
        cropped_left: 裁剪后的左图
        cropped_right: 裁剪后的右图
        final_bbox_left: 调整后的左图边框
        final_bbox_right: 调整后的右图边框
    """
    if bbox_left is None or bbox_right is None:
        return None, None, None, None
    
    # 计算两个边框的尺寸
    x1_left, y1_left, x2_left, y2_left = bbox_left
    x1_right, y1_right, x2_right, y2_right = bbox_right
    
    width_left = x2_left - x1_left
    height_left = y2_left - y1_left
    width_right = x2_right - x1_right
    height_right = y2_right - y1_right
    
    # 计算最大宽度和最大高度
    max_width = max(width_left, width_right)
    max_height = max(height_left, height_right)
    
    logger.info(f"左图原始尺寸: {width_left}x{height_left}")
    logger.info(f"右图原始尺寸: {width_right}x{height_right}")
    logger.info(f"统一尺寸: {max_width}x{max_height}")
    
    # 调整左图边框（保持中心位置不变，调整到最大尺寸）
    center_x_left = (x1_left + x2_left) / 2
    center_y_left = (y1_left + y2_left) / 2
    
    final_x1_left = int(max(0, center_x_left - max_width / 2))
    final_y1_left = int(max(0, center_y_left - max_height / 2))
    final_x2_left = int(min(image_left.shape[1], final_x1_left + max_width))
    final_y2_left = int(min(image_left.shape[0], final_y1_left + max_height))
    
    # 如果宽度或高度不足，调整起始位置
    if final_x2_left - final_x1_left < max_width:
        final_x1_left = max(0, final_x2_left - max_width)
    if final_y2_left - final_y1_left < max_height:
        final_y1_left = max(0, final_y2_left - max_height)
    
    final_bbox_left = (final_x1_left, final_y1_left, final_x2_left, final_y2_left)
    
    # 调整右图边框（保持中心位置不变，调整到相同尺寸）
    center_x_right = (x1_right + x2_right) / 2
    center_y_right = (y1_right + y2_right) / 2
    
    final_x1_right = int(max(0, center_x_right - max_width / 2))
    final_y1_right = int(max(0, center_y_right - max_height / 2))
    final_x2_right = int(min(image_right.shape[1], final_x1_right + max_width))
    final_y2_right = int(min(image_right.shape[0], final_y1_right + max_height))
    
    # 如果宽度或高度不足，调整起始位置
    if final_x2_right - final_x1_right < max_width:
        final_x1_right = max(0, final_x2_right - max_width)
    if final_y2_right - final_y1_right < max_height:
        final_y1_right = max(0, final_y2_right - max_height)
    
    final_bbox_right = (final_x1_right, final_y1_right, final_x2_right, final_y2_right)
    
    # 确保最终尺寸完全一致
    final_width_left = final_x2_left - final_x1_left
    final_height_left = final_y2_left - final_y1_left
    final_width_right = final_x2_right - final_x1_right
    final_height_right = final_y2_right - final_y1_right
    
    # 如果尺寸不一致，使用最小尺寸
    min_width = min(final_width_left, final_width_right)
    min_height = min(final_height_left, final_height_right)
    
    # 调整左图边框到最小尺寸
    final_bbox_left = (
        final_x1_left,
        final_y1_left,
        final_x1_left + min_width,
        final_y1_left + min_height
    )
    
    # 调整右图边框到最小尺寸
    final_bbox_right = (
        final_x1_right,
        final_y1_right,
        final_x1_right + min_width,
        final_y1_right + min_height
    )
    
    # 裁剪图像
    cropped_left = crop_image_by_bbox(image_left, final_bbox_left)
    cropped_right = crop_image_by_bbox(image_right, final_bbox_right)
    
    # 确保尺寸完全一致（如果由于边界限制导致不一致，使用resize）
    if cropped_left is not None and cropped_right is not None:
        h_left, w_left = cropped_left.shape[:2]
        h_right, w_right = cropped_right.shape[:2]
        
        if h_left != h_right or w_left != w_right:
            # 使用最小尺寸
            min_h = min(h_left, h_right)
            min_w = min(w_left, w_right)
            
            if h_left != min_h or w_left != min_w:
                cropped_left = cv2.resize(cropped_left, (min_w, min_h))
                logger.info(f"左图调整后尺寸: {min_w}x{min_h}")
            
            if h_right != min_h or w_right != min_w:
                cropped_right = cv2.resize(cropped_right, (min_w, min_h))
                logger.info(f"右图调整后尺寸: {min_w}x{min_h}")
    
    return cropped_left, cropped_right, final_bbox_left, final_bbox_right


def transform_points_to_roi_coordinates(points, bbox):
    """将点坐标从完整图像坐标系转换到ROI坐标系"""
    if bbox is None:
        return points
    
    x1, y1, _, _ = bbox
    transformed = points.copy()
    transformed[:, 0] = points[:, 0] - x1
    transformed[:, 1] = points[:, 1] - y1
    
    return transformed