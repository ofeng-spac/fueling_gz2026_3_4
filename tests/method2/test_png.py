import cv2
import numpy as np
from loguru import logger


def compare_images_pixel_perfect(img1, img2, show=True):
    """
    逐像素比较两张图像是否完全相同（支持灰度和彩色）。
    """
    # 1. 检查尺寸
    if img1.shape != img2.shape:
        logger.error(f"尺寸不同：img1={img1.shape}, img2={img2.shape}")
        return

    # 2. 转浮点计算MSE
    err = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    psnr = 10 * np.log10((255 ** 2) / err) if err != 0 else float('inf')

    # 3. 输出结果
    if err == 0:
        logger.info("✅ 两张图片在像素级完全相同。")
        return
    else:
        logger.warning(f"⚠️ 两张图片不同。MSE={err:.4f}, PSNR={psnr:.2f} dB")

    # 4. 差异图
    diff = cv2.absdiff(img1, img2)
    cv2.imwrite("difference_image.png", diff)
    logger.info("差异图已保存为 'difference_image.png'")

    # 5. 转为三通道以便拼接和叠加
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    else:
        img1_color, img2_color, diff_color = img1, img2, diff

    # 6. 拼接可视化
    vis = np.hstack((img1_color, img2_color, diff_color))
    cv2.putText(vis, "Image 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, "Image 2", (img1.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, "Difference", (img1.shape[1]*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite("comparison_visual.png", vis)
    logger.info("拼接图已保存为 'comparison_visual.png'")

    # 7. 叠加效果
    overlay = cv2.addWeighted(img1_color, 0.5, img2_color, 0.5, 0)
    cv2.imwrite("overlay_image.png", overlay)
    logger.info("叠加图已保存为 'overlay_image.png'")

    # 8. 显示（可选）
    if show:
        cv2.imshow("Comparison", vis)
        cv2.imshow("Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()