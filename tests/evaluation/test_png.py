import cv2
import numpy as np

def compare_images_pixel_perfect(image_path1, image_path2):
    """
    逐像素比较两张灰度图是否完全相同。

    参数:
    image_path1 (str): 第一张图片的路径。
    image_path2 (str): 第二张图片的路径。
    """
    # 1. 以灰度模式加载图片
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # 2. 检查图片是否成功加载
    if img1 is None:
        print(f"错误：无法加载图片 {image_path1}")
        return
    if img2 is None:
        print(f"错误：无法加载图片 {image_path2}")
        return

    # 3. 检查图片尺寸是否一致
    if img1.shape != img2.shape:
        print("错误：两张图片尺寸不同，无法进行像素级比较。")
        print(f"图片1尺寸: {img1.shape}, 图片2尺寸: {img2.shape}")
        return

    # 4. 计算均方误差 (MSE)
    # 将图像数据类型转为浮点型以避免计算溢出
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])

    # 5. 根据MSE结果得出结论
    if err == 0:
        print("结论：两张图片在像素级别上【完全相同】。")
    else:
        print(f"结论：两张图片【不相同】。")
        print(f"均方误差 (MSE): {err:.4f}")

        # 6. 如果不同，计算并显示差异图
        # cv2.absdiff可以直接计算两张图的差异绝对值
        diff = cv2.absdiff(img1, img2)

        print("正在显示差异图 (Difference Image)... 按任意键关闭窗口。")

        # 将原图和差异图拼接在一起显示，方便对比
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        h3, w3 = diff.shape
        vis = np.zeros((max(h1, h2, h3), w1 + w2 + w3), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1+w2] = img2
        vis[:h3, w1+w2:w1+w2+w3] = diff

        # 为拼接后的图像添加标签
        cv2.putText(vis, "Image 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, "Image 2", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, "Difference", (w1 + w2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Image1 vs Image2 vs Difference", vis)
        # 保存差异图像
        cv2.imwrite("difference_image.png", diff)
        print("差异图已保存为 'difference_image.png'")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 7. 叠加灰度图，查看是否有重影
        overlay = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        cv2.imshow("Overlay (Image1 + Image2)", overlay)
        cv2.imwrite("overlay_image.png", overlay)
        print("叠加图已保存为 'overlay_image.png'，请观察是否有重影。")

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# --- 使用示例 ---
if __name__ == "__main__":
    # 请将下面的路径替换为您自己的图片绝对路径
    #image1_path = "evaluation\\20251002_172750\images_output\\arm1\captured_left_ir.png"
    #image2_path = "20251002163816\\arm_1\ir_images\captured_left_ir.png"

    image1_path = r'1.png'
    image2_path = r'2.png'
    # 调用函数进行比较
    compare_images_pixel_perfect(image1_path, image2_path)