import os
import time

def rename_images_by_time(folder_path, padding=0):
    """
    按时间顺序重命名文件夹下的图片。

    :param folder_path: 图片所在的文件夹路径
    :param padding: 数字填充位数，例如 3 表示生成 001.jpg, 002.jpg。0 表示不填充。
    """
    # 1. 定义支持的图片扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print(f"错误：路径 '{folder_path}' 不存在。")
        return

    # 获取文件夹内所有文件
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 筛选出图片文件
    images = []
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            full_path = os.path.join(folder_path, filename)
            # 获取修改时间 (getmtime)。如果想用创建时间，Windows下可用 getctime
            timestamp = os.path.getmtime(full_path)
            images.append({'name': filename, 'path': full_path, 'time': timestamp, 'ext': ext})

    if not images:
        print("未找到图片文件。")
        return

    # 2. 按时间排序 (从小到大，即最早的在前)
    images.sort(key=lambda x: x['time'])

    print(f"找到 {len(images)} 张图片，正在处理...")

    # ==========================================
    # 步骤 A: 先全部重命名为临时名称
    # 这样做是为了防止文件名冲突（例如想把 A 改名为 1，但文件夹里已经有个旧的 1）
    # ==========================================
    temp_prefix = f"temp_rename_{int(time.time())}_"
    for idx, img in enumerate(images):
        temp_name = f"{temp_prefix}{idx}{img['ext']}"
        temp_path = os.path.join(folder_path, temp_name)
        os.rename(img['path'], temp_path)
        # 更新列表中的路径，供下一步使用
        images[idx]['temp_path'] = temp_path

    # ==========================================
    # 步骤 B: 重命名为最终的数字名称
    # ==========================================
    for idx, img in enumerate(images):
        # 生成新名字，例如 1.jpg 或 001.jpg
        num_str = str(idx + 1).zfill(padding)
        new_name = f"{num_str}{img['ext']}"
        new_path = os.path.join(folder_path, new_name)

        os.rename(img['temp_path'], new_path)
        print(f"重命名: {img['name']} -> {new_name}")

    print("✅ 所有图片重命名完成！")

# ==========================================
# 使用配置区
# ==========================================
if __name__ == '__main__':
    # 请在这里修改你的图片文件夹路径
    # Windows 路径示例: r"D:\Photos\MyTrip" (前面加 r 防止转义)
    # Mac/Linux 路径示例: "/Users/name/Pictures"
    target_folder = "./img/color_img"

    # 数字位数补全：
    # 0 = 正常 (1.jpg, ..., 10.jpg)
    # 3 = 补全 (001.jpg, ..., 010.jpg) —— 推荐，方便电脑排序
    digit_padding = 0

    rename_images_by_time(target_folder, digit_padding)