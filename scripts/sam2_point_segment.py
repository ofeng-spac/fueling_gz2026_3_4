"""
SAM2 交互式点击分割 (OpenCV版本)
===============================
在图像上点击指定点，SAM2 将自动分割目标物体并显示结果。

操作说明:
  左键单击   - 添加前景点（绿色圆点）
  右键单击   - 添加背景点（红色圆点）
  Enter      - 执行分割
  空格键     - 清除所有点
  n          - 切换到下一个掩码
  s          - 保存当前掩码到文件
  q / Esc    - 退出

用法:
  python sam2_point_segment.py <图像路径> [--checkpoint <权重文件>] [--model-cfg <配置文件>]

示例:
  python sam2_point_segment.py photo.jpg
  python sam2_point_segment.py photo.jpg --checkpoint ../sam2-main/checkpoints/sam2.1_hiera_large.pt
"""

import os
import sys
import argparse
# 必须在导入 cv2 之前设置 Qt 环境
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_PLUGIN_PATH', '/usr/lib64/qt5/plugins')
os.environ.setdefault('XDG_CONFIG_HOME', '/home/vision/.config')
# 尝试修复 OpenCV Qt 插件路径
_cv2_plugin_path = '/home/vision/mamba/envs/fueling/lib/python3.12/site-packages/cv2/qt/plugins'
if os.path.exists(_cv2_plugin_path):
    _system_plugin_path = '/usr/lib64/qt5/plugins/platforms'
    if os.path.exists(_system_plugin_path):
        # 复制系统插件到 OpenCV 目录
        import shutil
        _dest = os.path.join(_cv2_plugin_path, 'platforms')
        if not os.path.exists(_dest):
            os.makedirs(_dest, exist_ok=True)
            for f in os.listdir(_system_plugin_path):
                src = os.path.join(_system_plugin_path, f)
                if os.path.isfile(src):
                    shutil.copy(src, _dest)

import numpy as np
import cv2
from PIL import Image

# 将 SAM2 源码路径加入 sys.path
SAM2_DIR = os.path.join(os.path.dirname(__file__), "..", "sam2-main")
if os.path.exists(SAM2_DIR):
    sys.path.insert(0, os.path.abspath(SAM2_DIR))

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"[错误] 无法导入 SAM2: {e}")
    print("安装方法:")
    print("  cd sam2-main && pip install -e .")
    sys.exit(1)


# ──────────────────────────────────────────────────────────
# 默认配置
# ──────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints", "sam2.1_hiera_large.pt")
DEFAULT_MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

WINDOW_NAME = "SAM2 Point Segmentation"
INFO_Y_OFFSET = 30  # 文字信息的起始 Y 坐标


# ──────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────
def draw_overlay(canvas, mask, alpha=0.5):
    """在画布上叠加显示掩码（蓝色半透明）。"""
    if mask is None:
        return canvas

    # 确保掩码是布尔类型
    mask_bool = mask.astype(np.bool_) if mask.dtype != np.bool_ else mask

    # 确保掩码与画布尺寸一致
    if mask_bool.shape[:2] != canvas.shape[:2]:
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (canvas.shape[1], canvas.shape[0])) > 0

    # 创建蓝色掩码
    color = np.array([30, 144, 255], dtype=np.uint8)  # 蓝色
    colored_mask = np.zeros_like(canvas)
    colored_mask[mask_bool] = color

    # 混合
    return cv2.addWeighted(canvas, 1 - alpha, colored_mask, alpha, 0)


def draw_points(canvas, points, labels):
    """在画布上绘制所有点。"""
    for (x, y), label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # 绿=前景，红=背景
        cv2.circle(canvas, (int(x), int(y)), 8, color, -1)
        cv2.circle(canvas, (int(x), int(y)), 8, (255, 255, 255), 2)  # 白边
    return canvas


def draw_info(canvas, text_lines, y_start=INFO_Y_OFFSET):
    """在画布顶部绘制信息文字。"""
    y = y_start
    for line in text_lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 25
    return canvas


def mask_info(mask):
    """返回掩码的像素面积和边界框。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, None
    area = int(mask.sum())
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return area, bbox


# ──────────────────────────────────────────────────────────
# 主应用程序
# ──────────────────────────────────────────────────────────
class SAM2PointSegmentor:
    def __init__(self, image_path: str, checkpoint: str, model_cfg: str):
        self.image_path = image_path
        self.checkpoint = checkpoint
        self.model_cfg  = model_cfg

        # 状态变量
        self.points: list[list[float]] = []
        self.labels: list[int]         = []
        self.masks   = None
        self.scores  = None
        self.logits  = None
        self.mask_idx = 0
        self.save_count = 0  # 保存文件的计数

        self._load_image()
        self._load_model()

    # ── 初始化 ──────────────────────────────────────────
    def _load_image(self):
        print(f"加载图像: {self.image_path}")
        img = Image.open(self.image_path).convert("RGB")
        self.image = np.array(img)
        print(f"  尺寸: {self.image.shape[1]}×{self.image.shape[0]} (宽×高)")

    def _load_model(self):
        ckpt_abs = os.path.abspath(self.checkpoint)
        if not os.path.exists(ckpt_abs):
            print(f"[错误] 权重文件不存在: {ckpt_abs}")
            print("请执行以下命令下载:")
            print(f"  cd {SAM2_DIR}/checkpoints && bash download_ckpts.sh")
            sys.exit(1)

        print(f"正在加载 SAM2 模型: {ckpt_abs}")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"  运行设备: {self.device}")

        # build_sam2 的 model_cfg 路径以 sam2-main 目录为基准
        prev_dir = os.getcwd()
        os.chdir(SAM2_DIR)
        model = build_sam2(self.model_cfg, ckpt_abs, device=self.device)
        os.chdir(prev_dir)
        self.predictor = SAM2ImagePredictor(model)

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        print("正在计算图像嵌入...")
        with torch.inference_mode():
            self.predictor.set_image(self.image)
        print("准备就绪！\n")

    # ── 渲染 ──────────────────────────────────────────
    def _render(self):
        """渲染当前状态到窗口。"""
        # 复制原始图像
        canvas = self.image.copy()

        # 叠加掩码（如果有）
        if self.masks is not None:
            mask = self.masks[self.mask_idx]
            canvas = draw_overlay(canvas, mask)

        # 绘制点
        canvas = draw_points(canvas, self.points, self.labels)

        # 绘制信息
        lines = []
        lines.append("Controls: LClick=FG(green)  RClick=BG(red)  Enter=Segment  Space=Reset  n=NextMask  s=Save  q=Quit")

        if self.points:
            pt_info = f"Points: {len(self.points)} ("
            pt_info += "FGs: " + str([f"({int(p[0])},{int(p[1])})" for p, l in zip(self.points, self.labels) if l == 1])
            pt_info += " BGs: " + str([f"({int(p[0])},{int(p[1])})" for p, l in zip(self.points, self.labels) if l == 0])
            pt_info += ")"
            lines.append(pt_info)

        if self.masks is not None:
            score = self.scores[self.mask_idx]
            area, bbox = mask_info(self.masks[self.mask_idx])
            lines.append(f"Mask {self.mask_idx+1}/{len(self.masks)}: score={score:.3f}  area={area:,}px  bbox={bbox}")
        else:
            lines.append("Press Enter to segment...")

        canvas = draw_info(canvas, lines)

        # 调整窗口大小以适应图像
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, canvas.shape[1], canvas.shape[0])
        cv2.imshow(WINDOW_NAME, canvas)

    # ── 分割 ──────────────────────────────────────────
    def _segment(self):
        if not self.points:
            print("[提示] 请先点击添加点。")
            return

        pts = np.array(self.points, dtype=np.float32)
        lbs = np.array(self.labels, dtype=np.int32)

        print(f"\n正在分割... (点数: {len(pts)})")
        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=pts,
                point_labels=lbs,
                multimask_output=True,
            )

        # 按得分降序排列
        order = np.argsort(scores)[::-1]
        self.masks = masks[order]
        self.scores = scores[order]
        self.logits = logits[order]
        self.mask_idx = 0

        self._print_result()
        self._render()

    def _print_result(self):
        if self.masks is None:
            return
        print("\n══ 分割结果 ══")
        for i, (mask, score) in enumerate(zip(self.masks, self.scores)):
            area, bbox = mask_info(mask)
            marker = "★" if i == self.mask_idx else " "
            print(f"  {marker} 掩码 {i+1}: 得分={score:.4f}  面积={area:,}px  bbox={bbox}")
        print("══════════════")

        sel_mask = self.masks[self.mask_idx]
        sel_area, sel_bbox = mask_info(sel_mask)
        print(f"\n[返回值] 当前选中掩码 {self.mask_idx+1}")
        print(f"  mask shape : {sel_mask.shape}")
        print(f"  像素面积   : {sel_area:,} px")
        print(f"  边界框     : x1={sel_bbox[0]}, y1={sel_bbox[1]}, x2={sel_bbox[2]}, y2={sel_bbox[3]}")

    def _save_mask(self):
        """保存当前掩码到文件。"""
        if self.masks is None:
            print("[提示] 没有可保存的掩码。")
            return

        mask = self.masks[self.mask_idx]

        # 保存 PNG 掩码（二值图）
        mask_path = f"/tmp/sam2_mask_{self.save_count}.png"
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        # 同时保存带掩码的图像
        overlay = self.image.copy()
        overlay = draw_overlay(overlay, mask)
        overlay_path = f"/tmp/sam2_overlay_{self.save_count}.png"
        cv2.imwrite(overlay_path, overlay)

        self.save_count += 1
        print(f"已保存: {mask_path} 和 {overlay_path}")

    # ── 事件回调 ──────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键 = 前景点
            self.points.append([float(x), float(y)])
            self.labels.append(1)
            print(f"  添加点: ({x}, {y}) [前景]  (共 {len(self.points)} 个)")
            self.masks = None  # 新点导致旧掩码失效
            self._render()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键 = 背景点
            self.points.append([float(x), float(y)])
            self.labels.append(0)
            print(f"  添加点: ({x}, {y}) [背景]  (共 {len(self.points)} 个)")
            self.masks = None
            self._render()

    # ── 主循环 ──────────────────────────────────────────
    def run(self):
        print("\n[操作说明]")
        print("  左键点击 - 添加前景点（绿色）")
        print("  右键点击 - 添加背景点（红色）")
        print("  Enter    - 执行分割")
        print("  空格     - 清除所有点")
        print("  n        - 切换到下一个掩码")
        print("  s        - 保存当前掩码")
        print("  q / Esc  - 退出\n")

        # 设置鼠标回调
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        # 初始渲染
        self._render()

        # 主循环
        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 13:  # Enter
                self._segment()

            elif key == ord(' '):  # Space
                self.points.clear()
                self.labels.clear()
                self.masks = None
                self.scores = None
                self.logits = None
                self.mask_idx = 0
                print("已清除所有点。")
                self._render()

            elif key == ord('n') or key == ord('N'):
                if self.masks is not None:
                    self.mask_idx = (self.mask_idx + 1) % len(self.masks)
                    print(f"切换到掩码 {self.mask_idx+1}/{len(self.masks)}")
                    self._render()

            elif key == ord('s') or key == ord('S'):
                self._save_mask()

            elif key == ord('q') or key == ord('Q') or key == 27:  # q or Escape
                if self.masks is not None:
                    self._print_result()
                break

        cv2.destroyAllWindows()
        print("程序已退出。")


# ──────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SAM2 交互式点击分割 (OpenCV版)")
    p.add_argument("image", help="待分割的图像路径")
    p.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help=f"SAM2 权重文件路径 (默认: {DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--model-cfg", default=DEFAULT_MODEL_CFG,
        help=f"模型配置文件 (默认: {DEFAULT_MODEL_CFG})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"[错误] 图像不存在: {args.image}")
        sys.exit(1)

    app = SAM2PointSegmentor(
        image_path=args.image,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
    )
    app.run()
