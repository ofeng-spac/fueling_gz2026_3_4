import sys
import os

# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/vision/mamba/envs/fueling/lib/python3.12/site-packages/PyQt5/Qt5/plugins'
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use 'xcb' for better compatibility

# 尝试导入 PyQt5 - 必须在 cv2 和其他库之前导入，以避免 Qt 版本冲突
# 这通常解决 "Cannot mix incompatible Qt library" 错误
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                 QPushButton, QLabel, QComboBox, QTextEdit, QGroupBox, QGridLayout,
                                 QMessageBox, QSplitter, QFileDialog, QFrame, QSpinBox, QCheckBox)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot
    from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
except ImportError:
    print("错误: 未找到 PyQt5。请运行 'pip install PyQt5' 安装。")
    sys.exit(1)

# Now import PyQt and other modules
import cv2
import numpy as np
import time
from datetime import datetime

# 引入 pyqtgraph 和 OpenGL

import pyqtgraph.opengl as gl
import pyqtgraph as pg


from loguru import logger


# 导入后端
# 假设脚本从项目根目录或 fueling/initialization 运行
try:
    # 尝试作为模块导入
    from fueling.initialization.model_initializer_backend import FuelingModelBackend
except ImportError:
    # 尝试相对导入，添加路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from fueling.initialization.model_initializer_backend import FuelingModelBackend

class LoguruSink(QObject):
    """
    Loguru 自定义 Sink，将日志转发到 Qt 信号
    """
    new_log = pyqtSignal(str)

    def write(self, message):
        self.new_log.emit(message)

    def flush(self):
        pass

class Worker(QThread):
    """
    通用工作线程，用于执行耗时的后端任务
    """
    finished_signal = pyqtSignal(object)  # 任务完成信号，携带结果
    error_signal = pyqtSignal(str)        # 发生错误信号，携带错误信息

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished_signal.emit(result)
        except Exception as e:
            # 捕获所有未处理的异常
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))

class VideoThread(QThread):
    """
    视频流线程，循环获取帧
    """
    frame_received = pyqtSignal(object) # 发送 images dict

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self._running = False
        self.exposure = 20000
        self.gain = 100

    def set_params(self, exposure, gain):
        self.exposure = exposure
        self.gain = gain

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        self._running = True
        while self._running:
            if self.backend and self.backend.camera_client:
                images = self.backend.get_latest_frame(self.exposure, self.gain)
                if images:
                    self.frame_received.emit(images)
            # 简单的限频，避免过度占用 CPU，约 30fps
            QThread.msleep(30)

class ModelInitializerUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fueling Model Initializer")
        self.resize(1200, 800)

        # 后端实例
        self.backend = None
        self.worker = None
        self.video_thread = None

        # 默认配置路径
        self.default_config_path = os.path.join(os.path.dirname(__file__), "../../data/arm1/config.jsonnet")
        if not os.path.exists(self.default_config_path):
             self.default_config_path = ""

        self.init_ui()
        self.init_logging()

    def init_ui(self):
        """初始化 UI 布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局：左右分割
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- 左侧控制面板 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # 1. 配置区域
        config_group = QGroupBox("1. Configuration")
        config_layout = QGridLayout()

        config_layout.addWidget(QLabel("Config File:"), 0, 0)
        self.config_path_edit = QLabel(os.path.basename(self.default_config_path))
        self.config_path_edit.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        config_layout.addWidget(self.config_path_edit, 0, 1)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.select_config_file)
        config_layout.addWidget(btn_browse, 0, 2)

        config_layout.addWidget(QLabel("Camera:"), 1, 0)
        self.combo_camera = QComboBox()
        self.combo_camera.addItems(["OB"])
        config_layout.addWidget(self.combo_camera, 1, 1, 1, 2)

        config_layout.addWidget(QLabel("Light:"), 2, 0)
        self.combo_light = QComboBox()
        self.combo_light.addItems(["flood", "laser"])
        config_layout.addWidget(self.combo_light, 2, 1, 1, 2)

        btn_load_config = QPushButton("Load Config & Backend")
        btn_load_config.clicked.connect(self.load_backend)
        self.btn_load_config = btn_load_config # 保存引用以更改状态
        config_layout.addWidget(btn_load_config, 3, 0, 1, 3)

        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)

        # 2. 设备连接区域
        conn_group = QGroupBox("2. Connection")
        conn_layout = QVBoxLayout()

        hbox_conn = QHBoxLayout()
        self.btn_connect_robot = QPushButton("Connect Robot")
        self.btn_connect_robot.clicked.connect(self.connect_robot)
        self.btn_connect_robot.setEnabled(False)
        hbox_conn.addWidget(self.btn_connect_robot)

        self.lbl_robot_status = QLabel("Disconnected")
        self.lbl_robot_status.setStyleSheet("color: red")
        hbox_conn.addWidget(self.lbl_robot_status)
        conn_layout.addLayout(hbox_conn)

        hbox_cam = QHBoxLayout()
        self.btn_connect_camera = QPushButton("Connect Camera")
        self.btn_connect_camera.clicked.connect(self.connect_camera)
        self.btn_connect_camera.setEnabled(False)
        hbox_cam.addWidget(self.btn_connect_camera)

        self.lbl_camera_status = QLabel("Disconnected")
        self.lbl_camera_status.setStyleSheet("color: red")
        hbox_cam.addWidget(self.lbl_camera_status)
        conn_layout.addLayout(hbox_cam)

        # 相机参数控制 (新增)
        cam_param_layout = QGridLayout()
        cam_param_layout.addWidget(QLabel("Exposure:"), 0, 0)
        self.spin_exposure = QSpinBox()
        self.spin_exposure.setRange(1, 100000)
        self.spin_exposure.setValue(20000)
        self.spin_exposure.setSingleStep(100)
        self.spin_exposure.valueChanged.connect(self.update_camera_params)
        cam_param_layout.addWidget(self.spin_exposure, 0, 1)

        cam_param_layout.addWidget(QLabel("Gain:"), 1, 0)
        self.spin_gain = QSpinBox()
        self.spin_gain.setRange(0, 500)
        self.spin_gain.setValue(100)
        self.spin_gain.valueChanged.connect(self.update_camera_params)
        cam_param_layout.addWidget(self.spin_gain, 1, 1)

        self.chk_preview = QCheckBox("Real-time Preview")
        self.chk_preview.toggled.connect(self.toggle_preview)
        self.chk_preview.setEnabled(False) # 连接相机后启用
        cam_param_layout.addWidget(self.chk_preview, 2, 0, 1, 2)

        conn_layout.addLayout(cam_param_layout)

        hbox_ai = QHBoxLayout()
        self.btn_load_ai = QPushButton("Load AI Model")
        self.btn_load_ai.clicked.connect(self.load_ai_model)
        self.btn_load_ai.setEnabled(False)
        hbox_ai.addWidget(self.btn_load_ai)

        self.lbl_ai_status = QLabel("Not Loaded")
        self.lbl_ai_status.setStyleSheet("color: orange")
        hbox_ai.addWidget(self.lbl_ai_status)
        conn_layout.addLayout(hbox_ai)

        conn_group.setLayout(conn_layout)
        left_layout.addWidget(conn_group)

        # 3. 示教与采集流程
        step_group = QGroupBox("3. Operation Steps")
        step_layout = QVBoxLayout()

        self.steps_buttons = []

        # 辅助函数添加步骤按钮
        def add_step_btn(text, slot):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btn.setEnabled(False) # 初始禁用
            step_layout.addWidget(btn)
            self.steps_buttons.append(btn)
            return btn

        self.btn_rec_init = add_step_btn("(A) Record Init Pose", self.record_init_pose)
        self.btn_rec_capture = add_step_btn("(B) Record Capture Pose", self.record_capture_pose)
        self.btn_rec_fueling = add_step_btn("(C) Record Fueling Pose (Target)", self.record_fueling_pose)
        self.btn_save_pose = add_step_btn("(D) Save Poses to Config", self.save_poses)

        step_layout.addSpacing(10)
        step_layout.addWidget(QLabel("--- Action ---"))

        self.btn_move_capture = add_step_btn("1. Move to Capture Pose", self.move_to_capture)
        self.btn_capture = add_step_btn("2. Capture Images", self.capture_images)
        self.btn_compute = add_step_btn("3. Compute Stereo & Depth", self.compute_stereo)
        self.btn_gen_pcd = add_step_btn("4. Generate & Process PointCloud", self.process_pcd)
        self.btn_reset = add_step_btn("5. Return to Init", self.return_to_init)

        step_group.setLayout(step_layout)
        left_layout.addWidget(step_group)

        left_layout.addStretch() # 弹簧

        # --- 右侧显示面板 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # 图像显示区域 (Grid Layout)
        img_grid = QGridLayout()

        # 左图
        img_grid.addWidget(QLabel("Left Camera:"), 0, 0)
        self.lbl_img_left = QLabel()
        self.lbl_img_left.setStyleSheet("background-color: #222; border: 1px solid #555;")
        self.lbl_img_left.setMinimumSize(320, 240)
        self.lbl_img_left.setAlignment(Qt.AlignCenter)
        self.lbl_img_left.setText("No Image")
        img_grid.addWidget(self.lbl_img_left, 1, 0)

        # 右图
        img_grid.addWidget(QLabel("Right Camera:"), 0, 1)
        self.lbl_img_right = QLabel()
        self.lbl_img_right.setStyleSheet("background-color: #222; border: 1px solid #555;")
        self.lbl_img_right.setMinimumSize(320, 240)
        self.lbl_img_right.setAlignment(Qt.AlignCenter)
        self.lbl_img_right.setText("No Image")
        img_grid.addWidget(self.lbl_img_right, 1, 1)

        # 深度图/结果图
        img_grid.addWidget(QLabel("Depth Map / Result:"), 2, 0)
        self.lbl_img_depth = QLabel()
        self.lbl_img_depth.setStyleSheet("background-color: #222; border: 1px solid #555;")
        self.lbl_img_depth.setMinimumSize(320, 240)
        self.lbl_img_depth.setAlignment(Qt.AlignCenter)
        self.lbl_img_depth.setText("No Result")
        img_grid.addWidget(self.lbl_img_depth, 3, 0) # 跨两列 -> 改为单列

        # 点云显示区域 (新增)
        img_grid.addWidget(QLabel("Point Cloud Visualization:"), 2, 1)
        self.pcd_widget = gl.GLViewWidget()
        self.pcd_widget.setMinimumSize(320, 240)
        self.pcd_widget.opts['distance'] = 500  # 初始视角距离
        # 添加网格
        g = gl.GLGridItem()
        g.setSize(x=500, y=500, z=500)
        g.setSpacing(x=50, y=50, z=50)
        self.pcd_widget.addItem(g)
        img_grid.addWidget(self.pcd_widget, 3, 1)

        # 让图片区域尽可能大
        right_layout.addLayout(img_grid, 2)

        # 日志区域
        right_layout.addWidget(QLabel("Logs:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc; font-family: Monospace;")
        right_layout.addWidget(self.log_text, 1) # 比例 2:1

        # 添加到 Splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

    def init_logging(self):
        """配置 Loguru 转发"""
        self.log_sink = LoguruSink()
        self.log_sink.new_log.connect(self.append_log)

        # 移除默认handler并添加sink
        logger.remove()
        # 同时保留控制台输出以便调试，并添加GUI输出
        logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}")
        logger.add(
            self.log_sink,
            format="{time:HH:mm:ss} | {level} | {message}",
            level="INFO"
        )

    @pyqtSlot(str)
    def append_log(self, message):
        self.log_text.moveCursor(self.log_text.textCursor().End)
        self.log_text.insertPlainText(message)
        self.log_text.moveCursor(self.log_text.textCursor().End)

    # --- 逻辑控制 ---

    def run_worker(self, func, callback=None, *args, **kwargs):
        """
        运行后台任务的辅助函数
        :param func: 要运行的函数
        :param callback: 成功后的回调函数
        """
        # 禁用所有按钮防止重入（简单处理，实际可做更细致控制）
        # self.set_buttons_enabled(False)

        self.worker = Worker(func, *args, **kwargs)
        if callback:
            self.worker.finished_signal.connect(callback)
        self.worker.error_signal.connect(self.on_worker_error)
        self.worker.finished_signal.connect(lambda: self.on_worker_finished()) # 总是执行清理

        self.worker.start()

    def on_worker_error(self, err_msg):
        QMessageBox.critical(self, "Error", f"Operation Failed:\n{err_msg}")
        self.on_worker_finished()

    def on_worker_finished(self):
        # 恢复按钮状态等...
        # self.set_buttons_enabled(True)
        pass

    def select_config_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Config', os.path.dirname(self.default_config_path), 'Jsonnet Files (*.jsonnet)')
        if fname:
            self.default_config_path = fname
            self.config_path_edit.setText(os.path.basename(fname))

    # --- 槽函数 ---

    def load_backend(self):
        config_path = self.default_config_path
        if not config_path:
            QMessageBox.warning(self, "Warning", "Please select a config file first.")
            return

        light = self.combo_light.currentText()
        camera = self.combo_camera.currentText()

        self.log_text.append(f"Initializing Backend with {os.path.basename(config_path)}...")

        # 由于 __init__ 可能耗时（加载配置），也放到 worker
        def init_task():
            return FuelingModelBackend(config_path, light_type=light, camera_type=camera, cut_type="box")

        def on_loaded(backend_instance):
            self.backend = backend_instance
            self.log_text.append("Backend Initialized.")
            self.btn_load_config.setEnabled(False)
            self.btn_load_config.setText("Config Loaded")
            # 启用下一步
            self.btn_connect_robot.setEnabled(True)
            self.btn_connect_camera.setEnabled(True)
            self.btn_load_ai.setEnabled(True)

        self.run_worker(init_task, on_loaded)

    def connect_robot(self):
        if not self.backend: return
        self.log_text.append("Connecting to Robot...")

        def on_connected(success):
            if success:
                self.lbl_robot_status.setText("Connected")
                self.lbl_robot_status.setStyleSheet("color: green")
                self.btn_connect_robot.setEnabled(False)
                self.check_step_buttons()
            else:
                self.lbl_robot_status.setText("Failed")

        self.run_worker(self.backend.connect_robot, on_connected)

    def connect_camera(self):
        if not self.backend: return
        self.log_text.append("Connecting to Camera...")

        def on_connected(success):
            if success:
                self.lbl_camera_status.setText("Connected")
                self.lbl_camera_status.setStyleSheet("color: green")
                self.btn_connect_camera.setEnabled(False)
                self.chk_preview.setEnabled(True) # 启用预览勾选框

                # 初始化参数控件的值
                if self.backend.config:
                    try:
                        exp = self.backend.config['camera']['exposure']
                        gain = self.backend.config['camera']['gain']
                        self.spin_exposure.setValue(int(exp))
                        self.spin_gain.setValue(int(gain))
                    except:
                        pass

                self.check_step_buttons()
            else:
                self.lbl_camera_status.setText("Failed")

        self.run_worker(self.backend.connect_camera, on_connected)

    def update_camera_params(self):
        """当界面参数改变时更新线程参数"""
        if self.video_thread:
            self.video_thread.set_params(self.spin_exposure.value(), self.spin_gain.value())

    def toggle_preview(self, checked):
        """切换实时预览"""
        if checked:
            if not self.backend or not self.backend.camera_client:
                self.chk_preview.setChecked(False)
                return

            self.video_thread = VideoThread(self.backend)
            self.video_thread.set_params(self.spin_exposure.value(), self.spin_gain.value())
            self.video_thread.frame_received.connect(self.on_video_frame)
            self.video_thread.start()
            self.log_text.append("Preview started.")
        else:
            if self.video_thread:
                self.video_thread.stop()
                self.video_thread = None
            self.log_text.append("Preview stopped.")

    def on_video_frame(self, images):
        """处理视频流帧"""
        if 'left_ir' in images:
            self.display_numpy_image(images['left_ir'], self.lbl_img_left)
        if 'right_ir' in images:
            self.display_numpy_image(images['right_ir'], self.lbl_img_right)

    def display_numpy_image(self, img_np, label_widget):
        """直接显示 numpy数组图像"""
        if img_np is None: return

        # 可能是单通道 IR
        if len(img_np.shape) == 2:
            img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def load_ai_model(self):
        if not self.backend: return
        self.log_text.append("Loading AI Model (this may take a while)...")

        def on_loaded(success):
            if success:
                self.lbl_ai_status.setText("Loaded")
                self.lbl_ai_status.setStyleSheet("color: green")
                self.btn_load_ai.setEnabled(False)
                self.check_step_buttons()
            else:
                self.lbl_ai_status.setText("Failed")

        self.run_worker(self.backend.load_ai_model, on_loaded)

    def check_step_buttons(self):
        """检查依赖关系并启用操作按钮"""
        robot_ok = self.lbl_robot_status.text() == "Connected"
        cam_ok = self.lbl_camera_status.text() == "Connected"
        ai_ok = self.lbl_ai_status.text() == "Loaded"

        if robot_ok:
            self.btn_rec_init.setEnabled(True)
            self.btn_rec_capture.setEnabled(True)
            self.btn_rec_fueling.setEnabled(True)
            self.btn_move_capture.setEnabled(True)
            self.btn_reset.setEnabled(True)

        if robot_ok and cam_ok and ai_ok:
            # 这些需要组合条件，这里简单处理，有连接就允许点，backend 会再校验
            self.btn_capture.setEnabled(True)
            self.btn_compute.setEnabled(True)
            self.btn_gen_pcd.setEnabled(True)

        # 保存按钮需要三个位姿都OK，暂时一直开启，由 backend 校验
        self.btn_save_pose.setEnabled(True)

    # --- 流程函数 ---

    def log_pose_to_file(self, title, data):
        """Append recorded pose to the output text file."""
        try:
            output_path = os.path.join(os.path.dirname(__file__), "../../calibration/pose/output.txt")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(output_path, "a") as f:
                f.write(f"[{timestamp}] {title}: {data}\n")

            self.log_text.append(f"Logged {title} to output.txt")
        except Exception as e:
            logger.error(f"Failed to log pose: {e}")

    def record_init_pose(self):
        def on_success(r):
            self.log_pose_to_file("Init Pose", r)
            QMessageBox.information(self, "Info", f"Init Pose Recorded:\n{r}")
        self.run_worker(self.backend.record_init_pose, on_success)

    def record_capture_pose(self):
        def on_success(r):
            self.log_pose_to_file("Capture Pose", r)
            QMessageBox.information(self, "Info", f"Capture Pose Recorded:\n{r}")
        self.run_worker(self.backend.record_capture_pose, on_success)

    def record_fueling_pose(self):
        def on_success(r):
            self.log_pose_to_file("Fueling Pose", r)
            QMessageBox.information(self, "Info", f"Fueling Pose Recorded:\n{r}")
        self.run_worker(self.backend.record_fueling_pose, on_success)

    def save_poses(self):
        self.run_worker(self.backend.save_poses_to_file, lambda s: QMessageBox.information(self, "Info", "Poses Saved!" if s else "Save Failed! Check Logs."))

    def move_to_capture(self):
        self.run_worker(self.backend.move_to_capture_pose)

    def capture_images(self):
        def on_captured(result):
            left_path, right_path, _ = result
            if left_path and right_path:
                self.display_image(left_path, self.lbl_img_left)
                self.display_image(right_path, self.lbl_img_right)
            else:
                QMessageBox.warning(self, "Warning", "Capture returned empty paths.")

        self.run_worker(self.backend.capture_image, on_captured)

    def display_image(self, path, label_widget):
        if not os.path.exists(path): return

        # 必须在 GUI 线程加载和显示
        img = cv2.imread(path)
        if img is None: return

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 缩放以适应 Label
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def compute_stereo(self):
        def on_computed(depth_path):
            if (depth_path and isinstance(depth_path, str)):
                # 显示深度图（伪彩色显示可能更好，这里直接显示保存的 image）
                # 原始深度图可能是 16位 的，OpenCV 读取可能全黑
                # 我们读取并做一个简单的归一化显示
                img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # 简单的可视化：归一化到 0-255
                    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                    norm_img = np.uint8(norm_img)
                    # 伪彩色
                    color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

                    # 转换为 QImage 过程同上
                    h, w, ch = color_img.shape
                    bytes_per_line = ch * w
                    q_img = QImage(color_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.lbl_img_depth.setPixmap(pixmap.scaled(self.lbl_img_depth.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                else:
                    self.lbl_img_depth.setText(f"Saved to {depth_path}")
            else:
                QMessageBox.warning(self, "Warning", "Computation failed.")

        self.run_worker(self.backend.compute_stereo_and_depth, on_computed)

    def process_pcd(self):
        def on_pcd_processed(source_pcd):
            if source_pcd is not None:
                QMessageBox.information(self, "Done", "PointCloud Processed and Saved!")
                self.visualize_pointcloud(source_pcd)
            else:
                QMessageBox.warning(self, "Failed", "Failed to process point cloud.")

        self.run_worker(self.backend.process_and_save_pointcloud, on_pcd_processed)

    def visualize_pointcloud(self, pcd):
        """在 GLViewWidget 中显示点云"""
        if pcd is None or not pcd.has_points():
            return

        try:
            # 1. 获取点数据 (N, 3)
            points = np.asarray(pcd.points)

            # 2. 获取颜色数据 (N, 3) 归一化到 0-1
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                # pyqtgraph 需要 RGBA，追加 alpha 通道
                if colors.max() > 1.0:
                    colors = colors / 255.0
                colors_rgba = np.hstack((colors, np.ones((colors.shape[0], 1))))
            else:
                # 默认为白色
                colors_rgba = (1, 1, 1, 1)

            # 3. 清理旧的点云 Item
            for item in self.pcd_widget.items:
                if isinstance(item, gl.GLScatterPlotItem) and item != self.pcd_widget.items[0]: # 保留 grid
                    self.pcd_widget.removeItem(item)

            # 4. 创建新的散点图 Item
            # size=2 是点的大小，pxMode=True 表示大小单位是像素
            sp = gl.GLScatterPlotItem(pos=points, color=colors_rgba, size=2, pxMode=True)
            self.pcd_widget.addItem(sp)

            # 5. 调整相机视角以聚焦中心
            center = points.mean(axis=0)
            self.pcd_widget.opts['center'] = pg.Vector(*center)

            self.log_text.append(f"Visualizing {len(points)} points.")

        except Exception as e:
            self.log_text.append(f"Error visualizing point cloud: {e}")

    def return_to_init(self):
        self.run_worker(self.backend.return_to_init)

    def closeEvent(self, event):
        # 退出时可以在此添加清理代码
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置 Qt 样式表 (Dark Theme)
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.Window, QColor(53, 53, 53))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(25, 25, 25))
    palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipBase, Qt.white)
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.BrightText, Qt.red)
    palette.setColor(palette.Link, QColor(42, 130, 218))
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = ModelInitializerUI()
    window.show()
    sys.exit(app.exec_())
