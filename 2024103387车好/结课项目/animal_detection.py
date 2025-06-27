import sys
import os
import cv2
import torch
import numpy as np
import time
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QWidget, QFileDialog, QSlider, QComboBox, QGroupBox, QMessageBox,
                             QStatusBar, QAction, QSplitter, QProgressBar, QCheckBox, QTabWidget)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QIcon
from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, pyqtSlot

model_path = 'yolo11n.pt'
model = torch.hub.load('ultralytics/yolov11', 'custom', path=model_path)
model.eval()

# 动物类别名称
class_names = model.names


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self, source=0, detect_enabled=True, conf_threshold=0.5):
        super().__init__()
        self.source = source
        self.detect_enabled = detect_enabled
        self.conf_threshold = conf_threshold
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.detect_enabled:
                    results = model(frame)
                    frame = self.draw_detections(frame, results)
                self.change_pixmap_signal.emit(frame)
            else:
                break
            time.sleep(0.03)  # 控制帧率

        cap.release()
        self.finished.emit()

    def stop(self):
        self.running = False

    def draw_detections(self, frame, results):
        # 复制原始帧以避免修改原始数据
        frame_copy = frame.copy()

        # 获取检测结果
        detections = results.xyxy[0].cpu().numpy()

        # 设置颜色和字体
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128)]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 绘制检测框和标签
        for det in detections:
            x1, y1, x2, y2, conf, cls = map(float, det)
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)

            # 随机选择颜色
            color = colors[cls % len(colors)]

            # 绘制边界框
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            # 创建标签文本
            label = f"{class_names[cls]}: {conf:.2f}"

            # 获取文本大小
            (text_width, text_height), _ = cv2.getTextSize(label, font, 0.5, 1)

            # 绘制标签背景
            cv2.rectangle(frame_copy, (x1, y1 - text_height - 10),
                          (x1 + text_width, y1), color, -1)

            # 绘制文本
            cv2.putText(frame_copy, label, (x1, y1 - 5), font, 0.5, (255, 255, 255), 1)

        return frame_copy


class AnimalDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv5的动物识别系统")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(QIcon("icon.png"))

        # 初始化变量
        self.image_path = None
        self.video_path = None
        self.video_thread = None
        self.camera_active = False
        self.detection_enabled = True
        self.conf_threshold = 0.5

        # 创建UI
        self.init_ui()

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def init_ui(self):
        # 创建主部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # 创建选项卡
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # 图像检测选项卡
        image_tab = QWidget()
        tab_widget.addTab(image_tab, "图像检测")
        self.setup_image_tab(image_tab)

        # 视频检测选项卡
        video_tab = QWidget()
        tab_widget.addTab(video_tab, "视频检测")
        self.setup_video_tab(video_tab)

        # 实时检测选项卡
        realtime_tab = QWidget()
        tab_widget.addTab(realtime_tab, "实时检测")
        self.setup_realtime_tab(realtime_tab)

        # 创建菜单栏
        self.create_menus()

    def create_menus(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        open_image_action = QAction('打开图像', self)
        open_image_action.triggered.connect(self.open_image)
        file_menu.addAction(open_image_action)

        open_video_action = QAction('打开视频', self)
        open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(open_video_action)

        save_action = QAction('保存结果', self)
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)

        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = menubar.addMenu('视图')

        zoom_in_action = QAction('放大', self)
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction('缩小', self)
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

        reset_zoom_action = QAction('重置缩放', self)
        reset_zoom_action.triggered.connect(self.reset_zoom)
        view_menu.addAction(reset_zoom_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助')

        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_image_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # 创建水平布局用于按钮
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        # 打开图像按钮
        self.open_image_btn = QPushButton("打开图像")
        self.open_image_btn.clicked.connect(self.open_image)
        self.open_image_btn.setFixedSize(120, 40)
        button_layout.addWidget(self.open_image_btn)

        # 检测按钮
        self.detect_btn = QPushButton("检测图像")
        self.detect_btn.clicked.connect(self.detect_image)
        self.detect_btn.setFixedSize(120, 40)
        button_layout.addWidget(self.detect_btn)

        # 保存结果按钮
        self.save_image_btn = QPushButton("保存结果")
        self.save_image_btn.clicked.connect(self.save_image_result)
        self.save_image_btn.setFixedSize(120, 40)
        button_layout.addWidget(self.save_image_btn)

        # 置信度阈值滑块
        conf_layout = QHBoxLayout()
        layout.addLayout(conf_layout)

        conf_label = QLabel("置信度阈值:")
        conf_layout.addWidget(conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(int(self.conf_threshold * 100))
        self.conf_slider.valueChanged.connect(self.conf_slider_changed)
        conf_layout.addWidget(self.conf_slider)

        self.conf_value_label = QLabel(f"{self.conf_threshold:.2f}")
        self.conf_value_label.setFixedWidth(50)
        conf_layout.addWidget(self.conf_value_label)

        # 图像显示区域
        image_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(image_splitter, 1)

        # 原始图像区域
        self.original_image_box = QGroupBox("原始图像")
        original_layout = QVBoxLayout()
        self.original_image_box.setLayout(original_layout)

        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: #2C3E50;")
        original_layout.addWidget(self.original_label)

        # 检测结果区域
        self.result_image_box = QGroupBox("检测结果")
        result_layout = QVBoxLayout()
        self.result_image_box.setLayout(result_layout)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background-color: #2C3E50;")
        result_layout.addWidget(self.result_label)

        image_splitter.addWidget(self.original_image_box)
        image_splitter.addWidget(self.result_image_box)

        # 检测信息区域
        info_layout = QHBoxLayout()
        layout.addLayout(info_layout)

        # 检测统计
        self.stats_label = QLabel("检测统计: 无")
        self.stats_label.setStyleSheet("font-weight: bold; color: #3498DB;")
        info_layout.addWidget(self.stats_label)

        # 检测时间
        self.time_label = QLabel("检测时间: -")
        self.time_label.setStyleSheet("font-weight: bold; color: #3498DB;")
        info_layout.addWidget(self.time_label)

        # 设置默认占位图像
        self.set_placeholder_images()

    def setup_video_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # 按钮布局
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        # 打开视频按钮
        self.open_video_btn = QPushButton("打开视频")
        self.open_video_btn.clicked.connect(self.open_video)
        self.open_video_btn.setFixedSize(120, 40)
        button_layout.addWidget(self.open_video_btn)

        # 播放/暂停按钮
        self.play_video_btn = QPushButton("播放")
        self.play_video_btn.clicked.connect(self.toggle_video_playback)
        self.play_video_btn.setFixedSize(120, 40)
        self.play_video_btn.setEnabled(False)
        button_layout.addWidget(self.play_video_btn)

        # 停止按钮
        self.stop_video_btn = QPushButton("停止")
        self.stop_video_btn.clicked.connect(self.stop_video)
        self.stop_video_btn.setFixedSize(120, 40)
        self.stop_video_btn.setEnabled(False)
        button_layout.addWidget(self.stop_video_btn)

        # 保存视频按钮
        self.save_video_btn = QPushButton("保存结果")
        self.save_video_btn.clicked.connect(self.save_video_result)
        self.save_video_btn.setFixedSize(120, 40)
        self.save_video_btn.setEnabled(False)
        button_layout.addWidget(self.save_video_btn)

        # 控制布局
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        # 检测开关
        self.detect_checkbox = QCheckBox("启用检测")
        self.detect_checkbox.setChecked(True)
        self.detect_checkbox.stateChanged.connect(self.toggle_detection)
        control_layout.addWidget(self.detect_checkbox)

        # 置信度阈值滑块
        conf_layout = QHBoxLayout()
        control_layout.addLayout(conf_layout)

        conf_label = QLabel("置信度阈值:")
        conf_layout.addWidget(conf_label)

        self.video_conf_slider = QSlider(Qt.Horizontal)
        self.video_conf_slider.setRange(10, 95)
        self.video_conf_slider.setValue(int(self.conf_threshold * 100))
        self.video_conf_slider.valueChanged.connect(self.video_conf_slider_changed)
        conf_layout.addWidget(self.video_conf_slider)

        self.video_conf_value_label = QLabel(f"{self.conf_threshold:.2f}")
        self.video_conf_value_label.setFixedWidth(50)
        conf_layout.addWidget(self.video_conf_value_label)

        # 视频显示区域
        video_layout = QHBoxLayout()
        layout.addLayout(video_layout, 1)

        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #2C3E50;")
        video_layout.addWidget(self.video_label)

        # 进度条
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        layout.addWidget(self.video_progress)

        # 视频信息
        info_layout = QHBoxLayout()
        layout.addLayout(info_layout)

        self.video_stats_label = QLabel("检测统计: 无")
        self.video_stats_label.setStyleSheet("font-weight: bold; color: #3498DB;")
        info_layout.addWidget(self.video_stats_label)

        self.video_time_label = QLabel("帧率: -")
        self.video_time_label.setStyleSheet("font-weight: bold; color: #3498DB;")
        info_layout.addWidget(self.video_time_label)

        # 设置默认占位图像
        self.set_video_placeholder()

    def setup_realtime_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # 按钮布局
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        # 摄像头选择
        camera_layout = QHBoxLayout()
        button_layout.addLayout(camera_layout)

        camera_label = QLabel("选择摄像头:")
        camera_layout.addWidget(camera_label)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["摄像头0", "摄像头1", "摄像头2"])
        camera_layout.addWidget(self.camera_combo)

        # 开始摄像头按钮
        self.start_camera_btn = QPushButton("开始摄像头")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.start_camera_btn.setFixedSize(150, 40)
        button_layout.addWidget(self.start_camera_btn)

        # 停止摄像头按钮
        self.stop_camera_btn = QPushButton("停止摄像头")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setFixedSize(150, 40)
        self.stop_camera_btn.setEnabled(False)
        button_layout.addWidget(self.stop_camera_btn)

        # 保存快照按钮
        self.snapshot_btn = QPushButton("保存快照")
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        self.snapshot_btn.setFixedSize(150, 40)
        self.snapshot_btn.setEnabled(False)
        button_layout.addWidget(self.snapshot_btn)

        # 控制布局
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        # 检测开关
        self.realtime_detect_checkbox = QCheckBox("启用检测")
        self.realtime_detect_checkbox.setChecked(True)
        self.realtime_detect_checkbox.stateChanged.connect(self.toggle_realtime_detection)
        control_layout.addWidget(self.realtime_detect_checkbox)

        # 置信度阈值滑块
        conf_layout = QHBoxLayout()
        control_layout.addLayout(conf_layout)

        conf_label = QLabel("置信度阈值:")
        conf_layout.addWidget(conf_label)

        self.realtime_conf_slider = QSlider(Qt.Horizontal)
        self.realtime_conf_slider.setRange(10, 95)
        self.realtime_conf_slider.setValue(int(self.conf_threshold * 100))
        self.realtime_conf_slider.valueChanged.connect(self.realtime_conf_slider_changed)
        conf_layout.addWidget(self.realtime_conf_slider)

        self.realtime_conf_value_label = QLabel(f"{self.conf_threshold:.2f}")
        self.realtime_conf_value_label.setFixedWidth(50)
        conf_layout.addWidget(self.realtime_conf_value_label)

        # 实时显示区域
        camera_layout = QHBoxLayout()
        layout.addLayout(camera_layout, 1)

        # 摄像头显示标签
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #2C3E50;")
        camera_layout.addWidget(self.camera_label)

        # 实时信息
        info_layout = QHBoxLayout()
        layout.addLayout(info_layout)

        self.camera_stats_label = QLabel("检测统计: 无")
        self.camera_stats_label.setStyleSheet("font-weight: bold; color: #3498DB;")
        info_layout.addWidget(self.camera_stats_label)

        self.camera_fps_label = QLabel("帧率: -")
        self.camera_fps_label.setStyleSheet("font-weight: bold; color: #3498DB;")
        info_layout.addWidget(self.camera_fps_label)

        # 设置默认占位图像
        self.set_camera_placeholder()

    def set_placeholder_images(self):
        # 创建占位图像
        pixmap = QPixmap(800, 500)
        pixmap.fill(QColor("#2C3E50"))

        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 24))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "打开图像进行检测")
        painter.end()

        self.original_label.setPixmap(pixmap)
        self.result_label.setPixmap(pixmap)

    def set_video_placeholder(self):
        # 创建视频占位图像
        pixmap = QPixmap(800, 500)
        pixmap.fill(QColor("#2C3E50"))

        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 24))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "打开视频进行检测")
        painter.end()

        self.video_label.setPixmap(pixmap)

    def set_camera_placeholder(self):
        # 创建摄像头占位图像
        pixmap = QPixmap(800, 500)
        pixmap.fill(QColor("#2C3E50"))

        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 24))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "点击'开始摄像头'进行实时检测")
        painter.end()

        self.camera_label.setPixmap(pixmap)

    def open_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)",
            options=options
        )

        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.original_label.setPixmap(pixmap.scaled(
                self.original_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.result_label.clear()
            self.status_bar.showMessage(f"已加载图像: {Path(file_path).name}")
            self.detect_btn.setEnabled(True)
            self.save_image_btn.setEnabled(False)

    def detect_image(self):
        if not self.image_path:
            QMessageBox.warning(self, "警告", "请先打开一张图像")
            return

        self.status_bar.showMessage("正在检测图像...")
        QApplication.processEvents()  # 更新UI

        # 读取图像
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 记录开始时间
        start_time = time.time()

        # 进行检测
        results = model(image)

        # 记录结束时间
        end_time = time.time()
        detection_time = end_time - start_time

        # 获取检测结果
        detections = results.xyxy[0].cpu().numpy()

        # 统计检测结果
        stats = {}
        for det in detections:
            _, _, _, _, conf, cls = det
            if conf < self.conf_threshold:
                continue
            class_name = class_names[int(cls)]
            stats[class_name] = stats.get(class_name, 0) + 1

        # 绘制检测结果
        detected_image = results.render()[0]

        # 转换为QPixmap
        height, width, channel = detected_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(detected_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 显示结果
        self.result_label.setPixmap(pixmap.scaled(
            self.result_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

        # 更新统计信息
        stats_text = "检测统计: "
        if stats:
            stats_text += ", ".join([f"{k}: {v}" for k, v in stats.items()])
        else:
            stats_text += "未检测到目标"

        self.stats_label.setText(stats_text)
        self.time_label.setText(f"检测时间: {detection_time:.3f}秒")
        self.status_bar.showMessage(f"检测完成! 耗时: {detection_time:.3f}秒")
        self.save_image_btn.setEnabled(True)

    def save_image_result(self):
        if self.result_label.pixmap() is None:
            QMessageBox.warning(self, "警告", "没有检测结果可保存")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", "",
            "JPEG图像 (*.jpg);;PNG图像 (*.png);;所有文件 (*)",
            options=options
        )

        if file_path:
            self.result_label.pixmap().save(file_path)
            self.status_bar.showMessage(f"结果已保存到: {file_path}")

    def open_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开视频", "",
            "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)",
            options=options
        )

        if file_path:
            self.video_path = file_path
            self.status_bar.showMessage(f"已加载视频: {Path(file_path).name}")
            self.play_video_btn.setEnabled(True)
            self.stop_video_btn.setEnabled(True)
            self.save_video_btn.setEnabled(True)

            # 初始化视频线程
            self.video_thread = VideoThread(file_path, self.detect_checkbox.isChecked(), self.conf_threshold)
            self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
            self.video_thread.finished.connect(self.video_finished)

            # 获取视频信息
            cap = cv2.VideoCapture(file_path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            self.video_progress.setValue(0)
            self.video_stats_label.setText("检测统计: 等待开始...")
            self.video_time_label.setText(f"帧率: {self.fps:.1f} FPS")

    def toggle_video_playback(self):
        if not self.video_thread:
            return

        if self.play_video_btn.text() == "播放":
            self.video_thread.start()
            self.play_video_btn.setText("暂停")
            self.status_bar.showMessage("视频播放中...")
        else:
            self.video_thread.stop()
            self.play_video_btn.setText("播放")
            self.status_bar.showMessage("视频已暂停")

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            self.play_video_btn.setText("播放")
            self.play_video_btn.setEnabled(False)
            self.stop_video_btn.setEnabled(False)
            self.status_bar.showMessage("视频已停止")

    def video_finished(self):
        self.play_video_btn.setText("播放")
        self.play_video_btn.setEnabled(True)
        self.status_bar.showMessage("视频播放完成")

    @pyqtSlot(np.ndarray)
    def update_video_frame(self, frame):
        # 转换为QPixmap
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)

        # 更新视频标签
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

        # 更新进度条
        if hasattr(self, 'total_frames') and self.total_frames > 0:
            current_frame = self.video_thread.cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = int((current_frame / self.total_frames) * 100)
            self.video_progress.setValue(progress)

    def save_video_result(self):
        # 在实际应用中，这里需要实现视频保存功能
        QMessageBox.information(self, "保存视频", "视频保存功能将在完整版本中实现")
        self.status_bar.showMessage("视频保存功能将在完整版本中实现")

    def start_camera(self):
        if self.camera_active:
            return

        camera_index = self.camera_combo.currentIndex()
        self.status_bar.showMessage(f"正在启动摄像头 {camera_index}...")

        # 初始化摄像头线程
        self.camera_thread = VideoThread(camera_index, self.realtime_detect_checkbox.isChecked(), self.conf_threshold)
        self.camera_thread.change_pixmap_signal.connect(self.update_camera_frame)
        self.camera_thread.start()

        self.camera_active = True
        self.start_camera_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(True)
        self.snapshot_btn.setEnabled(True)
        self.status_bar.showMessage(f"摄像头 {camera_index} 已启动")

    def stop_camera(self):
        if not self.camera_active:
            return

        self.camera_thread.stop()
        self.camera_thread = None
        self.camera_active = False
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.snapshot_btn.setEnabled(False)
        self.status_bar.showMessage("摄像头已停止")

    @pyqtSlot(np.ndarray)
    def update_camera_frame(self, frame):
        # 转换为QPixmap
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)

        # 更新摄像头标签
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def save_snapshot(self):
        if not self.camera_active or self.camera_label.pixmap() is None:
            QMessageBox.warning(self, "警告", "没有图像可保存")
            return

        options = QFileDialog.Options()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存快照", f"snapshot_{timestamp}.jpg",
            "JPEG图像 (*.jpg);;PNG图像 (*.png);;所有文件 (*)",
            options=options
        )

        if file_path:
            self.camera_label.pixmap().save(file_path)
            self.status_bar.showMessage(f"快照已保存到: {file_path}")

    def toggle_detection(self, state):
        self.detection_enabled = state == Qt.Checked
        if self.video_thread:
            self.video_thread.detect_enabled = self.detection_enabled
        self.status_bar.showMessage(f"检测已 {'启用' if self.detection_enabled else '禁用'}")

    def toggle_realtime_detection(self, state):
        self.detection_enabled = state == Qt.Checked
        if self.camera_thread:
            self.camera_thread.detect_enabled = self.detection_enabled
        self.status_bar.showMessage(f"实时检测已 {'启用' if self.detection_enabled else '禁用'}")

    def conf_slider_changed(self, value):
        self.conf_threshold = value / 100.0
        self.conf_value_label.setText(f"{self.conf_threshold:.2f}")

    def video_conf_slider_changed(self, value):
        self.conf_threshold = value / 100.0
        self.video_conf_value_label.setText(f"{self.conf_threshold:.2f}")
        if self.video_thread:
            self.video_thread.conf_threshold = self.conf_threshold

    def realtime_conf_slider_changed(self, value):
        self.conf_threshold = value / 100.0
        self.realtime_conf_value_label.setText(f"{self.conf_threshold:.2f}")
        if self.camera_thread:
            self.camera_thread.conf_threshold = self.conf_threshold

    def zoom_in(self):
        # 在实际应用中，这里需要实现放大功能
        self.status_bar.showMessage("放大功能将在完整版本中实现")

    def zoom_out(self):
        # 在实际应用中，这里需要实现缩小功能
        self.status_bar.showMessage("缩小功能将在完整版本中实现")

    def reset_zoom(self):
        # 在实际应用中，这里需要实现重置缩放功能
        self.status_bar.showMessage("重置缩放功能将在完整版本中实现")

    def save_result(self):
        # 在实际应用中，这里需要实现保存功能
        self.status_bar.showMessage("保存功能将在完整版本中实现")

    def show_about(self):
        about_text = """
        <h2>基于YOLOv5的动物识别系统</h2>
        <p>版本: 1.0</p>
        <p>开发日期: 2023年10月</p>
        <p>本系统使用YOLOv5模型进行动物检测，支持图像、视频和实时摄像头检测。</p>
        <p>主要功能:</p>
        <ul>
            <li>图像检测与结果保存</li>
            <li>视频文件检测</li>
            <li>实时摄像头检测</li>
            <li>置信度阈值调整</li>
            <li>检测结果统计</li>
        </ul>
        <p>© 2023 动物识别系统开发团队</p>
        """
        QMessageBox.about(self, "关于", about_text)

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        if self.video_thread:
            self.video_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 设置应用样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #34495E;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #3498DB;
            border-radius: 5px;
            margin-top: 1ex;
            color: #ECF0F1;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
        }
        QLabel {
            color: #ECF0F1;
        }
        QPushButton {
            background-color: #3498DB;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 5px 10px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #2980B9;
        }
        QPushButton:disabled {
            background-color: #7FB3D5;
        }
        QSlider::groove:horizontal {
            background: #7F8C8D;
            height: 5px;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #3498DB;
            border: 1px solid #2C3E50;
            width: 15px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QProgressBar {
            border: 1px solid #34495E;
            border-radius: 3px;
            text-align: center;
            background: #2C3E50;
            color: #ECF0F1;
        }
        QProgressBar::chunk {
            background-color: #3498DB;
            width: 10px;
        }
        QTabWidget::pane {
            border: 1px solid #3498DB;
            background: #2C3E50;
        }
        QTabBar::tab {
            background: #34495E;
            color: #ECF0F1;
            padding: 8px 20px;
            border: 1px solid #3498DB;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #3498DB;
            color: white;
        }
    """)

    window = AnimalDetectionApp()
    window.show()
    sys.exit(app.exec_())
