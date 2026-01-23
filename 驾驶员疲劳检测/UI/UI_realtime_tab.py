# ui/realtime_tab.py
"""
实时识别标签页（MediaPipe版本）
"""
import os
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QTextEdit, QFrame, QGridLayout, QProgressBar,
    QComboBox, QSpinBox, QCheckBox, QSizePolicy, 
    QFileDialog, QRadioButton, QButtonGroup, QLineEdit, QSlider
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap, QFont

from .UI_worker import RealtimeDetectionWorker
from .UI_styles import COLORS, FONTS, STATUS_STYLES


class RealtimeTab(QWidget):
    """实时识别标签页（MediaPipe版本）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 工作线程
        self.worker = None
        
        # UI状态
        self.is_detecting = False
        self.is_recording = False
        self.alarm_active = False
        
        # 摄像头设置
        self.camera_index = 0
        self.video_file_path = ""
        self.use_video_file = False
        self.confidence = 0.5  # 固定置信度
        
        # 初始化UI
        self.init_ui()
        
        # 创建数据目录
        self._create_data_dirs()
        
        # 定时器（用于更新UI）
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_statistics)
        self.update_timer.start(1000)  # 1秒更新一次
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title_label = QLabel("实时疲劳检测 (MediaPipe)")
        title_label.setObjectName("titleLabel")
        title_label.setFont(FONTS['title'])
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 视频显示区域
        self.video_frame = QFrame()
        self.video_frame.setObjectName("videoFrame")
        self.video_frame.setMinimumSize(320, 240)  # 减小最小尺寸，防止布局溢出
        
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(1, 1)  # 允许缩小
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("摄像头未开启")
        self.video_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 14px;")
        
        video_layout = QVBoxLayout(self.video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0) # 移除边距以最大化显示
        video_layout.addWidget(self.video_label)
        
        # 创建水平内容布局 (左侧视频，右侧状态)
        content_layout = QHBoxLayout()
        content_layout.addWidget(self.video_frame, 7) # 视频占70%
        
        # 右侧面板布局
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 状态信息面板
        status_group = QGroupBox("状态信息")
        status_layout = QGridLayout()
        
        # 疲劳状态
        status_layout.addWidget(QLabel("疲劳状态:"), 0, 0)
        self.status_label = QLabel("未检测")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setStyleSheet(STATUS_STYLES['inactive'])
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumWidth(100)
        status_layout.addWidget(self.status_label, 0, 1)
        
        # 报警状态
        status_layout.addWidget(QLabel("报警状态:"), 0, 2)
        self.alarm_label = QLabel("正常")
        self.alarm_label.setStyleSheet(f"background-color: {COLORS['normal']}; color: white; padding: 2px 6px; border-radius: 3px;")
        self.alarm_label.setAlignment(Qt.AlignCenter)
        self.alarm_label.setMinimumWidth(80)
        status_layout.addWidget(self.alarm_label, 0, 3)
        
        # EAR值
        status_layout.addWidget(QLabel("EAR值:"), 1, 0)
        self.ear_label = QLabel("0.000")
        self.ear_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        status_layout.addWidget(self.ear_label, 1, 1)
        
        # MAR值
        status_layout.addWidget(QLabel("MAR值:"), 1, 2)
        self.mar_label = QLabel("0.000")
        self.mar_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        status_layout.addWidget(self.mar_label, 1, 3)
        
        # PERCLOS值
        status_layout.addWidget(QLabel("PERCLOS:"), 2, 0)
        self.perclos_label = QLabel("0.0%")
        self.perclos_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        status_layout.addWidget(self.perclos_label, 2, 1)
        
        # FPS
        status_layout.addWidget(QLabel("处理速度:"), 2, 2)
        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        status_layout.addWidget(self.fps_label, 2, 3)
        
        # 录制状态
        status_layout.addWidget(QLabel("录制状态:"), 3, 0)
        self.recording_label = QLabel("未录制")
        self.recording_label.setStyleSheet(f"color: {COLORS['text']};")
        status_layout.addWidget(self.recording_label, 3, 1)
        
        # 人脸检测状态
        status_layout.addWidget(QLabel("人脸检测:"), 3, 2)
        self.face_detect_label = QLabel("未检测")
        self.face_detect_label.setStyleSheet(f"color: {COLORS['text']};")
        status_layout.addWidget(self.face_detect_label, 3, 3)
        
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        # 统计信息面板
        stats_group = QGroupBox("统计信息")
        stats_layout = QGridLayout()
        
        # 处理帧数
        stats_layout.addWidget(QLabel("处理帧数:"), 0, 0)
        self.frames_label = QLabel("0")
        self.frames_label.setStyleSheet(f"color: {COLORS['text']};")
        stats_layout.addWidget(self.frames_label, 0, 1)
        
        # 眨眼次数
        stats_layout.addWidget(QLabel("眨眼次数:"), 0, 2)
        self.blink_label = QLabel("0")
        self.blink_label.setStyleSheet(f"color: {COLORS['text']};")
        stats_layout.addWidget(self.blink_label, 0, 3)
        
        # 打哈欠次数
        stats_layout.addWidget(QLabel("打哈欠次数:"), 1, 0)
        self.yawn_label = QLabel("0")
        self.yawn_label.setStyleSheet(f"color: {COLORS['text']};")
        stats_layout.addWidget(self.yawn_label, 1, 1)
        
        # 疲劳帧比例
        stats_layout.addWidget(QLabel("疲劳帧比例:"), 1, 2)
        self.fatigue_ratio_label = QLabel("0.0%")
        self.fatigue_ratio_label.setStyleSheet(f"color: {COLORS['text']};")
        stats_layout.addWidget(self.fatigue_ratio_label, 1, 3)
        
        # 疲劳进度条
        stats_layout.addWidget(QLabel("疲劳程度:"), 2, 0)
        self.fatigue_progress = QProgressBar()
        self.fatigue_progress.setRange(0, 100)
        self.fatigue_progress.setValue(0)
        stats_layout.addWidget(self.fatigue_progress, 2, 1, 1, 3)
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        right_layout.addStretch()
        content_layout.addWidget(right_panel, 3) # 状态占30%
        main_layout.addLayout(content_layout, 1)
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout()
        
        # 视频源选择
        source_group = QGroupBox("视频源")
        source_layout = QVBoxLayout()
        
        # 单选按钮组
        self.source_btn_group = QButtonGroup(self)
        
        # 摄像头选项
        self.camera_radio = QRadioButton("摄像头")
        self.camera_radio.setChecked(True)
        self.camera_radio.toggled.connect(self.toggle_source_mode)
        self.source_btn_group.addButton(self.camera_radio)
        source_layout.addWidget(self.camera_radio)
        
        # 摄像头选择下拉框
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("默认摄像头", 0)
        self.camera_combo.addItem("摄像头1", 1)
        self.camera_combo.addItem("摄像头2", 2)
        self.camera_combo.currentIndexChanged.connect(self.update_camera_index)
        source_layout.addWidget(self.camera_combo)
        
        # 视频文件选项
        self.file_radio = QRadioButton("本地视频文件")
        self.file_radio.toggled.connect(self.toggle_source_mode)
        self.source_btn_group.addButton(self.file_radio)
        source_layout.addWidget(self.file_radio)
        
        # 文件选择区域 (默认隐藏)
        self.file_selection_widget = QWidget()
        file_layout = QHBoxLayout(self.file_selection_widget)
        file_layout.setContentsMargins(0, 0, 0, 0)
        
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("请选择视频文件...")
        self.file_path_input.setReadOnly(True)
        file_layout.addWidget(self.file_path_input)
        
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse_video_file)
        file_layout.addWidget(self.browse_btn)
        
        source_layout.addWidget(self.file_selection_widget)
        self.file_selection_widget.setVisible(False) # 初始隐藏
        
        # 播放速度控制 (仅视频文件模式显示)
        self.speed_widget = QWidget()
        speed_layout = QHBoxLayout(self.speed_widget)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        
        speed_layout.addWidget(QLabel("播放速度:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 30)  # 0.1x - 3.0x
        self.speed_slider.setValue(10)     # 1.0x
        self.speed_slider.setSingleStep(1)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_value_label = QLabel("1.0x")
        self.speed_value_label.setMinimumWidth(40)
        self.speed_value_label.setStyleSheet(f"color: {COLORS['text']};")
        speed_layout.addWidget(self.speed_value_label)
        
        source_layout.addWidget(self.speed_widget)
        self.speed_widget.setVisible(False)
        
        source_group.setLayout(source_layout)
        control_layout.addWidget(source_group)

        # 置信度设置
        settings_layout = QHBoxLayout()
        
        # 固定置信度显示
        settings_layout.addWidget(QLabel("置信度:"))
        confidence_label = QLabel("0.5 (固定)")
        confidence_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
        confidence_label.setMinimumWidth(80)
        settings_layout.addWidget(confidence_label)
        
        settings_layout.addStretch()
        control_layout.addLayout(settings_layout)
        
        # 按钮控制
        button_layout = QHBoxLayout()
        
        # 开始按钮
        self.start_button = QPushButton("开始检测")
        self.start_button.setObjectName("startButton")
        self.start_button.setMinimumHeight(40)
        self.start_button.setFont(FONTS['subheading'])
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)
        
        # 停止按钮
        self.stop_button = QPushButton("停止检测")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setFont(FONTS['subheading'])
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        # 录制按钮
        self.record_button = QPushButton("开始录制")
        self.record_button.setObjectName("recordButton")
        self.record_button.setMinimumHeight(40)
        self.record_button.setFont(FONTS['subheading'])
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        button_layout.addWidget(self.record_button)
        
        # 重置按钮
        self.reset_button = QPushButton("重置统计")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.setFont(FONTS['subheading'])
        self.reset_button.clicked.connect(self.reset_statistics)
        self.reset_button.setEnabled(False)
        button_layout.addWidget(self.reset_button)
        
        button_layout.addStretch()
        control_layout.addLayout(button_layout)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # 连接信号
        # self.camera_combo.currentIndexChanged.connect(self.update_camera_index) # 已在上面连接
        
        
        # 日志输出
        log_group = QGroupBox("日志输出")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
    
    def _create_data_dirs(self):
        """创建数据目录"""
        directories = [
            "data/recordings",
            "data/results",
            "models"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def update_camera_index(self, index):
        """更新摄像头索引"""
        self.camera_index = self.camera_combo.itemData(index)
        if self.camera_index is None:
            self.camera_index = 0
    
    def toggle_source_mode(self):
        """切换视频源模式"""
        is_file_mode = self.file_radio.isChecked()
        self.use_video_file = is_file_mode
        
        # 切换控件可见性
        self.camera_combo.setVisible(not is_file_mode)
        self.file_selection_widget.setVisible(is_file_mode)
        self.speed_widget.setVisible(is_file_mode)
        
    def update_speed(self, value):
        """更新播放速度"""
        speed = value / 10.0
        self.speed_value_label.setText(f"{speed:.1f}x")
        if self.worker and self.is_detecting:
            self.worker.set_speed(speed)

    def browse_video_file(self):
        """浏览视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.video_file_path = file_path
            self.file_path_input.setText(os.path.basename(file_path))

    def start_detection(self):
        """开始检测"""
        # 如果正在检测，则不执行
        if self.is_detecting:
            return
            
        # 检查文件模式下是否选择了文件
        if self.use_video_file and not self.video_file_path:
            # 这里简单处理，如果没有选择文件，就不启动
            self.video_label.setText("请先选择视频文件")
            return
        
        # 初始化工作线程
        self.worker = RealtimeDetectionWorker()
        
        # 连接信号
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.status_updated.connect(self.update_status)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.recording_started.connect(self.handle_recording_started)
        self.worker.recording_stopped.connect(self.handle_recording_stopped)
        self.worker.alarm_triggered.connect(self.handle_alarm_triggered)
        self.worker.video_finished.connect(self.stop_detection)
        
        # 根据模式传递参数：如果是文件模式，传递路径字符串；如果是摄像头模式，传递索引整数
        source = self.video_file_path if self.use_video_file else self.camera_index
        
        # 初始化检测器
        if not self.worker.initialize(camera_index=source, confidence=self.confidence):
            self.log_message("错误", "检测器初始化失败")
            return
        
        # 设置初始速度
        if self.use_video_file:
            self.update_speed(self.speed_slider.value())
        
        # 开始检测
        self.worker.start_detection()
        self.worker.start()  # 启动线程
        
        # 更新UI状态
        self.is_detecting = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        
        # 禁用设置控件
        self.camera_combo.setEnabled(False)
        self.camera_radio.setEnabled(False)
        self.file_radio.setEnabled(False)
        self.browse_btn.setEnabled(False)
        
        self.log_message("信息", f"开始实时疲劳检测 (源: {source}, 置信度: {self.confidence})")
    
    def stop_detection(self):
        """停止检测"""
        if not self.worker or not self.is_detecting:
            return
        
        # 提前更新标志位，防止后续信号处理
        self.is_detecting = False
        self.is_recording = False
        self.alarm_active = False
        
        # 断开信号连接，确保不会再有画面更新
        try:
            self.worker.frame_ready.disconnect(self.update_video_frame)
            self.worker.status_updated.disconnect(self.update_status)
        except Exception:
            pass
        
        # 停止工作线程
        self.worker.stop_detection()
        self.worker = None
        
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.record_button.setText("开始录制")
        
        # 启用设置控件
        self.camera_combo.setEnabled(True)
        self.camera_radio.setEnabled(True)
        self.file_radio.setEnabled(True)
        self.browse_btn.setEnabled(True)
        
        # 清空视频显示
        self.video_label.clear()  # 先清除Pixmap
        self.video_label.setText("检测已停止")
        
        # 更新报警状态
        self.alarm_label.setText("正常")
        self.alarm_label.setStyleSheet(f"background-color: {COLORS['normal']}; color: white; padding: 2px 6px; border-radius: 3px;")
        
        self.log_message("信息", "停止实时疲劳检测")
    
    def toggle_recording(self):
        """切换录制状态"""
        if not self.worker:
            return
        
        if not self.is_recording:
            self.worker.start_recording()
        else:
            self.worker.stop_recording()
    
    def reset_statistics(self):
        """重置统计信息"""
        if self.worker and self.worker.fatigue_tracker:
            self.worker.fatigue_tracker.reset()
            self.log_message("信息", "统计信息已重置")
    
    @Slot(np.ndarray)
    def update_video_frame(self, frame):
        """更新视频帧显示"""
        if not self.is_detecting:
            return

        try:
            # 转换图像格式 (BGR -> RGB)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 获取图像尺寸
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            
            # 创建QImage
            qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 创建QPixmap并显示
            pixmap = QPixmap.fromImage(qt_image)
            
            # 使用video_label的当前大小进行缩放，确保不超出范围
            target_size = self.video_label.size()
            if target_size.width() < 1 or target_size.height() < 1:
                target_size = self.video_frame.size()
                
            scaled_pixmap = pixmap.scaled(
                target_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"更新视频帧错误: {e}")
    
    @Slot(dict)
    def update_status(self, status):
        """更新状态信息"""
        # 疲劳状态
        fatigue_state = status.get('fatigue_state', '未知')
        self.status_label.setText(fatigue_state)
        
        # 根据状态设置颜色
        if fatigue_state == '正常':
            self.status_label.setStyleSheet(STATUS_STYLES['normal'])
        elif fatigue_state == '轻度疲劳':
            self.status_label.setStyleSheet(STATUS_STYLES['mild'])
        elif fatigue_state == '重度疲劳':
            self.status_label.setStyleSheet(STATUS_STYLES['severe'])
        else:
            self.status_label.setStyleSheet(STATUS_STYLES['inactive'])
        
        # EAR值
        ear = status.get('ear', 0.0)
        self.ear_label.setText(f"{ear:.3f}")
        
        # MAR值
        mar = status.get('mar', 0.0)
        self.mar_label.setText(f"{mar:.3f}")
        
        # PERCLOS值
        perclos = status.get('perclos', 0.0)
        self.perclos_label.setText(f"{perclos:.1%}")
        
        # FPS
        fps = status.get('fps', 0.0)
        self.fps_label.setText(f"{fps:.1f} FPS")
        
        # 人脸检测状态
        face_detected = status.get('face_detected', False)
        self.face_detect_label.setText("已检测" if face_detected else "未检测")
        
        # 更新疲劳进度条 (基于PERCLOS值映射: 0.0-0.5 -> 0-100%)
        # 0.25 (25%) 为轻度疲劳阈值 -> 对应进度条 50%
        # 0.50 (50%) 为重度疲劳阈值 -> 对应进度条 100%
        fatigue_value = min(100, int(perclos * 200))
        self.fatigue_progress.setValue(fatigue_value)
        
        # 设置进度条颜色
        if fatigue_value < 50:  # < 0.25 PERCLOS (正常)
            color = COLORS['normal']
        elif fatigue_value < 100: # 0.25 - 0.5 PERCLOS (轻度)
            color = COLORS['mild']
        else: # >= 0.5 PERCLOS (重度)
            color = COLORS['severe']
        
        self.fatigue_progress.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)
    
    def update_statistics(self):
        """更新统计信息"""
        if not self.worker or not self.is_detecting:
            return
        
        # 直接从worker获取统计信息
        stats = self.worker.get_statistics()
        
        # 处理帧数
        total_frames = stats.get('total_frames', 0)
        self.frames_label.setText(str(total_frames))
        
        # 眨眼次数和打哈欠次数
        blink_count = stats.get('blink_count', 0)
        yawn_count = stats.get('yawn_count', 0)
        self.blink_label.setText(str(blink_count))
        self.yawn_label.setText(str(yawn_count))
        
        # 疲劳比例
        fatigue_percentage = stats.get('fatigue_percentage', 0.0)
        self.fatigue_ratio_label.setText(f"{fatigue_percentage:.1f}%")
        
        # 录制状态
        if self.is_recording:
            duration = stats.get('recording_duration', 0)
            self.recording_label.setText(f"录制中 ({duration:.0f}秒)")
        else:
            self.recording_label.setText("未录制")
    
    @Slot(bool)
    def handle_alarm_triggered(self, active):
        """处理报警触发"""
        self.alarm_active = active
        
        if active:
            self.alarm_label.setText("报警中")
            self.alarm_label.setStyleSheet(f"background-color: {COLORS['severe']}; color: white; padding: 2px 6px; border-radius: 3px;")
            self.log_message("警告", "检测到重度疲劳，触发报警！")
        else:
            self.alarm_label.setText("正常")
            self.alarm_label.setStyleSheet(f"background-color: {COLORS['normal']}; color: white; padding: 2px 6px; border-radius: 3px;")
    
    @Slot(str)
    def handle_error(self, error_msg):
        """处理错误"""
        self.log_message("错误", error_msg)
    
    @Slot(str)
    def handle_recording_started(self, file_path):
        """处理录制开始"""
        self.is_recording = True
        self.record_button.setText("停止录制")
        self.record_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
            }}
            QPushButton:hover {{
                background-color: #c0392b;
            }}
        """)
        
        self.log_message("信息", f"开始录制: {Path(file_path).name}")
    
    @Slot(str)
    def handle_recording_stopped(self, message):
        """处理录制停止"""
        self.is_recording = False
        self.record_button.setText("开始录制")
        self.record_button.setStyleSheet("")  # 恢复默认样式
        
        self.log_message("信息", message)
    
    def log_message(self, level, message):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        
        self.log_text.append(formatted_message)
        
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.worker and self.is_detecting:
            self.stop_detection()
        
        self.update_timer.stop()
        super().closeEvent(event)
        
        self.update_timer.stop()
        super().closeEvent(event)