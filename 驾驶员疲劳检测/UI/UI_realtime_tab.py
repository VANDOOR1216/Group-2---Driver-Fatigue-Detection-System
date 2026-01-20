# ui/realtime_tab.py
"""
实时识别标签页（MediaPipe版本）
"""
import os
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QTextEdit, QFrame, QGridLayout, QProgressBar,
    QComboBox, QSpinBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont

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
        
        # 摄像头设置（固定置信度为0.5）
        self.camera_index = 0
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
        self.video_frame.setMinimumSize(640, 480)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("摄像头未开启")
        self.video_label.setStyleSheet(f"color: {COLORS['text']}; font-size: 14px;")
        
        video_layout = QVBoxLayout(self.video_frame)
        video_layout.addWidget(self.video_label)
        
        main_layout.addWidget(self.video_frame, 1)
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout()
        
        # 摄像头设置
        settings_layout = QHBoxLayout()
        
        # 摄像头选择
        settings_layout.addWidget(QLabel("摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("默认摄像头", 0)
        self.camera_combo.addItem("摄像头1", 1)
        self.camera_combo.addItem("摄像头2", 2)
        settings_layout.addWidget(self.camera_combo)
        
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
        self.camera_combo.currentIndexChanged.connect(self.update_camera_index)
        
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
        main_layout.addWidget(status_group)
        
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
        main_layout.addWidget(stats_group)
        
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
    
    def start_detection(self):
        """开始检测"""
        # 获取当前设置
        camera_index = self.camera_index
        
        # 初始化工作线程，使用固定置信度0.5
        self.worker = RealtimeDetectionWorker()
        
        # 连接信号
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.status_updated.connect(self.update_status)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.recording_started.connect(self.handle_recording_started)
        self.worker.recording_stopped.connect(self.handle_recording_stopped)
        self.worker.alarm_triggered.connect(self.handle_alarm_triggered)
        
        # 初始化检测器，使用固定置信度0.5
        if not self.worker.initialize(camera_index=camera_index, confidence=0.5):
            self.log_message("错误", "检测器初始化失败")
            return
        
        # 开始检测
        self.worker.start_detection()
        self.worker.start()  # 启动线程
        
        # 更新UI状态
        self.is_detecting = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.camera_combo.setEnabled(False)
        
        self.log_message("信息", f"开始实时疲劳检测 (摄像头: {camera_index}, 置信度: 0.5)")
    
    def stop_detection(self):
        """停止检测"""
        if not self.worker or not self.is_detecting:
            return
        
        # 停止工作线程
        self.worker.stop_detection()
        self.worker = None
        
        # 更新UI状态
        self.is_detecting = False
        self.is_recording = False
        self.alarm_active = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.record_button.setText("开始录制")
        self.camera_combo.setEnabled(True)
        
        # 清空视频显示
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
    
    @pyqtSlot(np.ndarray)
    def update_video_frame(self, frame):
        """更新视频帧显示"""
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
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"更新视频帧错误: {e}")
    
    @pyqtSlot(dict)
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
        
        # 更新疲劳进度条
        fatigue_value = 0
        if fatigue_state == '轻度疲劳':
            fatigue_value = 50
        elif fatigue_state == '重度疲劳':
            fatigue_value = 100
        
        self.fatigue_progress.setValue(fatigue_value)
        
        # 设置进度条颜色
        if fatigue_value < 30:
            color = COLORS['normal']
        elif fatigue_value < 70:
            color = COLORS['mild']
        else:
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
    
    @pyqtSlot(bool)
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
    
    @pyqtSlot(str)
    def handle_error(self, error_msg):
        """处理错误"""
        self.log_message("错误", error_msg)
    
    @pyqtSlot(str)
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
    
    @pyqtSlot(str)
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