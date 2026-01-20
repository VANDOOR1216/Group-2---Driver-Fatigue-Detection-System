# ui/main_window.py
"""
主窗口（MediaPipe版本）
"""
import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
    QStatusBar, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from .UI_realtime_tab import RealtimeTab
from .UI_upload_tab import UploadTab
from .UI_styles import apply_dark_theme, COLORS, FONTS


class MainWindow(QMainWindow):
    """主窗口（MediaPipe版本）"""
    
    def __init__(self):
        super().__init__()
        
        # 窗口标题
        self.setWindowTitle("驾驶员疲劳检测系统 - MediaPipe版本")
        
        # 窗口尺寸
        self.resize(1200, 800)
        self.setMinimumSize(1000, 700)
        
        # 初始化UI
        self.init_ui()
        
        # 应用样式
        apply_dark_theme(QApplication.instance())
    
    def init_ui(self):
        """初始化UI"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 实时识别标签页
        self.realtime_tab = RealtimeTab()
        self.tab_widget.addTab(self.realtime_tab, "实时识别")
        
        # 视频上传标签页
        self.upload_tab = UploadTab()
        self.tab_widget.addTab(self.upload_tab, "视频上传")
        
        main_layout.addWidget(self.tab_widget)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 状态栏消息
        self.status_bar.showMessage("就绪 - 使用MediaPipe进行疲劳检测")
        
        # 创建数据目录
        self._create_data_dirs()
    
    def _create_data_dirs(self):
        """创建数据目录"""
        directories = [
            "data/recordings",
            "data/results",
            "data/uploads",
            "models"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 确认是否退出
        reply = QMessageBox.question(
            self, '确认退出',
            '确定要退出驾驶员疲劳检测系统吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 停止所有检测
            if hasattr(self.realtime_tab, 'stop_detection'):
                self.realtime_tab.stop_detection()
            
            if hasattr(self.upload_tab, 'stop_processing'):
                self.upload_tab.stop_processing()
            
            event.accept()
        else:
            event.ignore()
    
    def update_status(self, message):
        """更新状态栏消息"""
        self.status_bar.showMessage(message)