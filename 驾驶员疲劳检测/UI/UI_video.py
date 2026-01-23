# ui/upload_tab.py
"""
视频上传标签页
"""
import os
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QTextEdit, QFrame, QGridLayout, QProgressBar,
    QFileDialog, QListWidget, QListWidgetItem, QSplitter,
    QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QIcon

from .UI_styles import COLORS, FONTS


class VideoProcessingWorker(QThread):
    """视频处理工作线程"""
    
    # 信号定义
    progress_updated = Signal(int)    # 进度更新
    result_ready = Signal(dict)       # 结果就绪
    error_occurred = Signal(str)      # 错误信号
    finished = Signal()               # 完成信号
    
    def __init__(self, video_path, model_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.running = True
        
    def run(self):
        """运行视频处理"""
        try:
            import cv2
            import numpy as np
            import json
            import sys
            from pathlib import Path
            
            # 添加父目录到Python路径
            sys.path.append(str(Path(__file__).parent.parent.parent))
            
            from detectors.face_detector import FaceDetector
            from detectors.landmark_detector import LandmarkDetector
            from features.ear_calculator import EARCalculator
            from features.mar_calculator import MARCalculator
            from fatigue.state_tracker import FatigueTracker, FatigueState
            
            # 检查文件
            if not Path(self.video_path).exists():
                self.error_occurred.emit(f"视频文件不存在: {self.video_path}")
                return
            
            # 检查模型文件
            if not Path(self.model_path).exists():
                self.error_occurred.emit(f"模型文件不存在: {self.model_path}")
                return
            
            # 初始化检测器
            face_detector = FaceDetector(method='dlib')
            landmark_detector = LandmarkDetector(self.model_path)
            ear_calculator = EARCalculator()
            mar_calculator = MARCalculator()
            fatigue_tracker = FatigueTracker()
            
            # 打开视频文件
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit("无法打开视频文件")
                return
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if total_frames == 0:
                self.error_occurred.emit("视频文件为空或格式不支持")
                return
            
            # 处理视频
            frame_count = 0
            results = []
            fatigue_states = []
            ear_values = []
            mar_values = []
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 更新进度
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                self.progress_updated.emit(progress)
                
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 人脸检测
                faces = face_detector.detect(gray)
                
                if len(faces) > 0:
                    # 取面积最大的人脸
                    faces = sorted(faces, 
                                  key=lambda rect: (rect.right()-rect.left()) * (rect.bottom()-rect.top()),
                                  reverse=True)
                    main_face = faces[0]
                    
                    # 关键点检测
                    shape = landmark_detector.detect(gray, main_face)
                    
                    if shape is not None:
                        # 计算EAR
                        left_ear, right_ear, avg_ear = ear_calculator.calculate(shape)
                        ear_values.append(avg_ear)
                        
                        # 计算MAR
                        mar = mar_calculator.calculate(shape)
                        mar_values.append(mar)
                        
                        # 更新疲劳状态
                        state, stats = fatigue_tracker.update(avg_ear, mar)
                        fatigue_states.append(state.value)
                        
                        # 记录结果
                        results.append({
                            'frame': frame_count,
                            'ear': avg_ear,
                            'mar': mar,
                            'fatigue_state': state.value,
                            'perclos': stats['perclos']
                        })
                
                # 每处理10帧检查一次是否停止
                if frame_count % 10 == 0 and not self.running:
                    break
            
            # 释放视频
            cap.release()
            
            # 准备结果
            if results:
                # 统计信息
                normal_count = fatigue_states.count('正常')
                mild_count = fatigue_states.count('轻度疲劳')
                severe_count = fatigue_states.count('重度疲劳')
                total_fatigue = mild_count + severe_count
                
                avg_ear = np.mean(ear_values) if ear_values else 0
                avg_mar = np.mean(mar_values) if mar_values else 0
                
                result = {
                    'video_path': self.video_path,
                    'total_frames': total_frames,
                    'processed_frames': frame_count,
                    'normal_frames': normal_count,
                    'mild_fatigue_frames': mild_count,
                    'severe_fatigue_frames': severe_count,
                    'fatigue_percentage': (total_fatigue / len(results)) * 100 if results else 0,
                    'avg_ear': avg_ear,
                    'avg_mar': avg_mar,
                    'results': results[:1000],  # 只保留前1000帧结果，避免数据过大
                    'processing_time': datetime.now().isoformat()
                }
                
                self.result_ready.emit(result)
            
            self.finished.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"处理错误: {str(e)}")
    
    def stop(self):
        """停止处理"""
        self.running = False


class UploadTab(QWidget):
    """视频上传标签页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 模型文件路径
        self.model_path = "models/shape_predictor_68_face_landmarks.dat"
        
        # 工作线程
        self.worker = None
        
        # 视频文件列表
        self.video_files = []
        
        # 初始化UI
        self.init_ui()
        
        # 创建数据目录
        self._create_data_dirs()
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title_label = QLabel("视频上传与分析")
        title_label.setObjectName("titleLabel")
        title_label.setFont(FONTS['title'])
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板 - 视频选择和上传
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 上传区域
        upload_group = QGroupBox("视频上传")
        upload_layout = QVBoxLayout()
        
        # 上传按钮
        self.upload_button = QPushButton("选择视频文件")
        self.upload_button.setObjectName("uploadButton")
        self.upload_button.setMinimumHeight(40)
        self.upload_button.setFont(FONTS['subheading'])
        self.upload_button.clicked.connect(self.select_video_file)
        upload_layout.addWidget(self.upload_button)
        
        # 视频文件列表
        self.video_list = QListWidget()
        self.video_list.setMinimumWidth(300)
        self.video_list.itemDoubleClicked.connect(self.process_selected_video)
        upload_layout.addWidget(QLabel("已上传的视频文件:"))
        upload_layout.addWidget(self.video_list)
        
        upload_group.setLayout(upload_layout)
        left_layout.addWidget(upload_group)
        
        # 处理控制
        control_group = QGroupBox("处理控制")
        control_layout = QVBoxLayout()
        
        # 处理按钮
        self.process_button = QPushButton("开始处理")
        self.process_button.setObjectName("startButton")
        self.process_button.setMinimumHeight(40)
        self.process_button.setFont(FONTS['subheading'])
        self.process_button.clicked.connect(self.process_video)
        self.process_button.setEnabled(False)
        control_layout.addWidget(self.process_button)
        
        # 停止按钮
        self.stop_button = QPushButton("停止处理")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setFont(FONTS['subheading'])
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        # 右侧面板 - 结果显示
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 结果区域
        result_group = QGroupBox("分析结果")
        result_layout = QVBoxLayout()
        
        # 结果文本
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        
        # 导出按钮
        self.export_button = QPushButton("导出结果")
        self.export_button.setMinimumHeight(30)
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        result_layout.addWidget(self.export_button)
        
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group, 1)
        
        # 日志区域
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # 设置分割器
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter, 1)
    
    def _create_data_dirs(self):
        """创建数据目录"""
        directories = [
            "data/uploads",
            "data/results",
            "models"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def select_video_file(self):
        """选择视频文件"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mov *.mkv *.flv)")
        
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                file_path = file_paths[0]
                self.add_video_file(file_path)
    
    def add_video_file(self, file_path):
        """添加视频文件到列表"""
        try:
            # 检查文件是否已存在
            for i in range(self.video_list.count()):
                item = self.video_list.item(i)
                if item.data(Qt.UserRole) == file_path:
                    self.log_message("警告", f"视频文件已存在: {Path(file_path).name}")
                    return
            
            # 复制文件到上传目录
            upload_dir = Path("data/uploads")
            filename = Path(file_path).name
            target_path = upload_dir / filename
            
            # 如果目标文件已存在，添加时间戳
            if target_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_parts = filename.split('.')
                name_parts[0] = f"{name_parts[0]}_{timestamp}"
                filename = '.'.join(name_parts)
                target_path = upload_dir / filename
            
            # 复制文件
            import shutil
            shutil.copy2(file_path, target_path)
            
            # 添加到列表
            item = QListWidgetItem(Path(target_path).name)
            item.setData(Qt.UserRole, str(target_path))
            self.video_list.addItem(item)
            
            self.video_files.append(str(target_path))
            self.process_button.setEnabled(True)
            
            self.log_message("信息", f"已上传视频: {Path(target_path).name}")
            
        except Exception as e:
            self.log_message("错误", f"上传失败: {str(e)}")
    
    def process_video(self):
        """处理视频"""
        # 获取选中的视频
        selected_items = self.video_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一个视频文件")
            return
        
        # 检查模型文件
        if not Path(self.model_path).exists():
            self.log_message("错误", f"模型文件不存在: {self.model_path}")
            QMessageBox.critical(self, "错误", f"模型文件不存在: {self.model_path}")
            return
        
        item = selected_items[0]
        video_path = item.data(Qt.UserRole)
        
        # 创建并启动工作线程
        self.worker = VideoProcessingWorker(video_path, self.model_path)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.result_ready.connect(self.handle_result)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.handle_finished)
        
        # 更新UI状态
        self.process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.upload_button.setEnabled(False)
        
        # 清空结果
        self.result_text.clear()
        
        # 启动线程
        self.worker.start()
        
        self.log_message("信息", f"开始处理视频: {Path(video_path).name}")
    
    def process_selected_video(self, item):
        """双击处理视频"""
        self.process_video()
    
    def stop_processing(self):
        """停止处理"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
            
            self.log_message("信息", "处理已停止")
    
    @Slot(int)
    def update_progress(self, progress):
        """更新进度条"""
        self.progress_bar.setValue(progress)
    
    @Slot(dict)
    def handle_result(self, result):
        """处理分析结果"""
        try:
            # 显示结果
            result_text = f"""
视频分析结果
============

视频文件: {Path(result['video_path']).name}
处理时间: {result['processing_time']}
总帧数: {result['total_frames']}
处理帧数: {result['processed_frames']}

疲劳分析结果
------------
正常帧数: {result['normal_frames']}
轻度疲劳帧数: {result['mild_fatigue_frames']}
重度疲劳帧数: {result['severe_fatigue_frames']}
疲劳帧比例: {result['fatigue_percentage']:.1f}%

特征统计
--------
平均EAR值: {result['avg_ear']:.3f}
平均MAR值: {result['avg_mar']:.3f}

详细结果（前50帧）
---------------
"""
            # 添加前50帧的详细结果
            for i, r in enumerate(result['results'][:50]):
                result_text += f"帧 {r['frame']:4d}: EAR={r['ear']:.3f}, MAR={r['mar']:.3f}, 状态={r['fatigue_state']}\n"
            
            if len(result['results']) > 50:
                result_text += f"\n... 还有 {len(result['results']) - 50} 帧结果未显示\n"
            
            self.result_text.setText(result_text)
            
            # 保存结果到文件
            self.save_results(result)
            
            # 启用导出按钮
            self.export_button.setEnabled(True)
            
            self.log_message("信息", "视频分析完成")
            
        except Exception as e:
            self.log_message("错误", f"结果显示失败: {str(e)}")
    
    def save_results(self, result):
        """保存结果到文件"""
        try:
            import json
            
            # 创建结果目录
            results_dir = Path("data/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            video_name = Path(result['video_path']).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = results_dir / f"analysis_{video_name}_{timestamp}.json"
            
            # 保存结果
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.log_message("信息", f"结果已保存: {result_file.name}")
            
        except Exception as e:
            self.log_message("错误", f"保存结果失败: {str(e)}")
    
    @Slot(str)
    def handle_error(self, error_msg):
        """处理错误"""
        self.log_message("错误", error_msg)
        
        # 恢复UI状态
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.upload_button.setEnabled(True)
        self.progress_bar.setValue(0)
    
    @Slot()
    def handle_finished(self):
        """处理完成"""
        # 恢复UI状态
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.upload_button.setEnabled(True)
        self.progress_bar.setValue(100)
        
        self.log_message("信息", "视频处理完成")
    
    def export_results(self):
        """导出结果"""
        try:
            # 选择保存位置
            file_dialog = QFileDialog()
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            file_dialog.setNameFilter("文本文件 (*.txt);;所有文件 (*)")
            file_dialog.setDefaultSuffix("txt")
            
            if file_dialog.exec():
                file_paths = file_dialog.selectedFiles()
                if file_paths:
                    file_path = file_paths[0]
                    
                    # 保存文本内容
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(self.result_text.toPlainText())
                    
                    self.log_message("信息", f"结果已导出: {file_path}")
                    
        except Exception as e:
            self.log_message("错误", f"导出失败: {str(e)}")
    
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
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        
        super().closeEvent(event)