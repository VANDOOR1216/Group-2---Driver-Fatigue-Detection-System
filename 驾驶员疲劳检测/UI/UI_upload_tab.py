# ui/upload_tab.py
"""
视频上传标签页（MediaPipe版本）
"""
import os
from datetime import datetime
from pathlib import Path
import shutil
import cv2
import numpy as np
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QTextEdit, QFrame, QGridLayout, QProgressBar,
    QFileDialog, QListWidget, QListWidgetItem, QSplitter,
    QMessageBox, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont

from .UI_styles import COLORS, FONTS


class VideoProcessingWorker(QThread):
    """视频处理工作线程（MediaPipe版本）"""
    
    # 信号定义
    progress_updated = pyqtSignal(int)    # 进度更新
    result_ready = pyqtSignal(dict)       # 结果就绪
    error_occurred = pyqtSignal(str)      # 错误信号
    finished = pyqtSignal()               # 完成信号
    
    def __init__(self, video_path, confidence=0.5):
        super().__init__()
        self.video_path = video_path
        self.confidence = confidence
        self.running = True
        
    def run(self):
        """运行视频处理"""
        try:
            import sys
            from pathlib import Path
            
            # 添加父目录到Python路径
            sys.path.append(str(Path(__file__).parent.parent.parent))
            
            from detectors.face_detector import FaceDetector
            from detectors.landmark_detector import LandmarkDetector
            from features.ear_calculator import EARCalculator
            from features.mar_calculator import MARCalculator
            from fatigue.state_tracker import FatigueTracker, FatigueState
            from config import CONFIG
            
            # 检查文件
            if not Path(self.video_path).exists():
                self.error_occurred.emit(f"视频文件不存在: {self.video_path}")
                return
            
            # 初始化检测器（MediaPipe版本）
            print("初始化 MediaPipe 检测器...")
            face_detector = FaceDetector(min_detection_confidence=self.confidence)
            landmark_detector = LandmarkDetector(
                min_detection_confidence=self.confidence,
                min_tracking_confidence=self.confidence
            )
            ear_calculator = EARCalculator()
            mar_calculator = MARCalculator()
            fatigue_tracker = FatigueTracker()
            
            print("✓ MediaPipe 检测器初始化成功")
            
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
                # 如果无法获取总帧数，尝试估算
                total_frames = 1000  # 默认值
                self.error_occurred.emit("警告: 无法获取视频总帧数，使用默认值")
            
            # 处理视频
            frame_count = 0
            results = []
            fatigue_states = []
            ear_values = []
            mar_values = []
            no_face_count = 0
            
            # 疲劳状态标志
            has_mild_fatigue = False
            has_severe_fatigue = False
            
            print(f"开始处理视频: {Path(self.video_path).name}")
            print(f"视频信息: {width}x{height}, {fps:.1f} FPS, 总帧数: {total_frames}")
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 更新进度
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                self.progress_updated.emit(progress)
                
                # 调整尺寸为640x480（与实时检测一致）
                frame_resized = cv2.resize(frame, (640, 480))
                
                # 人脸检测（MediaPipe版本）
                main_face = face_detector.get_main_face(frame_resized)
                
                if main_face is not None:
                    no_face_count = 0
                    
                    # 获取人脸边界框
                    face_bbox = main_face.get('bbox', None)
                    
                    if face_bbox:
                        # 关键点检测（MediaPipe版本）
                        landmarks = landmark_detector.detect(frame_resized, face_bbox)
                        
                        if landmarks is not None:
                            # 计算EAR
                            left_ear, right_ear, avg_ear = ear_calculator.calculate(landmarks)
                            ear_values.append(avg_ear)
                            
                            # 计算MAR
                            mar = mar_calculator.calculate(landmarks)
                            mar_values.append(mar)
                            
                            # 更新疲劳状态
                            state, stats = fatigue_tracker.update(avg_ear, mar)
                            fatigue_states.append(state.value)
                            
                            # 记录疲劳状态标志
                            if state == FatigueState.MILD:
                                has_mild_fatigue = True
                            elif state == FatigueState.SEVERE:
                                has_severe_fatigue = True
                            
                            # 记录结果
                            results.append({
                                'frame': frame_count,
                                'ear': avg_ear,
                                'mar': mar,
                                'fatigue_state': state.value,
                                'perclos': stats['perclos'],
                                'blink_count': stats.get('blink_count', 0),
                                'yawn_count': stats.get('yawn_count', 0)
                            })
                else:
                    no_face_count += 1
                
                # 每处理10帧检查一次是否停止
                if frame_count % 10 == 0 and not self.running:
                    break
            
            # 释放视频
            cap.release()
            
            # 确定最终疲劳程度（只要检测到一次就记录）
            if has_severe_fatigue:
                final_fatigue_level = "重度疲劳"
            elif has_mild_fatigue:
                final_fatigue_level = "轻度疲劳"
            else:
                final_fatigue_level = "正常"
            
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
                    'video_name': Path(self.video_path).name,
                    'final_fatigue_level': final_fatigue_level,
                    'total_frames': total_frames,
                    'processed_frames': frame_count,
                    'frames_with_face': len(results),
                    'frames_without_face': no_face_count,
                    'normal_frames': normal_count,
                    'mild_fatigue_frames': mild_count,
                    'severe_fatigue_frames': severe_count,
                    'fatigue_percentage': (total_fatigue / len(results)) * 100 if results else 0,
                    'avg_ear': avg_ear,
                    'avg_mar': avg_mar,
                    'results': results[:100],  # 只保留前100帧结果，避免数据过大
                    'processing_time': datetime.now().isoformat(),
                    'confidence': self.confidence
                }
                
                self.result_ready.emit(result)
            
            self.finished.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """停止处理"""
        self.running = False


class UploadTab(QWidget):
    """视频上传标签页（MediaPipe版本）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 工作线程
        self.worker = None
        
        # 视频文件列表
        self.video_files = []
        self.confidence = 0.5
        
        # 所有检测结果的列表
        self.all_results = []
        
        # 结果文件路径
        self.results_file = "video_detection_results.json"
        
        # 初始化UI
        self.init_ui()
        
        # 创建数据目录
        self._create_data_dirs()
        
        # 加载已有的检测结果
        self.load_previous_results()
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title_label = QLabel("视频上传与分析 (MediaPipe)")
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
        
        # 批量上传按钮
        self.batch_upload_button = QPushButton("批量上传视频")
        self.batch_upload_button.setMinimumHeight(40)
        self.batch_upload_button.setFont(FONTS['normal'])
        self.batch_upload_button.clicked.connect(self.select_multiple_video_files)
        upload_layout.addWidget(self.batch_upload_button)
        
        # 置信度设置
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("置信度:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(30, 90)  # 30%到90%
        self.confidence_slider.setValue(50)  # 默认50%
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel("0.50")
        self.confidence_label.setMinimumWidth(40)
        confidence_layout.addWidget(self.confidence_label)
        
        upload_layout.addLayout(confidence_layout)
        
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
        
        # 批量处理按钮
        self.batch_process_button = QPushButton("批量处理所有视频")
        self.batch_process_button.setMinimumHeight(40)
        self.batch_process_button.setFont(FONTS['subheading'])
        self.batch_process_button.clicked.connect(self.process_all_videos)
        self.batch_process_button.setEnabled(False)
        control_layout.addWidget(self.batch_process_button)
        
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
        
        # 查看结果按钮
        self.view_results_button = QPushButton("查看所有检测结果")
        self.view_results_button.setMinimumHeight(40)
        self.view_results_button.setFont(FONTS['normal'])
        self.view_results_button.clicked.connect(self.view_all_results)
        control_layout.addWidget(self.view_results_button)
        
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
        
        # 清空结果按钮
        self.clear_results_button = QPushButton("清空结果列表")
        self.clear_results_button.setMinimumHeight(30)
        self.clear_results_button.clicked.connect(self.clear_results)
        result_layout.addWidget(self.clear_results_button)
        
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
    
    def load_previous_results(self):
        """加载之前的检测结果"""
        try:
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    self.all_results = json.load(f)
                self.log_message("信息", f"已加载 {len(self.all_results)} 条历史检测结果")
            else:
                self.all_results = []
                self.log_message("信息", "无历史检测结果")
        except Exception as e:
            self.log_message("错误", f"加载历史结果失败: {str(e)}")
            self.all_results = []
    
    def save_results_to_file(self):
        """保存所有检测结果到文件"""
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_results, f, indent=2, ensure_ascii=False)
            self.log_message("信息", f"结果已保存到 {self.results_file}")
        except Exception as e:
            self.log_message("错误", f"保存结果失败: {str(e)}")
    
    def update_confidence_label(self, value):
        """更新置信度标签"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        self.confidence = confidence
    
    def select_video_file(self):
        """选择视频文件"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)")
        
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                file_path = file_paths[0]
                self.add_video_file(file_path)
    
    def select_multiple_video_files(self):
        """批量选择视频文件"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)")
        
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            for file_path in file_paths:
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
            shutil.copy2(file_path, target_path)
            
            # 添加到列表
            item = QListWidgetItem(Path(target_path).name)
            item.setData(Qt.UserRole, str(target_path))
            self.video_list.addItem(item)
            
            self.video_files.append(str(target_path))
            self.process_button.setEnabled(True)
            self.batch_process_button.setEnabled(True)
            
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
        
        item = selected_items[0]
        video_path = item.data(Qt.UserRole)
        confidence = self.confidence
        
        # 创建并启动工作线程
        self.worker = VideoProcessingWorker(video_path, confidence)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.result_ready.connect(self.handle_result)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.handle_finished)
        
        # 更新UI状态
        self.process_button.setEnabled(False)
        self.batch_process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.upload_button.setEnabled(False)
        self.batch_upload_button.setEnabled(False)
        self.confidence_slider.setEnabled(False)
        
        # 清空结果
        self.result_text.clear()
        
        # 启动线程
        self.worker.start()
        
        self.log_message("信息", f"开始处理视频: {Path(video_path).name} (置信度: {confidence:.2f})")
    
    def process_all_videos(self):
        """批量处理所有视频"""
        if self.video_list.count() == 0:
            QMessageBox.warning(self, "警告", "没有可处理的视频文件")
            return
        
        # 获取所有视频路径
        video_paths = []
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            video_path = item.data(Qt.UserRole)
            video_paths.append(video_path)
        
        # 启动批量处理
        self.batch_process_videos(video_paths)
    
    def batch_process_videos(self, video_paths):
        """批量处理视频"""
        if not video_paths:
            return
        
        # 创建一个批量处理器
        self.batch_videos = video_paths
        self.current_batch_index = 0
        
        # 处理第一个视频
        self.process_next_video_in_batch()
    
    def process_next_video_in_batch(self):
        """批量处理中的下一个视频"""
        if self.current_batch_index >= len(self.batch_videos):
            self.log_message("信息", "批量处理完成")
            self.handle_batch_finished()
            return
        
        video_path = self.batch_videos[self.current_batch_index]
        confidence = self.confidence
        
        # 创建并启动工作线程
        self.worker = VideoProcessingWorker(video_path, confidence)
        self.worker.progress_updated.connect(self.update_batch_progress)
        self.worker.result_ready.connect(self.handle_batch_result)
        self.worker.error_occurred.connect(self.handle_batch_error)
        self.worker.finished.connect(self.handle_batch_finished)
        
        # 更新UI状态
        self.process_button.setEnabled(False)
        self.batch_process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.upload_button.setEnabled(False)
        self.batch_upload_button.setEnabled(False)
        self.confidence_slider.setEnabled(False)
        
        # 启动线程
        self.worker.start()
        
        self.log_message("信息", f"批量处理中 ({self.current_batch_index+1}/{len(self.batch_videos)}): {Path(video_path).name}")
    
    def process_selected_video(self, item):
        """双击处理视频"""
        self.process_video()
    
    def stop_processing(self):
        """停止处理"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
            
            self.log_message("信息", "处理已停止")
    
    @pyqtSlot(int)
    def update_progress(self, progress):
        """更新进度条"""
        self.progress_bar.setValue(progress)
    
    @pyqtSlot(int)
    def update_batch_progress(self, progress):
        """更新批量处理的进度条"""
        # 计算总体进度
        base_progress = (self.current_batch_index / len(self.batch_videos)) * 100
        current_progress = (progress / 100) * (100 / len(self.batch_videos))
        total_progress = base_progress + current_progress
        self.progress_bar.setValue(int(total_progress))
    
    @pyqtSlot(dict)
    def handle_result(self, result):
        """处理分析结果"""
        try:
            # 将结果添加到所有结果列表中
            self.all_results.append({
                'video_name': result['video_name'],
                'final_fatigue_level': result['final_fatigue_level'],
                'processing_time': result['processing_time'],
                'total_frames': result['total_frames'],
                'processed_frames': result['processed_frames'],
                'fatigue_percentage': result['fatigue_percentage']
            })
            
            # 保存到文件
            self.save_results_to_file()
            
            # 显示结果
            result_text = f"""
视频分析结果 (MediaPipe)
=======================

视频文件: {result['video_name']}
处理时间: {result['processing_time']}
置信度阈值: {result['confidence']:.2f}
最终疲劳程度: 【{result['final_fatigue_level']}】

视频信息
--------
总帧数: {result['total_frames']}
处理帧数: {result['processed_frames']}
检测到人脸的帧数: {result['frames_with_face']}
未检测到人脸的帧数: {result['frames_without_face']}

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

详细结果（前20帧）
---------------
"""
            # 添加前20帧的详细结果
            for i, r in enumerate(result['results'][:20]):
                result_text += f"帧 {r['frame']:4d}: EAR={r['ear']:.3f}, MAR={r['mar']:.3f}, 状态={r['fatigue_state']}\n"
            
            if len(result['results']) > 20:
                result_text += f"\n... 还有 {len(result['results']) - 20} 帧结果未显示\n"
            
            self.result_text.setText(result_text)
            
            # 保存详细结果到单独文件
            self.save_detailed_result(result)
            
            # 启用导出按钮
            self.export_button.setEnabled(True)
            
            self.log_message("信息", f"视频分析完成 - 最终疲劳程度: {result['final_fatigue_level']}")
            
        except Exception as e:
            self.log_message("错误", f"结果显示失败: {str(e)}")
    
    @pyqtSlot(dict)
    def handle_batch_result(self, result):
        """处理批量分析结果"""
        # 将结果添加到所有结果列表中
        self.all_results.append({
            'video_name': result['video_name'],
            'final_fatigue_level': result['final_fatigue_level'],
            'processing_time': result['processing_time'],
            'total_frames': result['total_frames'],
            'processed_frames': result['processed_frames'],
            'fatigue_percentage': result['fatigue_percentage']
        })
        
        # 更新UI显示当前结果
        self.result_text.append(f"【{result['video_name']}】 - 最终疲劳程度: {result['final_fatigue_level']} "
                              f"(疲劳帧比例: {result['fatigue_percentage']:.1f}%)")
        
        self.log_message("信息", f"批量处理完成 ({self.current_batch_index+1}/{len(self.batch_videos)}): "
                              f"{result['video_name']} - {result['final_fatigue_level']}")
    
    def save_detailed_result(self, result):
        """保存详细结果到文件"""
        try:
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
            
            self.log_message("信息", f"详细结果已保存: {result_file.name}")
            
        except Exception as e:
            self.log_message("错误", f"保存详细结果失败: {str(e)}")
    
    @pyqtSlot(str)
    def handle_error(self, error_msg):
        """处理错误"""
        self.log_message("错误", error_msg)
        
        # 恢复UI状态
        self.process_button.setEnabled(True)
        self.batch_process_button.setEnabled(len(self.video_files) > 0)
        self.stop_button.setEnabled(False)
        self.upload_button.setEnabled(True)
        self.batch_upload_button.setEnabled(True)
        self.confidence_slider.setEnabled(True)
        self.progress_bar.setValue(0)
    
    @pyqtSlot(str)
    def handle_batch_error(self, error_msg):
        """处理批量处理错误"""
        self.log_message("错误", f"批量处理错误: {error_msg}")
        
        # 继续处理下一个视频
        self.current_batch_index += 1
        self.process_next_video_in_batch()
    
    @pyqtSlot()
    def handle_finished(self):
        """处理完成"""
        # 恢复UI状态
        self.process_button.setEnabled(True)
        self.batch_process_button.setEnabled(len(self.video_files) > 0)
        self.stop_button.setEnabled(False)
        self.upload_button.setEnabled(True)
        self.batch_upload_button.setEnabled(True)
        self.confidence_slider.setEnabled(True)
        self.progress_bar.setValue(100)
        
        self.log_message("信息", "视频处理完成")
    
    @pyqtSlot()
    def handle_batch_finished(self):
        """处理批量完成"""
        # 处理下一个视频
        self.current_batch_index += 1
        
        if self.current_batch_index < len(self.batch_videos):
            # 还有视频要处理
            self.process_next_video_in_batch()
        else:
            # 所有视频处理完成
            # 恢复UI状态
            self.process_button.setEnabled(True)
            self.batch_process_button.setEnabled(len(self.video_files) > 0)
            self.stop_button.setEnabled(False)
            self.upload_button.setEnabled(True)
            self.batch_upload_button.setEnabled(True)
            self.confidence_slider.setEnabled(True)
            self.progress_bar.setValue(100)
            
            # 保存所有结果到文件
            self.save_results_to_file()
            
            self.log_message("信息", "批量处理全部完成")
            
            # 显示总结信息
            summary_text = f"""
批量处理总结
============

处理视频总数: {len(self.batch_videos)}
处理完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

各视频疲劳程度统计:
"""
            # 统计各种疲劳程度的数量
            fatigue_counts = {
                "正常": 0,
                "轻度疲劳": 0,
                "重度疲劳": 0
            }
            
            for result in self.all_results[-len(self.batch_videos):]:
                fatigue_level = result['final_fatigue_level']
                if fatigue_level in fatigue_counts:
                    fatigue_counts[fatigue_level] += 1
            
            summary_text += f"正常: {fatigue_counts['正常']} 个视频\n"
            summary_text += f"轻度疲劳: {fatigue_counts['轻度疲劳']} 个视频\n"
            summary_text += f"重度疲劳: {fatigue_counts['重度疲劳']} 个视频\n\n"
            
            summary_text += "详细结果列表:\n"
            for result in self.all_results[-len(self.batch_videos):]:
                summary_text += f"- {result['video_name']}: {result['final_fatigue_level']} "
                summary_text += f"(疲劳帧比例: {result.get('fatigue_percentage', 0):.1f}%)\n"
            
            self.result_text.setText(summary_text)
    
    def export_results(self):
        """导出结果"""
        try:
            # 选择保存位置
            file_dialog = QFileDialog()
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            file_dialog.setNameFilter("文本文件 (*.txt);;JSON文件 (*.json);;CSV文件 (*.csv);;所有文件 (*)")
            file_dialog.setDefaultSuffix("txt")
            
            if file_dialog.exec_():
                file_paths = file_dialog.selectedFiles()
                if file_paths:
                    file_path = file_paths[0]
                    
                    # 根据文件类型保存
                    if file_path.endswith('.json'):
                        import json
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
                    elif file_path.endswith('.csv'):
                        import csv
                        with open(file_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(['视频名称', '疲劳程度', '处理时间', '总帧数', '处理帧数', '疲劳帧比例'])
                            for result in self.all_results:
                                writer.writerow([
                                    result['video_name'],
                                    result['final_fatigue_level'],
                                    result['processing_time'],
                                    result['total_frames'],
                                    result['processed_frames'],
                                    f"{result.get('fatigue_percentage', 0):.1f}%"
                                ])
                    else:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write("视频疲劳检测结果汇总\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"总视频数: {len(self.all_results)}\n\n")
                            
                            f.write("视频名称\t疲劳程度\t处理时间\t总帧数\t处理帧数\t疲劳帧比例\n")
                            f.write("-" * 100 + "\n")
                            
                            for result in self.all_results:
                                f.write(f"{result['video_name']}\t")
                                f.write(f"{result['final_fatigue_level']}\t")
                                f.write(f"{result['processing_time']}\t")
                                f.write(f"{result['total_frames']}\t")
                                f.write(f"{result['processed_frames']}\t")
                                f.write(f"{result.get('fatigue_percentage', 0):.1f}%\n")
                    
                    self.log_message("信息", f"结果已导出: {file_path}")
                    
        except Exception as e:
            self.log_message("错误", f"导出失败: {str(e)}")
    
    def view_all_results(self):
        """查看所有检测结果"""
        if not self.all_results:
            QMessageBox.information(self, "信息", "暂无检测结果")
            return
        
        result_text = f"""
所有视频检测结果汇总
==================

总视频数: {len(self.all_results)}
最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

统计信息:
---------
"""
        # 统计各种疲劳程度的数量
        fatigue_counts = {
            "正常": 0,
            "轻度疲劳": 0,
            "重度疲劳": 0
        }
        
        for result in self.all_results:
            fatigue_level = result['final_fatigue_level']
            if fatigue_level in fatigue_counts:
                fatigue_counts[fatigue_level] += 1
        
        result_text += f"正常: {fatigue_counts['正常']} 个视频\n"
        result_text += f"轻度疲劳: {fatigue_counts['轻度疲劳']} 个视频\n"
        result_text += f"重度疲劳: {fatigue_counts['重度疲劳']} 个视频\n\n"
        
        result_text += "详细结果列表:\n"
        result_text += "序号\t视频名称\t疲劳程度\t处理时间\t疲劳帧比例\n"
        result_text += "-" * 100 + "\n"
        
        for i, result in enumerate(self.all_results, 1):
            result_text += f"{i:3d}\t{result['video_name']}\t"
            result_text += f"{result['final_fatigue_level']}\t"
            result_text += f"{result['processing_time'][:19]}\t"
            result_text += f"{result.get('fatigue_percentage', 0):.1f}%\n"
        
        self.result_text.setText(result_text)
    
    def clear_results(self):
        """清空结果列表"""
        reply = QMessageBox.question(
            self, '确认清空',
            '确定要清空所有检测结果吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.all_results = []
            self.save_results_to_file()
            self.log_message("信息", "所有检测结果已清空")
            self.result_text.clear()
    
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