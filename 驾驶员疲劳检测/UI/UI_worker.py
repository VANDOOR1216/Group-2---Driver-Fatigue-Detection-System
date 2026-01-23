# ui/worker.py
"""
工作线程：处理视频识别任务，避免阻塞UI（MediaPipe版本）
"""
import os
import cv2
import numpy as np
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import QThread, Signal, QMutex

# 导入疲劳检测模块
import sys
sys.path.append(str(Path(__file__).parent.parent))

from detectors.face_detector import FaceDetector
from detectors.landmark_detector import LandmarkDetector
from features.ear_calculator import EARCalculator
from features.mar_calculator import MARCalculator
from fatigue.state_tracker import FatigueTracker, FatigueState
from config import CONFIG


class RealtimeDetectionWorker(QThread):
    """实时检测工作线程（MediaPipe版本）"""
    
    # 信号定义
    frame_ready = Signal(np.ndarray)  # 视频帧信号
    status_updated = Signal(dict)     # 状态更新信号
    error_occurred = Signal(str)      # 错误信号
    recording_started = Signal(str)   # 录制开始信号
    recording_stopped = Signal(str)   # 录制停止信号
    alarm_triggered = Signal(bool)    # 报警触发信号
    video_finished = Signal()         # 视频播放结束信号
    
    def __init__(self):
        super().__init__()
        
        # 线程控制
        self.mutex = QMutex()
        self.running = False
        self.recording = False
        self.alarm_active = False
        
        # 报警线程控制
        self.alarm_thread = None
        self.alarm_running = False
        
        # 检测组件
        self.face_detector = None
        self.landmark_detector = None
        self.ear_calculator = None
        self.mar_calculator = None
        self.fatigue_tracker = None
        
        # 视频相关
        self.camera_index = 0
        self.video_path = None
        self.is_video_file = False
        self.cap = None
        self.frame_width = CONFIG['frame']['width']
        self.frame_height = CONFIG['frame']['height']
        self.confidence = 0.5
        
        # 录制相关
        self.video_writer = None
        self.recording_start_time = None
        self.recording_file = None
        
        # 结果记录
        self.detection_results = []
        self.result_file = None
        
        # 性能统计
        self.frame_count = 0
        self.start_time = None
        self.fps = 0
        
        # 人脸检测状态
        self.last_face = None
        self.no_face_count = 0
        
        # 疲劳帧计数
        self.fatigue_frame_count = 0
        
        # 播放速度控制
        self.video_fps = 30.0
        self.speed_factor = 1.0
    
    def initialize(self, camera_index=0, confidence=0.5):
        """初始化检测器（MediaPipe版本）"""
        try:
            print("初始化 MediaPipe 检测器...")
            
            # 初始化检测组件（MediaPipe版本）
            self.face_detector = FaceDetector(min_detection_confidence=confidence)
            self.landmark_detector = LandmarkDetector(
                min_detection_confidence=confidence,
                min_tracking_confidence=confidence
            )
            self.ear_calculator = EARCalculator()
            self.mar_calculator = MARCalculator()
            self.fatigue_tracker = FatigueTracker()
            
            # 设置参数
            self.camera_index = camera_index
            self.confidence = confidence
            
            # 判断是否为视频文件
            if isinstance(camera_index, str):
                self.is_video_file = True
                self.video_path = camera_index
            else:
                self.is_video_file = False
                self.video_path = None
            
            print("✓ MediaPipe 检测器初始化成功")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"初始化失败: {str(e)}")
            return False
    
    def start_detection(self):
        """开始检测"""
        self.mutex.lock()
        self.running = True
        self.mutex.unlock()
        
        # 打开摄像头
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.error_occurred.emit("无法打开摄像头")
                self.running = False
                return
            
            # 获取视频FPS
            if self.is_video_file:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    self.video_fps = fps
                else:
                    self.video_fps = 30.0
            
            # 只有在非视频文件模式（即摄像头模式）下才尝试设置分辨率
            # 视频文件应保持原始比例，或根据原始比例计算缩放
            if not self.is_video_file:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame']['width'])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame']['height'])
            
            # 获取实际/原始分辨率
            orig_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            orig_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # 计算保持比例的目标分辨率
            # 目标是限制在配置的最大宽度/高度内，但保持原始长宽比
            max_width = CONFIG['frame']['width']
            max_height = CONFIG['frame']['height']
            
            if orig_width > 0 and orig_height > 0:
                # 计算缩放比例
                scale_w = max_width / orig_width
                scale_h = max_height / orig_height
                scale = min(scale_w, scale_h)
                
                # 如果原始尺寸小于最大尺寸，且不需要放大，则保持原样（或者也可以选择放大）
                # 这里我们选择：如果太大则缩小，如果太小则保持原样（或者根据需求放大）
                # 为了性能，通常只做缩小处理
                if scale < 1.0:
                    self.frame_width = int(orig_width * scale)
                    self.frame_height = int(orig_height * scale)
                else:
                    # 原始尺寸小于配置尺寸，保持原始尺寸
                    self.frame_width = int(orig_width)
                    self.frame_height = int(orig_height)
            else:
                # 获取失败，回退到默认配置
                self.frame_width = max_width
                self.frame_height = max_height
                
            print(f"分辨率设置: 原始={orig_width}x{orig_height}, 实际处理={self.frame_width}x{self.frame_height}")
            
        except Exception as e:
            self.error_occurred.emit(f"摄像头打开失败: {str(e)}")
            self.running = False
            return
        
        # 重置统计
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_results = []
        self.no_face_count = 0
        self.fatigue_frame_count = 0
        
        # 创建结果文件
        self._create_result_file()
    
    def stop_detection(self):
        """停止检测"""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        
        # 停止报警
        self.alarm_active = False
        self._stop_alarm_thread()
        
        # 停止录制
        if self.recording:
            self.stop_recording()
        
        # 请求线程退出并等待
        self.quit()
        self.wait()
    
    def start_recording(self):
        """开始录制视频"""
        if self.recording:
            return
        
        # 创建录制目录
        recording_dir = Path("data/recordings")
        recording_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_file = recording_dir / f"recording_{timestamp}.mp4"
        
        # 获取视频参数
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 创建视频写入器
        self.video_writer = cv2.VideoWriter(
            str(self.recording_file), 
            fourcc, 
            fps, 
            (self.frame_width, self.frame_height)
        )
        
        if not self.video_writer.isOpened():
            self.error_occurred.emit("无法创建录制文件")
            return
        
        self.recording = True
        self.recording_start_time = time.time()
        
        # 发出信号
        self.recording_started.emit(str(self.recording_file))
    
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        recording_duration = time.time() - self.recording_start_time
        
        self.recording = False
        self.recording_start_time = None
        
        # 发出信号
        self.recording_stopped.emit(f"录制已保存: {self.recording_file.name} ({recording_duration:.1f}秒)")
    
    def _start_alarm_thread(self):
        """启动报警线程"""
        import winsound
        self.alarm_running = True
        
        def alarm_worker():
            while self.alarm_running and self.alarm_active:
                # 使用短促的连续鸣笛
                for _ in range(12):
                    if not self.alarm_running or not self.alarm_active:
                        break
                    try:
                        winsound.Beep(1000, 60)
                    except Exception:
                        pass
                
                # 间隔
                if self.alarm_running and self.alarm_active:
                    time.sleep(0.05)
            
            # 线程退出时确保标志位重置
            self.alarm_running = False
        
        self.alarm_thread = threading.Thread(target=alarm_worker, daemon=True)
        self.alarm_thread.start()
    
    def set_speed(self, speed):
        """设置播放速度"""
        self.speed_factor = max(0.1, float(speed))

    def _stop_alarm_thread(self):
        """停止报警线程"""
        self.alarm_running = False
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=0.5)
    
    def run(self):
        """主运行循环"""
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    self.error_occurred.emit("摄像头未打开")
                    time.sleep(0.5)
                    continue

                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    if self.is_video_file:
                        # 视频文件结束，停止检测
                        print("视频播放结束")
                        self.running = False
                        # 发送信号通知UI
                        self.video_finished.emit()
                        break
                    else:
                        # 如果只是暂时读不到，不要立即报错退出，而是等待重试
                        # 但如果是持续读不到，可能需要处理
                        time.sleep(0.01)
                        continue
                
                # 调整尺寸 (保持比例缩放以适应显示区域，或者强制缩放)
                # 为了保持一致性，我们还是缩放到标准尺寸
                if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                # 检测处理
                processed_frame, result = self._process_frame(frame)
                
                # 录制视频
                if self.recording and self.video_writer:
                    self.video_writer.write(processed_frame)
                
                # 记录结果
                self._record_result(result)
                
                # 更新统计
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # 发出信号
                self.frame_ready.emit(processed_frame)
                self.status_updated.emit(result)
                
                # 控制帧率
                if self.is_video_file and self.video_fps > 0:
                    # 计算目标帧间隔 (秒)
                    target_interval = 1.0 / self.video_fps
                    # 应用速度因子
                    actual_interval = target_interval / self.speed_factor
                    time.sleep(actual_interval)
                else:
                    time.sleep(0.01)  # 约100 FPS限制
                
            except Exception as e:
                self.error_occurred.emit(f"处理错误: {str(e)}")
                time.sleep(0.1)
        
        # 线程结束后的资源清理
        self._cleanup()

    def _cleanup(self):
        """资源清理"""
        # 释放摄像头
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 保存结果
        if self.result_file:
            self._save_results()
    
    def _process_frame(self, frame):
        """处理单帧图像（MediaPipe版本）"""
        result = {
            'frame_count': self.frame_count,
            'timestamp': time.time(),
            'fatigue_state': '未知',
            'ear': 0.0,
            'mar': 0.0,
            'perclos': 0.0,
            'fps': self.fps,
            'recording': self.recording,
            'face_detected': False,
            'alarm_active': self.alarm_active
        }
        
        # 获取帧的尺寸
        height, width = frame.shape[:2]
        
        # 人脸检测（MediaPipe版本）
        main_face = self.face_detector.get_main_face(frame)
        
        if main_face is not None:
            self.no_face_count = 0
            self.last_face = main_face
            
            # 获取人脸边界框
            face_bbox = main_face.get('bbox', None)
            
            if face_bbox:
                # 关键点检测（MediaPipe版本）
                landmarks = self.landmark_detector.detect(frame, face_bbox)
                
                if landmarks is not None:
                    # 计算EAR
                    left_ear, right_ear, avg_ear = self.ear_calculator.calculate(landmarks)
                    
                    # 计算MAR
                    mar = self.mar_calculator.calculate(landmarks)
                    
                    # 更新疲劳跟踪器
                    fatigue_state, stats = self.fatigue_tracker.update(avg_ear, mar)
                    
                    # 更新结果
                    result.update({
                        'fatigue_state': fatigue_state.value,
                        'ear': avg_ear,
                        'mar': mar,
                        'perclos': stats['perclos'],
                        'blink_count': stats.get('blink_count', 0),
                        'yawn_count': stats.get('yawn_count', 0),
                        'face_detected': True
                    })
                    
                    # 疲劳报警逻辑
                    if fatigue_state == FatigueState.SEVERE:
                        # 严重疲劳：视觉报警 + 声音报警
                        cv2.putText(frame, "FATIGUE ALERT!", (50, 200), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        
                        if not self.alarm_active:
                            self.alarm_active = True
                            self.alarm_triggered.emit(True)
                            # 启动报警线程
                            if not self.alarm_running:
                                self._start_alarm_thread()
                        
                        # 统计疲劳帧
                        self.fatigue_frame_count += 1
                        
                    elif fatigue_state == FatigueState.MILD:
                        # 轻度疲劳：仅视觉提示
                        cv2.putText(frame, "Warning: Drowsy", (50, 200), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                        if self.alarm_active:
                            self.alarm_active = False
                            self.alarm_triggered.emit(False)
                            self._stop_alarm_thread()
                        
                        # 统计疲劳帧
                        self.fatigue_frame_count += 1
                    else:
                        # 正常状态：关闭报警
                        if self.alarm_active:
                            self.alarm_active = False
                            self.alarm_triggered.emit(False)
                            self._stop_alarm_thread()
                    
                    # ********** 修复人脸框绘制问题 **********
                    # 根据你的face_detector.py，bbox格式是(x, y, width, height)
                    if 'bbox' in main_face and main_face['bbox']:
                        bbox = main_face['bbox']
                        
                        # 调试：打印bbox坐标和图像尺寸
                        print(f"DEBUG - bbox: {bbox}, width: {width}, height: {height}")
                        
                        # 确保bbox有4个值
                        if len(bbox) >= 4:
                            # face_detector返回的是(x, y, width, height)
                            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                            
                            # 转换为(x1, y1, x2, y2)格式
                            x1 = int(x)
                            y1 = int(y)
                            x2 = int(x + w)
                            y2 = int(y + h)
                            
                            print(f"DEBUG - 转换后: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            
                            # 确保坐标是整数并且不越界
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(width, x2)
                            y2 = min(height, y2)
                            
                            # 绘制人脸框
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # *************************************
                    
                    # 绘制状态文本
                    status_text = f"State: {fatigue_state.value}"
                    color = (0, 255, 0) if fatigue_state == FatigueState.NORMAL else \
                            (0, 255, 255) if fatigue_state == FatigueState.MILD else \
                            (0, 0, 255)
                    
                    cv2.putText(frame, status_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    cv2.putText(frame, f"MAR: {mar:.3f}", (10, 80),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # 添加眨眼和打哈欠计数显示
                    blink_count = stats.get('blink_count', 0)
                    yawn_count = stats.get('yawn_count', 0)
                    cv2.putText(frame, f"Blink: {blink_count}", (10, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"Yawn: {yawn_count}", (10, 120),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        else:
            self.no_face_count += 1
            if self.alarm_active:
                self.alarm_active = False
                self.alarm_triggered.emit(False)
            
            # 如果连续多帧未检测到人脸，显示警告
            if self.no_face_count > 30:
                cv2.putText(frame, "No face detected", 
                          (width // 2 - 100, height // 2),
                          0, 1, (0, 0, 255), 2)
        
        # 绘制FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width - 100, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制录制状态
        if self.recording:
            cv2.putText(frame, "Recording...", (width - 100, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # 绘制报警状态
        if self.alarm_active:
            cv2.putText(frame, "ALARM!", (width - 100, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame, result
    
    def _create_result_file(self):
        """创建结果文件"""
        results_dir = Path("data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_file = results_dir / f"detection_results_{timestamp}.json"
        
        # 写入文件头
        header = {
            "start_time": datetime.now().isoformat(),
            "camera_index": self.camera_index,
            "confidence": self.confidence,
            "config": CONFIG,
            "results": []
        }
        
        with open(self.result_file, 'w', encoding='utf-8') as f:
            json.dump(header, f, indent=2, ensure_ascii=False)
    
    def _record_result(self, result):
        """记录检测结果"""
        self.detection_results.append(result)
        
        # 每10帧保存一次到文件
        if len(self.detection_results) % 10 == 0 and self.result_file:
            self._save_results()
    
    def _save_results(self):
        """保存结果到文件"""
        try:
            if not self.result_file or not self.detection_results:
                return
            
            # 读取现有数据
            with open(self.result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 更新结果
            data['results'] = self.detection_results
            data['end_time'] = datetime.now().isoformat()
            data['total_frames'] = len(self.detection_results)
            
            # 计算统计
            if self.detection_results:
                fatigue_states = [r['fatigue_state'] for r in self.detection_results]
                data['statistics'] = {
                    'normal_count': fatigue_states.count(FatigueState.NORMAL.value),
                    'mild_count': fatigue_states.count(FatigueState.MILD.value),
                    'severe_count': fatigue_states.count(FatigueState.SEVERE.value),
                    'total_fatigue_frames': fatigue_states.count(FatigueState.MILD.value) + fatigue_states.count(FatigueState.SEVERE.value),
                    'avg_fps': np.mean([r.get('fps', 0) for r in self.detection_results if r.get('fps', 0) > 0]),
                    'avg_ear': np.mean([r.get('ear', 0) for r in self.detection_results if r.get('ear', 0) > 0]),
                    'avg_mar': np.mean([r.get('mar', 0) for r in self.detection_results if r.get('mar', 0) > 0])
                }
            
            # 写回文件
            with open(self.result_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def get_statistics(self):
        """获取统计信息"""
        if not self.fatigue_tracker:
            return {}
        
        # 直接从fatigue_tracker获取统计信息
        stats = self.fatigue_tracker.get_statistics()
        
        # 确保有这些字段
        result = {
            'total_frames': self.frame_count,
            'blink_count': stats.get('blink_count', 0),
            'yawn_count': stats.get('yawn_count', 0),
            'perclos': stats.get('perclos', 0.0),
            'current_state': stats.get('current_state', '未知'),
            'recording_duration': time.time() - self.recording_start_time if self.recording_start_time else 0,
            'alarm_active': self.alarm_active,
            'fatigue_percentage': 0.0
        }
        
        # 计算疲劳帧比例
        if self.frame_count > 0:
            result['fatigue_percentage'] = (self.fatigue_frame_count / self.frame_count) * 100
        
        return result