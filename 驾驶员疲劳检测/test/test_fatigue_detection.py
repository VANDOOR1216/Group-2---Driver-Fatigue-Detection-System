# tests/test_fatigue_detection.py
"""
视频疲劳检测测试脚本
"""
import os
import cv2
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import sys

# 添加父目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入疲劳检测模块
from detectors.face_detector import FaceDetector
from detectors.landmark_detector import LandmarkDetector
from features.ear_calculator import EARCalculator
from features.mar_calculator import MARCalculator
from fatigue.state_tracker import FatigueTracker, FatigueState
from utils.visualizer import Visualizer
from fatigue_detection. config import CONFIG


class VideoTester:
    """视频测试器"""
    
    def __init__(self, model_path):
        """
        初始化视频测试器
        
        Args:
            model_path: 关键点模型文件路径
        """
        # 检查模型文件是否存在
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 初始化组件
        self.face_detector = FaceDetector(method='dlib')
        self.landmark_detector = LandmarkDetector(model_path)
        self.ear_calculator = EARCalculator()
        self.mar_calculator = MARCalculator()
        self.visualizer = Visualizer()
        
        # 创建测试输出目录
        self.output_dir = Path("tests/test_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # 测试视频目录
        self.video_dir = Path("tests/test_videos")
        
        # 测试结果
        self.test_results = {}
    
    def test_single_video(self, video_path, output_video=True):
        """
        测试单个视频文件
        
        Args:
            video_path: 视频文件路径
            output_video: 是否输出处理后的视频
            
        Returns:
            dict: 测试结果
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"警告: 视频文件不存在 - {video_path}")
            return None
        
        print(f"\n开始测试视频: {video_path.name}")
        print("-" * 50)
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 - {video_path}")
            return None
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
        
        # 初始化疲劳跟踪器
        fatigue_tracker = FatigueTracker()
        
        # 准备输出视频（如果需要）
        video_writer = None
        if output_video:
            output_path = self.output_dir / f"processed_{video_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )
            print(f"输出视频将保存到: {output_path}")
        
        # 处理视频帧
        frame_count = 0
        processing_times = []
        fatigue_frames = 0
        severe_fatigue_frames = 0
        
        # 状态统计
        state_counts = {
            "正常": 0,
            "轻度疲劳": 0,
            "重度疲劳": 0
        }
        
        # EAR和MAR值记录
        ear_values = []
        mar_values = []
        
        print("开始处理视频...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 显示进度
                if frame_count % 100 == 0:
                    progress = frame_count / total_frames * 100
                    print(f"进度: {frame_count}/{total_frames} 帧 ({progress:.1f}%)")
                
                # 记录处理时间
                start_time = time.time()
                
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 人脸检测
                faces = self.face_detector.detect(gray)
                
                if len(faces) > 0:
                    # 取面积最大的人脸
                    faces = sorted(faces, 
                                 key=lambda rect: (rect.right()-rect.left()) * (rect.bottom()-rect.top()),
                                 reverse=True)
                    main_face = faces[0]
                    
                    # 关键点检测
                    shape = self.landmark_detector.detect(gray, main_face)
                    
                    if shape is not None:
                        # 计算EAR
                        left_ear, right_ear, avg_ear = self.ear_calculator.calculate(shape)
                        ear_values.append(avg_ear)
                        
                        # 计算MAR
                        mar = self.mar_calculator.calculate(shape)
                        mar_values.append(mar)
                        
                        # 更新疲劳状态
                        state, stats = fatigue_tracker.update(avg_ear, mar)
                        
                        # 统计状态
                        state_counts[state.value] += 1
                        if state == FatigueState.SEVERE:
                            severe_fatigue_frames += 1
                            fatigue_frames += 1
                        elif state == FatigueState.MILD:
                            fatigue_frames += 1
                        
                        # 可视化
                        self.visualizer.draw_face_bbox(frame, main_face)
                        self.visualizer.draw_landmarks(frame, shape)
                        self.visualizer.draw_status(frame, state, left_ear, right_ear, mar, stats)
                    else:
                        # 关键点检测失败
                        cv2.putText(frame, "关键点检测失败", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # 未检测到人脸
                    cv2.putText(frame, "未检测到人脸", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 记录处理时间
                processing_times.append(time.time() - start_time)
                
                # 写入输出视频
                if video_writer:
                    video_writer.write(frame)
                
                # 按ESC键可以中断处理
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    print("用户中断处理")
                    break
        
        except KeyboardInterrupt:
            print("\n处理被用户中断")
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
        finally:
            # 释放资源
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
        
        # 计算统计信息
        total_processed = len(processing_times)
        if total_processed > 0:
            avg_processing_time = np.mean(processing_times) * 1000  # 毫秒
            avg_fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0
            
            # 计算疲劳百分比
            fatigue_percentage = fatigue_frames / total_processed * 100
            severe_fatigue_percentage = severe_fatigue_frames / total_processed * 100
            
            # EAR和MAR统计
            avg_ear = np.mean(ear_values) if ear_values else 0
            avg_mar = np.mean(mar_values) if mar_values else 0
            
            # 整理结果
            result = {
                "video_name": video_path.name,
                "total_frames": total_frames,
                "processed_frames": total_processed,
                "processing_time_avg_ms": avg_processing_time,
                "fps": avg_fps,
                "fatigue_frames": fatigue_frames,
                "fatigue_percentage": fatigue_percentage,
                "severe_fatigue_frames": severe_fatigue_frames,
                "severe_fatigue_percentage": severe_fatigue_percentage,
                "state_distribution": state_counts,
                "avg_ear": avg_ear,
                "avg_mar": avg_mar,
                "ear_threshold": CONFIG['ear']['threshold'],
                "mar_threshold": CONFIG['mar']['threshold'],
                "perclos_mild": CONFIG['fatigue']['perclos_mild'],
                "perclos_severe": CONFIG['fatigue']['perclos_severe']
            }
            
            # 输出结果
            self._print_test_result(result)
            
            # 保存结果
            self.test_results[video_path.name] = result
            
            return result
        
        return None
    
    def test_all_videos(self, output_video=True):
        """
        测试所有视频文件
        
        Args:
            output_video: 是否输出处理后的视频
            
        Returns:
            dict: 所有测试结果
        """
        print("=" * 60)
        print("开始批量测试所有视频")
        print("=" * 60)
        
        # 检查视频目录
        if not self.video_dir.exists():
            print(f"警告: 测试视频目录不存在 - {self.video_dir}")
            return {}
        
        # 获取所有视频文件
        video_files = list(self.video_dir.glob("*.mp4"))
        video_files.extend(self.video_dir.glob("*.avi"))
        video_files.extend(self.video_dir.glob("*.mov"))
        
        if not video_files:
            print("未找到测试视频文件")
            return {}
        
        print(f"找到 {len(video_files)} 个视频文件")
        
        # 测试每个视频
        for video_file in video_files:
            self.test_single_video(video_file, output_video)
        
        # 生成汇总报告
        if self.test_results:
            self._generate_summary_report()
        
        return self.test_results
    
    def _print_test_result(self, result):
        """打印单个测试结果"""
        print("\n" + "=" * 50)
        print(f"视频测试结果: {result['video_name']}")
        print("=" * 50)
        print(f"总帧数: {result['total_frames']}")
        print(f"处理帧数: {result['processed_frames']}")
        print(f"平均处理时间: {result['processing_time_avg_ms']:.2f} ms")
        print(f"处理速度: {result['fps']:.1f} FPS")
        print(f"疲劳帧数: {result['fatigue_frames']} ({result['fatigue_percentage']:.1f}%)")
        print(f"重度疲劳帧数: {result['severe_fatigue_frames']} ({result['severe_fatigue_percentage']:.1f}%)")
        print(f"状态分布: {result['state_distribution']}")
        print(f"平均EAR: {result['avg_ear']:.3f} (阈值: {result['ear_threshold']})")
        print(f"平均MAR: {result['avg_mar']:.3f} (阈值: {result['mar_threshold']})")
        print("=" * 50)
    
    def _generate_summary_report(self):
        """生成汇总报告"""
        print("\n" + "=" * 60)
        print("测试汇总报告")
        print("=" * 60)
        
        # 汇总统计
        total_videos = len(self.test_results)
        total_frames = sum(r['total_frames'] for r in self.test_results.values())
        total_processed = sum(r['processed_frames'] for r in self.test_results.values())
        avg_fps = np.mean([r['fps'] for r in self.test_results.values()])
        
        print(f"测试视频总数: {total_videos}")
        print(f"总帧数: {total_frames}")
        print(f"总处理帧数: {total_processed}")
        print(f"平均处理速度: {avg_fps:.1f} FPS")
        
        # 按视频显示结果
        for video_name, result in self.test_results.items():
            print(f"\n{video_name}:")
            print(f"  疲劳比例: {result['fatigue_percentage']:.1f}%")
            print(f"  重度疲劳比例: {result['severe_fatigue_percentage']:.1f}%")
        
        # 保存报告到文件
        report_path = self.output_dir / "test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "test_date": datetime.now().isoformat(),
                "config": CONFIG,
                "results": self.test_results,
                "summary": {
                    "total_videos": total_videos,
                    "total_frames": total_frames,
                    "total_processed": total_processed,
                    "avg_fps": avg_fps
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细报告已保存到: {report_path}")
    
    def generate_test_videos(self, num_videos=3, duration_seconds=10):
        """
        生成测试视频（模拟数据）
        
        Args:
            num_videos: 生成视频数量
            duration_seconds: 每个视频时长（秒）
        """
        print("生成测试视频...")
        
        # 创建视频目录
        self.video_dir.mkdir(exist_ok=True)
        
        fps = 30
        width, height = 640, 480
        
        for i in range(num_videos):
            video_type = ["normal", "tired", "yawning"][i % 3]
            video_name = f"{video_type}_driving.mp4"
            video_path = self.video_dir / video_name
            
            print(f"生成视频: {video_name}")
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            # 生成视频帧
            for frame_idx in range(fps * duration_seconds):
                # 创建背景
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = (100, 100, 100)  # 灰色背景
                
                # 添加人脸区域（模拟）
                face_center = (width // 2, height // 2)
                face_size = 150
                
                # 根据视频类型设置眼睛和嘴巴状态
                if video_type == "normal":
                    # 正常驾驶：眼睛正常，嘴巴闭合
                    eye_state = "open"
                    mouth_state = "closed"
                elif video_type == "tired":
                    # 疲劳驾驶：眼睛经常闭合
                    eye_state = "closed" if frame_idx % 60 < 30 else "open"
                    mouth_state = "closed"
                else:  # yawning
                    # 打哈欠：嘴巴张开
                    eye_state = "open"
                    mouth_state = "open" if frame_idx % 90 < 45 else "closed"
                
                # 绘制人脸框
                cv2.rectangle(frame,
                            (face_center[0] - face_size, face_center[1] - face_size),
                            (face_center[0] + face_size, face_center[1] + face_size),
                            (255, 255, 255), 2)
                
                # 绘制眼睛
                eye_color = (0, 255, 0) if eye_state == "open" else (0, 0, 255)
                eye_radius = 15 if eye_state == "open" else 5
                
                # 左眼
                cv2.circle(frame,
                          (face_center[0] - 50, face_center[1] - 30),
                          eye_radius, eye_color, -1)
                # 右眼
                cv2.circle(frame,
                          (face_center[0] + 50, face_center[1] - 30),
                          eye_radius, eye_color, -1)
                
                # 绘制嘴巴
                mouth_color = (0, 255, 0) if mouth_state == "closed" else (0, 0, 255)
                mouth_height = 10 if mouth_state == "closed" else 40
                
                cv2.ellipse(frame,
                           (face_center[0], face_center[1] + 30),
                           (60, mouth_height), 0, 0, 180, mouth_color, -1)
                
                # 添加文字说明
                cv2.putText(frame, f"{video_type.upper()} DRIVING", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_idx}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 写入帧
                out.write(frame)
            
            # 释放视频写入器
            out.release()
            
            print(f"  已生成: {video_path} ({duration_seconds}秒)")
        
        print(f"测试视频已生成到: {self.video_dir}")


def main():
    """主函数"""
    # 模型文件路径
    model_path = "models/shape_predictor_68_face_landmarks.dat"
    
    # 检查模型文件是否存在
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在 - {model_path}")
        print("请下载模型文件并放置到 models/ 目录下")
        print("下载地址: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    # 创建视频测试器
    tester = VideoTester(model_path)
    
    # 检查测试视频是否存在，如果不存在则生成
    video_dir = Path("tests/test_videos")
    if not video_dir.exists() or not list(video_dir.glob("*.mp4")):
        print("未找到测试视频，正在生成模拟测试视频...")
        tester.generate_test_videos(num_videos=3, duration_seconds=15)
    
    # 测试所有视频
    print("\n" + "=" * 60)
    print("开始疲劳检测视频测试")
    print("=" * 60)
    
    # 测试选项
    output_video = True  # 是否输出处理后的视频
    
    # 运行测试
    results = tester.test_all_videos(output_video=output_video)
    
    if results:
        print("\n测试完成！")
        print(f"测试结果已保存到: tests/test_output/")
    else:
        print("\n测试未完成或未找到测试视频")


if __name__ == "__main__":
    main()