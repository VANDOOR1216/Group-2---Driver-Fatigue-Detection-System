"""
驾驶员疲劳检测系统 - MediaPipe版本
统一主程序入口，支持命令行和UI两种模式
"""
import argparse
import cv2
import numpy as np
import time
import sys
import os
import winsound
import threading
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 全局变量控制报警线程
alarm_active = False
alarm_thread = None

def alarm_worker():
    """报警线程函数"""
    global alarm_active
    while alarm_active:
        # 使用更短促的连续鸣笛（40ms），以极大减少停止延迟
        # 循环播放短音，每次检查状态
        # 增加循环次数以保持总时长接近 (12 * 40ms = 480ms)
        for _ in range(12):
            if not alarm_active:
                break
            try:
                winsound.Beep(1000, 60)
            except RuntimeError:
                pass
        
        # 间隔
        if alarm_active:
            time.sleep(0.05)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='驾驶员疲劳检测系统 - MediaPipe版本')
    parser.add_argument('--mode', type=str, default='ui', choices=['ui', 'cli'],
                       help='运行模式: ui(图形界面, 默认) 或 cli(命令行)')
    parser.add_argument('--camera', type=int, default=0, 
                       help='摄像头设备ID (默认: 0, 仅在cli模式下有效)')
    parser.add_argument('--video', type=str, default=None,
                       help='视频文件路径 (如果提供，将使用视频文件而不是摄像头)')
    parser.add_argument('--width', type=int, default=640,
                       help='视频宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='视频高度 (默认: 480)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频文件路径 (可选, 仅在cli模式下有效)')
    parser.add_argument('--headless', action='store_true',
                       help='无界面模式 (用于服务器部署, 仅在cli模式下有效)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='人脸检测置信度阈值 (默认: 0.5)')
    return parser.parse_args()


def run_cli_mode(args):
    """运行命令行模式 - 完全保持main1.py的功能"""
    global alarm_active, alarm_thread
    
    # 导入自定义模块
    from detectors.face_detector import FaceDetector
    from detectors.landmark_detector import LandmarkDetector
    from features.ear_calculator import EARCalculator
    from features.mar_calculator import MARCalculator
    from fatigue.state_tracker import FatigueTracker, FatigueState
    from utils.visualizer import Visualizer
    from config import CONFIG
    
    # 使用config中的参数（如果用户没有指定）
    if args.width == 640:
        args.width = CONFIG['frame']['width']
    if args.height == 480:
        args.height = CONFIG['frame']['height']
    
    # 初始化检测器（使用 MediaPipe Tasks API）
    print("初始化 MediaPipe 检测器...")
    face_detector = FaceDetector(min_detection_confidence=args.confidence)
    landmark_detector = LandmarkDetector(
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence
    )
    ear_calculator = EARCalculator()
    mar_calculator = MARCalculator()
    fatigue_tracker = FatigueTracker()
    visualizer = Visualizer()
    
    print("✓ MediaPipe 检测器初始化成功")
    
    # 视频源设置
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"无法打开视频文件: {args.video}")
            return
        print(f"使用视频文件: {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f"使用摄像头: {args.camera}")
    
    # 输出视频设置
    output_writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(
            args.output, fourcc, fps, (args.width, args.height)
        )
        print(f"输出视频: {args.output}")
    
    print("\n开始疲劳检测... (按 ESC 退出)")
    print("-" * 50)
    
    frame_count = 0
    start_time = time.time()
    last_face = None
    no_face_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流结束")
                break
            
            # 调整尺寸
            frame = cv2.resize(frame, (args.width, args.height))
            
            # 人脸检测
            main_face = face_detector.get_main_face(frame)
            
            if main_face is not None:
                no_face_count = 0
                last_face = main_face
                
                # 获取人脸边界框
                face_bbox = main_face['bbox']
                
                # 关键点检测（MediaPipe 直接处理彩色图像）
                landmarks = landmark_detector.detect(frame, face_bbox)
                
                if landmarks is not None:
                    # 计算EAR
                    left_ear, right_ear, avg_ear = ear_calculator.calculate(landmarks)
                    
                    # 计算MAR
                    mar = mar_calculator.calculate(landmarks)
                    
                    # 更新疲劳跟踪器
                    fatigue_state, stats = fatigue_tracker.update(avg_ear, mar)
                    
                    # 疲劳报警逻辑
                    if fatigue_state == FatigueState.SEVERE:
                        # 严重疲劳：视觉报警 + 声音报警
                        cv2.putText(frame, "FATIGUE ALERT!", (50, 200), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        
                        if not alarm_active:
                            alarm_active = True
                            # 检查线程是否已在运行（复用线程或启动新线程）
                            if alarm_thread and alarm_thread.is_alive():
                                pass  # 线程还活着，只需设置 alarm_active=True 即可复活/继续
                            else:
                                alarm_thread = threading.Thread(target=alarm_worker, daemon=True)
                                alarm_thread.start()
                            
                    elif fatigue_state == FatigueState.MILD:
                        # 轻度疲劳：仅视觉提示
                        cv2.putText(frame, "Warning: Drowsy", (50, 200), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                        alarm_active = False
                    else:
                        # 正常状态：关闭报警
                        alarm_active = False
                    
                    # 可视化 - 完全保持原来的可视化方式
                    if not args.headless:
                        visualizer.draw_face_bbox_dict(frame, main_face)
                        visualizer.draw_landmarks(frame, landmarks)
                        visualizer.draw_status(frame, fatigue_state, 
                                             left_ear, right_ear, mar, stats)
            else:
                no_face_count += 1
                alarm_active = False # 未检测到人脸时停止报警
                # 如果连续多帧未检测到人脸，显示警告
                if no_face_count > 30 and not args.headless:
                    cv2.putText(frame, "No face detected", 
                              (args.width // 2 - 100, args.height // 2),
                              0, 1, (0, 0, 255), 2)
            
            # 计算并显示FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                if not args.headless:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                              0, 0.7, (0, 255, 0), 2)
            
            # 写入输出视频
            if output_writer:
                output_writer.write(frame)
            
            # 显示结果
            if not args.headless:
                cv2.imshow("Driver Fatigue Detection System - MediaPipe", frame)
                
                # 按ESC退出
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r'):  # R键重置统计
                    fatigue_tracker.reset()
                    print("统计信息已重置")
            else:
                # 无界面模式，每处理100帧输出一次状态
                if frame_count % 100 == 0:
                    if main_face:
                        print(f"已处理 {frame_count} 帧，当前状态: {fatigue_state.value if 'fatigue_state' in locals() else 'N/A'}")
                    else:
                        print(f"已处理 {frame_count} 帧，未检测到人脸")
    
    except KeyboardInterrupt:
        print("\n检测被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 释放资源
        cap.release()
        if output_writer:
            output_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        
        # 输出统计信息
        print("\n" + "=" * 50)
        print("检测统计信息:")
        print("=" * 50)
        if fatigue_tracker:
            final_stats = fatigue_tracker.get_statistics()
            elapsed_time = time.time() - start_time
            print(f"总帧数: {frame_count}")
            print(f"运行时间: {elapsed_time:.1f} 秒")
            print(f"平均FPS: {frame_count/elapsed_time:.1f}")
            print(f"当前疲劳状态: {final_stats.get('current_state', 'N/A')}")
            print(f"眨眼次数: {final_stats.get('blink_count', 0)}")
            print(f"打哈欠次数: {final_stats.get('yawn_count', 0)}")
            print(f"PERCLOS: {final_stats.get('perclos', 0):.1%}")
        print("=" * 50)


def run_ui_mode():
    """运行UI模式"""
    from PySide6.QtWidgets import QApplication
    from UI.UI_main_windows import MainWindow
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("驾驶员疲劳检测系统 - MediaPipe")
    app.setApplicationDisplayName("驾驶员疲劳检测系统 - MediaPipe")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    return app.exec()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 根据模式运行
    if args.mode == 'ui':
        print("启动UI模式...")
        return run_ui_mode()
    else:
        print("启动命令行模式...")
        # 完全保持main1.py的命令行功能
        run_cli_mode(args)

if __name__ == "__main__":
    sys.exit(main())