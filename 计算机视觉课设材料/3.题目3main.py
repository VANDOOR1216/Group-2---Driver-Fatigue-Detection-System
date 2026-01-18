#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
驾驶员疲劳检测系统 - 主程序
Computer Vision Course Design - Driver Fatigue Detection System

作者：__________
学号：__________
日期：__________

本程序实现了基于计算机视觉的驾驶员疲劳检测功能，包括：
1. 人脸检测（使用Dlib的HOG+SVM方法）
2. 面部关键点提取（68点模型）
3. 眼睛纵横比(EAR)和嘴部纵横比(MAR)计算
4. 基于PERCLOS准则的疲劳状态判定
5. 实时可视化展示

使用方法：
    python main.py                    # 使用默认摄像头
    python main.py --camera 1         # 指定摄像头设备ID
    python main.py --video test.mp4   # 使用视频文件
"""

import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import dlib
import numpy as np

# ============================================================
# 配置参数
# ============================================================

class Config:
    """系统配置参数"""

    # 模型文件路径
    MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"

    # 视频参数
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30

    # EAR阈值（眼睛纵横比）
    EAR_THRESHOLD = 0.20          # 眨眼检测阈值
    EAR_CLOSED = 0.15             # 闭眼判定阈值

    # MAR阈值（嘴部纵横比）
    MAR_THRESHOLD = 0.50          # 打哈欠检测阈值

    # PERCLOS参数（疲劳判定）
    WINDOW_SIZE = 60              # 时间窗口大小（帧数）
    PERCLOS_MILD = 0.20           # 轻度疲劳阈值（20%）
    PERCLOS_SEVERE = 0.40         # 重度疲劳阈值（40%）

    # 告警参数
    YAWN_COUNT_THRESHOLD = 3      # 连续打哈欠次数阈值


# ============================================================
# 工具函数
# ============================================================

def create_directory(path: str) -> None:
    """创建目录（如果不存在）"""
    Path(path).mkdir(parents=True, exist_ok=True)


def check_model_file(model_path: str) -> bool:
    """检查模型文件是否存在"""
    if not Path(model_path).exists():
        print(f"错误：模型文件不存在 - {model_path}")
        print("请从以下地址下载模型文件：")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(f"\n并将文件放置到: {Path(model_path).parent.absolute()}/")
        return False
    return True


# ============================================================
# 核心算法类
# ============================================================

class EARCalculator:
    """眼睛纵横比（Eye Aspect Ratio）计算器"""

    # Dlib 68点模型中的眼睛关键点索引
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

    @staticmethod
    def calculate_ear(eye_points: np.ndarray) -> float:
        """
        计算单眼的纵横比

        EAR公式：EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

        参数:
            eye_points: 眼睛的6个关键点坐标，形状为(6, 2)

        返回:
            ear: 眼睛纵横比（通常睁眼时约为0.25-0.35）
        """
        # 计算垂直距离
        vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])

        # 计算水平距离
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])

        # EAR公式
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

        return ear

    @staticmethod
    def extract_eyes(shape: np.ndarray):
        """
        从68点模型中提取左右眼关键点

        参数:
            shape: 68个关键点坐标，形状为(68, 2)

        返回:
            left_eye: 左眼6个关键点
            right_eye: 右眼6个关键点
        """
        left_eye = shape[EARCalculator.LEFT_EYE_INDICES]
        right_eye = shape[EARCalculator.RIGHT_EYE_INDICES]
        return left_eye, right_eye

    @staticmethod
    def compute_avg_ear(shape: np.ndarray) -> tuple:
        """
        计算双眼平均EAR

        参数:
            shape: 68个关键点坐标

        返回:
            (avg_ear, left_ear, right_ear): 平均EAR、左眼EAR、右眼EAR
        """
        left_eye, right_eye = EARCalculator.extract_eyes(shape)
        left_ear = EARCalculator.calculate_ear(left_eye)
        right_ear = EARCalculator.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear, left_ear, right_ear


class MARCalculator:
    """嘴部纵横比（Mouth Aspect Ratio）计算器"""

    # 嘴部关键点索引（48-67）
    MOUTH_INDICES = list(range(48, 68))

    @staticmethod
    def calculate_mar(mouth_points: np.ndarray) -> float:
        """
        计算嘴部纵横比

        MAR公式：MAR = (|p66-p62| + |p64-p60|) / (2 * |p48-54|)

        参数:
            mouth_points: 嘴部的20个关键点坐标

        返回:
            mar: 嘴部纵横比（正常状态<0.5，打哈欠时>=0.5）
        """
        # 计算垂直距离（使用嘴唇上下边缘点）
        vertical_1 = np.linalg.norm(mouth_points[2] - mouth_points[10])  # 66-62
        vertical_2 = np.linalg.norm(mouth_points[4] - mouth_points[8])   # 64-60

        # 计算水平距离（使用嘴角点）
        horizontal = np.linalg.norm(mouth_points[0] - mouth_points[6])   # 48-54

        # MAR公式
        mar = (vertical_1 + vertical_2) / (2.0 * horizontal)

        return mar

    @staticmethod
    def compute_mar(shape: np.ndarray) -> float:
        """
        从68点模型中计算MAR

        参数:
            shape: 68个关键点坐标

        返回:
            mar: 嘴部纵横比
        """
        mouth = shape[48:68]
        return MARCalculator.calculate_mar(mouth)


# ============================================================
# 疲劳状态跟踪器
# ============================================================

class FatigueTracker:
    """疲劳状态跟踪器 - 基于PERCLOS准则"""

    def __init__(self, config: Config = None):
        """
        初始化跟踪器

        参数:
            config: 配置对象
        """
        self.config = config or Config()

        # 历史数据缓冲区
        self.ear_buffer = []
        self.mar_buffer = []

        # 统计计数
        self.blink_count = 0
        self.yawn_count = 0

        # 状态标记
        self.was_blinking = False
        self.is_yawning = False

    def update(self, ear: float, mar: float) -> None:
        """
        更新当前帧的特征值

        参数:
            ear: 眼睛纵横比
            mar: 嘴部纵横比
        """
        # 添加到缓冲区
        self.ear_buffer.append(ear)
        self.mar_buffer.append(mar)

        # 保持缓冲区大小
        if len(self.ear_buffer) > self.config.WINDOW_SIZE:
            self.ear_buffer.pop(0)
            self.mar_buffer.pop(0)

        # 检测眨眼（检测闭眼到睁眼的转换）
        if ear < self.config.EAR_CLOSED:
            if not self.was_blinking:
                self.blink_count += 1
                self.was_blinking = True
        else:
            self.was_blinking = False

        # 检测打哈欠
        if mar > self.config.MAR_THRESHOLD:
            if not self.is_yawning:
                self.yawn_count += 1
                self.is_yawning = True
        else:
            self.is_yawning = False

    def get_perclos(self) -> float:
        """
        计算PERCLOS值

        PERCLOS = 眼睛闭合时间占比
        定义：单位时间内EAR < 阈值的时间比例

        返回:
            perclos: PERCLOS值 [0, 1]
        """
        if len(self.ear_buffer) == 0:
            return 0.0

        closed_frames = sum(1 for ear in self.ear_buffer if ear < self.config.EAR_THRESHOLD)
        perclos = closed_frames / len(self.ear_buffer)
        return perclos

    def get_state(self) -> dict:
        """
        获取当前疲劳状态

        返回:
            state_dict: 包含状态、置信度等信息
        """
        perclos = self.get_perclos()

        # 判定疲劳状态
        if perclos >= self.config.PERCLOS_SEVERE or self.yawn_count >= self.config.YAWN_COUNT_THRESHOLD:
            state = "重度疲劳"
            confidence = "高"
            color = (0, 0, 255)  # 红色
        elif perclos >= self.config.PERCLOS_MILD:
            state = "轻度疲劳"
            confidence = "中"
            color = (0, 165, 255)  # 橙色
        else:
            state = "清醒"
            confidence = "正常"
            color = (0, 255, 0)  # 绿色

        return {
            "state": state,
            "perclos": perclos,
            "confidence": confidence,
            "color": color,
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "avg_ear": np.mean(self.ear_buffer) if self.ear_buffer else 0.0,
            "avg_mar": np.mean(self.mar_buffer) if self.mar_buffer else 0.0
        }


# ============================================================
# 视觉化工具
# ============================================================

class Visualizer:
    """结果可视化类"""

    @staticmethod
    def draw_landmarks(image: np.ndarray, shape: np.ndarray) -> None:
        """
        在图像上绘制面部关键点

        参数:
            image: 输入图像（会直接修改）
            shape: 68个关键点坐标
        """
        # 绘制所有关键点
        for i, (x, y) in enumerate(shape):
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)

        # 高亮眼睛区域（36-47）
        for i in range(36, 48):
            x, y = shape[i]
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)

        # 高亮嘴巴区域（48-67）
        for i in range(48, 68):
            x, y = shape[i]
            cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

    @staticmethod
    def draw_status(image: np.ndarray, state_info: dict, ear: float, mar: float) -> None:
        """
        在图像上绘制状态信息

        参数:
            image: 输入图像（会直接修改）
            state_info: 状态信息字典
            ear: 当前EAR值
            mar: 当前MAR值
        """
        h, w = image.shape[:2]

        # 绘制状态条（顶部）
        color = state_info["color"]
        cv2.rectangle(image, (0, 0), (w, 10), color, -1)

        # 状态文本
        state_text = f"状态: {state_info['state']}"
        cv2.putText(image, state_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 指标显示
        perclos_text = f"PERCLOS: {state_info['perclos']*100:.1f}%"
        ear_text = f"EAR: {ear:.3f}"
        mar_text = f"MAR: {mar:.3f}"
        blink_text = f"眨眼次数: {state_info['blink_count']}"
        yawn_text = f"哈欠次数: {state_info['yawn_count']}"

        # 左侧信息
        cv2.putText(image, perclos_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, ear_text, (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, mar_text, (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 右侧信息
        cv2.putText(image, blink_text, (w - 200, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, yawn_text, (w - 200, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 帧率信息
        fps_text = f"FPS: {int(state_info.get('fps', 0))}"
        cv2.putText(image, fps_text, (w - 100, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# ============================================================
# 主检测类
# ============================================================

class FatigueDetector:
    """疲劳检测系统主类"""

    def __init__(self, config: Config = None):
        """
        初始化检测系统

        参数:
            config: 配置对象
        """
        self.config = config or Config()

        # 初始化人脸检测器
        print("正在初始化人脸检测器...")
        self.face_detector = dlib.get_frontal_face_detector()

        # 初始化关键点检测器
        print(f"正在加载关键点模型: {self.config.MODEL_PATH}")
        if not check_model_file(self.config.MODEL_PATH):
            raise FileNotFoundError("模型文件缺失")
        self.landmark_detector = dlib.shape_predictor(self.config.MODEL_PATH)

        # 初始化疲劳跟踪器
        self.tracker = FatigueTracker(self.config)

        # 初始化可视化器
        self.visualizer = Visualizer()

        print("初始化完成！")

    def process(self, frame: np.ndarray) -> tuple:
        """
        处理单帧图像

        参数:
            frame: 输入图像（BGR格式）

        返回:
            (result_image, state_info): 处理后的图像和状态信息
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = self.face_detector(gray, 0)

        # 默认状态
        state_info = {
            "state": "未检测到人脸",
            "perclos": 0.0,
            "confidence": "无",
            "color": (128, 128, 128),
            "blink_count": 0,
            "yawn_count": 0,
            "avg_ear": 0.0,
            "avg_mar": 0.0
        }

        # 处理检测到的人脸（取第一个）
        if len(faces) > 0:
            face = faces[0]

            # 关键点检测
            shape = self.landmark_detector(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # 计算EAR
            avg_ear, left_ear, right_ear = EARCalculator.compute_avg_ear(shape)

            # 计算MAR
            mar = MARCalculator.compute_mar(shape)

            # 更新跟踪器
            self.tracker.update(avg_ear, mar)

            # 获取状态
            state_info = self.tracker.get_state()

            # 绘制关键点
            self.visualizer.draw_landmarks(frame, shape)

            # 绘制人脸边框
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), state_info["color"], 2)

        # 绘制状态信息
        self.visualizer.draw_status(frame, state_info,
                                    state_info.get("avg_ear", 0),
                                    state_info.get("avg_mar", 0))

        return frame, state_info


# ============================================================
# 命令行参数解析
# ============================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='驾驶员疲劳检测系统 - 计算机视觉课程设计',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py                    # 使用默认摄像头
  python main.py --camera 1         # 使用摄像头1
  python main.py --video test.mp4   # 使用视频文件
  python main.py --width 1280       # 指定视频宽度
        """
    )

    parser.add_argument('--camera', type=int, default=0,
                       help='摄像头设备ID (默认: 0)')
    parser.add_argument('--video', type=str, default=None,
                       help='视频文件路径（若指定则优先使用视频文件）')
    parser.add_argument('--width', type=int, default=640,
                       help='视频宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='视频高度 (默认: 480)')
    parser.add_argument('--model', type=str,
                       default='models/shape_predictor_68_face_landmarks.dat',
                       help='68点模型文件路径')
    parser.add_argument('--window-size', type=int, default=60,
                       help='PERCLOS时间窗口大小（帧数）')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示窗口（仅用于测试）')

    return parser.parse_args()


# ============================================================
# 主程序
# ============================================================

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 创建配置
    config = Config()
    config.MODEL_PATH = args.model
    config.FRAME_WIDTH = args.width
    config.FRAME_HEIGHT = args.height
    config.WINDOW_SIZE = args.window_size

    print("=" * 60)
    print("驾驶员疲劳检测系统")
    print("Computer Vision Course Design - Driver Fatigue Detection")
    print("=" * 60)
    print(f"视频尺寸: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print(f"时间窗口: {config.WINDOW_SIZE} 帧")
    print(f"模型路径: {config.MODEL_PATH}")
    print("=" * 60)

    # 初始化检测器
    try:
        detector = FatigueDetector(config)
    except FileNotFoundError as e:
        print(f"\n初始化失败: {e}")
        return 1

    # 打开视频源
    if args.video:
        print(f"\n正在打开视频文件: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 - {args.video}")
            return 1
    else:
        print(f"\n正在打开摄像头 {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 {args.camera}")
            return 1

    # 设置视频尺寸
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    print("开始检测（按ESC退出，按's'暂停）...")
    print("-" * 60)

    paused = False
    frame_count = 0
    start_time = time.time()

    # 主循环
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n视频结束或读取失败")
                break

            frame_count += 1

            # 计算FPS
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
            else:
                fps = 0

            # 处理帧
            result_frame, state_info = detector.process(frame.copy())
            state_info["fps"] = fps

            # 显示结果
            if not args.no_display:
                cv2.imshow("Fatigue Detection - Driver Safety System", result_frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # 暂停/继续
            paused = not paused
            print("已暂停" if paused else "继续播放")
        elif key == ord('q'):  # 退出
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 统计信息
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0

    print("-" * 60)
    print("检测结束")
    print(f"总帧数: {frame_count}")
    print(f"运行时间: {elapsed:.2f} 秒")
    print(f"平均FPS: {avg_fps:.2f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n程序发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
