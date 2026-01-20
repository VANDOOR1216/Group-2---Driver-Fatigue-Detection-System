# features/ear_calculator.py
import numpy as np
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'fatigue_detection'))
from config import CONFIG


class EARCalculator:
    """眼睛纵横比计算器 - 适配 MediaPipe"""
    
    # MediaPipe Face Mesh 眼睛关键点索引
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # 左眼6个点
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # 右眼6个点
    
    def __init__(self):
        """初始化EAR计算器"""
        self.ear_threshold = CONFIG['ear']['threshold']
        self.blink_threshold = CONFIG['ear']['blink_threshold']
    
    @staticmethod
    def calculate_ear_for_eye(eye_points):
        """
        计算单只眼睛的EAR值
        
        Args:
            eye_points: 眼睛的6个关键点坐标 (6, 2)
            
        Returns:
            float: EAR值
        """
        # MediaPipe 眼睛点顺序：
        # 0: 眼角外侧, 1: 上眼睑外, 2: 上眼睑中, 3: 眼角内侧, 4: 下眼睑中, 5: 下眼睑外
        
        # 计算垂直距离
        vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])  # 上外-下外
        vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])  # 上中-下中
        
        # 计算水平距离
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])  # 外角-内角
        
        # 计算EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)  # 避免除零
        
        return ear
    
    def calculate(self, landmarks):
        """
        计算双眼的EAR值
        
        Args:
            landmarks: 面部关键点坐标 (468, 2) - MediaPipe Face Mesh
            
        Returns:
            tuple: (左眼EAR, 右眼EAR, 平均EAR)
        """
        if landmarks is None or len(landmarks) < 468:
            return 0.0, 0.0, 0.0
        
        # 提取眼睛关键点
        left_eye_points = landmarks[self.LEFT_EYE_INDICES]
        right_eye_points = landmarks[self.RIGHT_EYE_INDICES]
        
        # 计算EAR
        left_ear = self.calculate_ear_for_eye(left_eye_points)
        right_ear = self.calculate_ear_for_eye(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        
        return left_ear, right_ear, avg_ear
    
    def is_eye_closed(self, ear):
        """判断眼睛是否闭合"""
        return ear < self.ear_threshold
    
    def is_blinking(self, ear):
        """判断是否眨眼（更严格的阈值）"""
        return ear < self.blink_threshold