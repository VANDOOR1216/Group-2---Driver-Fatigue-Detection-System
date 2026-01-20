# features/mar_calculator.py
import numpy as np
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'fatigue_detection'))
from config import CONFIG
 

class MARCalculator:
    """嘴部纵横比计算器 - 适配 MediaPipe"""
    
    # MediaPipe Face Mesh 嘴部关键点索引
    MOUTH_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    MOUTH_INNER_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
    
    # 用于计算 MAR 的关键点
    MOUTH_TOP = 13      # 上嘴唇中心
    MOUTH_BOTTOM = 14   # 下嘴唇中心
    MOUTH_LEFT = 61     # 嘴角左
    MOUTH_RIGHT = 291   # 嘴角右
    
    def __init__(self):
        """初始化MAR计算器"""
        self.mar_threshold = CONFIG['mar']['threshold']
    
    @staticmethod
    def calculate(landmarks):
        """
        计算嘴部纵横比
        
        Args:
            landmarks: 面部关键点坐标 (468, 2) - MediaPipe Face Mesh
            
        Returns:
            float: MAR值
        """
        if landmarks is None or len(landmarks) < 468:
            return 0.0
        
        # 使用更精确的嘴部点
        # 上嘴唇点
        upper_lip_top = landmarks[13]
        upper_lip_points = [landmarks[82], landmarks[13], landmarks[312]]
        
        # 下嘴唇点
        lower_lip_bottom = landmarks[14]
        lower_lip_points = [landmarks[87], landmarks[14], landmarks[317]]
        
        # 嘴角点
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        
        # 计算嘴部宽度（水平距离）
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        
        # 计算嘴部高度（垂直距离）- 使用多个点的平均值
        height_1 = np.linalg.norm(landmarks[13] - landmarks[14])  # 中心
        height_2 = np.linalg.norm(landmarks[82] - landmarks[87])  # 左侧
        height_3 = np.linalg.norm(landmarks[312] - landmarks[317])  # 右侧
        
        mouth_height = (height_1 + height_2 + height_3) / 3.0
        
        # 计算MAR
        mar = mouth_height / (mouth_width + 1e-6)  # 避免除零
        
        return mar
    
    def is_yawning(self, mar):
        """判断是否打哈欠"""
        return mar > self.mar_threshold