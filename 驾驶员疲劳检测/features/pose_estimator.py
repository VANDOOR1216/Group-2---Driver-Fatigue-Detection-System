# features/pose_estimator.py
import numpy as np
import cv2


class PoseEstimator:
    """头部姿态估计器"""
    
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        """
        初始化姿态估计器
        
        Args:
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
        """
        # 如果没有提供相机参数，使用默认值（近似值）
        if camera_matrix is None:
            # 假设相机参数（可根据实际情况调整）
            self.camera_matrix = np.array([
                [640, 0, 320],
                [0, 640, 240],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
            
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1))
        else:
            self.dist_coeffs = dist_coeffs
        
        # 3D面部模型点（基于68点模型的近似3D坐标）
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 鼻尖
            (0.0, -330.0, -65.0),        # 下巴
            (-225.0, 170.0, -135.0),     # 左眼左角
            (225.0, 170.0, -135.0),      # 右眼右角
            (-150.0, -150.0, -125.0),    # 左嘴角
            (150.0, -150.0, -125.0)      # 右嘴角
        ], dtype=np.float32)
        
        # 对应的2D关键点索引
        self.image_point_indices = [30, 8, 36, 45, 48, 54]
    
    def estimate(self, shape):
        """
        估计头部姿态
        
        Args:
            shape: 面部关键点坐标 (68, 2)
            
        Returns:
            tuple: (旋转向量, 平移向量, 欧拉角)
        """
        # 提取对应的2D点
        image_points = np.array([
            shape[self.image_point_indices[0]],  # 鼻尖
            shape[self.image_point_indices[1]],  # 下巴
            shape[self.image_point_indices[2]],  # 左眼左角
            shape[self.image_point_indices[3]],  # 右眼右角
            shape[self.image_point_indices[4]],  # 左嘴角
            shape[self.image_point_indices[5]]   # 右嘴角
        ], dtype=np.float32)
        
        # 使用PnP求解姿态
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None
        
        # 将旋转向量转换为欧拉角
        euler_angles = self.rotation_vector_to_euler(rotation_vector)
        
        return rotation_vector, translation_vector, euler_angles
    
    @staticmethod
    def rotation_vector_to_euler(rotation_vector):
        """将旋转向量转换为欧拉角（俯仰、偏航、滚转）"""
        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # 从旋转矩阵提取欧拉角
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0
        
        # 转换为度
        euler_angles = np.degrees([x, y, z])
        
        return euler_angles
    
    def is_head_nodding(self, euler_angles, threshold=15):
        """判断是否点头（俯仰角变化）"""
        if euler_angles is None:
            return False
        
        pitch = euler_angles[0]  # 俯仰角
        return abs(pitch) > threshold