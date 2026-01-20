# detectors/landmark_detector.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os
import time
from utils.path_utils import get_model_path


class LandmarkDetector:
    """面部关键点检测器 - 使用 MediaPipe Tasks API"""
    
    # MediaPipe Face Mesh 关键点索引
    # 映射到类似 Dlib 68点的关键区域
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # 左眼6个点
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # 右眼6个点
    MOUTH_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]  # 嘴外轮廓
    MOUTH_INNER_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]  # 嘴内轮廓
    
    def __init__(self, static_image_mode=False, max_num_faces=1, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化关键点检测器
        
        Args:
            static_image_mode: 是否为静态图像模式 (Tasks API 中对应 IMAGE 模式，这里忽略该参数，始终使用 IMAGE 模式以简化调用)
            max_num_faces: 最大检测人脸数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度 (Tasks API IMAGE 模式下可能不适用，但保留参数签名)
        """
        model_path = get_model_path('face_landmarker.task')
        
        # 读取模型文件内容以避开路径包含中文/空格导致 MediaPipe 无法读取的问题
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence, # 使用 detection confidence 近似
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
    
    def detect(self, image, face_bbox=None):
        """
        检测面部关键点
        
        Args:
            image: BGR 图像（彩色或灰度）
            face_bbox: 人脸边界框 (x, y, w, h)，可选
            
        Returns:
            numpy.ndarray or None: 关键点坐标数组 (N, 2)
        """
        try:
            # 如果是灰度图，转换为RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 如果提供了人脸框，裁剪图像以提高性能
            if face_bbox is not None:
                x, y, w, h = face_bbox
                # 扩展边界框以包含更多上下文
                padding = int(max(w, h) * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                roi = image_rgb[y1:y2, x1:x2]
                roi = np.ascontiguousarray(roi) # 确保内存连续
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi)
                
                detection_result = self.landmarker.detect(mp_image)
                
                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    # 转换坐标到原图
                    h_roi, w_roi = roi.shape[:2]
                    coords = np.array([
                        [int(lm.x * w_roi) + x1, int(lm.y * h_roi) + y1]
                        for lm in landmarks
                    ])
                    return coords
            else:
                # 处理整张图像
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                detection_result = self.landmarker.detect(mp_image)
                
                if detection_result.face_landmarks:
                    landmarks = detection_result.face_landmarks[0]
                    h, w = image.shape[:2]
                    coords = np.array([
                        [int(lm.x * w), int(lm.y * h)]
                        for lm in landmarks
                    ])
                    return coords
            
            return None
            
        except Exception as e:
            print(f"关键点检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_eye_points(self, landmarks):
        """提取眼睛关键点"""
        if landmarks is None:
            return None, None
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        return left_eye, right_eye
    
    def extract_mouth_points(self, landmarks):
        """提取嘴部关键点"""
        if landmarks is None:
            return None
        # 合并内外轮廓
        mouth_indices = self.MOUTH_OUTER_INDICES + self.MOUTH_INNER_INDICES
        mouth = landmarks[mouth_indices]
        return mouth
    
    def draw_landmarks(self, image, landmarks, color=(0, 255, 255), thickness=1):
        """绘制面部关键点"""
        if landmarks is None:
            return image
        
        # 绘制眼睛关键点
        for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                cv2.circle(image, (int(x), int(y)), 2, color, -1)
        
        # 绘制眼睛轮廓
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        cv2.polylines(image, [left_eye.astype(np.int32)], True, color, thickness)
        cv2.polylines(image, [right_eye.astype(np.int32)], True, color, thickness)
        
        # 绘制嘴部关键点和轮廓
        for idx in self.MOUTH_OUTER_INDICES:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                cv2.circle(image, (int(x), int(y)), 2, color, -1)
        
        mouth = landmarks[self.MOUTH_OUTER_INDICES]
        cv2.polylines(image, [mouth.astype(np.int32)], True, color, thickness)
        
        return image
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()