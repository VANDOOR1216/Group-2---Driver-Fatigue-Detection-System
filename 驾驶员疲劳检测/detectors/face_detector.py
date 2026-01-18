# detectors/face_detector.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os


class FaceDetector:
    """人脸检测器 - 使用 MediaPipe Tasks API"""
    
    def __init__(self, min_detection_confidence=0.5):
        """
        初始化人脸检测器
        
        Args:
            min_detection_confidence: 最小检测置信度 (0.0-1.0)
        """
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'blaze_face_short_range.tflite')
        
        # 读取模型文件内容以避开路径包含中文/空格导致 MediaPipe 无法读取的问题
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)
    
    def detect(self, image):
        """
        检测图像中的人脸
        
        Args:
            image: BGR 图像（彩色或灰度）
            
        Returns:
            list: 人脸检测结果列表，每个元素包含 (x, y, w, h, confidence)
        """
        # 如果是灰度图，转换为RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建 MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # 检测人脸
        detection_result = self.detector.detect(mp_image)
        
        faces = []
        if detection_result.detections:
            for detection in detection_result.detections:
                # 获取边界框
                bbox = detection.bounding_box
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                width = int(bbox.width)
                height = int(bbox.height)
                confidence = detection.categories[0].score
                
                # 确保坐标在图像范围内 (虽然 Tasks API 通常返回合法的，但双保险)
                h, w = image.shape[:2]
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': confidence
                })
        
        return faces
    
    def get_main_face(self, image):
        """
        获取图像中的主要人脸（面积最大的人脸）
        
        Args:
            image: BGR 图像
            
        Returns:
            dict or None: 主要人脸信息 {'bbox': (x, y, w, h), 'confidence': float}
        """
        faces = self.detect(image)
        
        if len(faces) == 0:
            return None
        
        # 按面积排序，返回最大的人脸
        faces = sorted(faces, 
                      key=lambda f: f['bbox'][2] * f['bbox'][3],
                      reverse=True)
        
        return faces[0]
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'detector'):
            self.detector.close()