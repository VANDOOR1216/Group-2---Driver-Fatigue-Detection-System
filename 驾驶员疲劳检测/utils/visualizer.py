# utils/visualizer.py
import cv2
import numpy as np
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'fatigue_detection'))
from config import CONFIG


class Visualizer:
    """可视化工具类"""
    
    # MediaPipe Face Mesh 关键点索引
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    MOUTH_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    MOUTH_INNER_INDICES = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
    
    def __init__(self):
        """初始化可视化器"""
        self.colors = CONFIG['visualization']['colors']
        self.text_config = CONFIG['visualization']['text']
        
    def draw_face_bbox_dict(self, image, face_info, color=None):
        """
        绘制人脸边界框（字典格式）
        
        Args:
            image: 输入图像
            face_info: 人脸信息字典 {'bbox': (x, y, w, h), 'confidence': float}
            color: 框的颜色 (B, G, R)
            
        Returns:
            numpy.ndarray: 绘制后的图像
        """
        if color is None:
            color = self.colors['bbox']
        
        # 获取边界框坐标
        x, y, w, h = face_info['bbox']
        confidence = face_info.get('confidence', 0.0)
        
        # 绘制矩形
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # 添加标签
        label = f"Face: {confidence:.2f}"
        cv2.putText(image, label, (x, y - 10), 
                   self.text_config['font'], self.text_config['scale'],
                   color, self.text_config['thickness'])
        
        return image
    
    def draw_face_bbox(self, image, face_rect, color=None):
        """
        绘制人脸边界框
        
        Args:
            image: 输入图像
            face_rect: 人脸矩形 (dlib.rectangle)
            color: 框的颜色 (B, G, R)
            
        Returns:
            numpy.ndarray: 绘制后的图像
        """
        if color is None:
            color = self.colors['bbox']
        
        # 获取矩形坐标
        x1, y1 = face_rect.left(), face_rect.top()
        x2, y2 = face_rect.right(), face_rect.bottom()
        
        # 绘制矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 添加标签
        cv2.putText(image, "Face", (x1, y1-10), 
                   self.text_config['font'], self.text_config['scale'],
                   color, self.text_config['thickness'])
        
        return image
    
    def draw_landmarks(self, image, shape, color=None):
        """
        绘制面部关键点
        
        Args:
            image: 输入图像
            shape: 关键点坐标 (478, 2)
            color: 关键点颜色
            
        Returns:
            numpy.ndarray: 绘制后的图像
        """
        if color is None:
            color = self.colors['landmark']
        
        # 绘制眼睛轮廓
        left_eye = shape[self.LEFT_EYE_INDICES]
        right_eye = shape[self.RIGHT_EYE_INDICES]
        
        # 闭合多边形
        cv2.polylines(image, [left_eye.astype(np.int32)], True, color, 1)
        cv2.polylines(image, [right_eye.astype(np.int32)], True, color, 1)
        
        # 绘制嘴部轮廓
        mouth_outer = shape[self.MOUTH_OUTER_INDICES]
        mouth_inner = shape[self.MOUTH_INNER_INDICES]
        
        cv2.polylines(image, [mouth_outer.astype(np.int32)], True, color, 1)
        cv2.polylines(image, [mouth_inner.astype(np.int32)], True, color, 1)
        
        # 可选：绘制关键点（仅绘制眼睛和嘴部点，避免过于杂乱）
        for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
            pt = shape[idx]
            cv2.circle(image, (int(pt[0]), int(pt[1])), 1, color, -1)
            
        return image
    
    def draw_status(self, image, state, left_ear, right_ear, mar, stats):
        """
        绘制状态信息
        
        Args:
            image: 输入图像
            state: 疲劳状态
            left_ear: 左眼EAR
            right_ear: 右眼EAR
            mar: MAR值
            stats: 统计信息字典
            
        Returns:
            numpy.ndarray: 绘制后的图像
        """
        # Determine state color
        if state.value == "Severe Fatigue":
            state_color = self.colors['severe']
        elif state.value == "Mild Fatigue":
            state_color = self.colors['mild']
        else:
            state_color = self.colors['normal']
        
        # Draw status bar
        bar_height = 30
        cv2.rectangle(image, (0, 0), (image.shape[1], bar_height),
                     state_color, -1)
        
        # Display status text
        status_text = f"State: {state.value}"
        cv2.putText(image, status_text, (10, 20),
                   self.text_config['font'], 0.7, self.colors['text'], 2)
        
        # Display EAR and MAR values
        y_offset = bar_height + 20
        
        if CONFIG['visualization']['show_ear']:
            ear_text = f"EAR: L={left_ear:.3f}  R={right_ear:.3f}"
            cv2.putText(image, ear_text, (10, y_offset),
                       self.text_config['font'], self.text_config['scale'],
                       self.colors['text'], self.text_config['thickness'])
            y_offset += 20
        
        if CONFIG['visualization']['show_mar']:
            mar_text = f"MAR: {mar:.3f}"
            cv2.putText(image, mar_text, (10, y_offset),
                       self.text_config['font'], self.text_config['scale'],
                       self.colors['text'], self.text_config['thickness'])
            y_offset += 20
        
        if CONFIG['visualization']['show_perclos'] and 'perclos' in stats:
            perclos_text = f"PERCLOS: {stats['perclos']:.1%}"
            cv2.putText(image, perclos_text, (10, y_offset),
                       self.text_config['font'], self.text_config['scale'],
                       self.colors['text'], self.text_config['thickness'])
            y_offset += 20
        
        # Display statistics
        blink_text = f"Blink: {stats.get('blink_count', 0)}"
        cv2.putText(image, blink_text, (image.shape[1] - 150, bar_height + 20),
                   self.text_config['font'], self.text_config['scale'],
                   self.colors['text'], self.text_config['thickness'])
        
        yawn_text = f"Yawn: {stats.get('yawn_count', 0)}"
        cv2.putText(image, yawn_text, (image.shape[1] - 150, bar_height + 40),
                   self.text_config['font'], self.text_config['scale'],
                   self.colors['text'], self.text_config['thickness'])
        
        # Add warning if state is abnormal
        if state.value != "Normal":
            warning_text = "WARNING: FATIGUE DRIVING!"
            cv2.putText(image, warning_text, 
                       (image.shape[1] // 2 - 150, image.shape[0] - 20),
                       self.text_config['font'], 0.8, self.colors['severe'], 2)
        
        return image
    
    def draw_alert(self, image, message, color=None):
        """
        绘制警告信息
        
        Args:
            image: 输入图像
            message: 警告信息
            color: 警告颜色
            
        Returns:
            numpy.ndarray: 绘制后的图像
        """
        if color is None:
            color = self.colors['severe']
        
        # 在图像中央绘制警告框
        h, w = image.shape[:2]
        box_height = 80
        box_width = w - 100
        
        cv2.rectangle(image, 
                     (50, h//2 - box_height//2),
                     (50 + box_width, h//2 + box_height//2),
                     color, -1)
        
        # 绘制警告文本
        text_size = cv2.getTextSize(message, 
                                   self.text_config['font'], 1.2, 2)[0]
        text_x = 50 + (box_width - text_size[0]) // 2
        text_y = h//2 + text_size[1]//2
        
        cv2.putText(image, message, (text_x, text_y),
                   self.text_config['font'], 1.2, self.colors['text'], 2)
        
        return image