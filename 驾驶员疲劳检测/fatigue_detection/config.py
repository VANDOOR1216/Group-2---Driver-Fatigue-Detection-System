# config.py
# 配置文件 - MediaPipe 版本
import cv2

CONFIG = {
    # 图像处理配置
    'frame': {
        'width': 640,
        'height': 480,
        'resize_factor': 1.0  # 图像缩放因子
    },
    
    # 人脸检测配置（MediaPipe）
    'face_detection': {
        'min_detection_confidence': 0.5,  # 最小检测置信度
        'model_selection': 0  # 0: 2米内检测, 1: 5米内检测
    },
    
    # 关键点检测配置（MediaPipe Face Mesh）
    'landmark_detection': {
        'static_image_mode': False,
        'max_num_faces': 1,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    },
    
    # EAR配置（MediaPipe 的 EAR 值范围可能不同）
    'ear': {
        'threshold': 0.18,        # EAR阈值（降低阈值以减少正常眨眼或小眼睛的误报，原0.2）
        'blink_threshold': 0.15,  # 眨眼阈值
        'frame_buffer': 3         # 眨眼确认的帧数缓冲
    },
    
    # MAR配置
    'mar': {
        'threshold': 0.6,         # MAR阈值（提高阈值以避免说话被误判为哈欠，原0.5）
        'yawn_duration': 5        # 打哈欠持续帧数（原10）
    },
    
    # 疲劳检测配置
    'fatigue': {
        'window_size': 150,       # PERCLOS计算窗口大小（帧数）- 约5秒
        'perclos_mild': 0.30,     # 轻度疲劳阈值（提高到30%以减少误报，原0.25）
        'perclos_severe': 0.5,    # 重度疲劳阈值（50%）
        'yawn_threshold': 4,      # 打哈欠告警次数
        'continuous_blink': 8,    # 连续眨眼告警次数
        'max_closed_frames': 30   # 最大连续闭眼帧数（约1秒，原25帧）
    },
    
    # 可视化配置
    'visualization': {
        'show_landmarks': True,   # 显示面部关键点
        'show_ear': True,         # 显示EAR值
        'show_mar': True,         # 显示MAR值
        'show_perclos': True,     # 显示PERCLOS值
        
        # 颜色配置
        'colors': {
            'normal': (0, 255, 0),     # 正常状态 - 绿色
            'mild': (0, 255, 255),     # 轻度疲劳 - 黄色
            'severe': (0, 0, 255),     # 重度疲劳 - 红色
            'bbox': (255, 0, 0),       # 人脸框 - 蓝色
            'landmark': (0, 255, 255), # 关键点 - 青色
            'text': (255, 255, 255)    # 文字 - 白色
        },
        
        # 文字显示配置
        'text': {
            'font': 0,  # 使用整数 0 代替 cv2.FONT_HERSHEY_SIMPLEX
            'scale': 0.5,
            'thickness': 1,
            'line_type': 16  # cv2.LINE_AA 的整数值为 16
        }
    },
    
    # 日志配置
    'logging': {
        'enabled': True,
        'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
        'file': 'fatigue_detection.log'
    }
}