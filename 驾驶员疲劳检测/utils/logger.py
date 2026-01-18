# utils/logger.py
import logging
import sys
from datetime import datetime
from fatigue_detection. config import CONFIG


class Logger:
    """日志记录器"""
    
    def __init__(self, name='FatigueDetection'):
        """初始化日志记录器"""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, CONFIG['logging']['level']))
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 控制台handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
            
            # 文件handler（如果启用）
            if CONFIG['logging']['enabled']:
                file_handler = logging.FileHandler(CONFIG['logging']['file'])
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)
    
    def debug(self, message):
        """记录调试信息"""
        self.logger.debug(message)
    
    def info(self, message):
        """记录一般信息"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录警告信息"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误信息"""
        self.logger.error(message)
    
    def critical(self, message):
        """记录严重错误信息"""
        self.logger.critical(message)
    
    def log_detection(self, frame_id, state, ear, mar, perclos):
        """记录检测结果"""
        log_msg = (f"帧 {frame_id}: 状态={state}, EAR={ear:.3f}, "
                  f"MAR={mar:.3f}, PERCLOS={perclos:.1%}")
        
        if state == "重度疲劳":
            self.warning(log_msg)
        elif state == "轻度疲劳":
            self.info(log_msg)
        else:
            self.debug(log_msg)


# 创建全局日志记录器实例
logger = Logger()