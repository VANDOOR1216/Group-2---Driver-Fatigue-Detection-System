import sys
import os

def get_base_path():
    """
    获取项目基础路径，兼容 PyInstaller 打包环境
    
    Returns:
        str: 项目根目录的绝对路径
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的运行环境
        if hasattr(sys, '_MEIPASS'):
            return sys._MEIPASS
        else:
            return os.path.dirname(sys.executable)
    else:
        # 开发环境：返回项目根目录
        # 此文件位于 utils/path_utils.py，所以根目录是上两级
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_model_path(model_name):
    """
    获取模型文件的绝对路径
    
    Args:
        model_name: 模型文件名
        
    Returns:
        str: 模型文件的完整路径
    """
    base_path = get_base_path()
    return os.path.join(base_path, 'models', model_name)
