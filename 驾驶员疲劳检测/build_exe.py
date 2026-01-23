import PyInstaller.__main__
import os
import shutil

# 确保在项目根目录下运行
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

print(f"工作目录: {os.getcwd()}")

# 清理旧的构建文件
if os.path.exists('build'):
    try:
        shutil.rmtree('build')
    except Exception as e:
        print(f"清理 build 目录失败: {e}")

if os.path.exists('dist'):
    try:
        shutil.rmtree('dist')
    except Exception as e:
        print(f"清理 dist 目录失败: {e}")

# PyInstaller 参数
params = [
    os.path.join('fatigue_detection', 'main.py'),  # 主程序入口
    '--name=FatigueDetector',                      # 生成的exe名称
    '--onefile',                                   # 打包成单个文件
    '--clean',                                     # 清理缓存
    '--collect-all=mediapipe',                     # 收集 mediapipe 的所有文件（包含 DLL 和数据）
    f'--add-data={os.path.join(project_root, "models")};models', # 添加模型文件夹 (使用绝对路径)
    
    # 隐藏导入（显式声明以防万一）
    '--hidden-import=mediapipe',
    '--hidden-import=mediapipe.python',
    '--hidden-import=mediapipe.tasks',
    '--hidden-import=mediapipe.tasks.python',
    '--hidden-import=mediapipe.tasks.c',
    '--hidden-import=mediapipe.tasks.python.vision',
    '--hidden-import=mediapipe.tasks.python.audio',
    '--hidden-import=mediapipe.tasks.python.text',
    '--hidden-import=mediapipe.tasks.python.components',
    '--hidden-import=mediapipe.tasks.python.core',
    '--hidden-import=winsound',
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    
    # 确保 config 和其他模块能被找到
    '--paths=.',
]

print("开始打包...")
try:
    PyInstaller.__main__.run(params)
    print("\n" + "="*50)
    print("打包完成！")
    print(f"可执行文件位于: {os.path.join(project_root, 'dist', 'FatigueDetector.exe')}")
    print("="*50)
except Exception as e:
    print(f"\n打包失败: {e}")
