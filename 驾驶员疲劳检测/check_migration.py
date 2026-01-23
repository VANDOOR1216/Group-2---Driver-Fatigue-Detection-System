import os
import re

def check_files(start_path):
    issues = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(r'from\s+PyQt[56]', content) or re.search(r'import\s+PyQt[56]', content):
                            issues.append(f"{file_path} (Import)")
                        if 'pyqtSignal' in content:
                            issues.append(f"{file_path} (pyqtSignal)")
                        if 'pyqtSlot' in content:
                            issues.append(f"{file_path} (pyqtSlot)")
                        if '.exec_()' in content and 'PySide6' in content:
                             issues.append(f"{file_path} (exec_)")
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")
    return issues

if __name__ == "__main__":
    start_path = r"c:\Users\VAN DOOR\Desktop\Group 2 - Driver Fatigue Detection System\驾驶员疲劳检测"
    issues = check_files(start_path)
    if issues:
        print("Found files with PyQt5/PyQt6 imports:")
        for issue in issues:
            print(issue)
    else:
        print("No PyQt5/PyQt6 imports found.")
