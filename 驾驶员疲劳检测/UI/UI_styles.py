# ui/styles.py
"""
UI样式定义
"""
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt

# 颜色定义
COLORS = {
    # 主色调
    'primary': '#3498db',      # 蓝色
    'secondary': '#2ecc71',    # 绿色
    'danger': '#e74c3c',       # 红色
    'warning': '#f39c12',      # 橙色
    'dark': '#2c3e50',         # 深蓝灰
    'light': '#ecf0f1',        # 浅灰
    
    # 状态颜色
    'normal': '#27ae60',       # 正常 - 绿色
    'mild': '#f39c12',         # 轻度疲劳 - 橙色
    'severe': '#e74c3c',       # 重度疲劳 - 红色
    
    # UI元素
    'background': '#34495e',   # 背景色
    'card_bg': '#2c3e50',      # 卡片背景
    'text': '#ecf0f1',         # 文字颜色
    'border': '#7f8c8d',       # 边框颜色
}

# 字体定义
FONTS = {
    'title': QFont('Arial', 16, QFont.Bold),
    'heading': QFont('Arial', 14, QFont.Bold),
    'subheading': QFont('Arial', 12, QFont.Bold),
    'normal': QFont('Arial', 10),
    'small': QFont('Arial', 9),
}

# 样式表
STYLESHEET = f"""
/* 主窗口样式 */
QMainWindow {{
    background-color: {COLORS['background']};
}}

/* 标签页样式 */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    background-color: {COLORS['card_bg']};
    border-radius: 5px;
}}

QTabBar::tab {{
    background-color: {COLORS['card_bg']};
    color: {COLORS['text']};
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}

QTabBar::tab:selected {{
    background-color: {COLORS['primary']};
    font-weight: bold;
}}

/* 按钮样式 */
QPushButton {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-weight: bold;
    font-size: 12px;
}}

QPushButton:hover {{
    background-color: #2980b9;
}}

QPushButton:pressed {{
    background-color: #21618c;
}}

QPushButton:disabled {{
    background-color: #7f8c8d;
    color: #bdc3c7;
}}

/* 开始按钮 */
QPushButton#startButton {{
    background-color: {COLORS['secondary']};
}}

QPushButton#startButton:hover {{
    background-color: #27ae60;
}}

/* 停止按钮 */
QPushButton#stopButton {{
    background-color: {COLORS['danger']};
}}

QPushButton#stopButton:hover {{
    background-color: #c0392b;
}}

/* 上传按钮 */
QPushButton#uploadButton {{
    background-color: {COLORS['warning']};
}}

QPushButton#uploadButton:hover {{
    background-color: #d68910;
}}

/* 标签样式 */
QLabel {{
    color: {COLORS['text']};
    font-size: 12px;
}}

QLabel#titleLabel {{
    font-size: 18px;
    font-weight: bold;
    color: white;
}}

QLabel#statusLabel {{
    font-size: 14px;
    font-weight: bold;
    padding: 5px 10px;
    border-radius: 3px;
}}

/* 文本框样式 */
QTextEdit, QPlainTextEdit {{
    background-color: {COLORS['dark']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 5px;
    font-family: Consolas, monospace;
}}

QLineEdit {{
    background-color: {COLORS['dark']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 5px;
}}

/* 组合框样式 */
QGroupBox {{
    color: {COLORS['text']};
    font-weight: bold;
    border: 2px solid {COLORS['border']};
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
}}

/* 进度条样式 */
QProgressBar {{
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    text-align: center;
    background-color: {COLORS['dark']};
}}

QProgressBar::chunk {{
    background-color: {COLORS['primary']};
    border-radius: 5px;
}}

/* 表格样式 */
QTableWidget {{
    background-color: {COLORS['dark']};
    color: {COLORS['text']};
    gridline-color: {COLORS['border']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
}}

QTableWidget::item {{
    padding: 5px;
}}

QHeaderView::section {{
    background-color: {COLORS['card_bg']};
    color: {COLORS['text']};
    padding: 5px;
    border: none;
    font-weight: bold;
}}

/* 滚动条样式 */
QScrollBar:vertical {{
    background-color: {COLORS['dark']};
    width: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['primary']};
    border-radius: 6px;
    min-height: 20px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background: none;
}}

/* 状态指示器 */
QFrame#statusIndicator {{
    border-radius: 7px;
    min-width: 14px;
    min-height: 14px;
    max-width: 14px;
    max-height: 14px;
}}

/* 视频显示区域 */
QFrame#videoFrame {{
    border: 2px solid {COLORS['border']};
    border-radius: 5px;
    background-color: black;
}}
"""

# 状态标签样式映射
STATUS_STYLES = {
    'normal': f"background-color: {COLORS['normal']}; color: white;",
    'mild': f"background-color: {COLORS['mild']}; color: white;",
    'severe': f"background-color: {COLORS['severe']}; color: white;",
    'inactive': f"background-color: {COLORS['border']}; color: {COLORS['text']};"
}

def apply_dark_theme(app):
    """应用深色主题"""
    app.setStyleSheet(STYLESHEET)
    
    # 设置调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS['background']))
    palette.setColor(QPalette.WindowText, QColor(COLORS['text']))
    palette.setColor(QPalette.Base, QColor(COLORS['dark']))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS['card_bg']))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS['text']))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS['text']))
    palette.setColor(QPalette.Text, QColor(COLORS['text']))
    palette.setColor(QPalette.Button, QColor(COLORS['primary']))
    palette.setColor(QPalette.ButtonText, QColor(COLORS['text']))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(COLORS['primary']))
    palette.setColor(QPalette.Highlight, QColor(COLORS['primary']))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)