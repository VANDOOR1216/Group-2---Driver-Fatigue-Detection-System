# 驾驶员疲劳检测系统 (MediaPipe + PySide6)

基于计算机视觉的实时驾驶员疲劳检测系统，采用 **MediaPipe** 进行高精度面部关键点检测，并配备现代化的 **PySide6** 图形用户界面。

## 📋 项目简介

本系统通过摄像头或视频文件实时监测驾驶员的面部特征，综合分析眼睛开合状态（EAR）、嘴部开合状态（MAR）和头部姿态，结合 PERCLOS 准则判断疲劳程度，并提供视听双重预警。

## ✨ 主要功能

### 1. 核心检测能力
- ✅ **高精度人脸检测**：使用 MediaPipe Face Detection，鲁棒性强。
- ✅ **面部关键点提取**：MediaPipe Face Mesh（468 点），精准捕捉面部细微变化。
- ✅ **多维疲劳指标**：
  - **EAR (Eye Aspect Ratio)**：监测眨眼频率、闭眼时长。
  - **MAR (Mouth Aspect Ratio)**：监测打哈欠行为。
  - **PERCLOS**：计算单位时间内眼睛闭合的百分比。

### 2. 现代化交互界面 (PySide6)
- ✅ **暗色主题 UI**：专业、护眼的深色界面设计。
- ✅ **实时数据可视化**：
  - 动态显示 EAR、MAR、PERCLOS 数值曲线。
  - 实时显示 FPS、眨眼次数、哈欠次数统计。
- ✅ **灵活的视频源选择**：
  - 支持多摄像头切换。
  - **本地视频文件分析**：支持导入 MP4/AVI 等视频文件进行离线检测。
  - **播放速度控制**：支持 0.1x - 3.0x 变速播放，方便快速回溯或慢动作分析。

### 3. 智能预警系统
- ✅ **视觉警报**：检测画面出现红色警告框和文字提示。
- ✅ **声音警报**：触发系统蜂鸣器（Beep）进行听觉提醒。
- ✅ **状态分级**：正常 (Green) -> 轻度疲劳 (Yellow) -> 重度疲劳 (Red)。

## 🛠️ 技术栈

- **编程语言**：Python 3.8+
- **GUI 框架**：PySide6 (Qt for Python)
- **核心算法**：MediaPipe (Google), OpenCV
- **数据处理**：NumPy, Pandas

## 🚀 快速开始

### 1. 环境要求
- Python 3.8 - 3.11
- 摄像头（用于实时检测）

### 2. 安装依赖

```bash
# 进入项目目录
cd fatigue_detection

# 安装所有依赖
pip install -r requirements.txt
```

> **注意**：如果安装 MediaPipe 遇到问题，请确保 Python 版本在 3.8-3.11 之间。

### 3. 运行程序

启动图形化界面：

```bash
python main.py
```

## 📖 使用说明

### 实时检测模式
1. 启动程序后，进入"实时检测"标签页。
2. **摄像头模式**：选择"摄像头"，点击"开始检测"。
3. **视频模式**：
   - 选择"本地视频文件"。
   - 点击"浏览"选择视频文件。
   - 使用下方的**速度滑块**调整播放速度（支持快进/慢放）。
   - 点击"开始检测"。

### 参数调整
可以在 `fatigue_detection/config.py` 中调整核心阈值：

```python
'fatigue': {
    'perclos_mild': 0.30,     # 轻度疲劳阈值 (30%)
    'perclos_severe': 0.5,    # 重度疲劳阈值 (50%)
    'max_closed_frames': 30   # 最大连续闭眼帧数
}
```

## 📁 项目结构

```
fatigue_detection/
├── main.py                 # 程序主入口
├── config.py               # 配置文件
├── requirements.txt        # 依赖列表
├── UI/                     # 界面模块
│   ├── UI_main_windows.py  # 主窗口
│   ├── UI_realtime_tab.py  # 实时检测页 (含视频流处理)
│   ├── UI_worker.py        # 工作线程 (含算法逻辑)
│   └── UI_styles.py        # 样式表
├── detectors/              # 视觉算法模块
│   ├── face_detector.py    # 人脸检测
│   └── landmark_detector.py # 关键点定位
├── features/               # 特征计算模块
│   ├── ear_calculator.py   # EAR 计算
│   └── mar_calculator.py   # MAR 计算
└── fatigue/                # 业务逻辑模块
    └── state_tracker.py    # 疲劳状态机
```

## ⚠️ 常见问题

**Q: 视频播放速度太快？**
A: 在视频文件模式下，使用界面上的"播放速度"滑块将速度调至 1.0x 或更低。

**Q: 报错 `ImportError: DLL load failed`？**
A: 通常是 OpenCV 或 PySide6 依赖缺失，尝试重装：`pip install --force-reinstall opencv-python PySide6`。

**Q: 摄像头无法打开？**
A: 检查摄像头是否被其他程序（如 Zoom、Teams）占用，或尝试更改 `main.py` 中的摄像头索引。
