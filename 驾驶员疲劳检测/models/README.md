# Models 目录

## MediaPipe 版本说明

本项目使用 **MediaPipe**，无需下载额外的模型文件！

MediaPipe 的模型已经内置在库中，安装后即可直接使用。

## 优势

- ✅ **无需下载**：模型内置，开箱即用
- ✅ **自动更新**：随 MediaPipe 库更新
- ✅ **高精度**：468 点面部标记
- ✅ **高性能**：优化的推理引擎

## 使用的模型

1. **Face Detection**
   - 用于检测人脸位置
   - 基于 BlazeFace 模型
   - 支持 2 米和 5 米检测范围

2. **Face Mesh**
   - 用于检测 468 个面部关键点
   - 实时性能优异
   - 支持多人脸检测

## 与 Dlib 的对比

| 特性 | MediaPipe | Dlib |
|------|-----------|------|
| 模型文件 | 内置 | 需下载 99.7MB |
| 安装难度 | 简单 | 需要编译器 |
| 关键点数 | 468 点 | 68 点 |
| 速度 | 快 | 较慢 |
| 精度 | 高 | 中等 |

## 参考资料

- [MediaPipe 官方文档](https://google.github.io/mediapipe/)
- [Face Detection Guide](https://google.github.io/mediapipe/solutions/face_detection)
- [Face Mesh Guide](https://google.github.io/mediapipe/solutions/face_mesh)
