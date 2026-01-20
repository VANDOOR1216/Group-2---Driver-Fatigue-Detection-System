import cv2
import sys
import time
from detectors.face_detector import FaceDetector
from detectors.landmark_detector import LandmarkDetector

def test_camera_and_face_detection():
    print("正在初始化摄像头...")
    # 尝试使用 CAP_DSHOW 加速 Windows 下的摄像头开启
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("无法打开摄像头 (index 0). 尝试不使用 CAP_DSHOW...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误: 无法打开任何摄像头。请检查连接。")
            return

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("摄像头已打开。正在初始化人脸检测器和关键点检测器...")
    try:
        detector = FaceDetector()
        print("人脸检测器初始化成功。")
        landmark_detector = LandmarkDetector()
        print("关键点检测器初始化成功。")
    except Exception as e:
        print(f"检测器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("开始视频流... 按 'q' 退出。")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧。")
            break

        # 检测人脸
        faces = detector.detect(frame)
        
        # 绘制人脸框和关键点
        for face in faces:
            x, y, w, h = face['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {face['confidence']:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 检测关键点
            landmarks = landmark_detector.detect(frame, face['bbox'])
            if landmarks is not None:
                for (lx, ly) in landmarks:
                    cv2.circle(frame, (lx, ly), 1, (0, 0, 255), -1)
        
        # 计算 FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Camera Test - Face & Landmarks', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("测试结束。")

if __name__ == "__main__":
    # 确保可以导入模块
    sys.path.append('.')
    test_camera_and_face_detection()
