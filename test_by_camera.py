from ultralytics import YOLO
import cv2
import tkinter as tk

if __name__ == '__main__':
    pass
    # 加载 YOLO 模型
    model = YOLO("best.pt")

    # 获取屏幕分辨率（使用 tkinter 更通用）
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.withdraw()  # 隐藏主窗口

    # 设置摄像头分辨率
    frame_width, frame_height = 1280, 960

    # 计算窗口左上角坐标，使窗口居中
    win_x = int((screen_width - frame_width) / 2)
    win_y = int((screen_height - frame_height) / 2)

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # 创建命名窗口并移动到屏幕中心
    cv2.namedWindow("YOLO Camera Detection", cv2.WINDOW_NORMAL)
    cv2.moveWindow("YOLO Camera Detection", win_x, win_y)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 推理
        results = model.predict(frame, conf=0.1, verbose=False)
        result = results[0]  # 当前帧的结果
        annotated_frame = result.plot()

        # 识别主体信息提取
        frame_detections = []
        boxes = result.boxes  # 检测框集合

        for box in boxes:
            cls_id = int(box.cls[0])  # 类别ID
            name = model.names[cls_id]  # 类别名称
            conf = float(box.conf[0])  # 置信度
            xyxy = box.xyxy[0].tolist()  # 边界框坐标 [x1, y1, x2, y2]

            det_info = {
                "name": name,
                "confidence": round(conf, 3),
                "bbox": [round(x, 2) for x in xyxy]
            }
            frame_detections.append(det_info)
            print(frame_detections)

        # 显示标注后的画面
        cv2.imshow("YOLO Camera Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()