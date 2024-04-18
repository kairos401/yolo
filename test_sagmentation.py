from ultralytics import YOLO
import cv2
import numpy as np
import random

model = YOLO("yolov8n-seg.pt")  # Load Model
conf = 0.5  # confidence

# Assign random color to each class
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
colors = {cls_id: random.choices(range(256), k=3) for cls_id in classes_ids}

# Check camera 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Can't open camera")
    exit()

# Realtime segmentation
while True:
    # Bring the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Analyzing
    results = model.predict(frame)

    # Rendering
    for result in results:
        
        # Case : Nothing detected
        if result.masks is None or result.boxes is None:
            print("result.boxes")
            continue
        
        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            color_number = int(box.cls[0])
            overlay = frame.copy() # 투명도 적용을 위해 복사본 생성
            cv2.fillPoly(overlay, points, colors[color_number])  # 복사본에 mask 적용
            alpha = 0.4  # 투명도 (0: 완전 투명, 1: 완전 불투명)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)  # 복사본과 원본 결합

    # Result
    cv2.imshow("YOLOv8 Segmentation", frame)

    # End the loop when press 'q' or ESC key
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

# Free Camera
cap.release()