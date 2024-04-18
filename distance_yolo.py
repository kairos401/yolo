import torch
#from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv5 라이브러리 가져오기
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# 카메라 내부 파라미터 (예시)
# 실제로 이 값은 카메라 캘리브레이션을 통해 얻어야 합니다.
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])

# 객체 크기를 픽셀 단위에서 실제 크기로 변환하기 위한 비율 (예시)
# 실제로 이 값은 측정된 객체의 크기에 따라 달라질 수 있습니다.
object_real_size = 0.1 # 미터 단위로 0.1m

# 객체 크기 추정 함수 정의
def estimate_object_size(object_bbox):
    # 경계 상자의 너비와 높이를 사용하여 객체의 크기를 추정
    bbox_width = object_bbox[2] - object_bbox[0]
    bbox_height = object_bbox[3] - object_bbox[1]
    object_size = max(bbox_width, bbox_height)
    return object_size

# 거리 추정 함수 정의
def estimate_distance(object_size_pixels):
    # 픽셀 단위의 객체 크기를 실제 크기로 변환
    object_size_real = object_size_pixels * object_real_size / object_real_size  # 수정된 부분

    # 삼각측량을 사용하여 거리 추정
    distance = (camera_matrix[0, 0] * object_real_size) / object_size_real
    return distance



while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        results = model(frame)
        annotated_frame = frame.copy()
        for r in results:
             for box in r.boxes:
                if box.cls.item() == 0:
                    #annotated_frame = results[0].plot()
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    xc = int((x1 + x2) / 2)
                    yc = int((y1 + y2) / 2) 

                    annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    point = (xc, yc)
                    cv2.circle(annotated_frame, point, 2, (255,0,0),-1)

                    # 객체 크기 추정
                    object_size_pixels = max(x2 - x1, y2 - y1)

                    # 거리 추정
                    distance = estimate_distance(object_size_pixels)
                    #print("Estimated distance to object:", distance)
                    # 거리를 화면에 표시
                    cv2.putText(annotated_frame, f"Distance: {distance:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        cv2.imshow("frame", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
