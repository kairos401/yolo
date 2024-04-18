import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize track history and last assigned ID dictionaries
track_history = defaultdict(list)
last_assigned_id = defaultdict(int)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=False)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Initialize object IDs dictionary
        object_ids = {}

        # Update object IDs if not already done
        for result in results:
            for box in result.boxes:
                obj_id = box.id.item()
                if obj_id not in object_ids:
                    last_assigned_id[obj_id] += 1
                    object_ids[obj_id] = last_assigned_id[obj_id]

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 90:  # Adjust as needed
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the ID of the tracked object
            cv2.putText(frame, f"ID: {track_id}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
