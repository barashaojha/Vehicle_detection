import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone

# Initialize YOLO model
model = YOLO('best.pt')

# Define variables
cy1 = 427
offset = 8
tracker = Tracker()
class_list = []

# Load class list from file
with open("coco1.txt", "r") as file:
    class_list = file.read().split("\n")

# Open video file
cap = cv2.VideoCapture('video (1).mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection using YOLO model
    results = model.predict(frame, conf=0.25, imgsz=320, save=True)
    boxes = pd.DataFrame(results.xyxy[0].numpy())

    # Initialize lists for different object classes
    class_boxes = [[] for _ in range(len(class_list))]

    # Iterate over detected boxes
    for _, row in boxes.iterrows():
        x1, y1, x2, y2, conf, class_id = row
        class_id = int(class_id)

        # Filter boxes based on confidence and class
        if conf > 0.25:
            class_boxes[class_id].append([int(x1), int(y1), int(x2), int(y2)])

    # Update object tracker with detected boxes
    for idx, class_box in enumerate(class_boxes):
        tracker.update(class_box, idx)

    # Draw bounding boxes and counts on frame
    for idx, bbox_idx in enumerate(tracker.bbox_idxs):
        for bbox in bbox_idx:
            x1, y1, x2, y2, id = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cy1 - offset < cy < cy1 + offset:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{id}', (x1, y1), 1, 1)

    # Display counts on frame
    for idx, count in enumerate(tracker.counts):
        cvzone.putTextRect(frame, f'count {class_list[idx]}: {count}', (50, 60 + 80 * idx), 2, 2)

    # Draw a line on frame
    cv2.line(frame, (254, 274), (884, 362), (255, 255, 255), 2)

    # Display frame
    cv2.imshow("RGB", frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
