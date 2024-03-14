#import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *

cap = cv2.VideoCapture(1)  #webcam
cap.set(3, 1280)
cap.set(4, 720)
#cap = cv2.VideoCapture("../Videos/ghy.mp4")  # For Video

model = YOLO("Datasets/runs/detect/train/weights/best.pt")

classNames = ["tractor",
              "car",
              "bike",
              "rickshaw",
              "bus",
              "traveller",
              "truck",
              "Auto-rickshaw",
              "tempo"]
# Load img and mask
mask = cv2.imread("mask.png")


# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []
#
prev_frame_time=0
new_frame_time=0
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    #imgRegion: None = cv2.bitwise_and(img, mask)

    #imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    #img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    #results = model(imgRegion, stream=True)

    #detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cvzone.putTextRect(img, f'{conf}',(max(0, x1), max(35, y1)))

            # Class Name

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            #cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.9, thickness= 1)

            #label = f'{currentClass}{conf}'
            #t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            # print(t_size)
            #c2 = x1 + t_size[0], y1 - t_size[1] - 3

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
            or currentClass == "bike" or currentClass == "tractor" or currentClass == "rickshaw" \
            or currentClass == "auto-rickshaw" or currentClass == "tempo" or currentClass == "traveller" and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.array([])
                detections = np.vstack(detections, currentArray)

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    #cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    #cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()


