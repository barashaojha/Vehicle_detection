import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone
import numpy as np


model = YOLO('Datasets/runs/detect/train/weights/best.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
        #print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('inshot/video (8).mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)
count=0
#area1 = [(255, 275), (830, 460), (875, 384), (456,241)]
#area2 = [(763, 17), (969, 343), (1016, 298), (924, 16)]

#tracker = Tracker()
#cy1=383
#cy2=500

cy1=274
offset=6


tracker=Tracker()
tracker1=Tracker()
tracker2=Tracker()
tracker3=Tracker()
tracker4=Tracker()
tracker5=Tracker()
tracker6=Tracker()
# tracker7=Tracker()
# tracker8=Tracker()
bus=[]
car=[]
truck=[]
auto_rickshaw=[]
motorcycle=[]
rickshaw=[]
tempo=[]
#traveller=[]
#tractor=[]


while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame, conf=0.25, imgsz=320, save=True)  #
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    #    print(px)
    list = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    # list7 = []
    # list8 = []

    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        d = int(row[5])
        c = class_list[d]

        if 'bus' in c:
            list.append([x1, y1, x2, y2])
        elif 'car' in c:
            list1.append([x1, y1, x2, y2])
        elif 'auto-rickshaw' in c:
            list2.append([x1, y1, x2, y2])
        elif 'motor-cycle' in c:
            list3.append([x1, y1, x2, y2])
        elif 'truck' in c:
            list4.append([x1, y1, x2, y2])
        elif 'rickshaw' in c:
            list5.append([x1, y1, x2, y2])
        elif 'tempo' in c:
            list6.append([x1, y1, x2, y2])
        #elif 'tractor' in c:
         #   list7.append([x1, y1, x2, y2])
        # elif 'traveller' in c:
        #     list8.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(list)
    bbox1_idx = tracker1.update(list1)
    bbox2_idx = tracker2.update(list2)
    bbox3_idx = tracker3.update(list3)
    bbox4_idx = tracker4.update(list4)
    bbox5_idx = tracker5.update(list5)
    bbox6_idx = tracker6.update(list6)
    # bbox7_idx = tracker7.update(list7)
    # bbox8_idx = tracker8.update(list8)


###########################BUS######################################
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if cy1<(cy+offset) and cy1>(cy-offset):
           cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
           cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
           if bus.count(id)==0:
              bus.append(id)
#####################################CAR#################################
    for bbox1 in bbox1_idx:
        x5,y5,x6,y6,id1=bbox1
        cx2=int(x5+x6)//2
        cy2=int(y5+y6)//2
        if cy1<(cy2+offset) and cy1>(cy2-offset):
           cv2.rectangle(frame,(x5,y5),(x6,y6),(0,255,0),2)
           cvzone.putTextRect(frame,f'{id1}',(x5,y5),1,1)
           if car.count(id1)==0:
              car.append(id1)
#################################auto-rikshaw############################
    for bbox2 in bbox2_idx:
        x7,y7,x8,y8,id2=bbox2
        cx3=int(x7+x8)//2
        cy3=int(y7+y8)//2
        if cy1<(cy3+offset) and cy1>(cy3-offset):
           cv2.rectangle(frame,(x7,y7),(x8,y8),(0,255,0),2)
           cvzone.putTextRect(frame,f'{id2}',(x7,y7),1,1)
           if auto_rickshaw.count(id2)==0:
              auto_rickshaw.append(id2)
#########################################motorcycle##############################
    for bbox3 in bbox3_idx:
        x9,y9,x10,y10,id3=bbox3
        cx4=int(x9+x10)//2
        cy4=int(y9+y10)//2
        if cy1<(cy4+offset) and cy1>(cy4-offset):
           cv2.rectangle(frame,(x9,y9),(x10,y10),(0,255,0),2)
           cvzone.putTextRect(frame,f'{id3}',(x9,y9),1,1)
           if motorcycle.count(id3)==0:
              motorcycle.append(id3)


#########################################TRUCK##############################
    for bbox4 in bbox4_idx:
        x11,y11,x12,y12,id3=bbox4
        cx5=int(x11+x12)//2
        cy5=int(y11+y12)//2
        if cy1<(cy5+offset) and cy1>(cy5-offset):
           cv2.rectangle(frame,(x11,y11),(x12,y12),(0,255,0),2)
           cvzone.putTextRect(frame,f'{id4}',(x11,y11),1,1)
           if truck.count(id4)==0:
              truck.append(id4)

#########################################Rickshaw##############################
    for bbox5 in bbox5_idx:
        x13,y13,x14,y14,id5=bbox5
        cx6=int(x13+x14)//2
        cy6=int(y13+y14)//2
        if cy1<(cy6+offset) and cy1>(cy6-offset):
           cv2.rectangle(frame,(x13,y13),(x14,y14),(0,255,0),2)
           cvzone.putTextRect(frame,f'{id5}',(x13,y13),1,1)
           if rickshaw.count(id5)==0:
              rickshaw.append(id5)

#########################################Tempo##############################
    for bbox6 in bbox6_idx:
        x15,y15,x16,y16,id6=bbox6
        cx7=int(x15+x16)//2
        cy7=int(y15+y16)//2
        if cy1<(cy7+offset) and cy1>(cy7-offset):
           cv2.rectangle(frame,(x15,y15),(x16,y16),(0,255,0),2)
           cvzone.putTextRect(frame,f'{id6}',(x15,y15),1,1)
           if tempo.count(id6)==0:
              tempo.append(id6)



    # Display counts and other information on the frame
    countbus=(len(bus))
    countcar=(len(car))
    countauto_rikshaw=(len(auto_rickshaw))
    countmotorcycle=(len(motorcycle))
    counttruck = (len(truck))
    countrickshaw = (len(rickshaw))
    counttempo = (len(tempo))
    # counttractor = (len(tractor))
    # counttraveller = (len(traveller))
    #


    cvzone.putTextRect(frame,f'countbus:-{countbus}',(50,40),1,1)
    cvzone.putTextRect(frame,f'countcar:-{countcar}',(50,90),1,1)
    cvzone.putTextRect(frame,f'countauto_rikshaw:-{countauto_rikshaw}',(800,40),1,1)
    cvzone.putTextRect(frame,f'countmotorcycle:-{countmotorcycle}',(800,90),1,1)

    cvzone.putTextRect(frame, f'countrickshaw:-{countrickshaw}', (50, 130), 1, 1)
    cvzone.putTextRect(frame, f'counttruck:-{counttruck}', (50, 170), 1, 1)
    # cvzone.putTextRect(frame, f'counttempo:-{counttempo}', (600, 120), 2, 2)
    # cvzone.putTextRect(frame, f'counttraveller:-{counttraveller}', (600, 150), 2, 2)

    cv2.line(frame,(1,274),(1018 ,274),(255,255,255),2)
    cv2.imshow("RGB", frame)
    #key = cv2.waitKey(delay_time)
    #if key == 27:
    if cv2.waitKey(1) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()




