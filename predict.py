from ultralytics import YOLO
#import glob
#import cv2
# import torch
# from PIL import image
# from IPython.display import Image, display
# from Ipython,display import HTML

#Load a customised model
model = YOLO('runs/detect/train/weights/last.pt')

# Run an image using opencv
source=('customised data/test/images')

# Run inference on the source
results= model.predict(source, conf=0.25, imgsz=320, save=True )

#View results
for r in results:
    print(r.probs)       # print the Probs object containing the detected class probabilities
    print(r.boxes)       # print the Boxes object containing the detection bounding boxes
    print(r.masks)       # print the Masks object containing the detected instance masks

# Access the Masks object
masks = results[0].masks

# Get the xy coordinates of the masks
#mask_coordinates = masks.xy