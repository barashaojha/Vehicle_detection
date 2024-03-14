from ultralytics import YOLO

# Load a customised model
model = YOLO('runs/detect/train/weights/best.pt')

# Customize validation settings
validation_results = model.val(data='customised data\data.yaml',
                               imgsz=640,
                               batch=16,
                               conf=0.25,
                               iou=0.6,
                               device='cpu')