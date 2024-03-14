from ultralytics import YOLO

# Load a trained model
model = YOLO('yolov8n.pt')

# train the model with custom dataset
metrics = model.train(data='customised data\data.yaml', epochs=30, imgsz=640)