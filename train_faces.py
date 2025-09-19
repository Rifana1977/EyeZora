from ultralytics import YOLO

# Load pretrained YOLOv8 small model
model = YOLO("yolov8s.pt")

# Train on your dataset
model.train(data="C:/Users/parve/OneDrive/Documents/EyeZora/face_dataset/data.yaml", epochs=50, imgsz=640)