from ultralytics import YOLO

# Load pretrained YOLOv8 small model
model = YOLO("yolov8s.pt")

# Train on your dataset
model.train(data="D:/ME/EyeZora/face_detection/data.yaml", epochs=50, imgsz=640)