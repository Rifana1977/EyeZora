from ultralytics import YOLO

# Load your trained model (best.pt from training)
model = YOLO("D:/ME/EyeZora/face_detection/runs/detect/train3/weights/best.pt")

# Path to your test images folder
test_images = "D:/ME/EyeZora/face_detection/test/images"

# Run prediction on the folder
results = model.predict(source=test_images, save=True, project="runs/detect", name="custom_test", exist_ok=True)

print("✅ Predictions completed! Check results inside: runs/detect/custom_test")
