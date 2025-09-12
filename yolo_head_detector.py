from ultralytics import YOLO
import cv2
from logger import log_event   # ✅ import our logger

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.5)

    persons = 0
    warnings = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls].lower()

            # Person detection
            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area > 5000:
                    persons += 1

            # Restricted objects
            if label in ["cell phone", "book", "paper"]:
                msg = f"Warning {label} detected! (conf {conf:.2f})"
                warnings.append(msg)
                log_event(msg, "WARNING")   # ✅ log event

            # Allowed objects
            elif label in ["bottle", "cup", "glass", "steel glass"]:
                msg = f"Detected: {label} ({conf:.2f})"
                cv2.putText(frame, msg, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                log_event(msg, "INFO")   # ✅ log event

    # Person detection rules
    if persons == 0:
        msg = "Warning No person detected!"
        warnings.append(msg)
        log_event(msg, "WARNING")
    elif persons > 1:
        msg = "Warning More than one person detected!"
        warnings.append(msg)
        log_event(msg, "WARNING")
    else:
        log_event("Exactly one person detected.", "INFO")

    # Show warnings on screen
    y_offset = 50
    for w in warnings:
        cv2.putText(frame, w, (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 40

    # Display output
    cv2.imshow("Exam Monitor", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
