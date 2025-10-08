# test.py
from ultralytics import YOLO
import cv2
from sound_alert import play_alert
from logger import log_event

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = "best_train.pt"
FRAME_WIDTH = 640        # webcam width
FRAME_HEIGHT = 480       # webcam height
FRAME_SKIP = 1           # process every Nth frame (0 = every frame)
BUZZER_THRESHOLD = 2     # number of faces to trigger buzzer
CONF_THRESHOLD = 0.6     # minimum confidence for detection
MIN_BOX_SIZE = 30        # ignore detections smaller than this

# ------------------------------
# Load YOLO model
# ------------------------------
model = YOLO(MODEL_PATH)
log_event("YOLO model loaded.", level="INFO")

# ------------------------------
# Open webcam
# ------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
log_event("Webcam started.", level="INFO")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        log_event("Failed to read frame from webcam.", level="ERROR")
        break

    # Optional: skip frames for speed
    if frame_count % (FRAME_SKIP + 1) != 0:
        frame_count += 1
        continue

    # Run YOLO detection with confidence threshold
    results = model(frame, conf=CONF_THRESHOLD, stream=True)
    face_count = 0

    # Count faces and draw boxes
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Skip tiny detections
            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                continue

            if "face" in label:
                face_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Face", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display face count
    cv2.putText(frame, f"Faces: {face_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # Log and play buzzer if threshold reached
    if face_count >= BUZZER_THRESHOLD:
        log_event(f"{face_count} faces detected! Buzzer triggered.", level="ALERT")
        play_alert()
    else:
        log_event(f"{face_count} faces detected.", level="INFO")

    # Show webcam feed
    cv2.imshow("Webcam - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        log_event("Webcam feed stopped by user.", level="INFO")
        break

    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
log_event("Webcam released and windows closed.", level="INFO")