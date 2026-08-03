from ultralytics import YOLO
import cv2
from sound_alert import play_alert
from logger import log_event

# ------------------------------
# Models
# ------------------------------
FACE_MODEL_PATH = "best_train.pt"
OBJECT_MODEL_PATH = "best_object_2.pt"



face_model = YOLO(FACE_MODEL_PATH)
object_model = YOLO(OBJECT_MODEL_PATH)
'''print("\n====== DEBUG: OBJECT MODEL CLASSES ======")
print(object_model.names)
print("========================================\n")'''
log_event("Face & Object models loaded.", level="INFO")

# ------------------------------
# Webcam configuration
# ------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SKIP = 1
BUZZER_THRESHOLD = 2
CONF_THRESHOLD = 0.6
MIN_BOX_SIZE = 30

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

log_event("Webcam started.", "INFO")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        log_event("Failed to read frame.", "ERROR")
        break
    # ------------------------------
    # OBJECT DETECTION (DEBUG + DRAW)
    # ------------------------------

    print("---- OBJECT DEBUG ----")
    print("Model classes:", object_model.names)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    object_results = object_model(rgb, conf=0.25, imgsz=640)

    for r in object_results:
        print("Boxes:", len(r.boxes))
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = object_model.names[cls].lower()

            print("Detected:", label, "@", conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


    # Skip frames if enabled
    if frame_count % (FRAME_SKIP + 1) != 0:
        frame_count += 1
        continue

    face_count = 0

    # ------------------------------
    # FACE DETECTION
    # ------------------------------
    face_results = face_model(frame, conf=CONF_THRESHOLD)

    for r in face_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = face_model.names[cls].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                continue

            if "face" in label:
                face_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (255, 0, 0), 2)
                cv2.putText(frame, "Face", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # ------------------------------
    # OBJECT DETECTION
    # ------------------------------
    object_results = object_model(frame, conf=0.5)

    for r in object_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = object_model.names[cls].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 200, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # ------------------------------
    # FACE COUNT DISPLAY & BUZZER
    # ------------------------------
    cv2.putText(frame, f"Faces: {face_count}", (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)

    if face_count >= BUZZER_THRESHOLD:
        log_event(f"{face_count} faces detected!", "ALERT")
        play_alert()
    else:
        log_event(f"{face_count} faces detected.", "INFO")

    cv2.imshow("Webcam - Face + Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        log_event("Stopped by user.", "INFO")
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
log_event("Webcam closed.", "INFO")
