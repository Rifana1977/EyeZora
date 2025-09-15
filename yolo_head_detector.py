from ultralytics import YOLO
import cv2
from logger import log_event
from sound_alert import play_warning_sound
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Load Haar cascades for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Open webcam
cap = cv2.VideoCapture(0)

# For smoothing gaze detection over frames
gaze_buffer = []
BUFFER_SIZE = 5  # check gaze over 5 frames before warning

def get_gaze_direction(eye_img):
    gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    _, thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)
    moments = cv2.moments(thresh)
    if moments['m00'] != 0:
        cx = int(moments['m10']/moments['m00'])
        w = eye_img.shape[1]
        if cx < w/3:
            return "Left"
        elif cx > 2*w/3:
            return "Right"
        else:
            return "Center"
    return "Unknown"

def check_gaze_buffer(buffer):
    if len(buffer) < BUFFER_SIZE:
        return "Center"  # not enough frames
    if all(g != "Center" for g in buffer):
        return "Away"
    return "Center"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    persons = 0
    warnings = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls].lower()

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area > 5000:
                    persons += 1

                # Eye-gaze detection inside face region
                face_region = frame[y1:y2, x1:x2]
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 4)

                eye_gazes = []
                for (ex, ey, ew, eh) in eyes[:2]:  # consider max 2 eyes
                    eye_img = face_region[ey:ey+eh, ex:ex+ew]
                    gaze = get_gaze_direction(eye_img)
                    eye_gazes.append(gaze)

                # Average gaze of both eyes
                if eye_gazes:
                    if "Left" in eye_gazes:
                        avg_gaze = "Left"
                    elif "Right" in eye_gazes:
                        avg_gaze = "Right"
                    else:
                        avg_gaze = "Center"

                    # Log the gaze direction
                    log_event(f"Gaze direction: {avg_gaze}", "INFO")

                    gaze_buffer.append(avg_gaze)
                    if len(gaze_buffer) > BUFFER_SIZE:
                        gaze_buffer.pop(0)

                    overall_gaze = check_gaze_buffer(gaze_buffer)
                    cv2.putText(frame, f"Gaze: {avg_gaze}", (x1, y1-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if overall_gaze == "Away":
                        msg = "Warning: Not looking at screen!"
                        warnings.append(msg)
                        log_event(msg, "WARNING")
                        play_warning_sound()

            # Restricted objects
            if label in ["cell phone", "book", "paper"]:
                msg = f"Warning {label} detected! (conf {conf:.2f})"
                warnings.append(msg)
                log_event(msg, "WARNING")
                play_warning_sound()

            # Allowed objects
            elif label in ["bottle", "cup", "glass", "steel glass"]:
                msg = f"Detected: {label} ({conf:.2f})"
                cv2.putText(frame, msg, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                log_event(msg, "INFO")

    # Person detection rules
    if persons == 0:
        msg = " No person detected!"
        warnings.append(msg)
        log_event(msg, "WARNING")
        play_warning_sound()
    elif persons > 1:
        msg = " More than one person detected!"
        warnings.append(msg)
        log_event(msg, "WARNING")
        play_warning_sound()
    else:
        log_event("Exactly one person detected.", "INFO")

    # Show warnings on screen (dynamic spacing)
    y_offset = 50
    for w in warnings:
        cv2.putText(frame, w, (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 35  # smaller spacing to avoid overlap

    # Display output
    cv2.imshow("Exam Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
