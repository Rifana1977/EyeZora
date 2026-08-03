"""
EyeZora AI Proctoring Service
FastAPI server that receives webcam frames and returns AI detection results.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health        — health check
    POST /analyze-frame — analyze a base64 encoded frame
    POST /verify-person — simple log endpoint (legacy)
"""

import base64
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import datetime
import os

# ── Lazy model loading to avoid startup crash if models aren't present ─────────
face_model = None
object_model = None
eye_cascade = None

def load_models():
    global face_model, object_model, eye_cascade
    try:
        from ultralytics import YOLO
        face_model = YOLO("best_train.pt")
        object_model = YOLO("yolov8s.pt")
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        print("✅ AI models loaded successfully")
    except Exception as e:
        print(f"⚠️  Could not load AI models: {e}")
        print("   Running in stub mode — will return mock results")

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EyeZora AI Proctoring Service",
    description="YOLO-based exam proctoring detection API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()

# ── Request / Response Schemas ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image: str           # base64 encoded JPEG/PNG
    student_id: str
    exam_id: str
    session_id: Optional[str] = None

class DetectionEvent(BaseModel):
    event: str
    confidence: float
    severity: str        # Low | Medium | High

class AnalyzeResponse(BaseModel):
    events: List[DetectionEvent]
    face_count: int
    timestamp: str

class LogRequest(BaseModel):
    student_id: str
    exam_id: str
    ai_result: str

# ── Gaze Detection Helpers ─────────────────────────────────────────────────────

gaze_buffer = []
BUFFER_SIZE = 5

def get_gaze_direction(eye_img):
    """Estimate gaze direction from eye image using pupil centroid."""
    try:
        gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        _, thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)
        moments = cv2.moments(thresh)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            w = eye_img.shape[1]
            if cx < w / 3:
                return "Left"
            elif cx > 2 * w / 3:
                return "Right"
            else:
                return "Center"
    except Exception:
        pass
    return "Unknown"

def check_gaze_buffer(buffer):
    """Returns 'Away' if recent gaze history is consistently non-center."""
    if len(buffer) < BUFFER_SIZE:
        return "Center"
    if all(g != "Center" for g in buffer):
        return "Away"
    return "Center"

# ── Main Analysis Endpoint ─────────────────────────────────────────────────────

@app.post("/analyze-frame", response_model=AnalyzeResponse)
async def analyze_frame(req: AnalyzeRequest):
    """
    Analyze a single webcam frame for proctoring violations.
    Returns a list of detected events with confidence scores.
    """
    print("AI processing frame")
    global gaze_buffer

    events: List[DetectionEvent] = []
    face_count = 0
    timestamp = datetime.datetime.now().isoformat()

    # ── Decode base64 image ──────────────────────────────────────────────────
    try:
        # Strip data URL prefix if present
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return AnalyzeResponse(events=[], face_count=0, timestamp=timestamp)

    except Exception as e:
        print(f"Frame decode error: {e}")
        return AnalyzeResponse(events=[], face_count=0, timestamp=timestamp)

    # ── Stub mode if models not loaded ──────────────────────────────────────
    if face_model is None:
        return AnalyzeResponse(
            events=[],
            face_count=1,
            timestamp=timestamp,
        )

    # ── Face Detection ────────────────────────────────────────────────────────
    try:
        face_results = face_model(frame, conf=0.5, verbose=False)

        for r in face_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = face_model.names[cls].lower()

                if label == "face":
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    area = (x2 - x1) * (y2 - y1)
                    if area > 5000:
                        face_count += 1

                    # ── Gaze detection ───────────────────────────────────
                    if eye_cascade is not None:
                        face_region = frame[y1:y2, x1:x2]
                        if face_region.size > 0:
                            gray_face = cv2.cvtColor(
                                face_region, cv2.COLOR_BGR2GRAY
                            )
                            eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 4)
                            eye_gazes = []
                            for (ex, ey, ew, eh) in eyes[:2]:
                                eye_img = face_region[ey:ey+eh, ex:ex+ew]
                                gaze = get_gaze_direction(eye_img)
                                eye_gazes.append(gaze)

                            if eye_gazes:
                                avg_gaze = (
                                    "Left" if "Left" in eye_gazes
                                    else "Right" if "Right" in eye_gazes
                                    else "Center"
                                )
                                gaze_buffer.append(avg_gaze)
                                if len(gaze_buffer) > BUFFER_SIZE:
                                    gaze_buffer.pop(0)

                                overall = check_gaze_buffer(gaze_buffer)
                                if overall == "Away":
                                    events.append(DetectionEvent(
                                        event="LOOKING_AWAY",
                                        confidence=round(conf * 100, 1),
                                        severity="Medium",
                                    ))

        # ── Face count violations ─────────────────────────────────────────────
        if face_count == 0:
            events.append(DetectionEvent(
                event="NO_FACE",
                confidence=95.0,
                severity="High",
            ))
        elif face_count > 1:
            events.append(DetectionEvent(
                event="MULTIPLE_FACES",
                confidence=98.0,
                severity="High",
            ))

    except Exception as e:
        print(f"Face detection error: {e}")

    # ── Object Detection ─────────────────────────────────────────────────────
    RESTRICTED_OBJECTS = {
        "cell phone": ("PHONE_DETECTED", "High"),
        "mobile phone": ("PHONE_DETECTED", "High"),
        "book": ("SUSPICIOUS_MOVEMENT", "Medium"),
        "calculator": ("SUSPICIOUS_MOVEMENT", "High"),
        "earphone": ("SUSPICIOUS_MOVEMENT", "Medium"),
    }

    try:
        object_results = object_model(frame, conf=0.5, verbose=False)

        for r in object_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = object_model.names[cls].lower()

                if label in RESTRICTED_OBJECTS:
                    event_name, severity = RESTRICTED_OBJECTS[label]
                    # Avoid duplicate events
                    if not any(e.event == event_name for e in events):
                        events.append(DetectionEvent(
                            event=event_name,
                            confidence=round(conf * 100, 1),
                            severity=severity,
                        ))

    except Exception as e:
        print(f"Object detection error: {e}")

    print("Detection complete")
    print("Result returned")

    return AnalyzeResponse(
        events=events,
        face_count=face_count,
        timestamp=timestamp,
    )

# ── Legacy Verify Endpoint ─────────────────────────────────────────────────────

@app.post("/verify-person")
async def verify_person(req: LogRequest):
    """Legacy endpoint for logging AI results."""
    log_message = (
        f"[{datetime.datetime.now().isoformat()}] "
        f"Student={req.student_id} Exam={req.exam_id} Result={req.ai_result}"
    )
    print(log_message)

    # Append to events log
    with open("events.log", "a") as f:
        f.write(log_message + "\n")

    return {"status": "logged", "ai_result": req.ai_result}

# ── Health Check ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "EyeZora AI service running",
        "models_loaded": face_model is not None,
        "timestamp": datetime.datetime.now().isoformat(),
    }
