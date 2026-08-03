# ── EyeZora AI Service (FastAPI + OpenCV + PyTorch + YOLO) ───────────────────
FROM python:3.11-slim

WORKDIR /app

# Install system-level native libraries required by OpenCV.
# libgl1-mesa-glx  → provides libGL.so.1 (OpenCV rendering backend)
# libglib2.0-0     → provides libgobject, required by OpenCV
# libgomp1         → provides OpenMP threading (used by PyTorch / YOLO)
# Cleaned up in the same layer to minimise image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps — install before copying model files ─────────────────────────
# Copying requirements.txt FIRST and pip-installing means Docker caches the
# heavy PyTorch/YOLO download layer. A code change will NOT retrigger it.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY main.py \
     yolo_head_detector.py \
     sound_alert.py \
     logger.py ./

# ── YOLO model weights (largest files — copied last for optimal caching) ─────
# Only models actually used by main.py are copied:
#   best_train.pt  → face detection model
#   yolov8s.pt     → object detection model
COPY best_train.pt yolov8s.pt ./

# Run as non-root for security
RUN useradd --system --uid 1001 --no-create-home aiuser && \
    chown -R aiuser:aiuser /app
USER aiuser

EXPOSE 8000

# Start Uvicorn — bind to 0.0.0.0 so Docker can route traffic to this port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
