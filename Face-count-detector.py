#!/usr/bin/env python3
"""
yolo_head_detector.py
High-accuracy detector using YOLOv8 (Ultralytics).
Detects persons (full/upper body) with much higher accuracy than Haar/HOG.

Usage:
- Webcam: python yolo_head_detector.py --source 0
- Image:  python yolo_head_detector.py --source test.jpg --output result.jpg
"""

import argparse
import cv2
import os
from datetime import datetime
from ultralytics import YOLO

def annotate_frame(frame, boxes):
    n = len(boxes)
    if n == 0:
        label = "No person detected"
        color = (0, 0, 255)
    elif n == 1:
        label = "Single person detected"
        color = (0, 255, 0)
    else:
        label = f"{n} people detected"
        color = (0, 255, 255)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame, label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='0', help="0 for webcam, or path to image/video")
    ap.add_argument('--output', default=None, help="Output file for annotated image")
    args = ap.parse_args()

    # Load YOLOv8 pretrained on COCO (person class = 0)
    model = YOLO("yolov8n.pt")  # 'n' = nano (fastest); try yolov8s.pt for better accuracy

    src = args.source
    try:
        src = int(src)
    except ValueError:
        pass

    # If source is webcam or video
    if str(src).isdigit() or os.path.splitext(str(src))[-1].lower() in ['.mp4','.avi','.mov']:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(":x: Error: Cannot open source", src)
            return
        print(":white_check_mark: YOLOv8 running (press 'q' to quit)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, verbose=False)
            persons = []
            for r in results[0].boxes:
                cls = int(r.cls)
                if cls == 0:  # 'person' class
                    x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
                    persons.append((x1,y1,x2,y2))

            frame, label = annotate_frame(frame, persons)
            ts = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, ts, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("YOLO Head Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # If single image
    else:
        img = cv2.imread(src)
        if img is None:
            print(":x: Error: Cannot read image", src)
            return
        results = model(img, verbose=False)
        persons = []
        for r in results[0].boxes:
            cls = int(r.cls)
            if cls == 0:
                x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
                persons.append((x1,y1,x2,y2))
        img, label = annotate_frame(img, persons)
        if args.output:
            cv2.imwrite(args.output, img)
            print(":white_check_mark: Wrote annotated image to", args.output)
        cv2.imshow("YOLO Head Detector", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

