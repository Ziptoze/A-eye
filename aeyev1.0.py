#!/usr/bin/env python3
"""
A-Eye — Iteration 1 — v1.0 (Fixed Basic)
Basic YOLOE object detection using prompt-based classes

Detects and displays only a few essential objects:
→ person, microwave, chair, bottle

Requirements:
    pip install ultralytics opencv-python torch numpy

Press 'q' or ESC to quit.
"""

import time
import cv2
import torch
import numpy as np
from ultralytics import YOLOE

# ----------------------------
# Configuration
# ----------------------------
PROMPT_CLASSES = ["person", "microwave", "chair", "bottle"]
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4
IMG_SZ = 640
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)

# ----------------------------
# Model Loading
# ----------------------------
print("[v1.0] Loading YOLOE model (prompt-based)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLOE("yoloe-11l-seg.pt")

print("[v1.0] Embedding basic prompt classes...")
embeddings = model.get_text_pe(PROMPT_CLASSES)
model.set_classes(PROMPT_CLASSES, embeddings)
print(f"[v1.0] Model ready with classes: {PROMPT_CLASSES}")

# ----------------------------
# Webcam Setup
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not found!")

# ----------------------------
# Helper: Draw Bounding Box
# ----------------------------
def draw_box(frame, xyxy, cls, conf):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    label = f"{cls} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), BOX_COLOR, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

# ----------------------------
# Main Loop
# ----------------------------
fps_smooth = 0
prev = time.time()

print("[v1.0] Running — press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.01)
        continue

    try:
        results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SZ, verbose=False)[0]
    except Exception as e:
        print("[ERROR] Inference failed:", e)
        continue

    # Draw boxes
    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls.item())
            cls = PROMPT_CLASSES[cls_id] if cls_id < len(PROMPT_CLASSES) else str(cls_id)
            conf = float(box.conf.item())
            draw_box(frame, xyxy, cls, conf)

    # FPS display
    now = time.time()
    fps = 1 / (now - prev) if now > prev else 0
    prev = now
    fps_smooth = 0.9 * fps_smooth + 0.1 * fps
    cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show
    cv2.imshow("A-Eye v1.0 — Basic YOLOE", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("[v1.0] Exited cleanly.")
