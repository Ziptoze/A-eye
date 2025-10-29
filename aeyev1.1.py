#!/usr/bin/env python3
"""
A-Eye v1.1 — Runtime with custom prompts and TensorRT support
Detects: person, microwave

Usage:
    python aeye_v1_1_runtime.py
"""

import time
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLOE

PROMPT_CLASSES = ["person", "microwave"]
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4
IMG_SZ = 640
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

ENGINE_PATH = "yoloe-11l-seg.engine"

# --------------------------
# Load Model (Engine or PyTorch)
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(ENGINE_PATH):
    print("[v1.1] Loading TensorRT engine...")
    model = YOLOE(ENGINE_PATH)
else:
    print("[v1.1] Loading PyTorch model...")
    model = YOLOE("yoloe-11l-seg.pt")
    embeddings = model.get_text_pe(PROMPT_CLASSES)
    model.set_classes(PROMPT_CLASSES, embeddings)

print(f"[v1.1] Ready. Classes: {PROMPT_CLASSES}")

# --------------------------
# Webcam Setup
# --------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not detected!")

# --------------------------
# Draw Function
# --------------------------
def draw_box(frame, xyxy, cls, conf):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    label = f"{cls} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), BOX_COLOR, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

# --------------------------
# Main Loop
# --------------------------
prev_time = time.time()
fps_smooth = 0

print("[v1.1] Running detection (press 'q' to quit)...")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SZ, verbose=False)[0]

    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls.item())
            cls = PROMPT_CLASSES[cls_id] if cls_id < len(PROMPT_CLASSES) else str(cls_id)
            conf = float(box.conf.item())
            draw_box(frame, xyxy, cls, conf)

    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    fps_smooth = 0.9 * fps_smooth + 0.1 * fps
    cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("A-Eye v1.1 — Custom Prompts", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[v1.1] Exited cleanly.")
