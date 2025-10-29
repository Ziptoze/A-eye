#!/usr/bin/env python3
"""
A-Eye v1.2 — Object Detection + Depth Estimation
-------------------------------------------------
Adds MiDaS depth map overlay alongside YOLOE detection.

Requirements:
    pip install ultralytics opencv-python torch torchvision numpy timm
"""

import time
import cv2
import torch
import numpy as np
from ultralytics import YOLOE

# ----------------------------
# Configuration
# ----------------------------
PROMPT_CLASSES = ["person", "microwave"]
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4
IMG_SZ = 640
DEPTH_SCALE = 3.0  # contrast multiplier for better depth visibility

# ----------------------------
# Device Setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[v1.2] Running on device: {device}")

# ----------------------------
# Load Models
# ----------------------------
print("[v1.2] Loading YOLOE model...")
yolo_model = YOLOE("yoloe-11l-seg.pt")
embeddings = yolo_model.get_text_pe(PROMPT_CLASSES)
yolo_model.set_classes(PROMPT_CLASSES, embeddings)

print("[v1.2] Loading MiDaS model for depth estimation...")
depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
depth_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
depth_model.to(device).eval()

print(f"[v1.2] Models ready. Classes: {PROMPT_CLASSES}")

# ----------------------------
# Webcam Setup
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not detected!")

# ----------------------------
# Draw Function
# ----------------------------
def draw_box(frame, xyxy, cls, conf):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{cls} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# ----------------------------
# Depth Estimation Function
# ----------------------------
def get_depth_map(frame):
    img_input = depth_transform(frame).to(device)
    with torch.no_grad():
        prediction = depth_model(img_input)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = cv2.convertScaleAbs(depth)
    depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth * DEPTH_SCALE), cv2.COLORMAP_INFERNO)
    return depth_colored

# ----------------------------
# Main Loop
# ----------------------------
fps_smooth = 0
prev_time = time.time()

print("[v1.2] Running detection + depth estimation (press 'q' to quit)...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Depth map
    depth_map = get_depth_map(frame)

    # Detection
    results = yolo_model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SZ, verbose=False)[0]

    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls.item())
            cls = PROMPT_CLASSES[cls_id] if cls_id < len(PROMPT_CLASSES) else str(cls_id)
            conf = float(box.conf.item())
            draw_box(frame, xyxy, cls, conf)

    # Combine displays: left = RGB, right = Depth
    depth_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
    combined = np.hstack((frame, depth_resized))

    # FPS
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    fps_smooth = 0.9 * fps_smooth + 0.1 * fps
    cv2.putText(combined, f"FPS: {fps_smooth:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show
    cv2.imshow("A-Eye v1.2 — Detection + Depth", combined)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[v1.2] Exited cleanly.")
