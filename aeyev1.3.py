#!/usr/bin/env python3
"""
A-Eye v1.3 — Detection + Depth + Distance Estimation
-----------------------------------------------------
Adds real-world distance estimation (in meters) per detected object
by averaging depth values inside YOLOE bounding boxes.

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
DEPTH_SCALE = 3.0
FOCAL_LENGTH = 550  # approximate pixel-based focal length for depth-to-meter conversion

# ----------------------------
# Device Setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[v1.3] Running on device: {device}")

# ----------------------------
# Load Models
# ----------------------------
print("[v1.3] Loading YOLOE model...")
yolo_model = YOLOE("yoloe-11l-seg.pt")
embeddings = yolo_model.get_text_pe(PROMPT_CLASSES)
yolo_model.set_classes(PROMPT_CLASSES, embeddings)

print("[v1.3] Loading MiDaS depth model...")
depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
depth_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
depth_model.to(device).eval()

print(f"[v1.3] Models ready. Classes: {PROMPT_CLASSES}")

# ----------------------------
# Webcam Setup
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not detected!")

# ----------------------------
# Helper Functions
# ----------------------------
def draw_box(frame, xyxy, cls, conf, distance_m=None):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{cls} {conf:.2f}"
    if distance_m is not None:
        label += f" | {distance_m:.2f} m"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

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
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_norm * DEPTH_SCALE), cv2.COLORMAP_INFERNO)
    return depth, depth_color

# ----------------------------
# Main Loop
# ----------------------------
fps_smooth = 0
prev_time = time.time()

print("[v1.3] Running Detection + Depth + Distance Estimation (press 'q' to quit)...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Depth Map
    depth_values, depth_vis = get_depth_map(frame)

    # Object Detection
    results = yolo_model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SZ, verbose=False)[0]

    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls.item())
            cls = PROMPT_CLASSES[cls_id] if cls_id < len(PROMPT_CLASSES) else str(cls_id)
            conf = float(box.conf.item())

            # Estimate distance
            x1, y1, x2, y2 = map(int, xyxy)
            roi = depth_values[y1:y2, x1:x2]
            if roi.size > 0:
                mean_depth = np.mean(roi)
                # Convert relative depth to approximate meters (scaled)
                distance_m = (1 / (mean_depth + 1e-6)) * (FOCAL_LENGTH / 100)
            else:
                distance_m = None

            draw_box(frame, xyxy, cls, conf, distance_m)

    # Combine depth + detection
    depth_resized = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))
    combined = np.hstack((frame, depth_resized))

    # FPS Calculation
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    fps_smooth = 0.9 * fps_smooth + 0.1 * fps

    cv2.putText(combined, f"FPS: {fps_smooth:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("A-Eye v1.3 — Detection + Distance", combined)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[v1.3] Exited cleanly.")
