#!/usr/bin/env python3
"""
A-Eye v1.4 — Directional Awareness + Distance Coloring
------------------------------------------------------
Builds on v1.3 by adding:
- Direction awareness (Left, Center, Right)
- Distance-based color coding (Green→Far, Red→Near)
- Sharpening filter for clarity

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
FOCAL_LENGTH = 550

# ----------------------------
# Device Setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[v1.4] Running on device: {device}")

# ----------------------------
# Load Models
# ----------------------------
print("[v1.4] Loading YOLOE model...")
yolo_model = YOLOE("yoloe-11l-seg.pt")
embeddings = yolo_model.get_text_pe(PROMPT_CLASSES)
yolo_model.set_classes(PROMPT_CLASSES, embeddings)

print("[v1.4] Loading MiDaS depth model...")
midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform_depth = midas_transforms.dpt_transform
midas_model.to(device).eval()

print(f"[v1.4] Models ready. Classes: {PROMPT_CLASSES}")

# ----------------------------
# Webcam Setup
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not detected!")

# ----------------------------
# Helper Functions
# ----------------------------
def sharpen(frame):
    """Applies a subtle sharpening filter for better detection."""
    alpha, beta = 1.5, -0.3
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    return cv2.addWeighted(frame, alpha, blur, beta, 0)

def get_depth_map(frame):
    """Generates a normalized depth map from MiDaS."""
    img_input = transform_depth(frame).to(device)
    with torch.no_grad():
        depth = midas_model(img_input)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_np = depth.cpu().numpy()
    depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_norm * DEPTH_SCALE), cv2.COLORMAP_INFERNO)
    return depth_np, depth_color

def direction_from_center(cx, width):
    """Determine if object is on left, right, or center."""
    if cx < width // 3:
        return "left"
    elif cx > 2 * width // 3:
        return "right"
    return "center"

def color_by_distance(distance):
    """Color gradient: Green → Far, Red → Near."""
    if distance < 0.8:
        return (0, 0, 255)  # Red (near)
    elif distance < 1.8:
        return (0, 165, 255)  # Orange (medium)
    else:
        return (0, 255, 0)  # Green (far)

def draw_box(frame, xyxy, cls, conf, distance_m, direction):
    """Draw bounding box with color-coded distance."""
    color = color_by_distance(distance_m)
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{cls} {distance_m:.2f}m {direction}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

# ----------------------------
# Main Loop
# ----------------------------
fps_smooth = 0
prev_time = time.time()

print("[v1.4] Running with Directional Awareness — press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = sharpen(frame)
    height, width = frame.shape[:2]

    # Generate depth map
    depth_vals, depth_vis = get_depth_map(frame)

    # Detect objects
    results = yolo_model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SZ, verbose=False)[0]

    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls.item())
            cls = PROMPT_CLASSES[cls_id] if cls_id < len(PROMPT_CLASSES) else str(cls_id)
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2

            # Extract depth ROI
            roi = depth_vals[y1:y2, x1:x2]
            if roi.size > 0:
                mean_depth = np.mean(roi)
                distance_m = (1 / (mean_depth + 1e-6)) * (550 / 100)
            else:
                distance_m = 2.5  # fallback

            direction = direction_from_center(cx, width)
            draw_box(frame, xyxy, cls, conf, distance_m, direction)

    # Add direction zone lines
    cv2.line(frame, (width // 3, 0), (width // 3, height), (255, 255, 255), 1)
    cv2.line(frame, (2 * width // 3, 0), (2 * width // 3, height), (255, 255, 255), 1)

    # Combine view
    depth_resized = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))
    combined = np.hstack((frame, depth_resized))

    # FPS calculation
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    fps_smooth = 0.9 * fps_smooth + 0.1 * fps
    cv2.putText(combined, f"FPS: {fps_smooth:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("A-Eye v1.4 — Directional Awareness", combined)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[v1.4] Exited cleanly.")
