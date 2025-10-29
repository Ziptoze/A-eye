#!/usr/bin/env python3
"""
A-eye Assist (Moondream & audio disabled)
- Press F to log nearest 3 objects (works repeatedly)
- Press Q to quit
"""

import time
import threading
import queue
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLOE

# ----------------------------
# Configuration
# ----------------------------
DEV_FLAGS = {
    "ENABLE_SPEECH": False,   # 🔇 Audio turned off
    "ENABLE_DEPTH": True,
    "CONF": 0.4,
    "IOU": 0.1,
    "IMG_SZ": 640,
    "DISPLAY_WIDTH": 640,
    "DISPLAY_HEIGHT": 480,
    "SHARPEN_ALPHA": 1.5,
    "SHARPEN_BETA": -0.3,
    "BOX_COLOR": (0, 255, 0),
    "YOLOE_ENGINE_PATH": "yoloe-11l-seg.engine",
    "ENABLE_MIDAS": True,
    "ENABLE_PROMPT_UI": True,
    "MOONDREAM_ID": None,     # 🚫 Moondream disabled
    "MOONDREAM_DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load YOLOE + MiDaS
# ----------------------------
print("[INFO] Loading YOLOE...")
yolo_model = YOLOE(DEV_FLAGS["YOLOE_ENGINE_PATH"])

midas, transform_depth = None, None
if DEV_FLAGS["ENABLE_DEPTH"] and DEV_FLAGS["ENABLE_MIDAS"]:
    try:
        print("[INFO] Loading MiDaS small...")
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform_depth = midas_transforms.small_transform
    except Exception as e:
        print("[WARN] MiDaS load failed, disabling depth:", e)
        DEV_FLAGS["ENABLE_DEPTH"] = False

# ----------------------------
# Capture Thread
# ----------------------------
class CaptureThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEV_FLAGS["DISPLAY_WIDTH"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEV_FLAGS["DISPLAY_HEIGHT"])
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.stop_event.set()
        try:
            self.cap.release()
        except Exception:
            pass

capture_thread = CaptureThread()
capture_thread.start()

# ----------------------------
# Inference Thread
# ----------------------------
class InferenceThread(threading.Thread):
    def __init__(self, capture: CaptureThread):
        super().__init__(daemon=True)
        self.capture = capture
        self.latest_display = None
        self.latest_objects = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    @staticmethod
    def _direction(cx, w):
        if cx < w // 3:
            return "left"
        elif cx > 2 * w // 3:
            return "right"
        return "ahead"

    @staticmethod
    def _sharpen(frame):
        a, b = DEV_FLAGS["SHARPEN_ALPHA"], DEV_FLAGS["SHARPEN_BETA"]
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        return cv2.addWeighted(frame, float(a), blur, float(b), 0)

    def run(self):
        while not self.stop_event.is_set():
            frame = self.capture.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            proc = self._sharpen(frame)
            h, w = proc.shape[:2]

            depth_map = None
            if DEV_FLAGS["ENABLE_DEPTH"] and midas and transform_depth:
                try:
                    input_batch = transform_depth(proc).to(device)
                    with torch.inference_mode():
                        pred = midas(input_batch)
                        pred = torch.nn.functional.interpolate(
                            pred.unsqueeze(1),
                            size=proc.shape[:2],
                            mode="bicubic",
                            align_corners=False
                        ).squeeze().cpu().numpy()
                        depth_map = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-9)
                except Exception:
                    depth_map = None

            objects = []
            try:
                results = yolo_model.predict(proc, conf=DEV_FLAGS["CONF"], iou=DEV_FLAGS["IOU"],
                                             imgsz=DEV_FLAGS["IMG_SZ"], verbose=False)[0]
                if hasattr(results, "boxes") and results.boxes is not None:
                    for b in results.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        lbl = results.names[int(b.cls.item())]
                        cx = (x1 + x2) // 2
                        dist = 2.0
                        if depth_map is not None:
                            roi = depth_map[y1:y2, x1:x2]
                            if roi.size > 0:
                                d = float(np.median(roi))
                                dist = 0.3 + (1 - d) * 5
                        dirc = self._direction(cx, w)
                        objects.append((dist, lbl, dirc, (x1, y1, x2, y2)))
            except Exception:
                objects = []

            disp = proc.copy()
            color = DEV_FLAGS["BOX_COLOR"]
            cv2.line(disp, (w//3, 0), (w//3, h), color, 1)
            cv2.line(disp, (2*w//3, 0), (2*w//3, h), color, 1)

            for dist, lbl, dirc, (x1, y1, x2, y2) in objects:
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                label = f"{lbl} {dist:.1f}ft"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(disp, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                cv2.putText(disp, label, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            with self.lock:
                self.latest_display = disp
                self.latest_objects = objects

    def get_display(self):
        with self.lock:
            return None if self.latest_display is None else self.latest_display.copy()

    def get_objects(self):
        with self.lock:
            return list(self.latest_objects)

    def stop(self):
        self.stop_event.set()

inference_thread = InferenceThread(capture_thread)
inference_thread.start()

# ----------------------------
# Main loop
# ----------------------------
def main():
    win = "A-eye Assist"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DEV_FLAGS["DISPLAY_WIDTH"], DEV_FLAGS["DISPLAY_HEIGHT"])

    fps, prev_time = 0.0, time.time()
    print("[INFO] Controls: F=log nearest 3 objects, Q=quit")

    while True:
        start = time.time()
        frame = inference_thread.get_display()
        if frame is None:
            frame = np.zeros((DEV_FLAGS["DISPLAY_HEIGHT"], DEV_FLAGS["DISPLAY_WIDTH"], 3), np.uint8)
            cv2.putText(frame, "Loading...", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        elapsed = start - prev_time
        fps = 0.9 * fps + 0.1 * (1 / elapsed) if elapsed > 0 else fps
        prev_time = start
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('f'):
            objs = inference_thread.get_objects()
            objs_sorted = sorted(objs, key=lambda x: x[0])[:5]
            if not objs_sorted:
                print("[LOG] No objects detected.")
            else:
                print("[LOG] Nearest 3 objects:")
                for dist, lbl, dirc, _ in objs_sorted:
                    print(f" - {lbl}: {dist:.2f}ft, {dirc}")

    shutdown()

# ----------------------------
# Shutdown
# ----------------------------
def shutdown():
    capture_thread.stop()
    inference_thread.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        shutdown()
