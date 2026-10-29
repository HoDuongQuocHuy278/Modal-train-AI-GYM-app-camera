"""Shared offline OpenCV source, pose extraction and skeleton rendering helpers."""
from pathlib import Path
import numpy as np

# MediaPipe 33 landmark INDEX ORDER, not the MediaPipe model/API.
EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
         (23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),
         (28,30),(30,32)]


def open_local_capture(source):
    import cv2
    if '://' in source:
        raise ValueError('Remote URL/network camera sources are disabled; use local file or camera index')
    if source.isdecimal():
        capture = cv2.VideoCapture(int(source))
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f'Local video not found: {path}')
        capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open camera/local file: {source}')
    return capture


def landmarks_from_bgr(model, frame, device='cpu'):
    import cv2
    from pose_model import IMAGE_SIZE, infer_rgb
    rgb = cv2.cvtColor(cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB)
    return infer_rgb(model, rgb, device)


def body_visible(points, threshold=0.4):
    p = np.asarray(points)
    return p.shape == (33, 4) and np.isfinite(p).all() and \
        int(np.sum(p[[11, 12, 23, 24], 3] >= threshold)) >= 3


def draw_skeleton(frame, points, threshold=0.5):
    import cv2
    h, w = frame.shape[:2]
    for a, b in EDGES:
        if min(points[a, 3], points[b, 3]) < threshold:
            continue
        pt1 = (int(points[a, 0] * (w - 1)), int(points[a, 1] * (h - 1)))
        pt2 = (int(points[b, 0] * (w - 1)), int(points[b, 1] * (h - 1)))
        cv2.line(frame, pt1, pt2, (0, 230, 140), 2)
    for i, point in enumerate(points):
        if point[3] >= threshold:
            pt = (int(point[0] * (w - 1)), int(point[1] * (h - 1)))
            cv2.circle(frame, pt, 3, (0, 170, 255), -1)
    return frame
