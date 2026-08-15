"""Render the annotated skeleton video (the labeling instrument).
Usage: python full_video.py <clip_id>"""
import sys
import time
 
import cv2
import mediapipe as mp
 
from manifest import load_clip
from video_io import read_frames
 
MODEL_PATH = "pose_landmarker_full.task"
 
clip_id = sys.argv[1]
cfg = load_clip(clip_id)
out_path = f"annotated_{clip_id}.mp4"
width, height, fps = cfg["width"], cfg["height"], cfg["fps"]
 
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
)
 
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
 
CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]
 
# Text/marker sizes scale with resolution so 1920x1080 doesn't get 1080x594's dot sizes
scale = height / 594
dot = max(2, int(4 * scale))
big_dot = max(6, int(12 * scale))
 
frame_index = -1
detected = 0
start = time.perf_counter()
 
with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
    for frame_index, frame in enumerate(read_frames(cfg["path"], flip=cfg["flip"])):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(frame_index * 1000 / fps))
 
        if result.pose_landmarks:
            detected += 1
            points = [(int(j.x * width), int(j.y * height)) for j in result.pose_landmarks[0]]
            for a, b in CONNECTIONS:
                cv2.line(frame, points[a], points[b], (0, 255, 0), max(1, int(2 * scale)))
            for px, py in points:
                cv2.circle(frame, (px, py), dot, (0, 0, 255), -1)
            cv2.circle(frame, points[15], big_dot, (255, 0, 0), -1)   # BLUE = landmark 15
            cv2.circle(frame, points[16], big_dot, (0, 0, 255), -1)   # RED  = landmark 16
 
        cv2.putText(frame, str(frame_index), (20, int(45 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 255, 255), max(2, int(3 * scale)))
        writer.write(frame)
 
elapsed = time.perf_counter() - start
writer.release()
 
frames = frame_index + 1
print(f"[{clip_id}] frames: {frames} | detected: {detected} ({100 * detected / frames:.1f}%)")
print(f"elapsed: {elapsed:.1f}s ({frames / elapsed:.1f} fps) | wrote {out_path}")