import time
import cv2
import mediapipe as mp

MODEL_PATH = "pose_landmarker_full.task"
VIDEO_PATH = "clips/heavybag.mp4"

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE, #tracks landmarks across frames
)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)

frame_index = 0
detected = 0
start = time.perf_counter()

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_index * 1000 / fps)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            detected += 1

        frame_index += 1

elapsed = time.perf_counter() - start
cap.release()

print(f"frames processed: {frame_index}")
print(f"pose detected in: {detected} ({100 * detected / frame_index:.1f}%)")
print(f"elapsed: {elapsed:.1f}s ({frame_index / elapsed:.1f} frames/sec)")