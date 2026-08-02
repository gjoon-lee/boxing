import csv
import cv2
import mediapipe as mp

MODEL_PATH = "pose_landmarker_full.task"
VIDEO_PATH = "clips/heavybag.mp4"
CSV_PATH = "landmarks.csv"

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0
rows_written = 0

with open(CSV_PATH, "w", newline="") as f, \
     mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:

    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "timestamp_ms", "landmark_id",
                         "x", "y", "z", "visibility", "presence"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_index * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            for idx, lm in enumerate(result.pose_landmarks[0]):
                csv_writer.writerow([frame_index, timestamp_ms, idx,
                                     lm.x, lm.y, lm.z,
                                     lm.visibility, lm.presence])
                rows_written += 1

        frame_index += 1

cap.release()
print(f"frames: {frame_index}, rows: {rows_written}")