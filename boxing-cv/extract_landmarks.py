import csv
import cv2
import mediapipe as mp

from video_io import read_frames, video_info

MODEL_PATH = "pose_landmarker_full.task"
VIDEO_PATH = "clips/heavybag.mp4"
CSV_PATH = "landmarks.csv"

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
)

fps = video_info(VIDEO_PATH)["fps"]
frame_index = -1
rows_written = 0

with open(CSV_PATH, "w", newline="") as f, \
     mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:

    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "timestamp_ms", "landmark_id",
                         "x", "y", "z", "visibility", "presence",
                         "wx", "wy", "wz"])

    for frame_index, frame in enumerate(read_frames(VIDEO_PATH)):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_index * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            img_lms = result.pose_landmarks[0]
            world_lms = result.pose_world_landmarks[0]
            for idx, (lm, wlm) in enumerate(zip(img_lms, world_lms)):
                csv_writer.writerow([frame_index, timestamp_ms, idx,
                                     lm.x, lm.y, lm.z,
                                     lm.visibility, lm.presence,
                                     wlm.x, wlm.y, wlm.z])
                rows_written += 1

print(f"frames: {frame_index + 1}, rows: {rows_written}")