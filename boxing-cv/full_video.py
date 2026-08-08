import time
import cv2
import mediapipe as mp

from video_io import read_frames, video_info

MODEL_PATH = "pose_landmarker_full.task"
VIDEO_PATH = "clips/heavybag.mp4"
OUT_PATH = "annotated.mp4"

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,  # tracks landmarks across frames
)

info = video_info(VIDEO_PATH)
fps = info["fps"]
width = info["width"]
height = info["height"]

# Create writer to annotate video — (width, height) order, not (h, w)
writer = cv2.VideoWriter(OUT_PATH,
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         fps,
                         (width, height))

# Lines that will be annotated
CONNECTIONS = [
    (11, 12),                      # shoulders
    (11, 13), (13, 15),            # left arm
    (12, 14), (14, 16),            # right arm
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27),            # left leg
    (24, 26), (26, 28),            # right leg
]

frame_index = -1
detected = 0
start = time.perf_counter()

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
    for frame_index, frame in enumerate(read_frames(VIDEO_PATH)):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_index * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            detected += 1

            # Getting the list of joints
            person = result.pose_landmarks[0]
            points = []
            for joint in person:
                px = int(joint.x * width)
                py = int(joint.y * height)
                points.append((px, py))

            # Draw line for bones
            for a, b in CONNECTIONS:
                cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

            # Draw circle around joint
            for px, py in points:
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

            cv2.circle(frame, points[15], 12, (255, 0, 0), -1)   # left wrist (BLUE) = jab hand
            cv2.circle(frame, points[16], 12, (0, 0, 255), -1)   # right wrist (RED)

        # Frame number on EVERY frame, detected or not (handoff decision)
        cv2.putText(frame, str(frame_index), (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Write every frame even if detection fails
        writer.write(frame)

elapsed = time.perf_counter() - start
writer.release()

frames = frame_index + 1
print(f"frames processed: {frames}")
print(f"pose detected in: {detected} ({100 * detected / frames:.1f}%)")
print(f"elapsed: {elapsed:.1f}s ({frames / elapsed:.1f} frames/sec)")