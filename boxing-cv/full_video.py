import time
import cv2
import mediapipe as mp

MODEL_PATH = "pose_landmarker_full.task"
VIDEO_PATH = "clips/heavybag.mp4"

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, #tracks landmarks across frames
)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)

#Get dimensions of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_index = 0
detected = 0
start = time.perf_counter()

#Create writer to annotate video
writer = cv2.VideoWriter("annotated.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         fps,
                         (width, height))

#Lines that will be annotated
CONNECTIONS = [
    (11, 12),                      # shoulders
    (11, 13), (13, 15),            # left arm
    (12, 14), (14, 16),            # right arm
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27),            # left leg
    (24, 26), (26, 28),            # right leg
]

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_index * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            detected += 1

            #Getting the list of joints
            person = result.pose_landmarks[0]
            points = []
            for joint in person:
                px = int(joint.x * width)
                py = int(joint.y * height)
                points.append((px, py))

            #Draw line for bones
            for a, b in CONNECTIONS:
                cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

            #Draw circle around joint
            for px, py in points:
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

        #Write every frame even if detection fails
        writer.write(frame)

        frame_index += 1

elapsed = time.perf_counter() - start
cap.release()
writer.release()

print(f"frames processed: {frame_index}")
print(f"pose detected in: {detected} ({100 * detected / frame_index:.1f}%)")
print(f"elapsed: {elapsed:.1f}s ({frame_index / elapsed:.1f} frames/sec)")