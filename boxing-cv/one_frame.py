import cv2
import mediapipe as mp

MODEL_PATH = "pose_landmarker_full.task"
VIDEO_PATH = "clips/heavybag.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
ret, frame = cap.read()
cap.release()
if not ret:
    raise SystemExit("Could not read frame 100")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
)
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

result = landmarker.detect(mp_image)
landmarker.close()

print(type(result.pose_landmarks))
print(len(result.pose_landmarks))

person = result.pose_landmarks[0]
print(len(person))
print(person[16])