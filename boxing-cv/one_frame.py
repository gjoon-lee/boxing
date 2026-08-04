import cv2
import mediapipe as mp

MODEL_PATH = "pose_landmarker_full.task"
VIDEO_PATH = "clips/heavybag.mp4"

#Return the index 30 frame
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = cap.read()
frame = cv2.flip(frame, 1)

cap.release()
if not ret:
    raise SystemExit("Could not read frame 30")

#Converting color sequence from BGR to RGB because MediaPipe expects RGB
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#Creates an instance of the PoseLandmarkerOptions class which holds the configurations
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
)
#Calls a factory method on PoseLandmarker, passing 'options' as config,
# to build a landmarker object that's loaded and ready
# detection happens later via landmarker.detect(image)
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

#Detection happens here
result = landmarker.detect(mp_image)
landmarker.close()

print(type(result.pose_landmarks))
print(len(result.pose_landmarks))
person = result.pose_landmarks[0]
print(len(person))

print(person[16])
print(person[15])

px_lw = int(width * person[15].x)
py_lw = int(height * person[15].y)

cv2.circle(frame, (px_lw, py_lw), 25, (0,0,255), 2)
cv2.imwrite("wrist_check.png", frame)