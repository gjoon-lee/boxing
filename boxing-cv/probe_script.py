import cv2

VIDEO_PATH = "clips/heavybag.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"fps: {fps:.2f}")
print(f"resolution: {width} x {height}")
print(f"reported frames: {reported_frames}")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

cap.release()

print(f"frames actually read: {frame_count}")
print(f"duration: {frame_count / fps:.2f} seconds")
