from video_io import read_frames

count = 0
for frame in read_frames("clips/heavybag.mp4"):
    count += 1

print(count)
