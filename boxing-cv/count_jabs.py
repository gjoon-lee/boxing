import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from video_io import video_info

CSV_PATH = "landmarks.csv"
LABELS_PATH = "labels.csv"
VIDEO_PATH = "clips/heavybag.mp4"

SHOULDER, WRIST = 11, 15   # left shoulder -> left wrist (jab hand, post-flip)
HEIGHT = 100               # naive threshold, px — eyeballed from the signal's p90

# --- rebuild the 2D extension signal from the bronze CSV ---
info = video_info(VIDEO_PATH)
W, H = info["width"], info["height"]

df = pd.read_csv(CSV_PATH)
shoulder = df[df["landmark_id"] == SHOULDER][["frame", "x", "y"]]
wrist = df[df["landmark_id"] == WRIST][["frame", "x", "y"]]
pair = shoulder.merge(wrist, on="frame", suffixes=("_s", "_w")).sort_values("frame")

dx = (pair["x_w"] - pair["x_s"]) * W
dy = (pair["y_w"] - pair["y_s"]) * H
signal = np.sqrt(dx**2 + dy**2).to_numpy()
frames = pair["frame"].to_numpy()

print(f"signal rows: {len(signal)} (expect 555)")

# --- naive peak detection ---
peaks, props = find_peaks(signal, height=HEIGHT)
detected_frames = frames[peaks]

labels = pd.read_csv(LABELS_PATH)["frame"].tolist()

print(f"height threshold: {HEIGHT}px")
print(f"peaks detected:   {len(peaks)}")
print(f"labeled jabs:     {len(labels)}")
print(f"detected frames:  {list(detected_frames)}")
print(f"labeled frames:   {labels}")

# --- plot: signal + detections vs ground truth ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(frames, signal, linewidth=1, label="shoulder->wrist extension (px)")
ax.plot(detected_frames, signal[peaks], "x", color="red", markersize=9,
        label=f"detected peaks ({len(peaks)})")
for i, lf in enumerate(labels):
    ax.axvline(lf, color="green", alpha=0.35,
               label=f"labeled jabs ({len(labels)})" if i == 0 else None)
ax.axhline(HEIGHT, color="gray", linestyle="--", alpha=0.7, label=f"height = {HEIGHT}px")
ax.set_xlabel("frame")
ax.set_ylabel("extension (px)")
ax.set_title("Stage 3.1 — naive find_peaks vs 14 labeled jabs")
ax.legend()
fig.tight_layout()
fig.savefig("peaks_naive.png", dpi=110)
print("wrote peaks_naive.png")