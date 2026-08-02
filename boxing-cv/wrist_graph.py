#Noticed that using raw coordinates to count punches do not match the reality
#Next step is to create a derived feature like the angle of joints which signal a jab

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("landmarks.csv")
print(df.shape)
print(df.head())

JAB_WRIST = 15   

wrist = df[df["landmark_id"] == JAB_WRIST]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

for ax, axis_name in zip(axes, ["x", "y", "z"]):
    ax.plot(wrist["frame"], wrist[axis_name])
    ax.set_ylabel(axis_name)

axes[-1].set_xlabel("frame")
fig.suptitle(f"Landmark {JAB_WRIST} — position over time")
fig.savefig("wrist_axes.png", dpi=120)


#Derived feature: Extension(shoulder to wrist length)
WRIST_ID = 15
SHOULDER_ID = 11
WIDTH, HEIGHT = 1080, 594

wrist = df[df["landmark_id"] == WRIST_ID][["frame", "x", "y"]]
shoulder = df[df["landmark_id"] == SHOULDER_ID][["frame", "x", "y"]]

merged = wrist.merge(shoulder, on="frame", suffixes=("_w", "_s"))

merged["dx"] = (merged["x_w"] - merged["x_s"]) * WIDTH
merged["dy"] = (merged["y_w"] - merged["y_s"]) * HEIGHT
merged["extension"] = (merged["dx"] ** 2 + merged["dy"] ** 2) ** 0.5

print(merged.shape)
print(merged.head())

plt.figure(figsize=(14, 5))
plt.plot(merged["frame"], merged["extension"])
plt.xlabel("frame")
plt.ylabel("shoulder-to-wrist distance (px)")
plt.savefig("extension.png", dpi=120)