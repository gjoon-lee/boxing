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
PUNCH_FRAMES = [36, 66, 93, 128, 216, 243, 272, 303, 394, 424, 460, 500, 509, 538]


def joint_pair(df, id_a, id_b, cols):
    """One row per frame with both joints' coordinates side by side."""
    a = df[df["landmark_id"] == id_a][["frame"] + cols]
    b = df[df["landmark_id"] == id_b][["frame"] + cols]
    return a.merge(b, on="frame", suffixes=("_w", "_s"))

# --- 2D, image space, pixels ---
m2 = joint_pair(df, WRIST_ID, SHOULDER_ID, ["x", "y"])
dx = (m2["x_w"] - m2["x_s"]) * WIDTH
dy = (m2["y_w"] - m2["y_s"]) * HEIGHT
m2["extension"] = (dx**2 + dy**2) ** 0.5

# --- 3D, world space, meters ---
m3 = joint_pair(df, WRIST_ID, SHOULDER_ID, ["wx", "wy", "wz"])
dwx = m3["wx_w"] - m3["wx_s"]
dwy = m3["wy_w"] - m3["wy_s"]
dwz = m3["wz_w"] - m3["wz_s"]
m3["extension"] = (dwx**2 + dwy**2 + dwz**2) ** 0.5

fig, (ax2d, ax3d) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax2d.plot(m2["frame"], m2["extension"])
ax2d.set_ylabel("2D extension (px)")

ax3d.plot(m3["frame"], m3["extension"])
ax3d.set_ylabel("3D extension (m)")
ax3d.set_xlabel("frame")

for ax in (ax2d, ax3d):
    for pf in PUNCH_FRAMES:
        ax.axvline(pf, color="red", alpha=0.4)

fig.suptitle("Jab-hand extension: image space vs world space")
fig.savefig("extension_2d_vs_3d.png", dpi=120)