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
SHOULDER, WRIST = 11, 15
TOL = 3   # labels are >=9 frames apart, so ±3 windows can never overlap -> matching is unambiguous


def load_signal():
    """Bronze CSV -> (frames, 2D extension px, wrist visibility)."""
    info = video_info(VIDEO_PATH)
    W, H = info["width"], info["height"]
    df = pd.read_csv(CSV_PATH)
    s = df[df["landmark_id"] == SHOULDER][["frame", "x", "y"]]
    w = df[df["landmark_id"] == WRIST][["frame", "x", "y", "visibility"]]
    pair = s.merge(w, on="frame", suffixes=("_s", "_w")).sort_values("frame")
    dx = (pair["x_w"] - pair["x_s"]) * W
    dy = (pair["y_w"] - pair["y_s"]) * H
    ext = np.sqrt(dx**2 + dy**2).to_numpy()
    return pair["frame"].to_numpy(), ext, pair["visibility"].to_numpy()


def match_events(detected, labeled, tol=TOL):
    """Greedy one-to-one matching: each detection claims its nearest
    still-unclaimed label within ±tol. Leftover detections are FPs,
    leftover labels are FNs. One-to-one is the point: a shadow peak
    can't ride on a label its sibling already claimed."""
    unclaimed = list(labeled)
    tp, fp = [], []
    for d in detected:
        cands = [l for l in unclaimed if abs(d - l) <= tol]
        if cands:
            best = min(cands, key=lambda l: abs(d - l))
            tp.append((int(d), int(best)))
            unclaimed.remove(best)
        else:
            fp.append(int(d))
    return tp, fp, unclaimed


def score(name, detected, labeled):
    tp, fp, fn = match_events(detected, labeled)
    p = len(tp) / len(detected) if len(detected) else 0.0
    r = len(tp) / len(labeled)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print(f"\n[{name}]")
    print(f"  detected ({len(detected)}): {list(map(int, detected))}")
    print(f"  TP {len(tp)} | FP {len(fp)} {fp} | FN {len(fn)} {fn}")
    print(f"  precision {p:.1%} | recall {r:.1%} | F1 {f1:.1%}")
    return tp, fp, fn


frames, signal, vis = load_signal()
labels = pd.read_csv(LABELS_PATH)["frame"].tolist()
print(f"signal rows: {len(signal)} (expect 555) | labels: {len(labels)}")

# Detector A — Stage 3.1 naive baseline
naive_pk, _ = find_peaks(signal, height=100)
score("naive: height=100px", frames[naive_pk], labels)

# Detector B — Stage 3.3 tuned => Prominence is kept at 40 to prevent deleting real double jabs
PROM, DIST = 40, 5
tuned_pk, _ = find_peaks(signal, prominence=PROM, distance=DIST)
tp, fp, fn = score(f"tuned: prominence={PROM}, distance={DIST}", frames[tuned_pk], labels)

# Instrument autopsy: wrist visibility at every label
print(f"\nwrist visibility: clip median {np.median(vis):.3f}")
print("label | caught? | visibility")
matched_labels = {l for _, l in tp}
for l in labels:
    v = vis[frames == l][0]
    print(f"  {l:3d} | {'yes' if l in matched_labels else ' NO'} | {v:.3f}")
occ = (frames >= 290) & (frames <= 315)
print(f"occlusion window 290-315: min visibility {vis[occ].min():.3f} @frame {int(frames[occ][np.argmin(vis[occ])])}")

# --- 3-row plot: naive vs tuned vs visibility ---
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, pk, ttl in [(axes[0], naive_pk, "naive: height=100px"),
                    (axes[1], tuned_pk, f"tuned: prominence={PROM}, distance={DIST}")]:
    ax.plot(frames, signal, linewidth=1)
    ax.plot(frames[pk], signal[pk], "x", color="red", markersize=9)
    for lf in labels:
        ax.axvline(lf, color="green", alpha=0.3)
    ax.set_ylabel("extension (px)")
    ax.set_title(f"{ttl} — {len(pk)} detected vs {len(labels)} labeled")
axes[2].plot(frames, vis, linewidth=1, color="purple")
axes[2].axhline(0.2, color="gray", linestyle="--", alpha=0.7)
for lf in labels:
    axes[2].axvline(lf, color="green", alpha=0.3)
axes[2].set_ylabel("wrist visibility")
axes[2].set_xlabel("frame")
axes[2].set_title("instrument health: MediaPipe visibility of landmark 15")
fig.tight_layout()
fig.savefig("evaluation.png", dpi=110)
print("\nwrote evaluation.png")