"""Detection + scoring harness.

Usage:
    python evaluate.py <clip_id>            score the shipped config
    python evaluate.py <clip_id> --ablation run the full ablation table
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detect import detect_punches
from features import arm_signals, visibility
from manifest import bronze_path, load_clip
from scoring import in_any, load_exclusions, match_events, prf

# --- shipped configuration (tuned on heavybag ONLY; see decisions.md #13) ---
SCALE_METHOD = "torso"
PROMINENCE = 0.25       # multiples of torso length — dimensionless, so it transfers
ARMS = ("left",)        # lead arm only; see decisions.md #12 for why both arms lost
REFRACTORY_MS = 167
TOL_MS = 100

ABLATION = [
    ("A  raw pixels, lead arm",       None,    ("left",),          40.0),
    ("B  + torso normalization",      "torso", ("left",),          0.25),
    ("C  + both arms",                "torso", ("left", "right"),  0.25),
    ("D  + both arms, retuned",       "torso", ("left", "right"),  0.30),
]


def evaluate(clip_id, scale_method, arms, prominence, verbose=False):
    cfg = load_clip(clip_id)
    df = pd.read_csv(bronze_path(clip_id))
    signals = arm_signals(df, cfg["width"], cfg["height"], scale_method)

    events = detect_punches(signals, cfg["fps"], prominence,
                            refractory_ms=REFRACTORY_MS, arms=arms)
    detected = [e[0] for e in events]

    labels = pd.read_csv("labels.csv")
    labels = labels[labels["clip_id"] == clip_id]["frame"].tolist()

    windows = load_exclusions(clip_id)
    kept_det = [d for d in detected if not in_any(d, windows)]
    kept_lab = [l for l in labels if not in_any(l, windows)]
    dropped = len(labels) - len(kept_lab)

    tol = max(1, round(TOL_MS * cfg["fps"] / 1000))
    tp, fp, fn = match_events(kept_det, kept_lab, tol)
    p, r, f1 = prf(tp, fp, fn)

    if verbose:
        excluded_frames = sum(b - a + 1 for a, b in windows)
        coverage = 1 - excluded_frames / len(signals["frames"])
        print(f"[{clip_id}] fps {cfg['fps']:.2f} | body scale {signals['scale']:.0f}px "
              f"| tol +/-{tol}f | coverage {coverage:.0%} ({dropped} labels excluded)")
        print(f"  detected {len(kept_det)} | TP {len(tp)} FP {len(fp)} FN {len(fn)} {fn}")
        print(f"  precision {p:.1%} | recall {r:.1%} | F1 {f1:.1%}")

    return dict(p=p, r=r, f1=f1, n_det=len(kept_det), n_lab=len(kept_lab),
                fn=fn, signals=signals, df=df, cfg=cfg, labels=labels,
                detected=kept_det, windows=windows)


def plot(clip_id, res):
    signals, df = res["signals"], res["df"]
    frames = signals["frames"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(frames, signals["left"], linewidth=1, label="lead arm")
    axes[0].plot(frames, signals["right"], linewidth=1, alpha=0.45, label="rear arm")
    for d in res["detected"]:
        axes[0].axvline(d, color="red", alpha=0.5, linewidth=0.9)
    for l in res["labels"]:
        axes[0].axvline(l, color="green", alpha=0.35)
    for a, b in res["windows"]:
        for ax in axes:
            ax.axvspan(a, b, color="gray", alpha=0.25)
    axes[0].set_ylabel("extension (x torso)")
    axes[0].set_title(f"{clip_id} — red=detected, green=labeled, gray=excluded")
    axes[0].legend(loc="upper right", fontsize=8)
    vf, vv = visibility(df, 15)
    axes[1].plot(vf, vv, linewidth=1, color="purple")
    axes[1].axhline(0.35, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("wrist visibility")
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(f"evaluation_{clip_id}.png", dpi=110)
    print(f"wrote evaluation_{clip_id}.png")


if __name__ == "__main__":
    clip_id = sys.argv[1]
    if "--ablation" in sys.argv:
        print(f"{'config':30s} | {'P':>6s} {'R':>6s} {'F1':>6s} | detections")
        print("-" * 66)
        for name, m, arms, prom in ABLATION:
            r = evaluate(clip_id, m, arms, prom)
            print(f"{name:30s} | {r['p']:6.1%} {r['r']:6.1%} {r['f1']:6.1%} | {r['n_det']}")
    else:
        res = evaluate(clip_id, SCALE_METHOD, ARMS, PROMINENCE, verbose=True)
        plot(clip_id, res)