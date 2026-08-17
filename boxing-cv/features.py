"""Signal construction from the bronze layer.

Everything here is a pure function of a landmarks DataFrame: no file paths,
no model, no plotting. That keeps the feature definitions testable and makes
them the single place where "what is a punch signal" is decided.
"""
import numpy as np

# MediaPipe pose landmark ids
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

ARMS = {"left": (L_SHOULDER, L_WRIST), "right": (R_SHOULDER, R_WRIST)}


def _xy(df, lid, W, H):
    """Landmark lid as pixel coordinates, indexed by frame."""
    sub = df[df["landmark_id"] == lid].sort_values("frame")
    return sub["frame"].to_numpy(), (sub["x"] * W).to_numpy(), (sub["y"] * H).to_numpy()


def _distance(df, a, b, W, H):
    fa, xa, ya = _xy(df, a, W, H)
    fb, xb, yb = _xy(df, b, W, H)
    assert np.array_equal(fa, fb), "landmark frames misaligned in bronze"
    return fa, np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2)


def body_scale(df, W, H, method="torso"):
    """One scale number per clip, in pixels — the ruler everything is measured against.

    Clip-median, not per-frame: dividing a noisy signal by a noisy denominator
    doubles the noise. A single stable number just rescales the axis.

    torso    — midpoint(shoulders) to midpoint(hips). Survives the boxer turning,
               because rotating about the vertical axis barely changes it.
    shoulder — shoulder-to-shoulder width. Simpler, but collapses toward zero
               as the boxer squares up or turns side-on.
    """
    if method == "shoulder":
        _, d = _distance(df, L_SHOULDER, R_SHOULDER, W, H)
        return float(np.median(d))

    if method == "torso":
        f, lsx, lsy = _xy(df, L_SHOULDER, W, H)
        _, rsx, rsy = _xy(df, R_SHOULDER, W, H)
        _, lhx, lhy = _xy(df, L_HIP, W, H)
        _, rhx, rhy = _xy(df, R_HIP, W, H)
        sx, sy = (lsx + rsx) / 2, (lsy + rsy) / 2
        hx, hy = (lhx + rhx) / 2, (lhy + rhy) / 2
        return float(np.median(np.sqrt((hx - sx) ** 2 + (hy - sy) ** 2)))

    raise ValueError(f"unknown body_scale method: {method!r}")


def arm_signals(df, W, H, scale_method=None):
    """Per-arm extension signals.

    Returns {"frames": ndarray, "left": ndarray, "right": ndarray, "scale": float}.
    With scale_method=None the signals are raw pixels (the old, clip-specific
    behaviour). With a method set, they are multiples of body size, so the same
    threshold means the same posture on any clip at any resolution.
    """
    scale = 1.0 if scale_method is None else body_scale(df, W, H, scale_method)
    out = {"scale": scale}
    for side, (shoulder, wrist) in ARMS.items():
        frames, ext = _distance(df, shoulder, wrist, W, H)
        out["frames"] = frames
        out[side] = ext / scale
    return out


def visibility(df, lid):
    sub = df[df["landmark_id"] == lid].sort_values("frame")
    return sub["frame"].to_numpy(), sub["visibility"].to_numpy()