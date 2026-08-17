"""Matching and metrics — the part that must never know how detection works."""
import pandas as pd


def load_exclusions(clip_id, path="exclusions.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return []
    df = df[df["clip_id"] == clip_id]
    return [(int(r.start_frame), int(r.end_frame)) for r in df.itertuples()]


def in_any(frame, windows):
    return any(a <= frame <= b for a, b in windows)


def match_events(detected, labeled, tol):
    """Greedy 1:1 — each detection claims its nearest unclaimed label within tol frames."""
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


def prf(tp, fp, fn):
    p = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) else 0.0
    r = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1