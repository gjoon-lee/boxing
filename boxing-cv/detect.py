"""Turn extension signals into punch events."""
import numpy as np
from scipy.signal import find_peaks


def _ms_to_frames(ms, fps):
    return max(1, round(ms * fps / 1000))


def detect_punches(signals, fps, prominence, refractory_ms=167,
                   cross_arm_ms=100, arms=("left", "right")):
    """Return a list of (frame, side, height) sorted by frame.

    Refractory is applied PER ARM: the same hand physically cannot fire twice
    within ~170 ms, but a 1-2 combination puts the other hand in much sooner.
    A separate, shorter cross-arm window then removes mirror artefacts, where
    one physical punch shows up on both arms in the same instant.
    """
    frames = signals["frames"]
    distance = _ms_to_frames(refractory_ms, fps)

    events = []
    for side in arms:
        peaks, props = find_peaks(signals[side], prominence=prominence, distance=distance)
        for idx, p in enumerate(peaks):
            events.append((int(frames[p]), side, float(props["prominences"][idx])))

    events.sort(key=lambda e: e[0])
    if len(arms) < 2:
        return events

    gap = _ms_to_frames(cross_arm_ms, fps)
    kept = []
    for ev in events:
        if kept and ev[0] - kept[-1][0] <= gap and ev[1] != kept[-1][1]:
            if ev[2] > kept[-1][2]:      # same instant, both arms -> keep the stronger
                kept[-1] = ev
            continue
        kept.append(ev)
    return kept