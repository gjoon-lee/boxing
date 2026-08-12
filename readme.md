# Boxing CV — pose-based punch analysis

Turning raw training footage into measured punch events with MediaPipe, OpenCV, and an
evaluation harness that grades the detector against hand-labeled ground truth.

I am a student of datascience and wanted to apply technical knowledge to my hobby: boxing. Hopefully this project becomes useful before my next fight.

**Current scoreboard (heavy-bag clip, 555 frames, 14 labeled jabs):**

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Naive (`height=100px`) | 76.9% | 71.4% | 74.1% |
| Tuned (`prominence=40, distance=5`) | **85.7%** | **85.7%** | **85.7%** |

![Evaluation: naive vs tuned vs instrument visibility]

## How it works

The system is built as an explicit ETL pipeline — extraction, transformation, analytics —
because that is the mental model it was designed with:

```
video ──► read_frames()          ingestion (mirror-normalized at the source)
      ──► MediaPipe PoseLandmarker (Tasks API, VIDEO mode, 33 landmarks/frame)
      ──► landmarks.csv          bronze layer: long format, one row per (frame, landmark),
                                 image-space AND world-space coordinates side by side
      ──► extension signal       2D shoulder→wrist pixel distance (landmarks 11→15)
      ──► find_peaks()           prominence + refractory distance
      ──► evaluation harness     1:1 matching vs labels.csv → precision / recall / F1
```

The long (tidy) bronze format is deliberate: two-person sparring analysis later means
adding a `person_id` column, not 33 more columns.

## Results, honestly

- **Both residual misses are instrument failures, not detector failures.** Wrist
  visibility at the two missed jabs is 0.109 and 0.158 against a clip median of 0.365;
  the occlusion window (frames 290–315) bottoms out at 0.062. MediaPipe could not see
  the wrist — those stretches are flagged as instrument-blind, not reported as
  "no punch."
- **Both false positives are real events of the wrong kind:** guard drops, confirmed by
  scrubbing the numbered source video. The scalar extension signal measures distance,
  not direction — a known, documented blind spot, deliberately deferred to the
  guard-drop detector (Stage 5), where that same signal becomes the target.
- **Aggregate counts lie.** The naive detector's "13 of 14" hid 3 double-counts
  canceling against 4 misses; the tuned detector found exactly 14 — and 2 of them were
  wrong. Detectors here are graded on *which*, never on *how many*.
- The 100%-precision configuration (`prominence=60`) was rejected because it silently
  deletes the second punch of double jabs — rapid combinations don't retract fully
  between punches, so the second peak has structurally low prominence.

## Engineering notes — what broke and what it taught

- **The source video was mirrored** (front-camera artifact), silently swapping which
  hand was analyzed. Fix: normalize at ingestion — every consumer receives corrected
  pixels; no downstream code ever sees raw frames.
- **Accurate ≠ discriminative.** 3D world-space extension is metrically honest
  (peaks plateau at my actual ~0.5 m reach) but anatomy caps its range; the 2D
  image-space signal shows far stronger punch/baseline contrast at this camera angle
  and won a labeled A/B. Features are hired for contrast, not correctness.
- **Two tuning knobs against 14 labels is already the overfitting edge.** Tuning
  stopped there; further gains require more data, not more parameters.
