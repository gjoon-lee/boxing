# Architecture decisions

Three lines per entry: what the situation was, what was decided, why. Newest last.
Entries marked *(retroactive)* were made earlier in the project and are recorded here
for the paper trail.

---

**#1 — Normalize the mirror at ingestion** *(retroactive, Stage 2)*
- **Context:** The source clip is horizontally mirrored (front-camera artifact); early analysis silently tracked the wrong hand.
- **Decision:** `cv2.flip(frame, 1)` immediately after the read guard, inside the ingestion layer, before any consumer touches pixels. `flip=True` is the default.
- **Why:** Fix data defects once, at the boundary. No downstream script should ever need to know the source was dirty.

**#2 — Long (tidy) format for the bronze layer** *(retroactive, Stage 2)*
- **Context:** 33 landmarks × 11 fields per frame could be stored wide (one row per frame) or long (one row per frame×landmark).
- **Decision:** Long format: `frame, timestamp_ms, landmark_id, x, y, z, visibility, presence, wx, wy, wz`.
- **Why:** Phase B (two-person sparring) becomes an added `person_id` column, not 33 new columns. Schema decisions are cheap now, expensive later.

**#3 — 2D image-space extension as the detection feature** *(retroactive, Stage 2)*
- **Context:** Punch signal could use 3D world-space distance (meters, hip-anchored) or 2D image-space pixel distance (landmarks 11→15).
- **Decision:** Detect on 2D; keep 3D for measurement and sanity checks (extension peaks plateau at real arm reach).
- **Why:** Labeled A/B showed far stronger punch/baseline contrast in 2D at this camera angle. Accurate ≠ discriminative — features are hired for contrast.

**#4 — Migrate only load-bearing scripts to `video_io`** *(Stage 3.0)*
- **Context:** Six copies of capture plumbing across five scripts; only two scripts are part of the ongoing pipeline.
- **Decision:** Migrate `extract_landmarks.py` and `full_video.py`; leave `probe_script.py`, `one_frame.py`, `full_img_ver.py` as unmigrated scaffolding. Each migration verified by A/B run — outputs byte-identical.
- **Why:** Migrate what carries load; scaffolding must be available, not beautiful. `video_info()` supersedes the probe anyway.

**#5 — `prominence=40, distance=5` over the 100%-precision config** *(Stage 3.3 — ratify or overturn: originally argued by Claude)*
- **Context:** `prominence=60` scores P=100% with zero false positives, but drops label 509.
- **Decision:** Ship `prominence=40, distance=5` (P/R/F1 all 85.7%).
- **Why:** 509 is the second punch of a real double jab; rapid combinations don't fully retract, so second hits are structurally low-prominence. Buying precision by deleting real punches of a combination type actually thrown is the wrong trade.

**#6 — Defer guard-drop discrimination to Stage 5** *(Decision 1, today)*
- **Context:** Both false positives (frames 190, 380) are guard drops — the scalar extension signal measures distance, not direction.
- **Decision:** No discriminator in the jab counter for now; document the blind spot; treat 190/380 as the first labeled guard-drop events for Stage 5.
- **Why:** A directional fix tuned against two examples on one clip would overfit; abundant sparring footage is available for doing it properly later.

**#7 — Repo carries irreplaceable inputs, not regenerable outputs** *(hygiene pass — pending ratification)*
- **Context:** Tracked files include the dev clip, the model weights, the bronze CSV, and generated plots/videos.
- **Decision:** Keep `clips/heavybag.mp4` (only dev dataset; cannot re-film until discharge) and `labels.csv` (ground truth). Untrack the model (re-downloadable by URL), the bronze CSV, and generated media (all reproducible from code). Keep one curated `docs/evaluation.png` for the README.
- **Why:** clip + labels + code = anyone can reproduce the scoreboard. Outputs that regenerate in one command are noise in version control; inputs that cannot be recreated are the crown jewels.

**#8 — Phase B timing: YOLO now vs finish single-person taxonomy first** *(Decision 3 — PENDING)*
- **Context:** …
- **Decision:** …
- **Why:** …