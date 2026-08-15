import csv
 
from video_io import video_info
 
MANIFEST_PATH = "clips.csv"
 
TRUE_STRINGS = {"true", "t", "yes", "y", "1"}
FALSE_STRINGS = {"false", "f", "no", "n", "0"}
 
 
def to_bool(text, field):
    """Converts CSV text to boolean"""
    key = text.strip().lower()
    if key in TRUE_STRINGS:
        return True
    if key in FALSE_STRINGS:
        return False
    raise ValueError(f"{field}: cannot read {text!r} as a boolean")
 
 
def load_clip(clip_id):
    """Return manifest config for clip_id, merged with live file metadata."""
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["clip_id"] == clip_id:
                break
        else:
            raise KeyError(f"{clip_id!r} not in {MANIFEST_PATH}")
 
    cfg = dict(row)
    cfg["flip"] = to_bool(row["flip"], "flip")
    cfg["stance"] = row["stance"].strip() or None   # blank = not yet verified
    cfg.update(video_info(cfg["path"]))             # fps/width/height read from the file itself
    return cfg
 
 
def bronze_path(clip_id):
    """DECISION A (provisional — ratify or overrule): one bronze file per clip.
    Single-file alternative is one line: return 'landmarks.csv'."""
    return f"landmarks/{clip_id}.csv"