"""Run metadata: one JSON file with everything needed to show or RECREATE the annotated clip (boxes,
dog_IDs, colours, barks and the barking dog) from the original video alone, without running the
pipeline again (render.render_metadata, the app's player, praat_export.py). Results only: how the run went
(stages, timings, settings) is in the run folder's stage_log.txt.

metadata.json (schema_version 2):
  video     file, path, size_bytes, fps, width, height, frames, duration_seconds (to find and check the
            original video)
  style     colours / thicknesses / blink cadence of render.py (a recreation looks the same)
  dogs      [{dog_id, color_rgb, first_time, last_time, frames_shown, track_ids}]
  barks_source  where the bark intervals came from ("typed", or the detector's name)
  barks     [{event_id, start, end, status, barker_dog_id}]: barker_dog_id = the dog chosen as the barker
            (None: no dog visible, or a hidden uncertain pick); it blinks while start <= t <= end
  box_tolerance_px  largest difference (pixels, per coordinate) between a stored-box interpolation and the
            pipeline's exact box (0 = exact)
  boxes     {dog_id: [segment, ...]}, stored ADAPTIVELY as keyframes: every segment keeps >= 95% of its
            boxes (MIN_STORED_FRACTION); the frames left out are in the stillest stretches, so a fast dog
            keeps ~all. segment = {"track_id", "keys": [[frame, x1, y1, x2, y2], ...]}: the dog has a box on
            EVERY frame from the first to the last key; between two keys the box is their linear
            interpolation, rounded half up (box_at()). frame = 0-based video frame, time = frame / fps.
            At a frame, boxes are drawn in increasing track_id order.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import video_renderer as R

SCHEMA_VERSION = 2
BOX_TOLERANCE_PX = 0.0  # 0 = the metadata recreates the pipeline video pixel-identically; 1 px halves the keyframes again
MIN_STORED_FRACTION = 0.95  # every segment stores at least this share of its boxes (the rest: still stretches)


def keyframes(frames: np.ndarray, boxes: np.ndarray, tol: float = BOX_TOLERANCE_PX,
              min_fraction: float = MIN_STORED_FRACTION) -> list[int]:
    """Indices of one segment's keyframes (consecutive frames): Ramer-Douglas-Peucker on the 4 box
    coordinates over time. A frame is kept where the rounded linear interpolation between the kept frames
    would miss the exact box by more than tol (the first and last frame always). Then, until min_fraction
    of the frames are kept, the middle of the longest remaining gap is added -- the frames left out are
    spread over the stillest stretches (a still dog keeps ~min_fraction, a fast one ~all) -- and every gap
    is checked again (an added key is a rounded box, which can shift the interpolation next to it)."""
    def refine(keep: set) -> set:
        ks = sorted(keep)
        stack = list(zip(ks, ks[1:]))
        while stack:
            i, j = stack.pop()
            if j - i < 2:
                continue
            w = ((frames[i + 1:j] - frames[i]) / (frames[j] - frames[i]))[:, None]
            err = np.abs(boxes[i + 1:j] - np.floor(boxes[i] + w * (boxes[j] - boxes[i]) + 0.5)).max(1)
            k = int(err.argmax())
            if err[k] > tol:
                keep.add(i + 1 + k); stack += [(i, i + 1 + k), (i + 1 + k, j)]
        return keep

    keep = refine({0, len(frames) - 1})
    target = math.ceil(min_fraction * len(frames))
    if len(keep) < target:
        import heapq
        ks = sorted(keep)
        gaps = [(-(y - x), x, y) for x, y in zip(ks, ks[1:]) if y - x > 1]  # longest gap first
        heapq.heapify(gaps)
        while len(keep) < target and gaps:
            _, x, y = heapq.heappop(gaps)
            m = (x + y) // 2; keep.add(m)
            for u, v in ((x, m), (m, y)):
                if v - u > 1:
                    heapq.heappush(gaps, (-(v - u), u, v))
        keep = refine(keep)
    return sorted(keep)


def box_at(keys: list, f: int) -> tuple[int, int, int, int] | None:
    """The box of one segment at frame f (None outside it): the same rule as the player (web_player.py)."""
    if f < keys[0][0] or f > keys[-1][0]:
        return None
    lo, hi = 0, len(keys) - 1
    while lo < hi:  # last key with frame <= f
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if keys[mid][0] <= f else (lo, mid - 1)
    a, b = keys[lo], keys[min(lo + 1, len(keys) - 1)]
    w = 0.0 if b[0] == a[0] else (f - a[0]) / (b[0] - a[0])
    return tuple(int(math.floor(a[i] + w * (b[i] - a[i]) + 0.5)) for i in range(1, 5))


def adaptive_boxes(per_frame: list, tol: float = BOX_TOLERANCE_PX, min_fraction: float = MIN_STORED_FRACTION) -> list[dict]:
    """[[frame, x1, y1, x2, y2, track_id], ...] of one dog -> segments of consecutive frames per track,
    each stored as keyframes."""
    out, by_track = [], {}
    for b in per_frame:
        by_track.setdefault(b[5], []).append(b)
    for tid, rows in by_track.items():
        rows.sort(key=lambda r: r[0])
        fr = np.array([r[0] for r in rows]); bx = np.array([r[1:5] for r in rows], float)
        for seg in np.split(np.arange(len(fr)), np.where(np.diff(fr) != 1)[0] + 1):  # a gap ends a segment
            idx = seg[keyframes(fr[seg], bx[seg], tol, min_fraction)]
            out.append({"track_id": int(tid), "keys": [[int(fr[i]), *[int(v) for v in bx[i]]] for i in idx]})
    return sorted(out, key=lambda s: (s["keys"][0][0], s["track_id"]))


def readable_json(obj, indent: int = 0) -> str:
    """JSON that is easy to read: objects indented, lists of plain values (a keyframe, a colour) on one line, lists
    of lists / objects one item per line. Same content as json.dumps (json.loads gives the same object)."""
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{inner}{json.dumps(str(k))}: {readable_json(v, indent + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, (list, tuple)):
        if not obj:
            return "[]"
        if all(not isinstance(x, (dict, list, tuple)) for x in obj):
            return "[" + ", ".join(json.dumps(x, default=str) for x in obj) + "]"
        return "[\n" + ",\n".join(inner + readable_json(x, indent + 1) for x in obj) + "\n" + pad + "]"
    return json.dumps(obj, default=str)


def write_metadata(video_path: Path, tracks_csv: Path, predictions_csv: Path, track_to_profile_csv: Path,
                   output_path: Path, barks_source: str = "typed",
                   tol: float = BOX_TOLERANCE_PX, sound_types_csv: Path | None = None) -> Path:
    st = R.prepare(Path(video_path), Path(tracks_csv), Path(predictions_csv), Path(track_to_profile_csv))
    info, fps = st["info"], st["fps"]
    boxes: dict[str, list] = {}
    for f in range(int(info["frames"])):
        for tid, pid, x1, y1, x2, y2 in R.boxes_at(f, st):
            boxes.setdefault(pid, []).append([f, x1, y1, x2, y2, tid])
    order = st["profile_order"]
    dogs = []
    for pid in sorted(boxes, key=lambda p: order.get(p, 1e9)):
        fr = [b[0] for b in boxes[pid]]
        dogs.append({"dog_id": pid, "color_rgb": list(R._color_for_profile(pid, order)[::-1]),  # render colours are BGR
                     "first_time": round(min(fr) / fps, 3), "last_time": round(max(fr) / fps, 3),
                     "frames_shown": len(set(fr)), "track_ids": sorted({b[5] for b in boxes[pid]})})
    barks = []
    if Path(predictions_csv).is_file():
        t2p = st["track_to_profile"]
        types = {}  # event_id -> (type, confidence) from vocalization_detector.py --method dogvoc
        if sound_types_csv is not None and Path(sound_types_csv).is_file():
            types = {r.event_id: (r.type, float(r.confidence)) for r in pd.read_csv(sound_types_csv).itertuples()}
        rows = list(pd.read_csv(predictions_csv).itertuples())
        tracks_npz = Path(sound_types_csv).with_name("type_tracks.npz") if sound_types_csv is not None else None
        if tracks_npz is not None and tracks_npz.is_file() and rows:  # each single sound typed on its own (a detected
            # sound can hold several, e.g. 7.7 s of whines that typed whole looks like one howl)
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audio"))
            from sound_models.sound_types import classify_tracks, load_model, load_tracks
            piece = classify_tracks(load_model(), load_tracks(tracks_npz),
                                    [(float(r.start_time_sec), float(r.end_time_sec)) for r in rows])
        else:
            piece = [types.get(r.event_id, (None, None)) for r in rows]
        for r, (typ, conf) in zip(rows, piece):
            tid = None if pd.isna(r.predicted_track_id) or r.predicted_track_id == "" else int(r.predicted_track_id)
            barks.append({"event_id": r.event_id, "start": float(r.start_time_sec), "end": float(r.end_time_sec),
                          "status": r.status, "barker_dog_id": t2p.get(tid) if tid is not None else None,
                          "type": typ, "type_confidence": None if conf is None else round(conf, 3)})
    video = Path(video_path).resolve()
    meta = {"schema_version": SCHEMA_VERSION,
            "video": {"file": video.name, "path": str(video), "size_bytes": video.stat().st_size, "fps": fps,
                      "width": st["width"], "height": st["height"], "frames": int(info["frames"]),
                      "duration_seconds": round(float(info["duration_seconds"]), 3)},
            "style": {"dark_scale": R.DARK_SCALE, "normal_thickness": R.NORMAL_BOX_THICKNESS,
                      "barking_thickness": R.BARKING_BOX_THICKNESS, "flash_interval_frames": R.FLASH_INTERVAL_FRAMES},
            "dogs": dogs, "barks_source": barks_source, "barks": barks, "box_tolerance_px": tol,
            "boxes": {pid: adaptive_boxes(b, tol) for pid, b in boxes.items()}}
    Path(output_path).write_text(readable_json(meta) + "\n")
    return Path(output_path)


if __name__ == "__main__":  # self-check of the keyframe rule
    f = np.arange(100)
    still = np.tile([10, 10, 50, 50], (100, 1)).astype(float)
    assert keyframes(f, still, 0, 0) == [0, 99]                                 # pure keyframes: a still dog, 2 keys
    line = np.stack([f, f, f + 40, f + 40], 1).astype(float)
    assert keyframes(f, line, 0, 0) == [0, 99]                                  # constant speed: 2 keys
    k90 = keyframes(f, still, 0, 0.9)
    assert len(k90) == 90 and max(np.diff(k90)) <= 2                            # >= 90% stored, gaps spread out
    jump = still.copy(); jump[50] += 10
    assert {49, 50, 51} <= set(keyframes(f, jump, 1, 0))                        # a sudden move is kept
    seg = adaptive_boxes([[g, *map(int, b), 7] for g, b in zip(f, jump)], 1)[0]["keys"]
    assert all(max(abs(p - q) for p, q in zip(box_at(seg, g), jump[g])) <= 1 for g in f)  # within tolerance
    sample = {"a": [1, 2.5, None, "x"], "b": [{"k": [[1, 2], [3, 4]]}, []], "c": {}, "d": "q\"uote"}
    assert json.loads(readable_json(sample)) == sample                         # readable layout, same content
    print("metadata keyframe self-check OK")
