"""Custom bounding-box renderer: dark color per persistent dog ID, brighter +
blinking when that dog is the predicted active barker for the current bark
event.

Key difference from the lab's earlier renderer (score_bark_events.py, REU
project): that one colors/labels boxes by the raw per-segment `track_id` from
Stage 1, which is NOT stable across an occlusion or exit/re-entry gap. This
renderer joins through Stage 2's track_to_all_dog_profile.csv and colors/
labels by the persistent `profile_id` instead, so a dog keeps the same color
and label for the whole video even if it leaves and re-enters frame.

`interpolate_box` and `read_video_metadata` (box between track samples, video fps/size) are adapted from
that renderer (same behaviour).
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd



def interpolate_box(track_rows: pd.DataFrame, target_frame: int, max_gap_frames: int):
    """Box of one track at target_frame: linear between its two neighbouring samples (if they are at most
    2 * max_gap_frames apart), else the nearest sample within max_gap_frames, else None. (Same behaviour as the
    earlier renderer's helper.)"""
    frames = track_rows["frame"].to_numpy(dtype=np.int64)
    boxes = track_rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
    if len(frames) == 0:
        return None
    pos = int(np.searchsorted(frames, target_frame))
    cand = [i for i in (pos, pos - 1) if 0 <= i < len(frames)]
    nearest = min(cand, key=lambda i: abs(int(frames[i]) - target_frame))
    if abs(int(frames[nearest]) - target_frame) > max_gap_frames:
        return None
    if 0 < pos < len(frames):
        lf, rf = int(frames[pos - 1]), int(frames[pos])
        if lf <= target_frame <= rf and 0 < rf - lf <= 2 * max_gap_frames:
            a = (target_frame - lf) / (rf - lf)
            return tuple(float(v) for v in boxes[pos - 1] * (1.0 - a) + boxes[pos] * a)
    return tuple(float(v) for v in boxes[nearest])


def read_video_metadata(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps, w = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h, n = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or w <= 0 or h <= 0 or n <= 0:
        raise RuntimeError(f"Invalid video metadata: fps={fps}, width={w}, height={h}, frames={n}")
    return {"fps": fps, "width": w, "height": h, "frames": n, "duration_seconds": n / fps}

# Base hues; each persistent profile_id gets one, cycling if there are more
# dogs than colors. Bright = this tuple as-is. Dark = scaled down.
BASE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]
DARK_SCALE = 0.30
NORMAL_BOX_THICKNESS = 2
BARKING_BOX_THICKNESS = 5
# label verb per sound type (same as the browser player, web_player.py)
VERB = {"bark": "BARKING", "howl": "HOWLING", "whine": "WHINING", "growl": "GROWLING", "grunt": "GRUNTING", "moan": "MOANING", "pant": "PANTING"}
FLASH_INTERVAL_FRAMES = 6  # same cadence as the existing renderer's blink
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_BOX_GAP_SEC = 0.5


def _dark(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(round(c * DARK_SCALE)) for c in color)


def _color_for_profile(profile_id: str, profile_order: dict[str, int]) -> tuple[int, int, int]:
    idx = profile_order[profile_id]
    return BASE_COLORS[idx % len(BASE_COLORS)]


def _draw_label(frame: np.ndarray, x1: int, y1: int, text: str, color: tuple[int, int, int]) -> None:
    font_scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    top = max(0, y1 - th - baseline - 12)
    left = max(0, min(x1, frame.shape[1] - 1 - tw - 12))  # keep the whole label inside the frame
    right = min(frame.shape[1] - 1, left + tw + 12)
    bottom = min(frame.shape[0] - 1, top + th + baseline + 12)
    cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
    text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
    cv2.putText(frame, text, (left + 6, bottom - baseline - 6), FONT, font_scale, text_color, thickness)


def _load_barking_windows(predictions_csv: Path, track_to_profile: dict[int, str]) -> list[tuple[float, float, str, None]]:
    """Returns [(start_time_sec, end_time_sec, barker_profile_id), ...] for
    every event that has a predicted (winning) track, translated to the
    persistent profile_id via Stage 2's mapping."""
    if not predictions_csv.is_file():
        return []
    df = pd.read_csv(predictions_csv)
    windows = []
    for _, row in df.iterrows():
        predicted = row.get("predicted_track_id")
        if pd.isna(predicted) or predicted == "":
            continue
        track_id = int(predicted)
        profile_id = track_to_profile.get(track_id)
        if profile_id is None:
            continue
        windows.append((float(row["start_time_sec"]), float(row["end_time_sec"]), profile_id, None))
    return windows


def prepare(video_path: Path, tracks_csv: Path, predictions_csv: Path, track_to_profile_csv: Path) -> dict:
    """Everything the renderer and the metadata writer (metadata.py) share: tracks per track_id, their
    observed spans, the track -> persistent dog_id map, colour order, barking windows, video geometry."""
    tracks = pd.read_csv(tracks_csv)
    if tracks.empty:
        raise RuntimeError(f"No tracks found in {tracks_csv}")

    profile_map_df = pd.read_csv(track_to_profile_csv)
    track_to_profile = dict(zip(profile_map_df["track_id"].astype(int), profile_map_df["persistent_profile_id"]))

    # Stable color assignment: order profiles by first appearance so re-runs
    # on the same video are deterministic.
    first_seen_order = (
        tracks.assign(profile_id=tracks["track_id"].astype(int).map(track_to_profile))
        .dropna(subset=["profile_id"])
        .sort_values("frame")
        .drop_duplicates("profile_id")["profile_id"]
        .tolist()
    )
    profile_order = {pid: i for i, pid in enumerate(first_seen_order)}

    barking_windows = _load_barking_windows(predictions_csv, track_to_profile)

    info = read_video_metadata(video_path)
    fps = float(info["fps"])
    width, height = int(info["width"]), int(info["height"])
    max_gap_frames = max(1, int(round(MAX_BOX_GAP_SEC * fps)))

    grouped_tracks = {
        int(track_id): group.reset_index(drop=True)
        for track_id, group in tracks.groupby("track_id")
    }
    # observed span per track, computed once (was recomputed per frame x track: 530k pandas calls on 723)
    spans = {tid: (int(g["frame"].min()), int(g["frame"].max())) for tid, g in grouped_tracks.items()}
    seen = {tid: np.sort(g["frame"].to_numpy()) for tid, g in grouped_tracks.items()}  # observed frames per track

    return {"grouped_tracks": grouped_tracks, "spans": spans, "seen": seen, "track_to_profile": track_to_profile,
            "profile_order": profile_order, "barking_windows": barking_windows, "info": info, "fps": fps,
            "width": width, "height": height, "max_gap_frames": max_gap_frames}


def boxes_at(frame_index: int, st: dict):
    """(track_id, profile_id, x1, y1, x2, y2) drawn at this frame, in drawing order; at most one box per dog.
    Two pieces of one dog can both have a box here when one of them is only being held across a short gap
    (interpolate_box) while the other has a real detection: the box nearest to a detection wins (then the
    piece with more detections), so a dog is never drawn twice."""
    width, height = st["width"], st["height"]
    best: dict[str, tuple] = {}
    for track_id, track_rows in st["grouped_tracks"].items():
        # interpolate_box() (reused unmodified from score_bark_events.py)
        # holds the box at a track's first/last known position for up
        # to max_gap_frames beyond its actually-observed range -- by
        # design for bridging brief mid-track gaps, but the same
        # behavior also applies at the very start/end of a track,
        # which visibly draws a box before the dog has appeared (or
        # after it has left) for up to MAX_BOX_GAP_SEC. Guard against
        # that here without touching the shared, unmodified function:
        # only allow it to interpolate within a track's observed span.
        first_observed, last_observed = st["spans"][track_id]
        if frame_index < first_observed or frame_index > last_observed:
            continue
        box = interpolate_box(track_rows, frame_index, st["max_gap_frames"])
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        profile_id = st["track_to_profile"].get(track_id)
        if profile_id is None:
            continue
        seen = st["seen"][track_id]; k = np.searchsorted(seen, frame_index)
        dist = min(abs(int(seen[j]) - frame_index) for j in (k - 1, k) if 0 <= j < len(seen))
        rank = (dist, -len(seen))
        if profile_id not in best or rank < best[profile_id][0]:
            best[profile_id] = (rank, (track_id, profile_id, x1, y1, x2, y2))
    yield from (b for _, b in sorted(best.values(), key=lambda v: v[1][0]))

def _write_annotated(video_path: Path, output_path: Path, fps: float, width: int, height: int,
                     boxes_fn, color_of, barking_windows, style: dict | None = None) -> Path:
    """Read the video, draw each frame's boxes, write H.264 + the original audio (Opus).
    boxes_fn(frame_index) -> [(dog_id, x1, y1, x2, y2), ...] in drawing order; color_of(dog_id) -> BGR."""
    st = style or {}
    flash = int(st.get("flash_interval_frames", FLASH_INTERVAL_FRAMES))
    thick_bark = int(st.get("barking_thickness", BARKING_BOX_THICKNESS))
    thick = int(st.get("normal_thickness", NORMAL_BOX_THICKNESS))
    dark = lambda c: tuple(int(round(v * st.get("dark_scale", DARK_SCALE))) for v in c)  # noqa: E731
    output_path.parent.mkdir(parents=True, exist_ok=True)
    silent_path = output_path.with_name(output_path.stem + ".silent_tmp.mp4")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create: {silent_path}")

    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            time_sec = frame_index / fps

            for profile_id, x1, y1, x2, y2 in boxes_fn(frame_index):
                base_color = color_of(profile_id)

                bark = next((w for w in barking_windows if w[2] == profile_id and w[0] <= time_sec <= w[1]), None)

                if bark:
                    flash_on = (frame_index // flash) % 2 == 0
                    color = base_color if flash_on else dark(base_color)
                    thickness = thick_bark
                    label = f"{profile_id} | {VERB.get(bark[3], 'BARKING')}"
                else:
                    color = dark(base_color)
                    thickness = thick
                    label = f"{profile_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                _draw_label(frame, x1, y1, label, color)

            writer.write(np.ascontiguousarray(frame, dtype=np.uint8))
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    # Mux the original audio back in. H.264 + Opus so the file plays everywhere: browsers play it
    # directly (no Gradio re-encode), and VS Code's media player, which has no AAC decoder, plays the
    # sound too (AAC tracks are silent there).
    import subprocess
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(silent_path), "-i", str(video_path), "-map", "0:v:0", "-map", "1:a:0?",
         *enc, "-c:a", "libopus", "-b:a", "128k", "-shortest", str(output_path)],
        capture_output=True, text=True)
    if result.returncode != 0 or not output_path.is_file():
        # audio could not be muxed: keep a playable video without sound
        result = subprocess.run(["ffmpeg", "-y", "-i", str(silent_path), *enc, str(output_path)],
                                capture_output=True, text=True)
    if result.returncode != 0 or not output_path.is_file():
        silent_path.rename(output_path)
    else:
        silent_path.unlink(missing_ok=True)

    return output_path


def render(
    video_path: Path,
    tracks_csv: Path,
    predictions_csv: Path,
    track_to_profile_csv: Path,
    output_path: Path,
) -> Path:
    """The annotated video from the pipeline's own outputs (exact boxes)."""
    st = prepare(video_path, tracks_csv, predictions_csv, track_to_profile_csv)
    return _write_annotated(
        Path(video_path), Path(output_path), st["fps"], st["width"], st["height"],
        lambda f: [(pid, x1, y1, x2, y2) for _, pid, x1, y1, x2, y2 in boxes_at(f, st)],
        lambda pid: _color_for_profile(pid, st["profile_order"]), st["barking_windows"])


def load_metadata(metadata_json: Path, video_path: Path | None = None) -> tuple[dict, Path]:
    """Read a metadata.json and check the video belongs to it (same size, frame count and frame rate).
    video_path defaults to the path recorded in the metadata. -> (metadata, video path)"""
    import json
    m = json.loads(Path(metadata_json).read_text())
    if m.get("schema_version", 1) < 2:
        raise ValueError(f"{Path(metadata_json).name}: metadata schema 2 needed (re-run the pipeline)")
    v = m["video"]
    video_path = Path(video_path or v["path"])
    if not video_path.is_file():
        raise ValueError(f"video not found: {video_path} (give the original clip, {v['file']})")
    info = read_video_metadata(video_path)
    if (video_path.stat().st_size != v["size_bytes"] or int(info["frames"]) != v["frames"]
            or abs(float(info["fps"]) - v["fps"]) > 1e-6):
        raise ValueError(f"this video is not the one the metadata was made from: expected {v['file']} "
                         f"({v['size_bytes']} bytes, {v['frames']} frames at {v['fps']:.3f} fps), got "
                         f"{video_path.name} ({video_path.stat().st_size} bytes, {int(info['frames'])} frames)")
    return m, video_path


def render_metadata(metadata_json: Path, output_path: Path, video_path: Path | None = None) -> Path:
    """The annotated video from metadata.json and the original video ONLY (no pipeline outputs): boxes from
    the stored keyframes, dog colours, barks and style from the metadata. video_path defaults to the
    recorded path; the video must match the recorded size, frame count and frame rate."""
    from metadata_writer import box_at
    m, video_path = load_metadata(metadata_json, video_path)
    v = m["video"]
    per_frame: dict[int, list] = {}  # frame -> [(track_id, dog_id, x1, y1, x2, y2)]
    for dog, segments in m["boxes"].items():
        for sg in segments:
            for f in range(sg["keys"][0][0], sg["keys"][-1][0] + 1):
                per_frame.setdefault(f, []).append((sg["track_id"], dog, *box_at(sg["keys"], f)))
    colour = {d["dog_id"]: tuple(d["color_rgb"][::-1]) for d in m["dogs"]}  # BGR for OpenCV
    windows = [(b["start"], b["end"], b["barker_dog_id"], b.get("type")) for b in m["barks"] if b.get("barker_dog_id")]
    return _write_annotated(
        video_path, Path(output_path), v["fps"], v["width"], v["height"],
        lambda f: [b[1:] for b in sorted(per_frame.get(f, []), key=lambda b: b[0])],  # draw order = track_id
        lambda dog: colour[dog], windows, m.get("style"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Recreate an annotated clip from metadata.json + the original video only "
                                             "(no pipeline run).")
    ap.add_argument("metadata", type=Path, help="a run's metadata.json (schema 2)")
    ap.add_argument("--video", type=Path, default=None, help="the original video (default: the path in the metadata)")
    ap.add_argument("--out", type=Path, default=None, help="output .mp4 (default: annotated_from_metadata.mp4 next to it)")
    a = ap.parse_args()
    print(render_metadata(a.metadata, a.out or a.metadata.with_name("annotated_from_metadata.mp4"), a.video))
