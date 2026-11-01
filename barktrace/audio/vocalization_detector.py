#!/usr/bin/env python3
"""Stage 0: automatic dog bark detection -> timestamps.csv (event_id,start_time,end_time), the
input format of the barker stage, plus bark_detection.json (events, settings, timing).

Method:
  dogvoc           (since 2026-10-17) Every dog vocalization (bark, howl, whine,
                   growl, grunt, moan; not panting): the vocalization BEATs trained with background
                   noise x FlexSED dog prompts (geometric mean): a sound starts where that reaches 0.80 and
                   extends while the BEATs alone stays >= 0.9 (--preset precise; balanced / recall: see
                   DOGVOC_PRESETS); also where the BEATs is >= 0.95 for >= 1 s with FlexSED >= 0.3 on average, and
                   where both BEATs models are >= 0.95 for >= 0.2 s with FlexSED >= 0.5 (AGREE).
                   User's 45 videos: event P 86 / R 88. Writes sound_types.csv: the type of each event
                   (sound_models/sound_types.py; types from FlexSED + BEATs scores alone were tried and rejected),
                   and type_tracks.npz, the type models' frame
                   features, from which the pipeline types each single sound (visual/metadata_writer.py).
Events: frames kept by the rule, >= 0.1 s, then split at BEATs dips between barks (split_touching).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sound_models.beats_crnn import load_audio, to_events  # noqa: E402

MIN_DUR = 0.1
FT_BEATS_DOG = Path(os.environ.get("BARKTRACE_MODELS", Path(__file__).resolve().parents[2] / "models")) / "lab_beats_ft" / "final_dog.pt"
FT_BEATS_VOCAL = FT_BEATS_DOG.with_name("final_vocal_noise.pt")
# The all-dog-sound BEATs cannot be replaced by the other two (2026-10-28): with d = p1 the
# event P / R barely move (86.0 / 87.9 -> 85.2 / 87.9), but on 723 only it hears the quiet barks at 10.0-10.4 s
# (d 0.96-0.98, p1 and FlexSED < 0.4), so the bark burst is cut there.
# dogvoc presets: (core, extend). A detection starts where sqrt(BEATs x FlexSED) >= core and extends over neighbouring
# frames while the vocalization BEATs alone stays >= extend, so a continuous barking run is covered in full
# (user's 45 videos, event P / R and labelled time covered):
# A detection also starts where the BEATs alone is SUSTAINED and sure (>= SURE for >= SURE_SEC) while FlexSED
# agrees on average (>= SURE_FLEX): FlexSED underrates some yips under music (723 chihuahua, 5-9 s).
#   precise: event P 86.4 / R 81.8, 84% of the labelled time (without the sustained rule 87.7 / 80.9, 73%)
DOGVOC_PRESETS = {"precise": (0.80, 0.9), "balanced": (0.75, 0.9), "recall": (0.75, 0.5)}
SURE, SURE_SEC, SURE_FLEX = 0.95, 1.0, 0.3
# Quiet sounds between two detections: a gap <= BRIDGE_SEC is filled when the all-dog-sound BEATs (final_dog.pt) is
# >= 0.9 on >= BRIDGE_NEED of it (723: 9.7-10.5 s, quiet barks both new models miss; user's videos: event P / R
# 86.4 / 81.8 -> 85.5 / 82.7, labelled time 84 -> 85%; play sessions and Barkopedia unchanged)
BRIDGE_SEC, BRIDGE_NEED = 1.0, 0.6
# Both BEATs sure, FlexSED moderate (2026-10-26): a detection also starts
# where the vocalization BEATs and the all-dog-sound BEATs are both >= AGREE for >= AGREE_SEC with FlexSED >= AGREE_FLEX
# (missed sounds have FlexSED median 0.63 vs 0.82). User's videos P / R 85.5 / 86.0 -> 86.0 / 87.9; Barkopedia
# 99.3 / 64.7 -> 99.4 / 67.4; play sessions R 57.4 -> 57.8; no new false detection anywhere (no-dog videos, music, noise)
AGREE, AGREE_FLEX, AGREE_SEC = 0.95, 0.5, 0.2
SPLIT_BELOW = 0.7  # one event per sound: cut at the lowest point where the BEATs dips below this between two peaks
                   # >= 0.9; the pieces touch, so no time is lost (723: 5.2-9.7 s -> 5.2-7.5, 7.5-9.2, 9.2-9.7)


def dogvoc_keep(p1, q, d, fps, core, extend, extra=None):
    """dogvoc frames that are a dog sound, from the vocalization BEATs p1, FlexSED q and the all-dog-sound BEATs d (same
    frame grid): start where sqrt(p1 x q) >= core (or the sustained-BEATs or both-BEATs rule), extend while p1 >= extend, bridge
    short gaps where d is sure (see the constants above). extra: more start frames (bool mask), for experiments."""
    start = np.sqrt(p1 * q) >= core  # geometric mean: both must hear a dog
    agree = np.flatnonzero(np.diff(np.r_[0, ((p1 >= AGREE) & (d >= AGREE) & (q >= AGREE_FLEX)).astype(int), 0]))
    for a, z in zip(agree[::2], agree[1::2]):  # both BEATs sure, FlexSED moderate
        if (z - a) / fps >= AGREE_SEC:
            start[a:z] = True
    if extra is not None:
        start |= extra
    sure = np.flatnonzero(np.diff(np.r_[0, (p1 >= SURE).astype(int), 0]))
    for a, z in zip(sure[::2], sure[1::2]):  # sustained, sure BEATs that FlexSED does not contradict
        if (z - a) / fps >= SURE_SEC and q[a:z].mean() >= SURE_FLEX:
            start[a:z] = True
    on = p1 >= extend; keep = start.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    for a, z in zip(edges[::2], edges[1::2]):
        if start[a:z].any():  # extend a start over its whole BEATs region
            keep[a:z] = True
    ed = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0])); runs = list(zip(ed[::2], ed[1::2]))
    for (_, z1), (a2, _) in zip(runs[:-1], runs[1:]):
        if (a2 - z1) / fps <= BRIDGE_SEC and (d[z1:a2] >= 0.9).mean() >= BRIDGE_NEED:
            keep[z1:a2] = True
    return keep


def split_touching(events, p, fps, low=SPLIT_BELOW, peak=0.9, min_dur=0.1):
    """One event per sound without losing time: an event is cut at the lowest point of every dip of p below `low`
    between two frames >= `peak`, and the pieces meet at the cut, so the quieter frames between two barks stay in
    an event (the cut pieces cover exactly the original event)."""
    out = []
    for s, e in events:
        a, b = int(round(s * fps)), max(int(round(s * fps)) + 1, int(round(e * fps)))
        v = p[a:b]; hi = np.flatnonzero(v >= peak)
        cuts = [(a + i + 1 + int(np.argmin(v[i + 1:j]))) / fps for i, j in zip(hi[:-1], hi[1:])
                if j - i > 1 and v[i + 1:j].min() < low]
        bounds = [s] + [round(float(c), 3) for c in cuts] + [e]
        out += [(x, y) for x, y in zip(bounds[:-1], bounds[1:]) if y - x >= min_dur]
    return out


def detect(video, method="dogvoc", threshold=None, device="cuda:0", preset=None):
    """dogvoc events of a video; threshold overrides the preset's core."""
    assert method == "dogvoc", method
    from sound_models.beats_crnn import BeatsSED
    from sound_models.flexsed import FlexSEDModel
    t0 = time.time()
    audio = load_audio(video); dur = len(audio) / 16000
    t1 = time.time()
    m = BeatsSED(device, ckpt=FT_BEATS_VOCAL); fx = FlexSEDModel(device); t2 = time.time()
    fps, p1 = m.fps, m.frame_probs(audio)
    f2, q = fx.dog_probs(audio)
    q = q[np.clip(np.round(np.arange(len(p1)) / fps * f2).astype(int), 0, len(q) - 1)]
    core, extend = DOGVOC_PRESETS[preset or "precise"]
    if threshold is not None:
        core = threshold
    md = BeatsSED(device, ckpt=FT_BEATS_DOG); d = md.frame_probs(audio)[:len(p1)]; del md, m
    d = np.pad(d, (0, len(p1) - len(d)))
    probs = dogvoc_keep(p1, q, d, fps, core, extend).astype(np.float32)
    del fx
    events = split_touching(to_events(probs, fps, 0.5, min_dur=MIN_DUR, duration=dur), p1, fps)
    types = None
    if events:  # bark / howl / whine / growl / grunt / moan per event (sound_models/sound_types.py: ATST-F + FineLAP)
        import torch
        torch.cuda.empty_cache()
        from sound_models.sound_types import DogTypeClassifier
        clf = DogTypeClassifier(device); tracks = clf.tracks(audio)  # frame features, kept: the pipeline types each
        types = [(t, round(c, 3)) for t, c in clf.classify(audio, events, tracks)]  # single sound from them (metadata.py)
    t3 = time.time()
    return {"method": method, "threshold": {"core": core, "extend": extend, "preset": preset or "precise"},
            "duration": round(dur, 3), "events": events, "types": types, "type_tracks": tracks if events else None,
            "seconds": {"decode": round(t1 - t0, 1), "model_load": round(t2 - t1, 1), "inference": round(t3 - t2, 1)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--method", choices=["dogvoc"], default="dogvoc")
    ap.add_argument("--threshold", type=float, default=None, help="dogvoc core threshold (default: the preset's)")
    ap.add_argument("--preset", choices=list(DOGVOC_PRESETS), default=None,
                    help="dogvoc (core, extend) preset, default precise: " + str(DOGVOC_PRESETS) + "; --threshold overrides core")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    info = detect(a.video, a.method, a.threshold, a.device, a.preset)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with open(a.output_dir / "timestamps.csv", "w") as f:
        f.write("event_id,start_time,end_time\n")
        for i, (s, e) in enumerate(info["events"]):
            f.write(f"event_{i:03d},{s},{e}\n")
    if info.get("types") is not None:  # kept apart from timestamps.csv (the barker stage's input format)
        with open(a.output_dir / "sound_types.csv", "w") as f:
            f.write("event_id,type,confidence\n")
            for i, (typ, conf) in enumerate(info["types"]):
                f.write(f"event_{i:03d},{typ},{conf}\n")
    tracks = info.pop("type_tracks")
    if tracks is not None:  # ATST-F / FineLAP frame features: metadata.py types each single sound from them
        from sound_models.sound_types import save_tracks
        save_tracks(tracks, a.output_dir / "type_tracks.npz")
    (a.output_dir / "bark_detection.json").write_text(json.dumps(info, indent=2))
    print(f"{len(info['events'])} bark events ({a.method}, thr {info['threshold']}): {info['events']}")
    print(f"timing: {info['seconds']}")


if __name__ == "__main__":
    main()
