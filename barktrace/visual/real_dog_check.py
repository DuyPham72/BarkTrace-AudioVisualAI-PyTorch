#!/usr/bin/env python3
"""Stage 2b: zero-shot profile verification (SigLIP2, google/siglip2-so400m-patch14-384).

Drops persistent profiles that clearly are not a real dog: cat, stuffed toy, statue, a dog on a
poster/screen, a mirror reflection, a person, or empty background. Per profile, up to --k crops
spread over time are classified zero-shot and averaged; the profile is kept if mean P(dog) >= --thr.

The threshold is deliberately low (0.1): on the 21-video benchmark SigLIP2 is only trustworthy when
it is near-certain (a cat scored P(dog) = 0.003), while some real dogs score 0.3-0.7 (a fluffy dog
read as "cat", a tiny far-away dog read as "toy").

Rewrites track_to_all_dog_profile.csv in place (the unfiltered mapping is kept as
track_to_all_dog_profile_unverified.csv) and writes verify_scores.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(os.environ.get("BARKTRACE_MODELS", Path(__file__).resolve().parents[2] / "models")) / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL = "google/siglip2-so400m-patch14-384"
LABELS = ["a photo of a dog", "a photo of a cat", "a photo of a stuffed toy", "a photo of a dog statue or figurine",
          "a picture of a dog on a poster or screen", "a reflection of a dog in a mirror or window",
          "a photo of a person", "a photo of an empty background"]


def _feat(o):
    """transformers 5.x returns an output object; older versions a tensor."""
    return (o if torch.is_tensor(o) else o.pooler_output).float()


def read_crops(video, reqs) -> list:
    """RGB crops for (frame, x1, y1, x2, y2) requests (0-based frames), in request order, from ONE
    sequential pass over the video; None where the frame could not be read. Seeking per crop
    (cap.set) decodes from the last keyframe every time: 524 crops took 25 s on a 78 s clip, a
    single pass 1 s, with identical frames."""
    by_frame: dict[int, list] = {}
    for i, (f, *box) in enumerate(reqs):
        by_frame.setdefault(int(f), []).append((i, box))
    out = [None] * len(reqs)
    cap = cv2.VideoCapture(str(video)); f = -1; last = max(by_frame, default=-1)
    while f < last:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        for i, (x1, y1, x2, y2) in by_frame.get(f, ()):
            c = img[max(0, int(y1)):max(int(y1) + 2, int(y2)), max(0, int(x1)):max(int(x1) + 2, int(x2))]
            out[i] = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


class DogVerifier:
    def __init__(self, device: str = "cuda:0"):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).eval().to(device)
        with torch.no_grad():
            t = self.proc(text=LABELS, return_tensors="pt", padding="max_length", max_length=64).to(device)
            self.text = torch.nn.functional.normalize(_feat(self.model.get_text_features(**t)), dim=-1)

    @torch.no_grad()
    def probs(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        x = self.proc(images=[Image.fromarray(c) for c in crops_rgb], return_tensors="pt").to(self.device)
        v = torch.nn.functional.normalize(_feat(self.model.get_image_features(pixel_values=x["pixel_values"].half())), dim=-1)
        logits = v @ self.text.T * self.model.logit_scale.exp().float() + self.model.logit_bias.float()
        return torch.softmax(logits, dim=-1).cpu().numpy()


_VERIFIERS: dict = {}


def get_verifier(device: str = "cuda:0") -> "DogVerifier":
    """One SigLIP2 verifier per process and device (model_worker.py keeps it between runs)."""
    if device not in _VERIFIERS:
        _VERIFIERS[device] = DogVerifier(device)
    return _VERIFIERS[device]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--stage2-dir", type=Path, required=True, help="precision merger output dir")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    pieces_csv, map_csv = a.stage2_dir / "tracks_pieces.csv", a.stage2_dir / "track_to_all_dog_profile.csv"
    mapping = pd.read_csv(map_csv)
    shutil.copyfile(map_csv, a.stage2_dir / "track_to_all_dog_profile_unverified.csv")
    if mapping.empty:
        (a.stage2_dir / "verify_scores.json").write_text("{}")
        return
    tp = pd.read_csv(pieces_csv).merge(mapping[["track_id", "persistent_profile_id"]], on="track_id")
    ver = get_verifier(a.device)
    groups = list(tp.groupby("persistent_profile_id"))
    reqs, owner = [], []
    for pid, g in groups:  # tracker frames are 0-based
        for r in g.iloc[np.unique(np.linspace(0, len(g) - 1, min(a.k, len(g))).astype(int))].itertuples():
            reqs.append((r.frame, r.x1, r.y1, r.x2, r.y2)); owner.append(pid)
    all_crops = read_crops(a.video, reqs)
    scores, drop = {}, []
    for pid, g in groups:
        crops = [c for c, o in zip(all_crops, owner) if o == pid and c is not None]
        p = ver.probs(crops).mean(0) if crops else np.r_[1.0, np.zeros(len(LABELS) - 1)]
        scores[pid] = {"p_dog": round(float(p[0]), 4), "best_other": LABELS[1 + int(p[1:].argmax())], "boxes": int(len(g))}
        if p[0] < a.thr:
            drop.append(pid)
    mapping[~mapping.persistent_profile_id.isin(drop)].to_csv(map_csv, index=False)
    (a.stage2_dir / "verify_scores.json").write_text(json.dumps({"thr": a.thr, "dropped": drop, "profiles": scores}, indent=2))
    print(f"Verification: kept {mapping.persistent_profile_id.nunique() - len(drop)} profiles, dropped {drop}")


if __name__ == "__main__":
    main()
