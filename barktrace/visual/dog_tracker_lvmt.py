#!/usr/bin/env python3
"""Stage 1a, LVMT mode: dog detection + tracking with LVMT (Norouzi et al., WACV 2027, arXiv 2609.34895;
github.com/tue-mps/lvmt, MIT): online video instance segmentation with a frozen DINOv2-L ViT and queries carried
across frames by a GRU memory, so query i is the same object in every frame.

Writes <output-dir>/tracks.csv (frame 0-based, time_sec, track_id, x1..y2, confidence,
class_name), so Stage 2 (precision merger), 2b (real-dog check), 1b (barker) and the player are unchanged.
  - every frame is processed, in WINDOW-frame chunks with the memory carried over; each query's mask -> its box
  - per query, the frames with dog probability >= PF form stretches (gaps <= GAP frames bridged); a stretch with
    >= MIN_FRAMES frames and mean dog probability >= VS becomes a track (Stage 2 links the stretches of one dog)
Checkpoint: models/lvmt/ovis_dinov2_online_55.6.pth (OVIS, occlusion-heavy; dog is a class). Code: LVMT and
Detectron2 (as pure Python; its compiled ops are not needed) in third_party/ (setup_third_party.sh).
Benchmark with Stage 2 + 2b (21 tracking-GT videos):
HOTA 52.6 vs 51.6 (SAM3), IDF1 56.8 vs 55.5, identity purity 97.6 vs 97.8, ID switches 38 vs 56, missed boxes 3331 vs 3545, false boxes 2313 vs 2196.
Usage: dog_tracker_lvmt.py --video V --output-dir D [--device cuda:0] [--vs 0.9] [--pf 0.5]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODELS = Path(os.environ.get("BARKTRACE_MODELS", REPO / "models"))
TP = REPO / "third_party"  # setup_third_party.sh
CKPT = MODELS / "lvmt" / "ovis_dinov2_online_55.6.pth"
CFG = "configs/ovis/lvmt/vit-large/lvmt_online_ViTL_dinov2.yaml"
WINDOW, VS, PF, MIN_FRAMES, GAP = 8, 0.9, 0.5, 15, 5  # best of a sweep on the 21 GT videos
COLUMNS = ["frame", "time_sec", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_name"]


def to_tracks(rows: np.ndarray, nq: int, fps: float, vs: float = VS, pf: float = PF, min_frames: int = MIN_FRAMES,
              gap: int = GAP) -> pd.DataFrame:
    """rows: (frame, query, x1, y1, x2, y2, mask area, per-frame dog prob) -> tracks.csv rows (see module doc).
    Per query, the frames with dog probability >= pf form stretches (gaps <= gap frames bridged); a stretch is kept as
    its own track (id = query * 1000 + stretch number) if it has >= min_frames frames and a mean dog probability >= vs.
    (A query's mask lingers after its dog has gone and a slot can be re-used later, so a whole-query score both
    dilutes real dogs and lets ghosts through; Stage 2 links the stretches of one dog again.)"""
    if not len(rows):
        return pd.DataFrame(columns=COLUMNS)
    r = rows[rows[:, 7] >= pf]
    if not len(r):  # no frame confident enough
        return pd.DataFrame(columns=COLUMNS)
    r = r[np.lexsort((r[:, 0], r[:, 1]))]
    q, fr = r[:, 1].astype(int), r[:, 0].astype(int)
    new = np.r_[True, (q[1:] != q[:-1]) | (fr[1:] - fr[:-1] > gap + 1)]  # start of a stretch
    seg = np.cumsum(new) - 1
    n = np.bincount(seg); mean = np.bincount(seg, weights=r[:, 7]) / n
    keep = (n[seg] >= min_frames) & (mean[seg] >= vs)
    num = seg - np.r_[0, np.cumsum(np.bincount(q[new], minlength=nq))[:-1]][q]  # stretch index within its query
    r, tid = r[keep], (q * 1000 + num + 1)[keep]
    t = pd.DataFrame({"frame": r[:, 0].astype(int), "time_sec": r[:, 0] / fps, "track_id": tid,
                      "x1": r[:, 2].round(1), "y1": r[:, 3].round(1), "x2": r[:, 4].round(1), "y2": r[:, 5].round(1),
                      "confidence": r[:, 7].round(4), "class_name": "dog"})
    return t.sort_values(["frame", "track_id"]).reset_index(drop=True)


def build(device: str):
    os.environ.setdefault("HF_HOME", str(MODELS / "hf_cache"))
    sys.path[:0] = [str(TP / "detectron2"), str(TP / "LVMT"), str(TP / "lvmt_stubs")]
    import timm
    import torch
    # the LVMT checkpoint holds the trained encoder, so the backbone needs no pretrained DINOv2 download (LVMT's own
    # offline loader also looks in HF_HOME instead of HF_HOME/hub and fails under the pipeline's HF_HUB_OFFLINE=1)
    _create = timm.create_model
    timm.create_model = lambda *a, **kw: _create(*a, **{**kw, "pretrained": False})
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config
    from lvmt.config import add_lvmt_config
    from lvmt.data_video.datasets.ytvis import OVIS_CATEGORIES
    cfg = get_cfg(); add_deeplab_config(cfg); add_lvmt_config(cfg)
    cfg.merge_from_file(str(TP / "LVMT" / CFG))
    cfg.MODEL.WEIGHTS = str(CKPT); cfg.MODEL.DEVICE = device
    cfg.freeze()
    model = build_model(cfg).eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    dog = next(i for i, c in enumerate([c for c in OVIS_CATEGORIES if c["isthing"] == 1]) if c["name"].lower() == "dog")
    return torch, cfg, model, dog


def infer(video: Path, device: str = "cuda:0", built=None):
    """-> rows (frame, query, x1, y1, x2, y2, mask area, per-frame dog prob), number of queries, frames, fps."""
    torch, cfg, model, dog = built or build(device)
    F = torch.nn.functional
    cap = cv2.VideoCapture(str(video)); H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    k = cfg.INPUT.MIN_SIZE_TEST / min(H, W)
    if max(H, W) * k > cfg.INPUT.MAX_SIZE_TEST:
        k = cfg.INPUT.MAX_SIZE_TEST / max(H, W)
    h, w = int(round(H * k)), int(round(W * k))
    mean = torch.tensor(cfg.MODEL.PIXEL_MEAN, device=device).view(3, 1, 1); std = torch.tensor(cfg.MODEL.PIXEL_STD, device=device).view(3, 1, 1)
    div = cfg.MODEL.BACKBONE.SIZE_DIVISIBILITY; ph, pw = -(-h // div) * div, -(-w // div) * div
    rows, nfr, nq = [], 0, cfg.MODEL.BACKBONE.NUM_OBJECT_QUERIES
    with torch.no_grad():
        while True:
            frames = []
            for _ in range(WINDOW):
                ok, im = cap.read()
                if not ok:
                    break
                frames.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (w, h)))
            if not frames:
                break
            x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(device)
            x = F.pad((x - mean) / std, (0, pw - w, 0, ph - h))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.backbone(x, resume=nfr > 0)
            pdog = torch.softmax(out["pred_logits"][0].float(), -1)[..., dog]   # (t, q)
            ms = out["pred_masks"][0].float() > 0                                   # (q, t, mh, mw)
            nq = ms.shape[0]; mh, mw = ms.shape[-2:]; sy, sx = ph / mh / k, pw / mw / k
            for t in range(ms.shape[1]):
                m = ms[:, t]; area = m.flatten(1).sum(1); qs = torch.nonzero(area > 0).flatten()
                if not len(qs):
                    continue
                ys, xs = m[qs].any(2).float(), m[qs].any(1).float()
                y1, y2 = ys.argmax(1), mh - 1 - ys.flip(1).argmax(1)
                x1, x2 = xs.argmax(1), mw - 1 - xs.flip(1).argmax(1)
                b = torch.stack([x1 * sx, y1 * sy, (x2 + 1) * sx, (y2 + 1) * sy], 1)
                b[:, [0, 2]] = b[:, [0, 2]].clamp(0, W); b[:, [1, 3]] = b[:, [1, 3]].clamp(0, H)
                b = b.cpu().numpy(); a = area[qs].cpu().numpy() * sx * sy; p = pdog[t, qs].cpu().numpy()
                rows += [(nfr + t, int(qq), *b[j], a[j], p[j]) for j, qq in enumerate(qs.tolist())]
            nfr += len(frames)
    cap.release()
    return np.array(rows, np.float32).reshape(-1, 8), nq, nfr, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--vs", type=float, default=VS); ap.add_argument("--pf", type=float, default=PF)
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows, nq, nfr, fps = infer(Path(a.video), a.device)
    tracks = to_tracks(rows, nq, fps, a.vs, a.pf)
    tracks.to_csv(out / "tracks.csv", index=False)
    meta = {"status": "complete", "engine": "lvmt", "weights": str(CKPT), "source_fps": fps, "sampled_frames": nfr,
            "frame_step": 1, "frame_offset": 0, "track_score_min": a.vs, "frame_dog_prob_min": a.pf,
            "lvmt_seconds": round(time.time() - t0, 1), "tracks_kept": int(tracks.track_id.nunique())}
    (out / "tracking_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"LVMT: {nfr} frames in {meta['lvmt_seconds']} s ({nfr / max(meta['lvmt_seconds'], 1e-6):.1f} fps), "
          f"{meta['tracks_kept']} dog track(s), {len(tracks)} boxes")


if __name__ == "__main__":
    main()
