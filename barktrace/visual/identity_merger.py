#!/usr/bin/env python3
"""Precision-first persistent-identity merger (replaces Stage 1c + Stage 2).

Design goal (user requirement): an assigned dog_id must almost never be
wrong. Recall may drop; runtime does not matter. Every step errs towards
"keep apart" when unsure.

  1. Pieces. Cut every raw track at detection gaps (> --gap raw frames) and at
     abrupt appearance changes (window-mean DINOv3 cosine < --split-threshold).
     Identity swaps between touching/overlapping dogs almost always coincide
     with a detection gap, so pieces are nearly pure.
  2. Embeddings. DINOv3 ViT-B/16 (CLS, letterbox) for EVERY detection.
  3. Constraints (no ground truth):
       must-link   detections inside one piece
       cannot-link two pieces from different raw tracks sharing >= 3 frames
                   in which their boxes are clearly apart (IoU < 0.2): two
                   dogs on screen together, in different places.
  4. Per-video adaptation. If cannot-links exist, train small projection
     heads (several seeds) on those constraints with a hard-mined triplet
     loss, following self-supervised video face clustering (must-link =
     same track, cannot-link = co-occurring tracks).
  5. Clustering. Agglomerative over pieces with
       - hard cannot-link between clusters (any member pair),
       - average-linkage score must clear the threshold AND the weakest
         member pair must clear a floor (no single-link chaining),
       - per-video calibration: a merge must be more similar than every
         known different-dog (cannot-link) pair in this video plus a margin,
         in every embedding space used.
  6. Tiny pieces (< --min-piece dets) never seed merges. They are attached
     only if confident, otherwise kept apart or dropped (--small-policy).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE))
import appearance_backends  # noqa: E402  (sets HF_HOME / offline before transformers)
from appearance_extractor import AppearanceExtractor  # noqa: E402


@dataclass
class Params:
    dinov3_variant: str = "b"      # DINOv3 size for the box embeddings: b (ViT-B/16) | s (ViT-S/16, smaller)
    gap: int = 4                   # raw-frame gap that ends a piece (odd sampling -> consecutive = 2)
    gap_mode: str = "multidog"     # when gaps cut: always | context (another track within
                                   # gap_context frames) | multidog (the video ever shows two raw
                                   # tracks together in different places -> cut every gap)
    gap_context: int = 30
    big_gap: int = 10              # gaps longer than this always cut, in any video (-1 = off)
    split_window: int = 2
    size_jump: float = 1.6         # cut where consecutive box area changes by more than this
    jump_cut: float = -1.0         # > 0: cut a track where its box centre moves more than this x the larger box
                                   # diagonal between consecutive detections <= jump_max_frames apart (a swap)
    jump_max_frames: int = 4
                                   # factor (either way); -1 = off
    partner_events: bool = True    # multi-dog videos: cut a track where another track starts or
                                   # ends next to it (occlusion swap point)
    partner_expand: float = 0.25   # 'next to': boxes overlap after growing each side by this
                                   # fraction of their width/height
    partner_window: int = 6        # partner box within this many raw frames of the event
    split_threshold: float = 0.50  # DINOv3 window-mean cosine below this cuts a piece
    multidog_split_threshold: float = 0.90  # stricter cut in multi-dog videos (swaps possible there)
    min_common: int = 3            # cannot-link: shared frames, all with boxes apart
    cl_rule: str = "all"           # all: every shared frame apart (short pieces) | count: >= min_common
                                   # apart frames suffice (long tracks: one dog is never in two places)
    apart_iou: float = 0.20
    apart_metric: str = "iomin"    # iou | iomin (overlap / smaller box: a head box inside a body box
                                   # is NOT 'apart', which plain IoU would wrongly say it is)
    adapt_seeds: int = 3
    adapt_steps: int = 600
    link_threshold: float = 0.65   # average-linkage similarity needed to merge (original space)
    link_floor: float = 0.55       # weakest cross pair must still clear this (anti-chaining)
    adapted_threshold: float = 0.50  # same, in the adapted space (when adaptation ran)
    cannot_link_margin: float = -0.07  # merge threshold vs known different-dog pairs (+ = stricter; -0.07 = balanced preset)
    calib_quantile: float = 1.0    # calibrate on this quantile of known different-dog pairs (1 = max)
    orig_veto: float = 0.45        # in adapted videos the original space only vetoes below this
    orig_veto_floor: float = -1.0  # decide mode: weakest original-space pair (-1 = average only)
    self_train_rounds: int = 2     # decide mode: retrain the heads with confident clusters as
                                   # extra must-links, then re-cluster (VideoClusterNet-style)
    adapted_mode: str = "auto"     # veto: both spaces must agree (original calibrated) | decide |
                                   # auto: decide only if the video's dogs are NOT look-alikes
    lookalike_threshold: float = 0.70  # auto: known different-dog pairs this similar (original space)
    lookalike_quantile: float = 1.0  # look-alike index = this quantile of known different-dog pair
                                   # similarities (1 = max; lower = robust to one blob-inflated pair)
    min_cl_decide: int = 0         # auto: 'decide' mode also needs at least this many cannot-link pairs
                                   # (an adaptation trained on 1-2 constraints is too weak to trust)
                                       # mark a look-alike video -> stay in veto mode
    decide_threshold: float = 0.85  # decide mode: adapted-space average similarity to merge
    decide_floor: float = 0.70      # decide mode: weakest adapted piece pair
    min_piece: int = 3             # pieces smaller than this never seed merges
    panel_sim: float = -1.0        # split-screen copies: two simultaneous, apart boxes of equal size
                                   # with appearance cosine >= this are ONE dog shown twice (edited
                                   # tiles), never 'different dogs'; they are linked. -1 = off
    part_iomin: float = -1.0       # part-box suppression: in a frame, a box lying >= this fraction inside
                                   # another box (overlap / its own area) and at most part_area_ratio of
                                   # that box's area is a PART of that dog (SAM 3 head/front segments)
                                   # and is dropped; -1 = off
    part_area_ratio: float = 0.70
    dup_track_sim: float = -1.0    # duplicate tracks: two pieces that overlap (>= 0.5 of the smaller box) in
                                   # >= 80% of their shared frames AND look alike per frame (cosine >= this)
                                   # are ONE dog tracked twice: linked as the same dog. -1 = off
    same_id_iomin: float = -1.0    # post-merge: two boxes with the SAME dog id overlapping by >= this
                                   # (overlap / smaller box) in one frame are one dog detected twice:
                                   # keep the more confident; apart same-id boxes (split-screen) stay
    dedup_iou: float = -1.0        # cross-track NMS: in each frame drop the lower-confidence of two boxes
                                   # with IoU >= this (duplicate detections of one dog); -1 = off
    min_profile_dets: int = 0      # drop whole profiles with fewer detections (false-positive filter)
    min_profile_conf: float = 0.0  # drop whole profiles whose mean detector confidence is below this
    weak_rejoin: bool = False      # rejoin consecutive pieces of one raw track split ONLY by a short
                                   # gap (no partner event / size jump / appearance change / big gap)
    weak_min_sim: float = 0.60     # ...if their original-space similarity is at least this
    rejoin_appearance_cuts: bool = False  # also treat appearance-change cuts inside one raw track as
                                   # weak links (rejoined if similarity >= weak_min_sim): for dense
                                   # trackers (SAM 3) whose tracks are usually right
    complete_min_sim: float = -9.0  # > -9: while a video has more profiles than the most dogs on screen at once,
                                   # merge mutual-best compatible profiles at least this similar (decision space)
    complete_margin: float = 0.20   # ...whose similarity beats both sides' next-best compatible profile by this
    complete_orig_min: float = -9.0  # ...and whose original-space similarity is at least this
    complete_shared_ok: bool = False  # margin: ignore a competitor that also clears complete_min_sim with the other
                                   # side (a third piece of the same dog, not an alternative dog)
    complete_min_dets: int = 50     # ...and that both hold at least this many boxes (two tiny profiles with no
                                   # other compatible partner are mutual-best by default)
    cl_min_dets: int = 1            # pieces with fewer boxes create no cannot-links (a few blurred boxes otherwise
                                   # teach the adaptation that two halves of one dog differ); one that ends up in a
                                   # profile it co-occurs with is dropped from it
    rejoin_track_cuts: bool = False  # after clustering, rejoin the clusters on both sides of a cut
                                   # inside one raw track made WITHOUT a swap sign (gap / size only; the
                                   # tracker kept one id there) if the two pieces' similarity >=
                                   # weak_min_sim, raised above every known different-dog pair;
                                   # see rejoin_track_cuts()
    rejoin_max_lookalike: float = 0.80  # ...but not in look-alike videos (a known different-dog pair
                                   # at least this similar), where the tracker itself swaps dogs
    parallel_seeds: bool = False   # learn the adapt_seeds heads in worker processes at the same time:
                                   # bit-identical, ~2x faster. Worker processes re-import the calling
                                   # script, so it needs an `if __name__ == "__main__"` guard (the CLI has one)
    piece_sim: str = "mean"        # piece appearance: mean = one average embedding per piece |
                                   # kmeans_max / kmeans_best = K view prototypes per piece (k-means++, K
                                   # chosen per piece by silhouette, 1..kmeans_max_k, >= kmeans_min_frames
                                   # frames per view); two pieces score their single best-matching view
                                   # pair (max) or the mean of each view's best match (best)
    kmeans_max_k: int = 4
    kmeans_min_frames: int = 8
    small_policy: str = "drop"     # keep | drop : what to do with unattached tiny pieces
    small_margin: float = 0.10     # tiny piece attaches only if best cluster beats 2nd by this
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- embeddings
_EXTRACTORS: dict = {}  # DINOv3 variant -> extractor (one per process)


# Frame number the embedding loops give the video's first frame. Stage-1 tracks are 0-based, so this
# is 0. It was 1 until 2026-09-30 (every box cropped from the PREVIOUS frame, frame 0 never
# embedded); fixing it left the clean benchmark identical.
# Embedding caches record it, and a cache made with another value is not reused.
FIRST_FRAME = 0


def get_extractor(device: str, variant: str = "b") -> AppearanceExtractor:
    """DINOv3 box-embedding extractor: variant "b" (ViT-B/16, live) or "s" (ViT-S/16, smaller)."""
    if variant not in _EXTRACTORS:
        backend = appearance_backends.build_backend("dinov3", device, dinov3_variant=variant,
                                                    dinov3_pooling="cls", dinov3_aspect="letterbox")
        _EXTRACTORS[variant] = AppearanceExtractor("osnet_x0_25", device, backend=backend)
    return _EXTRACTORS[variant]


def embed(video: Path, tracks: pd.DataFrame, p: "Params", device: str, cache: Path | None = None) -> np.ndarray:
    return embed_all(video, tracks, device, cache, p.dinov3_variant)


EMBED_BATCH = int(os.environ.get("DOG_EMBED_BATCH", "1"))  # crops per DINOv3 call across frames (1 = per frame)


def embed_all(video: Path, tracks: pd.DataFrame, device: str, cache: Path | None = None, variant: str = "b") -> np.ndarray:
    if cache is not None and cache.exists():
        z = np.load(cache)
        made_with = int(z["first_frame"]) if "first_frame" in z.files else 1  # older caches: made with 1
        if made_with == FIRST_FRAME and len(z["frame"]) == len(tracks) and (z["frame"] == tracks.frame.to_numpy()).all():
            return z["emb"]
    ex = get_extractor(device, variant)
    emb = np.zeros((len(tracks), ex.dim), dtype=np.float32)
    rows_by_frame = tracks.groupby("frame").indices
    cap = cv2.VideoCapture(str(video)); n = FIRST_FRAME - 1
    pend_crops, pend_rows = [], []

    def flush():  # EMBED_BATCH > 1: encode crops of several frames at once
        if pend_crops:
            emb[pend_rows] = ex.backend.encode(pend_crops)
            pend_crops.clear(); pend_rows.clear()
    while rows_by_frame:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        idx = rows_by_frame.pop(n, None)
        if idx is not None:
            boxes = [tracks.loc[i, ["x1", "y1", "x2", "y2"]].to_numpy(float) for i in idx]
            if EMBED_BATCH <= 1 or ex.backend is None:
                emb[idx] = ex(img, boxes)
            else:
                crops, ci = ex.backend_crops(img, boxes)
                pend_crops += [c.copy() for c in crops]; pend_rows += [int(idx[k]) for k in ci]
                if len(pend_crops) >= EMBED_BATCH:
                    flush()
    flush()
    cap.release()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, frame=tracks.frame.to_numpy(), emb=emb, first_frame=FIRST_FRAME)
    return emb


# ----------------------------------------------------------------------------- pieces
def window_scores(emb: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(emb) + 1, np.nan)
    for i in range(w, len(emb) - w + 1):
        a, b = emb[i - w:i].mean(0), emb[i:i + w].mean(0)
        out[i] = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return out


def appearance_cuts(emb: np.ndarray, w: int, thr: float) -> list[int]:
    if len(emb) < 2 * w:
        return []
    s = window_scores(emb, w)
    i = int(np.nanargmin(s))
    if s[i] >= thr:
        return []
    return appearance_cuts(emb[:i], w, thr) + [i] + [i + j for j in appearance_cuts(emb[i:], w, thr)]


def make_pieces(tracks: pd.DataFrame, emb: np.ndarray, p: Params) -> list[np.ndarray]:
    return make_pieces_with_links(tracks, emb, p)[0]


def make_pieces_with_links(tracks: pd.DataFrame, emb: np.ndarray, p: Params):
    """Returns (pieces, weak_links, track_links): weak_links are (i, i+1) piece-index pairs of one
    raw track whose boundary was cut only because of a short gap in a multi-dog video; track_links
    are the (i, i+1) boundaries inside one raw track cut WITHOUT a sign of an identity swap (no
    partner event, no appearance change): only gap / big gap / size jump."""
    pieces, weak_links, track_links = [], [], []
    all_fr, all_tid = tracks.frame.to_numpy(), tracks.track_id.to_numpy()

    multidog = p.gap_mode == "always" or (p.gap_mode == "multidog" and raw_tracks_apart(tracks, p, emb))

    def other_dog_near(tid, f0, f1):
        if p.gap_mode != "context":
            return multidog
        m = (all_tid != tid) & (all_fr >= f0 - p.gap_context) & (all_fr <= f1 + p.gap_context)
        return bool(m.any())

    box = tracks[["x1", "y1", "x2", "y2"]].to_numpy(float)
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    events = partner_event_frames(tracks, box, p) if (p.partner_events and multidog) else {}

    for tid in sorted(tracks.track_id.unique()):
        idx = tracks.index[tracks.track_id == tid].to_numpy()
        idx = idx[np.argsort(tracks.frame.to_numpy()[idx], kind="stable")]
        fr = tracks.frame.to_numpy()[idx]
        ev = np.array(sorted(events.get(tid, ())))

        def cut_reasons(k):
            r = set()
            if fr[k] - fr[k - 1] > p.gap and other_dog_near(tid, fr[k - 1], fr[k]):
                r.add("gap")
            if p.big_gap >= 0 and fr[k] - fr[k - 1] > p.big_gap:
                r.add("big_gap")
            if p.jump_cut > 0 and fr[k] - fr[k - 1] <= p.jump_max_frames:
                b0, b1 = box[idx[k - 1]], box[idx[k]]
                shift = np.hypot((b1[0] + b1[2] - b0[0] - b0[2]) / 2, (b1[1] + b1[3] - b0[1] - b0[3]) / 2)
                diag = max(np.hypot(b0[2] - b0[0], b0[3] - b0[1]), np.hypot(b1[2] - b1[0], b1[3] - b1[1]))
                if shift > p.jump_cut * diag:
                    r.add("jump")
            if p.size_jump > 0:
                ratio = area[idx[k]] / max(area[idx[k - 1]], 1e-9)
                if ratio > p.size_jump or ratio < 1 / p.size_jump:
                    r.add("size")
            # a partner event at frame e cuts between the detections straddling it
            if len(ev) and ((ev > fr[k - 1]) & (ev <= fr[k])).any():
                r.add("partner")
            return r

        reasons = {k: cut_reasons(k) for k in range(1, len(idx))}
        bounds = [0] + [k for k in range(1, len(idx)) if reasons[k]] + [len(idx)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = idx[a:b]
            thr = p.multidog_split_threshold if multidog else p.split_threshold
            cuts = [0] + appearance_cuts(emb[seg], p.split_window, thr) + [len(seg)]
            first = len(pieces)
            pieces += [seg[c0:c1] for c0, c1 in zip(cuts[:-1], cuts[1:]) if c1 > c0]
            if p.rejoin_appearance_cuts:
                weak_links += [(k, k + 1) for k in range(first, len(pieces) - 1)]
            # the boundary between this segment's first piece and the previous segment's last
            if a > 0 and reasons[a] == {"gap"} and first > 0:
                weak_links.append((first - 1, first))
            if a > 0 and "partner" not in reasons[a] and first > 0:
                track_links.append((first - 1, first))
    return pieces, weak_links, track_links


def _grow(b: np.ndarray, f: float) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] - f * w, b[1] - f * h, b[2] + f * w, b[3] + f * h])


def partner_event_frames(tracks: pd.DataFrame, box: np.ndarray, p: Params) -> dict[int, set[int]]:
    """Events: every time a raw track U's detections stop (its last detection, or one followed by
    a gap > p.gap) or resume (its first detection, or one after such a gap) -- a dog
    disappearing into, or re-emerging from, an occlusion. Any other track T whose box, within
    partner_window frames of the event, touches U's box (both grown by partner_expand) gets a
    cut at that event frame: T may have swapped onto the other dog there."""
    fr, tid = tracks.frame.to_numpy(), tracks.track_id.to_numpy()
    out: dict[int, set[int]] = {}
    for u in np.unique(tid):
        iu = np.where(tid == u)[0]
        iu = iu[np.argsort(fr[iu])]
        d = np.diff(fr[iu])
        ends = [iu[-1]] + [iu[k] for k in np.where(d > p.gap)[0]]
        starts = [iu[0]] + [iu[k + 1] for k in np.where(d > p.gap)[0]]
        for k, side in [(k, "end") for k in ends] + [(k, "start") for k in starts]:
            e = fr[k]
            # start event: cut before the partner's first frame; end event: just after its last
            cut_at = e if side == "start" else e + 1
            bu = _grow(box[k], p.partner_expand)
            near = np.where((tid != u) & (np.abs(fr - e) <= p.partner_window))[0]
            for j in near:
                bt = _grow(box[j], p.partner_expand)
                if box_iou(bu, bt) > 0:
                    out.setdefault(int(tid[j]), set()).add(int(cut_at))
    return out


def is_copy(i: int, j: int, box: np.ndarray, emb: np.ndarray | None, p: Params) -> bool:
    """Split-screen copy: same-size boxes whose crops are near-identical (one dog shown twice)."""
    if p.panel_sim <= 0 or emb is None:
        return False
    ai = (box[i, 2] - box[i, 0]) * (box[i, 3] - box[i, 1]); aj = (box[j, 2] - box[j, 0]) * (box[j, 3] - box[j, 1])
    return float(emb[i] @ emb[j]) >= p.panel_sim and min(ai, aj) / max(ai, aj, 1e-9) >= 0.8


def raw_tracks_apart(tracks: pd.DataFrame, p: Params, emb: np.ndarray | None = None) -> bool:
    """True if two raw tracks share >= min_common frames with boxes apart (two dogs at once).
    Split-screen copies of one dog do not count."""
    box = tracks[["x1", "y1", "x2", "y2"]].to_numpy(float)
    rows = {t: dict(zip(tracks.frame.to_numpy()[idx].tolist(), idx.tolist())) for t, idx in tracks.groupby("track_id").indices.items()}
    ids = list(rows)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            A, B = rows[ids[a]], rows[ids[b]]
            apart = sum(box_overlap(box[A[f]], box[B[f]], p.apart_metric) < p.apart_iou and not is_copy(A[f], B[f], box, emb, p)
                        for f in set(A) & set(B))
            if apart >= p.min_common:
                return True
    return False


def box_overlap(a, b, metric: str = "iou"):
    if metric == "iou":
        return box_iou(a, b)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) + 1e-9)


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def copy_pairs(tracks: pd.DataFrame, pieces: list[np.ndarray], p: Params, emb: np.ndarray | None) -> list[tuple[int, int]]:
    """Piece pairs that are split-screen copies of one dog on most of their shared frames."""
    if p.panel_sim <= 0 or emb is None:
        return []
    fr = tracks.frame.to_numpy(); tid = tracks.track_id.to_numpy(); box = tracks[["x1", "y1", "x2", "y2"]].to_numpy(float)
    by_frame = [dict(zip(fr[q].tolist(), q.tolist())) for q in pieces]
    out = []
    for a in range(len(pieces)):
        for b in range(a + 1, len(pieces)):
            if tid[pieces[a][0]] == tid[pieces[b][0]]:
                continue
            common = set(by_frame[a]) & set(by_frame[b])
            if len(common) >= p.min_common and np.mean([is_copy(by_frame[a][f], by_frame[b][f], box, emb, p) for f in common]) >= 0.8:
                out.append((a, b))
    return out


def duplicate_pairs(tracks: pd.DataFrame, pieces: list[np.ndarray], p: Params, emb: np.ndarray) -> list[tuple[int, int]]:
    """Piece pairs that are the same dog tracked twice (SAM 3 sometimes keeps two instances on one
    dog, e.g. one box including a hand): consistently overlapping and alike frame by frame."""
    if p.dup_track_sim <= 0:
        return []
    fr = tracks.frame.to_numpy(); tid = tracks.track_id.to_numpy(); box = tracks[["x1", "y1", "x2", "y2"]].to_numpy(float)
    by_frame = [dict(zip(fr[q].tolist(), q.tolist())) for q in pieces]
    out = []
    for a in range(len(pieces)):
        for b in range(a + 1, len(pieces)):
            if tid[pieces[a][0]] == tid[pieces[b][0]]:
                continue
            common = sorted(set(by_frame[a]) & set(by_frame[b]))
            if len(common) < p.min_common:
                continue
            ov = [box_overlap(box[by_frame[a][f]], box[by_frame[b][f]], "iomin") >= 0.5 for f in common]
            sim = np.median([float(emb[by_frame[a][f]] @ emb[by_frame[b][f]]) for f in common])
            if np.mean(ov) >= 0.8 and sim >= p.dup_track_sim:
                out.append((a, b))
    return out


def cannot_links(tracks: pd.DataFrame, pieces: list[np.ndarray], p: Params, emb: np.ndarray | None = None) -> np.ndarray:
    fr = tracks.frame.to_numpy(); tid = tracks.track_id.to_numpy()
    box = tracks[["x1", "y1", "x2", "y2"]].to_numpy(float)
    by_frame = [dict(zip(fr[q].tolist(), q.tolist())) for q in pieces]
    cl = np.zeros((len(pieces), len(pieces)), bool)
    for a in range(len(pieces)):
        for b in range(a + 1, len(pieces)):
            if tid[pieces[a][0]] == tid[pieces[b][0]]:
                continue
            common = set(by_frame[a]) & set(by_frame[b])
            if len(common) < p.min_common:
                continue
            apart = [box_overlap(box[by_frame[a][f]], box[by_frame[b][f]], p.apart_metric) < p.apart_iou
                     and not is_copy(by_frame[a][f], by_frame[b][f], box, emb, p) for f in common]
            if all(apart) if p.cl_rule == "all" else sum(apart) >= p.min_common:
                cl[a, b] = cl[b, a] = True
    return cl


# ----------------------------------------------------------------------------- adaptation
def adapt(x: np.ndarray, piece_of: np.ndarray, cl: np.ndarray, seed: int, steps: int, device: str) -> np.ndarray:
    """piece_of may be any grouping (piece id, or cluster id in self-training rounds); cl is the
    cannot-link matrix at that same grouping level."""
    torch.manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    po = torch.tensor(piece_of, device=device); clt = torch.tensor(cl, device=device)
    head = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 512), torch.nn.ReLU(), torch.nn.Linear(512, 128)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    # only detections in pieces that take part in some cannot-link give a full triplet
    active = torch.where(clt.any(1)[po.clamp(min=0)] & (po >= 0))[0]
    eye_cache = {}
    for _ in range(steps):
        idx = active[torch.randperm(len(active), device=device)[:256]]
        z = F.normalize(head(xt[idx]), dim=1); s = z @ z.T; q = po[idx]
        n = len(idx)
        if n not in eye_cache:
            eye_cache[n] = torch.eye(n, dtype=torch.bool, device=device)
        pos = (q[:, None] == q[None, :]) & ~eye_cache[n]
        neg = clt[q][:, q]
        has = pos.any(1) & neg.any(1)
        if not has.any():
            continue
        hp = torch.where(pos, s, torch.full_like(s, 2.0)).min(1).values
        hn = torch.where(neg, s, torch.full_like(s, -2.0)).max(1).values
        loss = F.relu(hn - hp + 0.3)[has].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return F.normalize(head(xt), dim=1).cpu().numpy()


# ----------------------------------------------------------------------------- clustering
def view_prototypes(x: np.ndarray, max_k: int, min_frames: int) -> np.ndarray:
    """K normalized view centroids of one piece's frame embeddings: k-means++ (deterministic), K in
    1..max_k with every view >= min_frames frames, chosen by the best silhouette (K=1 unless a split
    scores > 0.1, i.e. the piece really shows distinct views)."""
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.metrics import silhouette_score  # noqa: PLC0415
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    best, best_s = xn.mean(0, keepdims=True), 0.1
    for k in range(2, min(max_k, len(x) // min_frames) + 1):
        km = KMeans(k, init="k-means++", n_init=4, random_state=0).fit(xn)
        if np.bincount(km.labels_, minlength=k).min() < min_frames:
            continue
        sc = silhouette_score(xn, km.labels_, metric="cosine")
        if sc > best_s:
            best, best_s = km.cluster_centers_, sc
    return best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-9)


def piece_similarity(emb: np.ndarray, pieces: list[np.ndarray], p: "Params | None" = None) -> np.ndarray:
    if p is None or p.piece_sim == "mean":
        c = np.stack([emb[q].mean(0) for q in pieces])
        c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-9
        return c @ c.T
    protos = [view_prototypes(emb[q], p.kmeans_max_k, p.kmeans_min_frames) for q in pieces]
    n = len(pieces); S = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M = protos[i] @ protos[j].T
            S[i, j] = S[j, i] = M.max() if p.piece_sim == "kmeans_max" else 0.5 * (M.max(1).mean() + M.max(0).mean())
    return S


def cluster(spaces: list[tuple[np.ndarray, float, bool]], cl: np.ndarray, sizes: np.ndarray, p: Params,
            init: list[list[int]] | None = None):
    """spaces: [(piece-similarity matrix, threshold, calibrate?, floor)]; a merge must pass in
    every space. Calibrated spaces raise their threshold above every known different-dog pair."""
    n = len(sizes)
    big = [i for i in range(n) if sizes[i] >= p.min_piece]
    covered: set[int] = set()
    clusters = []
    for ch in init or []:  # pre-linked chains (weak rejoin) start as clusters
        if sizes[ch].sum() >= p.min_piece:
            clusters.append(list(ch)); covered |= set(ch)
    clusters += [[i] for i in big if i not in covered]
    # per-video calibration: beat every known different-dog pair in each space
    thr = []
    for S, t, calibrate, _ in spaces:
        known = S[cl & np.outer(sizes >= p.min_piece, sizes >= p.min_piece)]
        thr.append(max(t, float(np.quantile(known, p.calib_quantile)) + p.cannot_link_margin) if calibrate and known.size else t)
    while True:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                A, B = clusters[a], clusters[b]
                if cl[np.ix_(A, B)].any():
                    continue
                ok, score = True, 0.0
                for (S, _, _, floor), t in zip(spaces, thr):
                    sub = S[np.ix_(A, B)]
                    w = np.outer(sizes[A], sizes[B])
                    avg = float((sub * w).sum() / w.sum())
                    if avg < t or sub.min() < min(floor, t):
                        ok = False
                        break
                    score += avg
                if ok and (best is None or score > best[0]):
                    best = (score, a, b)
        if best is None:
            break
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    # tiny pieces: attach only if clearly best and above threshold in every space, never across a cannot-link
    small = [i for i in range(n) if sizes[i] < p.min_piece and i not in covered]
    unattached = []
    for i in small:
        scores = []
        for k, C in enumerate(clusters):
            if cl[i, C].any():
                continue
            per = [float(S[i, C].mean()) for S, _, _, _ in spaces]
            if all(s >= t for s, t in zip(per, thr)):
                scores.append((min(per), k))
        scores.sort(reverse=True)
        if scores and (len(scores) == 1 or scores[0][0] - scores[1][0] >= p.small_margin):
            clusters[scores[0][1]].append(i)
        else:
            unattached.append(i)
    if p.small_policy == "keep":
        clusters += [[i] for i in unattached]
    return clusters, thr


def dedup(tracks: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Cross-track NMS per frame: two boxes with IoU >= thr are one dog detected twice (test-time
    augmentation and BoT-SORT both produce these); keep the more confident box."""
    drop = []
    box = tracks[["x1", "y1", "x2", "y2"]].to_numpy(float); conf = tracks.confidence.to_numpy()
    for _, idx in tracks.groupby("frame").indices.items():
        if len(idx) < 2:
            continue
        order = idx[np.argsort(-conf[idx])]
        kept = []
        for i in order:
            if any(box_iou(box[i], box[j]) >= thr for j in kept):
                drop.append(i)
            else:
                kept.append(i)
    return tracks.drop(index=drop).reset_index(drop=True)


def drop_part_boxes(out: pd.DataFrame, profile_of: dict, iomin: float, area_ratio: float,
                    same_id_iomin: float = -1.0) -> set:
    """Post-merge, per frame: a box lying >= iomin inside a clearly larger box (<= area_ratio of its
    area) is dropped when it has the SAME dog id as the larger box (a head / front part segmented as
    its own instance: one dog drawn twice), or when it is LESS confident than the larger box (a stray
    fragment on another dog). A confident box inside a less confident, different-dog box is kept
    and the LARGER box is dropped instead: it is an oversized blob spanning two dogs (e.g. collage
    panels), not a whole dog."""
    rows = out[out.track_id.isin(list(profile_of))]
    box = rows[["x1", "y1", "x2", "y2"]].to_numpy(float)
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    conf = rows.confidence.to_numpy(float)
    prof = np.array([profile_of[t] for t in rows.track_id])
    index = rows.index.to_numpy()
    drop = set()
    for _, pos in rows.groupby("frame").indices.items():
        if len(pos) < 2:
            continue
        for i in pos:
            for j in pos:
                if i == j:
                    continue
                x1, y1 = max(box[i, 0], box[j, 0]), max(box[i, 1], box[j, 1])
                x2, y2 = min(box[i, 2], box[j, 2]), min(box[i, 3], box[j, 3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if same_id_iomin > 0 and prof[i] == prof[j] and inter >= same_id_iomin * min(area[i], area[j]) \
                        and (conf[i], -i) < (conf[j], -j):
                    drop.add(index[i])  # the same dog boxed twice: keep the more confident box
                    break
                inside = iomin > 0 and area[i] <= area_ratio * area[j] and inter >= iomin * area[i]
                if not inside:
                    continue
                if prof[i] == prof[j] or conf[i] < conf[j]:
                    drop.add(index[i])  # part of the same dog, or a stray fragment
                    break
                drop.add(index[j])  # j swallows a different, more confident dog: an oversized blob
    return drop


# ----------------------------------------------------------------------------- main API
def rejoin_track_cuts(clusters, links, S_orig, cl, sizes, p: Params):
    """After clustering: the two pieces of a track_link (one raw track, cut with no sign of a swap,
    so the tracker kept one id across it) join their clusters -- or a dropped tiny piece joins its
    neighbour's cluster -- when they look alike above every known different-dog pair and no
    cannot-link separates the two groups. Off in look-alike videos. Only ever merges."""
    big = sizes >= p.min_piece
    known = S_orig[cl & np.outer(big, big)]
    thr = p.weak_min_sim
    if known.size:
        if known.max() >= p.rejoin_max_lookalike:
            return clusters
        thr = max(thr, float(np.quantile(known, p.calib_quantile)) + p.cannot_link_margin)
    groups = [list(C) for C in clusters]
    for a, b in links:
        if S_orig[a, b] < thr or cl[a, b]:
            continue
        ga = next((g for g in groups if a in g), None)
        gb = next((g for g in groups if b in g), None)
        if ga is gb or (ga is None and gb is None):
            continue
        if ga is None or gb is None:  # attach the unclustered piece
            g, i = (gb, a) if ga is None else (ga, b)
            if not cl[i, g].any():
                g.append(i)
        elif not cl[np.ix_(ga, gb)].any():
            ga += gb
            groups.remove(gb)
    return groups


_POOL = None


def complete_to_cooccurrence(clusters, S, S_orig, cl, sizes, frames, p: Params):
    """More profiles than the most dogs ever on screen together means some dog is split. Merge the pair
    that is mutually most similar (no cannot-link, >= complete_min_sim, beating both sides' next-best
    compatible profile by complete_margin, both >= complete_min_dets boxes), one pair at a time, until the counts match or no pair
    qualifies. Only ever merges."""
    clusters = [list(C) for C in clusters]
    while len(clusters) > 1:
        present: dict[int, int] = {}
        for C in clusters:
            for f in set(np.concatenate([frames[i] for i in C]).tolist()):
                present[f] = present.get(f, 0) + 1
        if len(clusters) <= max(present.values(), default=0):
            break
        n = len(clusters); M = np.full((n, n), -np.inf); O = np.full((n, n), -np.inf)
        big = [sizes[C].sum() >= p.complete_min_dets for C in clusters]
        for a in range(n):
            for b in range(a + 1, n):
                A, B = clusters[a], clusters[b]
                if big[a] and big[b] and not cl[np.ix_(A, B)].any():
                    w = np.outer(sizes[A], sizes[B])
                    M[a, b] = M[b, a] = float((S[np.ix_(A, B)] * w).sum() / w.sum())
                    O[a, b] = O[b, a] = float((S_orig[np.ix_(A, B)] * w).sum() / w.sum())
        best = None
        for a in range(n):
            b = int(np.argmax(M[a]))
            if M[a, b] < p.complete_min_sim or O[a, b] < p.complete_orig_min or int(np.argmax(M[b])) != a or b < a:
                continue
            rival = np.ones(n, bool); rival[[a, b]] = False
            if p.complete_shared_ok:  # c similar to both a and b is the same dog's third piece, not a rival
                rival &= ~((M[a] >= p.complete_min_sim) & (M[b] >= p.complete_min_sim))
            nxt = max(M[a][rival].max(initial=-np.inf), M[b][rival].max(initial=-np.inf))
            if M[a, b] - nxt >= p.complete_margin and (best is None or M[a, b] > best[0]):
                best = (M[a, b], a, b)
        if best is None:
            break
        _, a, b = best
        clusters[a] += clusters[b]
        del clusters[b]
    return clusters


def _warm(device: str) -> None:
    torch.zeros(1, device=device)  # import torch + create the CUDA context ahead of the first real task


def _seed_pool(n: int, device: str):
    """Worker processes for adapt(), started once and warmed in the background."""
    global _POOL
    if _POOL is None:
        import multiprocessing as mp  # noqa: PLC0415
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
        _POOL = ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"))
        for _ in range(n):
            _POOL.submit(_warm, device)
    return _POOL


def adapt_heads(x, groups, cl, p: Params, device: str) -> list:
    """adapt() for seeds 0..adapt_seeds-1; in parallel worker processes if p.parallel_seeds (same
    results: each seed is deterministic on its own), falling back to one after another on any error."""
    if p.parallel_seeds:
        try:
            n = p.adapt_seeds
            return list(_seed_pool(n, device).map(adapt, [x] * n, [groups] * n, [cl] * n, range(n),
                                                  [p.adapt_steps] * n, [device] * n))
        except Exception as e:  # e.g. out of GPU memory in a worker
            print(f"parallel adapt failed ({e}); running the seeds one after another", file=sys.stderr)
    return [adapt(x, groups, cl, s, p.adapt_steps, device) for s in range(p.adapt_seeds)]


def run(video: Path, tracks: pd.DataFrame, p: Params, device: str = "cuda:0", cache: Path | None = None):
    """Returns (pieces_tracks_df, mapping_df, info). pieces_tracks_df has track_id = piece id."""
    tracks = tracks.sort_values(["frame", "track_id"]).reset_index(drop=True)
    if p.parallel_seeds:
        _seed_pool(p.adapt_seeds, device)  # worker start-up overlaps with the embedding below
    if p.dedup_iou > 0 and len(tracks):
        tracks = dedup(tracks, p.dedup_iou)
    empty_map = pd.DataFrame(columns=["track_id", "persistent_profile_id"])
    if tracks.empty:
        return tracks, empty_map, {}
    emb = embed(video, tracks, p, device, cache)
    pieces, weak_links, track_links = make_pieces_with_links(tracks, emb, p)
    piece_of = np.full(len(tracks), -1)
    for k, q in enumerate(pieces):
        piece_of[q] = k
    cl = cannot_links(tracks, pieces, p, emb)
    copies = copy_pairs(tracks, pieces, p, emb) + duplicate_pairs(tracks, pieces, p, emb)
    for a, b in copies:  # a copy / duplicate of one dog is never a 'different dog' constraint
        cl[a, b] = cl[b, a] = False
    sizes = np.array([len(q) for q in pieces])
    cl_full = cl.copy()
    if p.cl_min_dets > 1:
        solid = sizes >= p.cl_min_dets
        cl = cl & np.outer(solid, solid)
    S_orig = piece_similarity(emb, pieces, p)
    chains = None
    if (p.weak_rejoin and weak_links) or copies:
        parent = list(range(len(pieces)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for a, b in (weak_links if p.weak_rejoin else []):
            if S_orig[a, b] >= p.weak_min_sim and not cl[a, b]:
                parent[find(b)] = find(a)
        for a, b in copies:  # split-screen copies of one dog are the same dog
            parent[find(b)] = find(a)
        groups: dict[int, list[int]] = {}
        for i in range(len(pieces)):
            groups.setdefault(find(i), []).append(i)
        chains = [g for g in groups.values() if len(g) > 1]
    spaces = [(S_orig, p.link_threshold, True, p.link_floor)]
    adapted = False
    mode, lookalike = None, None
    if cl.any():
        heads = adapt_heads(emb, piece_of, cl, p, device)
        # ensemble: a pair is only as similar as the least-convinced head
        S_ad = np.min(np.stack([piece_similarity(h, pieces, p) for h in heads]), axis=0)
        # Adapted video: the adapted space decides (calibrated); the original space,
        # where look-alikes are by definition inseparable, only vetoes clear mismatches.
        big = sizes >= p.min_piece
        known = S_orig[cl & np.outer(big, big)]
        lookalike = float(np.quantile(known, p.lookalike_quantile)) if known.size else 1.0
        mode = p.adapted_mode
        if mode == "auto":
            mode = "decide" if (lookalike < p.lookalike_threshold and cl.sum() // 2 >= p.min_cl_decide) else "veto"
        if mode == "decide":
            # distinct-looking dogs: the adapted space decides with strict gates;
            # the original space only vetoes clear mismatches.
            spaces = [(S_ad, p.decide_threshold, True, p.decide_floor), (S_orig, p.orig_veto, False, p.orig_veto_floor)]
        else:
            spaces = [(S_orig, p.link_threshold, True, p.link_floor), (S_ad, p.adapted_threshold, True, p.adapted_threshold)]
        adapted = True
    clusters, thr = cluster(spaces, cl, sizes, p, chains)
    for _ in range(p.self_train_rounds if (adapted and mode == "decide") else 0):
        # groups = current clusters of real (>= min_piece) pieces; every other piece its own group
        group_of_piece = np.arange(len(pieces)) + len(pieces)
        for g, C in enumerate(clusters):
            if len(C) > 1:
                group_of_piece[C] = g
        uniq, gid = np.unique(group_of_piece, return_inverse=True)
        cl_g = np.zeros((len(uniq), len(uniq)), bool)
        ai, bi = np.where(cl)
        cl_g[gid[ai], gid[bi]] = True
        np.fill_diagonal(cl_g, False)
        group_of = gid[piece_of]
        heads = adapt_heads(emb, group_of, cl_g, p, device)
        S_ad = np.min(np.stack([piece_similarity(h, pieces, p) for h in heads]), axis=0)
        spaces = [(S_ad, p.decide_threshold, True, p.decide_floor), (S_orig, p.orig_veto, False, p.orig_veto_floor)]
        clusters, thr = cluster(spaces, cl, sizes, p, chains)
    if p.rejoin_track_cuts:
        clusters = rejoin_track_cuts(clusters, track_links, S_orig, cl, sizes, p)
    if p.complete_min_sim > -9:
        fr = tracks.frame.to_numpy()
        clusters = complete_to_cooccurrence(clusters, spaces[0][0], S_orig, cl, sizes, [fr[q] for q in pieces], p)
    if p.cl_min_dets > 1:  # a tiny piece on screen together with its own profile is another dog's: drop it
        clusters = [[i for i in C if sizes[i] >= p.cl_min_dets or not cl_full[i, C].any()] for C in clusters]
        clusters = [C for C in clusters if C]
    if p.min_profile_dets > 0 or p.min_profile_conf > 0:
        conf = tracks.confidence.to_numpy()
        keep = []
        for C in clusters:
            rows = np.concatenate([pieces[i] for i in C])
            if len(rows) >= p.min_profile_dets and conf[rows].mean() >= p.min_profile_conf:
                keep.append(C)
        clusters = keep
    out = tracks.copy()
    out["raw_track_id"] = out.track_id
    out["track_id"] = piece_of
    rows = []
    order = sorted(clusters, key=lambda C: min(tracks.frame.to_numpy()[pieces[i]].min() for i in C))
    for k, C in enumerate(order, start=1):
        for i in C:
            f = tracks.frame.to_numpy()[pieces[i]]
            rows.append({"track_id": i, "persistent_profile_id": f"dog_{k}", "first_frame": int(f.min()),
                         "last_frame": int(f.max()), "track_rows": int(len(f))})
    mapping = pd.DataFrame(rows, columns=["track_id", "persistent_profile_id", "first_frame", "last_frame", "track_rows"])
    part_dropped = 0
    orig_kept = set(mapping.track_id) if len(mapping) else set()  # before any part-box splitting
    if (p.part_iomin > 0 or p.same_id_iomin > 0) and len(mapping):
        profile_of = dict(zip(mapping.track_id, mapping.persistent_profile_id))
        dropped = drop_part_boxes(out, profile_of, p.part_iomin, p.part_area_ratio, p.same_id_iomin)
        part_dropped = len(dropped)
        if dropped:
            # Split each piece where boxes were removed (both halves keep the dog id), so the renderer,
            # which interpolates short gaps inside a piece, cannot redraw a removed box.
            out = out.copy()
            is_dropped = out.index.isin(sorted(dropped))
            next_id = int(out.track_id.max()) + 1
            for tid in [t for t in out.track_id.unique() if t in profile_of]:
                idx = out.index[out.track_id == tid]
                idx = idx[np.argsort(out.loc[idx, "frame"].to_numpy(), kind="stable")]
                seg = np.cumsum(is_dropped[out.index.get_indexer(idx)])
                keep = ~is_dropped[out.index.get_indexer(idx)]
                for k, sid in enumerate(np.unique(seg[keep])):
                    if k == 0:
                        continue  # the first segment keeps the original piece id
                    out.loc[idx[keep & (seg == sid)], "track_id"] = next_id
                    profile_of[next_id] = profile_of[tid]
                    next_id += 1
            out = out[~is_dropped]
            g = out[out.track_id.isin(list(profile_of))].groupby("track_id").frame
            mapping = pd.DataFrame({"track_id": list(g.groups), "persistent_profile_id": [profile_of[t] for t in g.groups],
                                    "first_frame": g.min().to_numpy(), "last_frame": g.max().to_numpy(),
                                    "track_rows": g.size().to_numpy()}).sort_values(["persistent_profile_id", "first_frame"])
    kept = set(mapping.track_id) if len(mapping) else set()
    info = {"pieces": len(pieces), "cannot_link_pairs": int(cl.sum() // 2), "adapted": adapted,
            "mode": mode if adapted else "original-only", "lookalike_index": round(lookalike, 4) if adapted else None,
            "thresholds": [round(float(t), 4) for t in thr], "profiles": len(order),
            "dropped_pieces": int(len(pieces) - len(orig_kept)), "dropped_detections": int((~out.track_id.isin(kept)).sum()),
            "part_boxes_dropped": int(part_dropped)}
    return out, mapping, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True, help="Stage 1a tracks.csv (raw tracker output)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    for k, v in Params().__dict__.items():
        if k != "extra":
            # bool("false") is True, so booleans need an explicit parser
            typ = (lambda x: str(x).lower() in ("1", "true", "yes")) if isinstance(v, bool) else type(v)
            ap.add_argument(f"--{k.replace('_', '-')}", type=typ, default=v)
    a = ap.parse_args()
    p = Params(**{k: getattr(a, k) for k in Params().__dict__ if k != "extra"})
    tracks = pd.read_csv(a.tracks) if a.tracks.stat().st_size else pd.DataFrame()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    pieces_df, mapping, info = run(a.video, tracks, p, a.device)
    pieces_df.to_csv(a.output_dir / "tracks_pieces.csv", index=False)
    mapping.to_csv(a.output_dir / "track_to_all_dog_profile.csv", index=False)
    (a.output_dir / "precision_merger_info.json").write_text(json.dumps({"params": p.__dict__, **info}, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
