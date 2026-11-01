"""Which dog made each detected sound (Stage 1b).

Candidates = the dogs with a box during the sound (+-0.05 s). Then:
  no dog on screen   -> "outside"
  one dog            -> that dog if its sync >= SINGLE_MIN, else "outside" (a silent dog is on screen
                        while an off-screen dog makes the sound)
  several dogs       -> score = sync + MOTION_W x (motion share - 1/n); the best dog if its score >= MULTI_MIN,
                        else "outside"
sync   = Synchformer (Iashin et al., ICASSP 2024; AudioSet offset model 24-01-04T16-39-21) on a 5 s window centred on
         the sound and a 256x256 square crop following the dog's box (1.4x the box), with dog fine-tuned heads
         (FT_DIR, 2026-10-26: top transformer block + a "this crop made the sound" head, trained on mined single-dog
         barks and synthetic two-dog scenes, 3 seeds averaged): P(this crop made the
         sound). Code: third_party/synchformer (patched for transformers 5); weights: models/synchformer.
motion = mean absolute grey-level change inside the dog's box (64x64) over the sound +-0.3 s; share = this dog's
         motion / all candidates' motion.
On the 297 labelled sounds (72 videos), cut-offs chosen leave-one-video-out, each video scored by heads that never saw
it: 75.1% right, wrong dog 2.7% (checkpoint offset score: 70.7% / 5.7%). FINAL v2 (FT_DIR = None, SINGLE_MIN 0.2):
75.8% / 9.8% with its deployed cut-offs. Runtime about the same (+3% for the barker stage).
Several barkers (EXTRA_MIN): among several dogs, every other dog scoring >= EXTRA_MIN is also named, one extra
predictions row per dog (same event_id, status "also_synced"). Off by default: on the 110 labelled sounds (no two-barker
labels yet) a cut-off of 0.9 added 6 wrong dogs and 0 right ones (2026-10-25), so it waits for the "together" labels.
Per bark (--split, on for detected sounds since 2026-10-26): a detected sound often holds several barks (median 1.2 s,
up to 12.7 s, vs 0.57 s per labelled bark), so it is first cut into single barks (split_barks) and each bark gets its
own decision and predictions row (same event_id). End to end on the 14 fully labelled videos:
61.3% right / 2.8% wrong dog -> 65.1% / 0.0%; the 12 extra named pieces were all checked as real barks.
Known limit: the SINGLE_MIN threshold rests on few real off-screen examples.

Burst context (2026-10-26): among several dogs, a bark called "outside" goes to the best-scoring dog if it scores
>= CONTEXT_MIN and a bark within CONTEXT_GAP s was given to that dog (status "context"; the one-dog rule is unchanged).
Held out on the 297 labelled sounds 76.1% -> 77.1% right with no new wrong dog (all 72 folds chose these values); end
to end with splitting 65.1% -> 67.9%, wrong dog 0%.
Burst chain (2026-10-28, for 723 chihuahua 7.4-10.9 s, where dog_2 leads 0.52-0.76 but was never named): a bark still
"outside" among several dogs that touches a named bark (gap 0) goes to its best dog if that scores >= CHAIN_MIN and
leads the second by >= CHAIN_LEAD, repeated so a whole burst follows (status "chain"). No change on the 297 set or the
no-bark set; end to end it changes only 723, wrong dog 0%. Rests on one clip.

predictions.csv columns (read by metadata_writer.py and video_renderer.py): event_id, start_time_sec, end_time_sec,
status (single | synced | context | chain | also_synced | outside), predicted_track_id (the chosen dog's track with most boxes during the sound),
barker_dog_id, candidates (the dogs on screen), reason, scores (per candidate).
Usage: barker_attribution.py --video V --events timestamps.csv --stage2-dir S2 --output predictions.csv [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

HERE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("BARKTRACE_MODELS", HERE.parents[1] / "models"))
SF_DIR = HERE.parents[1] / "third_party" / "synchformer"  # Synchformer (MIT), patched for transformers 5
CKPT = MODELS / "synchformer" / "24-01-04T16-39-21"
FT_DIR = MODELS / "synchformer" / "dog_ft_20261026"  # dog fine-tuned heads (seed*.pt); None = the checkpoint's offset score

COLUMNS = ["event_id", "start_time_sec", "end_time_sec", "status", "predicted_track_id", "barker_dog_id", "candidates",
           "reason", "scores"]
PAD_SEC = 0.05     # boxes this close to the sound count (Fast mode has a box every second frame)
SINGLE_MIN = 0.4   # one dog on screen: below this sync it is not the barker (0.2 with FT_DIR = None)
MULTI_MIN = 0.8    # several dogs: the best score must reach this (0.9 until 2026-10-28; user asked for a lower bar: end to end
                   # 67.9% -> 69.8% right, wrong dog 0%; 297 set wrong dog 1.7% -> 3.4%)
EXTRA_MIN = None   # several dogs: others scoring >= this are also barkers; None = one dog per sound (see docstring)
MOTION_W = 0.5
CONTEXT_GAP, CONTEXT_MIN = 3.0, 0.5  # several dogs, "outside": the best dog if it scores >= 0.5 and a bark <= 3 s away was given to it
CHAIN_MIN, CHAIN_LEAD = 0.5, 0.2    # several dogs, "outside", touching a named bark: the best dog if >= 0.5 and 0.2 ahead
VFPS, AFPS, WIN, SIZE, SCALE, CTX = 25, 16000, 5.0, 256, 1.4, 0.3


SPLIT_DROP, SPLIT_MIN = 0.3, 0.15  # bark split: a loudness dip below 30% of both neighbouring peaks, pieces >= 0.15 s
_BAND = butter(4, [300, 4000], btype="band", fs=AFPS, output="sos")


def loudness(audio: np.ndarray) -> np.ndarray:
    """300-4000 Hz RMS envelope of 16 kHz audio, 100 values per second (10 ms windows, 30 ms smoothing)."""
    x = sosfiltfilt(_BAND, audio)
    e = np.sqrt(np.convolve(x ** 2, np.ones(AFPS // 100) / (AFPS // 100), "same")[::AFPS // 100])
    return np.convolve(e, np.ones(3) / 3, "same")


def split_barks(env: np.ndarray, start: float, end: float) -> list[tuple[float, float]]:
    """One detected sound -> its single barks: cut at the deepest point between two loudness peaks when it falls
    below SPLIT_DROP x the smaller peak; the pieces touch. Separates dogs that take turns inside one detected sound
    (the detector's own split needs its BEATs score to dip, which fast back-and-forth barking does not do)."""
    a, b = int(start * 100), int(end * 100); v = env[a:b]
    bounds = [start]
    if len(v) >= 2 * SPLIT_MIN * 100:
        pk, _ = find_peaks(v, distance=8, prominence=v.max() * 0.1)
        for i, j in zip(pk[:-1], pk[1:]):
            k = i + int(np.argmin(v[i:j])); c = round((a + k) / 100, 3)
            if v[k] < SPLIT_DROP * min(v[i], v[j]) and c - bounds[-1] >= SPLIT_MIN and end - c >= SPLIT_MIN:
                bounds.append(c)
    return list(zip(bounds, bounds[1:] + [end]))


def decide(cands: list[str], sync: dict[str, float], motion: dict[str, float],
           extra_min: float | None = None) -> tuple[str | None, str, str, dict, list[str]]:
    """-> (dog or None, status, reason, scores, other barkers). Pure decision rule (no models), see the module docstring."""
    if not cands:
        return None, "outside", "no dog on screen", {}, []
    if len(cands) == 1:
        d = cands[0]; s = sync.get(d, 0.0)
        if s >= SINGLE_MIN:
            return d, "single", f"only dog on screen, sync {s:.2f}", {d: round(s, 3)}, []
        return None, "outside", f"only dog on screen is not in sync ({s:.2f} < {SINGLE_MIN})", {d: round(s, 3)}, []
    tot = sum(motion.get(d, 0.0) for d in cands) or 1.0
    sc = {d: sync.get(d, 0.0) + MOTION_W * (motion.get(d, 0.0) / tot - 1.0 / len(cands)) for d in cands}
    best = max(sc, key=sc.get)
    sc = {d: round(v, 3) for d, v in sc.items()}
    if sc[best] >= MULTI_MIN:
        extra = [] if extra_min is None else sorted((d for d in cands if d != best and sc[d] >= extra_min), key=lambda d: -sc[d])
        return best, "synced", f"best of {len(cands)} dogs, score {sc[best]:.2f}", sc, extra
    return None, "outside", f"no dog of {len(cands)} in sync (best {sc[best]:.2f} < {MULTI_MIN})", sc, []


def burst_context(barks: list[tuple[float, float, list[str], str | None, dict]]) -> list[tuple[str, str] | None]:
    """barks: (start, end, candidates, decided dog or None, scores) in one video -> for each "outside" bark among several
    dogs, (dog, "context" | "chain") from its neighbours, else None. context: the best-scoring dog, if it reaches
    CONTEXT_MIN and a bark within CONTEXT_GAP s was given to that dog by decide() (one pass). chain: then, repeatedly, the
    best dog of a bark touching a named bark, if it reaches CHAIN_MIN and leads the second by CHAIN_LEAD."""
    out = []
    for a, b, cands, dog, sc in barks:
        best = max(sc, key=sc.get) if sc else None
        near = {d for a2, b2, _, d, _ in barks if d and a2 - b <= CONTEXT_GAP and a - b2 <= CONTEXT_GAP}
        out.append((best, "context") if dog is None and len(cands) > 1 and sc[best] >= CONTEXT_MIN and best in near else None)
    changed = True
    while changed:
        changed = False
        for i, (a, b, cands, dog, sc) in enumerate(barks):
            if dog or out[i] or len(cands) < 2 or not sc:
                continue
            y, y2 = sorted(sc, key=sc.get, reverse=True)[:2]
            touch = any(j != i and (barks[j][3] or out[j]) and a2 - b <= 0 and a - b2 <= 0
                        for j, (a2, b2, _, _, _) in enumerate(barks))
            if touch and sc[y] >= CHAIN_MIN and sc[y] - sc[y2] >= CHAIN_LEAD:
                out[i], changed = (y, "chain"), True
    return out


def source_head(model):
    """The trainable top of Synchformer on the frozen extractor features (projections, 3-layer transformer, offset head)
    plus a 2-way "this crop made the sound" head (V2)."""
    import torch

    class SourceHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vproj, self.aproj, self.tr = model.vproj, model.aproj, model.transformer
            self.sync_head = torch.nn.Linear(self.tr.config.n_embd, 2)

        def forward(self, v, a):  # v (B, S, tv, D), a (B, S, ta, D) -> offset logits, source logits
            v, a = self.vproj(v.float()), self.aproj(a.float())
            x = self.tr(v.flatten(1, 2), a.flatten(1, 2), attempt_to_apply_heads=False)[:, 0]
            return self.tr.off_head(x), self.sync_head(x)
    return SourceHead()


class Sync:
    """score(frames RGB 256x256 list at 25 fps, audio 16 kHz 5 s) -> with FT_DIR: the mean over its heads of
    P(this crop made the sound); without: the offset model's P(|offset| <= 0.2 s)."""

    def __init__(self, device: str, ft_dir: Path | None = FT_DIR):
        import copy
        import torch
        from omegaconf import OmegaConf
        sys.path[:0] = [str(SF_DIR), str(SF_DIR / "model" / "modules" / "feat_extractors" / "visual")]
        from dataset.transforms import make_class_grid
        from scripts.train_utils import get_model, get_transforms
        cfg = OmegaConf.load(CKPT / f"cfg-{CKPT.name}.yaml")
        cfg.model.params.afeat_extractor.params.ckpt_path = None
        cfg.model.params.vfeat_extractor.params.ckpt_path = None
        cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(".modules.feature_selector.", ".sync_model.")
        _, self.model = get_model(cfg, torch.device(device))
        self.model.load_state_dict(torch.load(CKPT / f"{CKPT.name}.pt", map_location="cpu", weights_only=False)["model"])
        self.model.eval()
        self.heads = []
        for f in sorted(Path(ft_dir).glob("seed*.pt")) if ft_dir else []:  # each a copy of the top, fine-tuned parts loaded
            h = copy.deepcopy(source_head(self.model)).to(device).eval()
            h.load_state_dict(torch.load(f, map_location="cpu"), strict=False)
            self.heads.append(h)
        assert self.heads or not ft_dir, f"no seed*.pt in {ft_dir}"
        self.tf, self.cfg, self.torch, self.device = get_transforms(cfg, ["test"])["test"], cfg, torch, device
        self.grid = make_class_grid(-cfg.data.max_off_sec, cfg.data.max_off_sec,
                                    cfg.model.params.transformer.params.off_head_cfg.params.out_features).numpy()

    def score(self, frames: list[np.ndarray], audio: np.ndarray) -> float:
        torch = self.torch
        v = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
        item = self.tf({"video": v, "audio": torch.from_numpy(audio), "path": "", "split": "test",
                        "meta": {"video": {"fps": [VFPS]}, "audio": {"framerate": [AFPS]}},
                        "targets": {"v_start_i_sec": 0.0, "offset_sec": 0.0}})
        vis, aud = item["video"][None].to(self.device), item["audio"][None].to(self.device)
        with torch.no_grad():
            if self.heads:  # extractors once, then every head (as the heads were trained and scored)
                with torch.autocast("cuda", enabled=True):
                    vf = self.model.extract_vfeats(vis, False).half(); af = self.model.extract_afeats(aud, False).half()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return float(np.mean([torch.softmax(h(vf, af)[1].float(), -1)[0, 1].item() for h in self.heads]))
            with torch.autocast("cuda", enabled=bool(self.cfg.training.use_half_precision)):
                _, logits = self.model(vis, aud)
        p = torch.softmax(logits.float(), -1)[0].cpu().numpy()
        z = int(np.argmin(np.abs(self.grid)))
        return float(p[max(0, z - 1):z + 2].sum())


def _square(box, W, H):
    x1, y1, x2, y2 = box; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = min(max(x2 - x1, y2 - y1) * SCALE, W, H)
    return int(np.clip(cx - s / 2, 0, W - s)), int(np.clip(cy - s / 2, 0, H - s)), int(s)


def _frames(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx[0])); cur, ims, last = int(idx[0]), {}, None
    for i in sorted(set(int(j) for j in idx)):
        while cur <= i:
            ok, im = cap.read(); cur += 1
            if ok:
                last = im
        ims[i] = last
    ok = [i for i, im in ims.items() if im is not None]  # a frame right after a seek may fail to decode
    if not ok:
        raise RuntimeError(f"could not decode frames {min(ims)}-{max(ims)}")
    return {i: im if im is not None else ims[min(ok, key=lambda j: abs(j - i))] for i, im in ims.items()}


def assign(video: Path, events: pd.DataFrame, tracks_csv: Path, track_to_profile_csv: Path, device: str = "cuda:0",
           log=print, split: bool = False) -> pd.DataFrame:
    """events: event_id, start_time, end_time. split: first cut every sound into single barks (split_barks), one
    decision and one predictions row per bark, all keeping the sound's event_id."""
    tr = pd.read_csv(tracks_csv)
    t2p = dict(pd.read_csv(track_to_profile_csv)[["track_id", "persistent_profile_id"]].values)
    tr = tr[tr.track_id.isin(t2p)].assign(dog=lambda d: d.track_id.map(t2p))
    cap = cv2.VideoCapture(str(video))
    fps, nfr = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    audio = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-ac", "1", "-ar", str(AFPS), "-f", "s16le", "-"],
                                         capture_output=True).stdout, np.int16).astype(np.float32) / 32768
    dur = len(audio) / AFPS if len(audio) else nfr / fps
    if split and len(audio):
        env = loudness(audio)
        events = pd.DataFrame([(e.event_id, a, b) for e in events.itertuples() for a, b in split_barks(env, e.start_time, e.end_time)],
                              columns=["event_id", "start_time", "end_time"])
    sync_model = None
    rows = []
    for e in events.itertuples():
        w = tr[(tr.time_sec >= e.start_time - PAD_SEC) & (tr.time_sec <= e.end_time + PAD_SEC)]
        cands = sorted(w.dog.unique())
        sync, motion = {}, {}
        if cands and len(audio):
            if sync_model is None:
                sync_model = Sync(device)
            t0 = float(np.clip((e.start_time + e.end_time) / 2 - WIN / 2, 0, max(0.0, dur - WIN)))
            idx = np.minimum(np.round((t0 + np.arange(int(WIN * VFPS)) / VFPS) * fps).astype(int), nfr - 1)
            a = audio[int(t0 * AFPS):int(t0 * AFPS) + int(WIN * AFPS)]
            a = np.pad(a, (0, int(WIN * AFPS) - len(a)))
            f0, f1 = max(0, int((e.start_time - CTX) * fps)), min(nfr - 1, int((e.end_time + CTX) * fps) + 1)
            ims = _frames(cap, np.concatenate([idx, np.arange(f0, f1 + 1)]))
            for dog in cands:
                d = tr[tr.dog == dog]; fr = d.frame.to_numpy()
                crops = []
                for i in idx:
                    r = d.iloc[int(np.argmin(np.abs(fr - i)))]
                    x0, y0, s = _square((r.x1, r.y1, r.x2, r.y2), W, H)
                    crops.append(cv2.resize(cv2.cvtColor(ims[int(i)][y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2RGB), (SIZE, SIZE)))
                sync[dog] = sync_model.score(crops, a)
                diffs, prev = [], None
                for f in range(f0, f1 + 1):
                    j = int(np.argmin(np.abs(fr - f)))
                    if abs(fr[j] - f) > 3:
                        prev = None; continue
                    r = d.iloc[j]
                    x1, y1, x2, y2 = int(max(0, r.x1)), int(max(0, r.y1)), int(min(W, r.x2)), int(min(H, r.y2))
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        prev = None; continue
                    g = cv2.resize(cv2.cvtColor(ims[f][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32), (64, 64))
                    if prev is not None:
                        diffs.append(float(np.abs(g - prev).mean()))
                    prev = g
                motion[dog] = float(np.mean(diffs)) if diffs else 0.0
        rows.append((e, w, cands, *decide(cands, sync, motion, EXTRA_MIN)))
    cap.release()
    ctx = burst_context([(e.start_time, e.end_time, cands, dog, sc) for e, _, cands, dog, _, _, sc, _ in rows])
    out = []
    for (e, w, cands, dog, status, reason, scores, extra), c in zip(rows, ctx):
        if c and c[1] == "context":
            dog, status, reason = c[0], "context", f"best of {len(cands)} dogs, score {scores[c[0]]:.2f}, and a bark <= {CONTEXT_GAP:g} s away is {c[0]}'s"
        elif c:
            dog, status, reason = c[0], "chain", f"best of {len(cands)} dogs, score {scores[c[0]]:.2f}, inside a burst with a named bark"
        for d, st, why in [(dog, status, reason)] + [(x, "also_synced", f"also in sync, score {scores[x]:.2f}") for x in extra]:
            track = int(w[w.dog == d].track_id.value_counts().idxmax()) if d else None
            out.append([e.event_id, e.start_time, e.end_time, st, track, d, " ".join(cands), why, json.dumps(scores)])
            log(f"{e.event_id} {e.start_time:.2f}-{e.end_time:.2f} s: {d or 'outside'} ({why})")
    return pd.DataFrame(out, columns=COLUMNS)
    return pd.DataFrame(rows, columns=COLUMNS)


def summary(pred: pd.DataFrame) -> str:
    n = pred.event_id.nunique(); st = pred.status.value_counts()
    return (f"barker (dog fine-tuned Synchformer + motion, FINAL v3): {n} sound(s), {len(pred)} decision(s): {int(st.get('single', 0))} to the only dog on screen, "
            f"{int(st.get('synced', 0))} to the best-synced of several dogs, {int(st.get('context', 0)) + int(st.get('chain', 0))} from a neighbouring bark, {int(st.get('outside', 0))} outside, "
            f"{int(st.get('also_synced', 0))} extra barker(s) "
            f"({int((pred.candidates.fillna('') == '').sum())} with no dog on screen)")


def _self_check():
    assert decide([], {}, {})[:2] == (None, "outside")
    assert decide(["dog_1"], {"dog_1": 0.5}, {})[:2] == ("dog_1", "single")
    assert decide(["dog_1"], {"dog_1": 0.1}, {})[:2] == (None, "outside")
    assert decide(["dog_1", "dog_2"], {"dog_1": 0.95, "dog_2": 0.2}, {"dog_1": 1, "dog_2": 1})[:2] == ("dog_1", "synced")
    assert decide(["dog_1", "dog_2"], {"dog_1": 0.5, "dog_2": 0.4}, {"dog_1": 1, "dog_2": 1})[:2] == (None, "outside")
    two = (["dog_1", "dog_2", "dog_3"], {"dog_1": 0.97, "dog_2": 0.95, "dog_3": 0.1}, {"dog_1": 1, "dog_2": 1, "dog_3": 1})
    assert decide(*two)[4] == [] and decide(*two, extra_min=0.9)[4] == ["dog_2"]  # second barker only when switched on
    t = np.arange(0, 1, 1 / AFPS); burst = lambda c: np.sin(2 * np.pi * 800 * t) * np.exp(-((t - c) / 0.05) ** 2)
    env = loudness((burst(0.2) + burst(0.6)).astype(np.float32))
    assert [round(x, 1) for x, _ in split_barks(env, 0.0, 1.0)] == [0.0, 0.4]  # two barks -> cut in the gap
    assert split_barks(env, 0.1, 0.3) == [(0.1, 0.3)]                          # one bark stays whole
    # motion breaks a near-tie: dog_2 moves 3x more -> +0.25 vs -0.25
    assert decide(["dog_1", "dog_2"], {"dog_1": 0.9, "dog_2": 0.8}, {"dog_1": 1, "dog_2": 3})[0] == "dog_2"
    bk = [(0.0, 0.5, ["dog_1", "dog_2"], "dog_1", {"dog_1": 0.95, "dog_2": 0.2}),
          (0.6, 1.0, ["dog_1", "dog_2"], None, {"dog_1": 0.6, "dog_2": 0.3}),    # neighbour is dog_1's -> dog_1
          (1.2, 1.5, ["dog_1", "dog_2"], None, {"dog_2": 0.6, "dog_1": 0.3}),    # best is dog_2, not the neighbour's
          (1.6, 1.9, ["dog_1"], None, {"dog_1": 0.3}),                           # one dog: the one-dog rule stands
          (9.0, 9.5, ["dog_1", "dog_2"], None, {"dog_1": 0.8, "dog_2": 0.3})]    # no named bark within 3 s
    assert burst_context(bk) == [None, ("dog_1", "context"), None, None, None]
    # chain (723-like): dog_1 named, then touching barks where dog_2 leads -> dog_2 along the burst; a near-tie stays out
    ch = [(0.0, 1.0, ["dog_1", "dog_2"], "dog_1", {"dog_1": 0.96, "dog_2": 0.3}),
          (1.0, 2.0, ["dog_1", "dog_2"], None, {"dog_2": 0.55, "dog_1": 0.2}),
          (2.0, 3.0, ["dog_1", "dog_2"], None, {"dog_2": 0.52, "dog_1": 0.25}),
          (3.0, 4.0, ["dog_1", "dog_2"], None, {"dog_2": 0.6, "dog_1": 0.45}),     # lead 0.15 < 0.2
          (4.5, 5.0, ["dog_1", "dog_2"], None, {"dog_2": 0.9, "dog_1": 0.1})]      # not touching
    assert burst_context(ch) == [None, ("dog_2", "chain"), ("dog_2", "chain"), None, None]
    print("assign_barker self-check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--events"); ap.add_argument("--stage2-dir"); ap.add_argument("--output")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--split", action="store_true", help="cut every sound into single barks first (detected sounds)")
    a = ap.parse_args()
    if a.self_check or not a.video:
        _self_check()
    else:
        s2 = Path(a.stage2_dir)
        p = assign(Path(a.video), pd.read_csv(a.events), s2 / "tracks_pieces.csv", s2 / "track_to_all_dog_profile.csv", a.device, split=a.split)
        p.to_csv(a.output, index=False)
        print(summary(p))
