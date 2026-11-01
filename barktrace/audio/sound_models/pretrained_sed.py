"""Frame-level AudioSet-Strong sound event detectors (Schmid et al., "Effective Pre-Training of Audio
Transformers for Sound Event Detection", ICASSP 2025; github.com/fschmid56/PretrainedSED, cloned into
models/PretrainedSED). Transformers fine-tuned on AudioSet frame-level labels: 447 classes including Bark,
Bow-wow, Yip, Dog, Growling, Howl, Whimper (dog), Pant (dog).

Inference as the repo's inference.py: 16 kHz, consecutive 10 s chunks (last one zero-padded),
sigmoid of the strong head; weights in models/PretrainedSED/resources.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("BARKTRACE_MODELS", Path(__file__).resolve().parents[3] / "models")) / "PretrainedSED"  # code (setup_third_party.sh) + resources/*_strong_1.pt
sys.path.insert(0, str(REPO))
from data_util import audioset_classes  # noqa: E402

CLASSES = audioset_classes.as_strong_train_classes
IDX = {c: i for i, c in enumerate(CLASSES)}
BARK_CLASSES = ("Bark", "Bow-wow", "Yip")
DOG_CLASSES = BARK_CLASSES + ("Dog", "Growling", "Howl", "Whimper (dog)", "Pant (dog)")


def _build(name: str):
    from models.prediction_wrapper import PredictionsWrapper
    if name == "BEATs":
        from models.beats.BEATs_wrapper import BEATsWrapper
        return PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
    if name == "ATST-F":
        from models.atstframe.ATSTF_wrapper import ATSTWrapper
        return PredictionsWrapper(ATSTWrapper(), checkpoint="ATST-F_strong_1")
    if name == "M2D":
        from models.m2d.M2D_wrapper import M2DWrapper
        m2d = M2DWrapper()
        return PredictionsWrapper(m2d, checkpoint="M2D_strong_1", embed_dim=m2d.m2d.cfg.feature_d)
    if name.startswith("frame_mn"):
        from models.frame_mn.Frame_MN_wrapper import FrameMNWrapper
        from models.frame_mn.utils import NAME_TO_WIDTH
        fm = FrameMNWrapper(NAME_TO_WIDTH(name))
        return PredictionsWrapper(fm, checkpoint=f"{name}_strong_1", embed_dim=fm.state_dict()["frame_mn.features.16.1.bias"].shape[0])
    raise ValueError(name)


class StrongSED:
    """frame_probs(audio16k) -> (fps, probs[T, 447])."""

    def __init__(self, name: str = "BEATs", device: str = "cuda:0"):
        cwd = os.getcwd()
        os.chdir(REPO)  # the repo resolves its relative "resources" folder (weights) from the cwd
        try:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):  # it prints "Loading pretrained checkpoint: <name>"
                self.model = _build(name)
        finally:
            os.chdir(cwd)
        self.model.eval(); self.model.to(device)
        self.device, self.name = device, name

    @torch.no_grad()
    def frame_probs(self, audio: np.ndarray, emb: bool = False):
        """-> (fps, probs[T, 447]); emb=True: (fps, probs, emb[T, D]), the sequence model's frame embeddings that
        the strong head reads."""
        seg = 160000
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None].to(self.device)
        n = x.shape[1]; chunks, embs = [], []
        hook = self.model.seq_model.register_forward_hook(
            lambda m, i, o: embs.append(o[0] if isinstance(o, tuple) else o)) if emb else None
        try:
            for s in range(0, n, seg):
                c = x[:, s:s + seg]
                if c.shape[1] < seg:
                    c = torch.nn.functional.pad(c, (0, seg - c.shape[1]))
                y, _ = self.model(self.model.mel_forward(c))
                chunks.append(y)
        finally:
            if hook:
                hook.remove()
        y = torch.sigmoid(torch.cat(chunks, dim=2))[0].float().cpu().numpy().T  # (T, 447)
        fps = y.shape[0] / (len(chunks) * 10.0); t = int(np.ceil(n / 16000 * fps))
        if not emb:
            return fps, y[:t]
        e = torch.cat(embs, dim=1)[0].float().cpu().numpy()  # (T, D)
        assert len(e) == len(y), (e.shape, y.shape)
        return fps, y[:t], e[:t]


def class_max(probs: np.ndarray, names=BARK_CLASSES) -> np.ndarray:
    return probs[:, [IDX[c] for c in names]].max(1)
