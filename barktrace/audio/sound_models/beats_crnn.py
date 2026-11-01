"""Dog vocalization detection with the lab's fine-tuned BEATs SED model.

Model (the lab's BEATs SED folder, BARKTRACE_BEATS_SED): a CRNN over frozen BEATs
(BEATS_iter3_plus_AS2M) frame embeddings with one class, "Dog", trained DESED-style. It outputs one
probability per 64 ms frame (15.625 frames/s), so thresholding gives each vocalization's start/end.

Inference replicates the lab's inference script: 16 kHz mono, 10 s windows with a 5 s hop, per-frame
maximum over overlapping windows, clips shorter than 10 s zero-padded. Their librosa.load is replaced
by ffmpeg decoding + soxr "HQ" resampling (librosa's default resampler); librosa itself needs a numba
that does not support this env's numpy.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soxr
import torch
import yaml
from torchaudio.transforms import MelSpectrogram

# the lab's BEATs SED folder (desed_task code, hparams.yaml, BEATS_iter3_plus_AS2M.pt, epoch=109-step=6160.ckpt; not in this
# repository, README.md "Lab BEATs SED")
BEATS_DIR = Path(os.environ.get("BARKTRACE_BEATS_SED", Path(__file__).resolve().parents[3] / "third_party" / "beats_sed"))
sys.path.insert(0, str(BEATS_DIR / "beats_work"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # the model calls use_deterministic_algorithms
from desed_task.nnet.CRNN_beats import CRNN  # noqa: E402
import importlib.util as _ilu  # noqa: E402
# desed_task.utils/__init__ imports dcase_util (only needed for training-time label encoding): load the scaler file alone
_spec = _ilu.spec_from_file_location("_desed_scaler", BEATS_DIR / "beats_work/desed_task/utils/scaler.py")
_scaler = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_scaler)
TorchScaler = _scaler.TorchScaler

SR = 16000


def load_audio(path, sr: int = SR) -> np.ndarray:
    """Mono float32 audio at `sr` from any file ffmpeg reads (video or audio)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    if not probe:
        raise ValueError(f"no audio stream in {path}")
    native = int(probe.split(",")[0])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    return y if native == sr else soxr.resample(y, native, sr, quality="HQ").astype(np.float32)


class BeatsSED:
    """frame_probs(audio16k) -> P(dog vocalization) per 64 ms frame."""

    def __init__(self, device: str = "cuda:0", ckpt: str | Path | None = None):
        """ckpt: optional fine-tuned CRNN state dict; default the lab checkpoint."""
        self.device = device
        cfg = yaml.safe_load(open(BEATS_DIR / "hparams.yaml"))
        self.cfg = cfg
        net = dict(cfg["net"])
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):  # the lab CRNN prints '新参数' ("new parameters") when it
            # creates its head; the full state dict is loaded (strictly) just below
            self.model = CRNN(unfreeze_atst_layer=cfg["opt"]["tfm_trainable_layers"], **net, model_init=None,
                              beats_init=str(BEATS_DIR / "BEATS_iter3_plus_AS2M.pt"))
        state = (torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt else
                 torch.load(BEATS_DIR / "epoch=109-step=6160.ckpt", map_location="cpu", weights_only=False)["sed_student"])
        self.model.load_state_dict(state, strict=True)
        self.model.eval()  # their train() override returns None, so no chaining
        self.model.to(device)
        f = cfg["feats"]
        self.mel = MelSpectrogram(sample_rate=f["sample_rate"], n_fft=f["n_window"], win_length=f["n_window"],
                                  hop_length=f["hop_length"], f_min=f["f_min"], f_max=f["f_max"], n_mels=f["n_mels"],
                                  window_fn=torch.hamming_window, wkwargs={"periodic": False}, power=1).to(device)
        self.scaler = TorchScaler("instance", cfg["scaler"]["normtype"], cfg["scaler"]["dims"])
        self.fs = cfg["data"]["fs"]
        self.win = int(cfg["data"]["audio_max_len"]) * self.fs
        self.hop = 5 * self.fs
        self.frame_hop = f["hop_length"] * cfg["data"]["net_subsample"]  # samples per output frame
        self.fps = self.fs / self.frame_hop  # 15.625

    def _t2f(self, n_samples: int) -> int:
        return int(int(n_samples / self.cfg["feats"]["hop_length"]) / self.cfg["data"]["net_subsample"])

    @torch.no_grad()
    def _chunk(self, x: torch.Tensor) -> np.ndarray:
        x = x.unsqueeze(0)
        feats = self.scaler(torch.clamp(self.mel(x), min=1e-5).log10().mul(20.0).clamp(min=-50, max=80))
        strong, _ = self.model(feats, x)
        return strong[0, 0].float().cpu().numpy()

    def frame_probs(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(self.device)
        if n < self.win:
            x = torch.nn.functional.pad(x, (0, self.win - n))
        if x.numel() <= self.win:
            out = self._chunk(x)
        else:  # 10 s windows, 5 s hop, frame-wise maximum (as the lab's inference script)
            total_chunks = (x.numel() - (self.win - self.hop)) // self.hop + 1
            total = total_chunks * self.hop + (self.win - self.hop)
            pad = total - x.numel()
            xp = torch.nn.functional.pad(x, (0, pad))
            chunks = [self._chunk(xp[i * self.hop:i * self.hop + self.win]) for i in range(total_chunks)]
            T = len(chunks[0]); hop_f = self._t2f(self.hop)
            dec = np.zeros(self._t2f(total))
            dec[:T] = chunks[0]
            for i in range(1, len(chunks)):
                dec[i * hop_f:i * hop_f + T] = np.maximum(chunks[i], dec[i * hop_f:i * hop_f + T])
            pad_f = max(self._t2f(pad), 1) if pad > 0 else 0
            out = dec[:-pad_f] if pad_f else dec
        return out[:int(np.ceil(n / self.fs * self.fps))]


def _split(probs, s, e, thr, low, high, split_on=None):
    """Cut a frame run [s, e) at each valley below `low` that lies between two frames >= `high`; trim
    the pieces' frames below thr. Valleys and peaks are found in split_on (default: probs)."""
    seg = probs[s:e]; v = seg if split_on is None else split_on[s:e]; hi = np.flatnonzero(v >= high)
    cuts = [a + 1 + int(np.argmin(v[a + 1:b])) for a, b in zip(hi[:-1], hi[1:])
            if b - a > 1 and v[a + 1:b].min() < low]
    out = []
    for a, b in zip([0] + [c + 1 for c in cuts], cuts + [len(seg)]):
        on = np.flatnonzero(seg[a:b] >= thr)
        if len(on):
            out.append((s + a + on[0], s + a + on[-1] + 1))
    return out


def to_events(probs: np.ndarray, fps: float, thr: float, merge_gap: float = 0.5, min_dur: float = 0.5,
              duration: float | None = None, split_below: float | None = None, peak: float = 0.9,
              split_on: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Frames >= thr -> [(start, end)] seconds; events closer than merge_gap are merged and ones
    shorter than min_dur dropped (the settings of the lab's inference script). split_below: afterwards, an
    event is cut at every dip below this score between two peaks >= `peak` (one event per bark), found in
    split_on (default: probs)."""
    on = np.r_[False, probs >= thr, False]
    edges = np.flatnonzero(on[1:] != on[:-1])
    runs = []
    for s, e in zip(edges[::2], edges[1::2]):
        if runs and (s - runs[-1][1]) / fps < merge_gap:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    if split_below is not None:
        runs = [p for s, e in runs for p in _split(probs, s, e, thr, split_below, peak, split_on)]
    ev = [(s / fps, e / fps) for s, e in runs]
    if duration is not None:
        ev = [(max(0.0, s), min(duration, e)) for s, e in ev]
    return [(round(float(s), 3), round(float(e), 3)) for s, e in ev if e - s >= min_dur]
