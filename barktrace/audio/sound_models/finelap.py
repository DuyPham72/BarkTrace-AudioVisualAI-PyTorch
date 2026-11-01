"""FineLAP (Li et al., ACL 2026; huggingface.co/AndreasXi/FineLAP): open-vocabulary frame-level audio-text
alignment (EAT audio encoder + RoBERTa). frame_probs(audio16k, prompt) -> (fps, P(prompt) per frame).
The HF loader calls from_pretrained inside the model's __init__, which transformers 5.x refuses under its
meta-device init; so the class is built directly and the weights loaded from the snapshot (all keys match).
Input as in its load_audio: 10.24 s windows (1024 fbank frames), 64 output frames each (6.25 fps)."""
from __future__ import annotations

import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

REPO = "AndreasXi/FineLAP"
WIN = 1024  # fbank frames (10 ms) per window


class FineLAPSED:
    fps = 64 / 10.24

    def __init__(self, device: str = "cuda:0"):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
        self.model = get_class_from_dynamic_module(cfg.auto_map["AutoModel"], REPO)(cfg)
        sd = {}
        for f in glob.glob(f"{snapshot_download(REPO)}/*.safetensors"):
            sd.update(load_file(f))
        r = self.model.load_state_dict(sd, strict=False)
        assert not r.missing_keys and not r.unexpected_keys, r
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def frame_embeds(self, audio16k: np.ndarray) -> np.ndarray:
        """dense frame embeddings [T, D] (6.25 fps), the audio side of the frame-text similarity"""
        wav = torch.from_numpy(np.ascontiguousarray(audio16k, dtype=np.float32))
        wav = wav - wav.mean()
        mel = torchaudio.compliance.kaldi.fbank(wav[None], htk_compat=True, sample_frequency=16000, use_energy=False,
                                                window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
        mel = (mel + 4.268) / (4.569 * 2)
        n = mel.shape[0]
        chunks = [mel[s:s + WIN] for s in range(0, max(n, 1), WIN)]
        chunks = [F.pad(c, (0, 0, 0, WIN - c.shape[0])) for c in chunks]
        m = self.model
        out = []
        for i in range(0, len(chunks), 8):
            batch = torch.stack(chunks[i:i + 8])[:, None].to(self.device)
            feats = m.audio_encoder.extract_features(batch)
            raw = feats["x"] if isinstance(feats, dict) else feats
            B, T, D = raw[:, 1:, :].shape
            patches = raw[:, 1:, :].reshape(B, T // 8, 8, D).mean(dim=2)
            dense = m.local_audio_proj(patches)
            dense = dense[0] if m.local_audio_proj_type == "rnn" else dense
            if m.config.normalize_dense_audio_embeds:
                dense = F.normalize(dense, dim=-1)
            out.append(dense.reshape(-1, dense.shape[-1]).float())
        return torch.cat(out)[:int(np.ceil(len(audio16k) / 16000 * self.fps))].cpu().numpy()

    @torch.no_grad()
    def prompt_probs(self, dense: np.ndarray, prompts: list[str]) -> np.ndarray:
        """P(prompt) per frame [T, len(prompts)] from frame_embeds"""
        m = self.model
        text = m.get_global_text_embeds(prompts, self.device)  # (P, D)
        sim = torch.from_numpy(dense).to(self.device) @ text.T
        if hasattr(m, "temp_local"):
            sim = sim / m.temp_local
        if hasattr(m, "b_local"):
            sim = sim + m.b_local
        return torch.sigmoid(sim).float().cpu().numpy()

    def frame_probs(self, audio16k: np.ndarray, prompt: str = "dog barking"):
        return self.fps, self.prompt_probs(self.frame_embeds(audio16k), [prompt])[:, 0]
