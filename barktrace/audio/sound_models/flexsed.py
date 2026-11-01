"""FlexSED (Hai et al., WASPAA 2025; github.com/JHU-LCAP/FlexSED, MIT): open-vocabulary SED, Dasheng encoder +
query-conditioned decoder, CLAP text queries "The sound of <event>". frame_probs(audio16k, event) -> (25 fps,
probs). Same steps as its api.py (10 s chunks, peak-normalised), without its librosa file loading."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("BARKTRACE_MODELS", Path(__file__).resolve().parents[3] / "models")) / "FlexSED"  # code + ckpts/flexsed_as.pt (from huggingface.co/Higobeatz/FlexSED)
DOG_EVENTS = ("Dog", "Bark", "Growling", "Howl", "Whimper (dog)")  # dog vocalizations: best of these per frame


class FlexSEDModel:
    fps = 25.0

    def __init__(self, device: str = "cuda:0"):
        sys.path.insert(0, str(REPO))
        from transformers import AutoTokenizer, ClapTextModelWithProjection
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):  # its dasheng module prints "attention mode is flash"
            from src.models.sed_decoder import Decoder, TSED_Wrapper
            from src.models.transformer import Dasheng_Encoder
        from src.utils import load_yaml_with_includes
        cwd = os.getcwd(); os.chdir(REPO)  # its config includes are relative
        try:
            params = load_yaml_with_includes("src/configs/model.yml")
        finally:
            os.chdir(cwd)
        sd = torch.load(REPO / "ckpts" / "flexsed_as.pt", map_location="cpu", weights_only=False)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):  # its Dasheng setup prints "Expected missing keys: adaln" for
            # layers that only exist in FlexSED; they are filled by the checkpoint loaded (strictly) just below
            self.model = TSED_Wrapper(Dasheng_Encoder(**params["encoder"]), Decoder(**params["decoder"]),
                                      params["ft_blocks"], params["frozen_encoder"])
        self.model.load_state_dict(sd["model"]); self.model.to(device).eval()
        self.clap = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused").eval()
        self.tok = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
        self.device = device

    @torch.no_grad()
    def frame_probs(self, audio16k: np.ndarray, event: str = "Dog bark"):
        q = self.clap(**self.tok([f"The sound of {event.replace('_', ' ').capitalize()}"], return_tensors="pt")).text_embeds
        q = q[:, None].to(self.device)
        n, out = 160000, []
        for s in range(0, max(len(audio16k), 1), n):
            c = np.asarray(audio16k[s:s + n], np.float32)
            c = torch.tensor(np.pad(c, (0, max(0, 16000 - len(c)))))[None].to(self.device)  # a tail of a few samples breaks the STFT; pad to 1 s (trimmed below)
            c = c / (c.abs().max() + 1e-9)
            out.append(torch.sigmoid(self.model(self.model.forward_to_spec(c), q)).reshape(-1).float().cpu())
        p = torch.cat(out).numpy()
        return self.fps, p[:int(np.ceil(len(audio16k) / 16000 * self.fps))]

    def dog_probs(self, audio16k: np.ndarray):
        """any dog vocalization: the best of DOG_EVENTS per frame."""
        outs = [self.frame_probs(audio16k, ev) for ev in DOG_EVENTS]
        return outs[0][0], np.max([p for _, p in outs], axis=0)
