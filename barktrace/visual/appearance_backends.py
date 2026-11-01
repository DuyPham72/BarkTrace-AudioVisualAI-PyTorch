#!/usr/bin/env python3
"""Pluggable appearance-embedding backends for the dog re-ID pipeline.

The pipeline historically had exactly one appearance model: OSNet-x0.25,
fine-tuned on dog identities. Both
embedding sites -- `AppearanceExtractor` in appearance_extractor.py (per-frame,
inside StableTracker) and the per-track encoder of the profile merger --
hardcoded it.

This module adds DINOv3 as an opt-in alternative, plus a concatenation mode.
The motivation is that every embedding model tried so far in this project
(stock OSNet, MegaDescriptor-T-224, five yolo26*-reid ONNX models) shares an
architecture family and scale with the incumbent: small, supervised,
retrieval-specialised. DINOv3 is categorically different -- a large
self-supervised ViT trained on 1.7B images -- and is expected to degrade more
gracefully on blur, low resolution (clips here range 360p-1080p) and
truncated subjects, which is exactly where a small person-ReID backbone is
weakest.

IMPORTANT -- HF_HOME ordering: every entry point that imports this module
must set os.environ["HF_HOME"] to the in-project cache *before* transformers
or huggingface_hub is imported anywhere in the process. huggingface_hub
resolves its cache path at import time, so setting it afterwards silently
leaks multi-hundred-MB downloads into the real $HOME.
`ensure_hf_home()` below does this and must be called first.

All backends expose the same contract:
    .encode(crops_bgr: list[np.ndarray]) -> np.ndarray, shape (N, dim), float32
    .dim: int
Every returned row is L2-normalized, so cosine similarity remains the
comparison metric everywhere downstream (the merger's thresholds, the
hybrid-gallery bonus, StableTracker's Hungarian cost) without change.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

PROJECT_HF_CACHE = str(Path(os.environ.get("BARKTRACE_MODELS", Path(__file__).resolve().parents[2] / "models")) / "hf_cache")

DINOV3_REPOS = {
    "s": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_hf_home() -> None:
    """Point HuggingFace at the in-project cache. Call before importing transformers."""
    os.environ.setdefault("HF_HOME", PROJECT_HF_CACHE)
    # The weights are gated on the Hub but already cached locally, so nothing
    # here should ever need the network or a token.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


# Run at import, not just inside the encoder: by the time anything constructs
# a backend, transformers may already have been imported transitively by
# another module, and the cache path is fixed at *its* import. setdefault keeps
# an explicit caller override working.
ensure_hf_home()


def letterbox_square(crop_rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect, pad to square with black.

    Dog crops vary widely in aspect (a dog standing side-on is wide, a dog
    facing the camera is tall). Stretching distorts body proportions, which
    are a real identity cue; letterboxing preserves them at the cost of some
    wasted input area. Which one actually helps is measured, not assumed.
    """
    height, width = crop_rgb.shape[:2]
    if height == 0 or width == 0:
        return np.zeros((size, size, 3), dtype=crop_rgb.dtype)

    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


class DINOv3Encoder:
    """DINOv3 ViT appearance embeddings from BGR crops."""

    def __init__(
        self,
        device: str,
        variant: str = "s",
        aspect: str = "letterbox",
        image_size: int = 224,
        pooling: str = "cls",
    ) -> None:
        ensure_hf_home()
        import torch  # noqa: PLC0415 - imported after HF_HOME is set
        from transformers import AutoModel  # noqa: PLC0415

        variant = variant.lower()
        if variant not in DINOV3_REPOS:
            raise ValueError(
                f"Unknown DINOv3 variant {variant!r}; expected one of {sorted(DINOV3_REPOS)}"
            )
        if aspect not in {"letterbox", "stretch"}:
            raise ValueError(f"Unknown aspect mode {aspect!r}; expected letterbox or stretch")
        if pooling not in {"cls", "cls+mean"}:
            raise ValueError(f"Unknown pooling {pooling!r}; expected cls or cls+mean")

        self._torch = torch
        self.device = torch.device(device)
        self.variant = variant
        self.aspect = aspect
        self.image_size = int(image_size)
        self.pooling = pooling

        self.model = AutoModel.from_pretrained(DINOV3_REPOS[variant]).eval().to(self.device)
        hidden = int(self.model.config.hidden_size)
        self.dim = hidden * 2 if pooling == "cls+mean" else hidden
        # 1 CLS + N register tokens precede the patch tokens.
        self._num_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0))

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        if self.aspect == "letterbox":
            square = letterbox_square(rgb, self.image_size)
        else:
            square = cv2.resize(
                rgb,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        arr = square.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(arr, (2, 0, 1))

    def encode(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.empty((0, self.dim), dtype=np.float32)

        torch = self._torch
        batch = np.stack([self._preprocess(c) for c in crops_bgr])
        tensor = torch.from_numpy(batch).to(self.device)

        with torch.no_grad():
            out = self.model(pixel_values=tensor)
            cls = out.last_hidden_state[:, 0]
            if self.pooling == "cls+mean":
                patches = out.last_hidden_state[:, self._num_prefix:].mean(dim=1)
                feats = torch.cat([cls, patches], dim=1)
            else:
                feats = cls
            feats = feats / torch.linalg.vector_norm(
                feats, dim=1, keepdim=True
            ).clamp(min=1e-6)

        return feats.float().cpu().numpy().astype(np.float32)


class OSNetEncoder:
    """OSNet-x0.25 embeddings from BGR crops.

    Mirrors the preprocessing the pipeline has always used (256x128, ImageNet
    normalization, L2-normalized output). Exists so the concat backend can own
    both halves without reaching back into either call site.
    """

    def __init__(self, model_name: str, device: str, checkpoint: str | None = None) -> None:
        import torch  # noqa: PLC0415
        import torchreid  # noqa: PLC0415
        import torchvision.transforms as T  # noqa: PLC0415

        self._torch = torch
        self.device = torch.device(device)
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
            ]
        )
        self.model = torchreid.models.build_model(model_name, num_classes=1, pretrained=True)
        if checkpoint is not None:
            torchreid.utils.load_pretrained_weights(self.model, str(checkpoint))
        self.model.eval().to(self.device)
        self.dim = 512

    def encode(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.empty((0, self.dim), dtype=np.float32)

        torch = self._torch
        tensors = [
            self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in crops_bgr
        ]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
            feats = feats / torch.linalg.vector_norm(
                feats, dim=1, keepdim=True
            ).clamp(min=1e-6)
        return feats.float().cpu().numpy().astype(np.float32)


class ConcatEncoder:
    """Weighted concatenation of two L2-normalized embeddings.

    Each half is unit-norm already, so `dinov3_weight` sets the relative
    contribution directly: the DINOv3 block is scaled by w, the OSNet block by
    (1 - w), and the result is re-normalized so downstream cosine similarity is
    still bounded in [-1, 1]. w=0 reproduces OSNet alone (up to renormalization)
    and w=1 reproduces DINOv3 alone.

    Rationale: the two models should fail on different inputs. OSNet is
    dog-specific but small and blur-sensitive; DINOv3 is generic but robust.
    Whether that complementarity survives contact with the benchmark is an open
    question -- the weight is a flag, deliberately not pinned to a guess.
    """

    def __init__(self, osnet: OSNetEncoder, dinov3: DINOv3Encoder, dinov3_weight: float = 0.5) -> None:
        if not 0.0 <= dinov3_weight <= 1.0:
            raise ValueError(f"dinov3_weight must be in [0, 1], got {dinov3_weight}")
        self.osnet = osnet
        self.dinov3 = dinov3
        self.dinov3_weight = float(dinov3_weight)
        self.dim = osnet.dim + dinov3.dim

    def encode(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.empty((0, self.dim), dtype=np.float32)

        left = self.osnet.encode(crops_bgr) * (1.0 - self.dinov3_weight)
        right = self.dinov3.encode(crops_bgr) * self.dinov3_weight
        joined = np.concatenate([left, right], axis=1)
        norms = np.linalg.norm(joined, axis=1, keepdims=True)
        return (joined / np.clip(norms, 1e-6, None)).astype(np.float32)


def build_backend(
    backend: str,
    device: str,
    osnet_model: str = "osnet_x0_25",
    osnet_checkpoint: str | Path | None = None,
    dinov3_variant: str = "s",
    dinov3_weight: float = 0.5,
    dinov3_aspect: str = "letterbox",
    dinov3_pooling: str = "cls",
):
    """Build an appearance backend, or return None for the default OSNet path.

    Returning None (for backend == "osnet") is deliberate: it lets the two call
    sites keep their original, untouched code path, so the default behaviour is
    byte-identical rather than merely equivalent.
    """
    backend = (backend or "osnet").lower()
    if backend == "osnet":
        return None

    checkpoint = None if osnet_checkpoint is None else str(osnet_checkpoint)

    if backend == "dinov3":
        return DINOv3Encoder(
            device,
            variant=dinov3_variant,
            aspect=dinov3_aspect,
            pooling=dinov3_pooling,
        )

    if backend in {"osnet+dinov3", "concat"}:
        return ConcatEncoder(
            OSNetEncoder(osnet_model, device, checkpoint=checkpoint),
            DINOv3Encoder(
                device,
                variant=dinov3_variant,
                aspect=dinov3_aspect,
                pooling=dinov3_pooling,
            ),
            dinov3_weight=dinov3_weight,
        )

    raise ValueError(
        f"Unknown appearance backend {backend!r}; expected osnet, dinov3, or osnet+dinov3"
    )


def add_backend_cli_args(parser) -> None:
    """Shared CLI surface, so both entry points expose identical flag names."""
    parser.add_argument(
        "--appearance-backend",
        default="osnet",
        choices=["osnet", "dinov3", "osnet+dinov3"],
        help=(
            "Appearance embedding model. 'osnet' (default) is the fine-tuned "
            "OSNet-x0.25 that every current threshold was calibrated against. "
            "'dinov3' and 'osnet+dinov3' are opt-in alternatives whose "
            "similarity distributions differ -- recalibrate the "
            "existing thresholds before trusting them."
        ),
    )
    parser.add_argument(
        "--dinov3-variant", default="s", choices=["s", "b"],
        help="DINOv3 ViT size: s = ViT-S/16 (384-d), b = ViT-B/16 (768-d).",
    )
    parser.add_argument(
        "--dinov3-weight", type=float, default=0.5,
        help="Relative weight of the DINOv3 half in --appearance-backend osnet+dinov3.",
    )
    parser.add_argument(
        "--dinov3-aspect", default="letterbox", choices=["letterbox", "stretch"],
        help="Aspect handling for DINOv3's square input.",
    )
    parser.add_argument(
        "--dinov3-pooling", default="cls", choices=["cls", "cls+mean"],
        help="DINOv3 embedding: CLS token, or CLS concatenated with mean patch token.",
    )
