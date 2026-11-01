#!/usr/bin/env python3
"""
appearance_extractor.py

Adapted from the lab's earlier dog-tracking script (track_dogs.py, REU project). The ONLY
functional changes from the original are: (1) load the detector via `ultralytics.YOLO` instead of
`ultralytics.RTDETR`, and (2) default to the user's fine-tuned
models/dog_detection/yolo26s-dog.pt instead of rtdetr-l.pt. Everything else
-- StableTracker (Kalman filter + OSNet appearance, Hungarian IoU+appearance
matching), the CLI, and the tracks.csv/tracking_metadata.json output format
-- is byte-identical to the original.

Why a change was needed at all: `ultralytics.RTDETR` and `ultralytics.YOLO`
use different postprocessing internally (RT-DETR's fixed-query output vs.
YOLO's anchor-free grid output). The original script's `--detector-weights`
flag only ever let you swap between different RT-DETR *checkpoints* -- it
still always loaded them via the `RTDETR` class, which crashes
(`split_with_sizes` shape mismatch) if you point it at a YOLO-family model
like this one. Everything downstream (StableTracker, the CLI, the output
schema) is architecture-agnostic and needed no changes.

Run the single-video dog detection and stable-tracking stage used by the
active-barker inference pipeline.

Input:
    raw video

Outputs:
    <output-dir>/tracks.csv
    <output-dir>/tracking_metadata.json

Tracking stack:
    yolo26s-dog (fine-tuned, user-provided)
    Ultralytics BoT-SORT IDs
    OSNet x0.25 appearance embeddings
    custom stable IDs
    Kalman filter
    Hungarian IoU + appearance matching
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torchreid
import torchvision.transforms as T
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

import appearance_backends


DEFAULT_DETECTOR = Path(os.environ.get("BARKTRACE_MODELS", Path(__file__).resolve().parents[2] / "models")) / "dog_detection" / "yolo26l-base.pt"

TRACK_COLUMNS = [
    "frame",
    "time_sec",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "class_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and stably track every dog in one video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=DEFAULT_DETECTOR,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracking-fps", type=float, default=1000.0)
    parser.add_argument("--conf-threshold", type=float, default=0.60)
    parser.add_argument("--augment", action="store_true",
                        help="Ultralytics test-time augmentation (flips + scales) for the detector; off by default")
    parser.add_argument("--imgsz", type=int, default=640, help="detector inference size (default 640)")
    parser.add_argument("--manual-tta", action="store_true",
                        help="test-time augmentation done here (original + 0.83x flipped + 0.67x, boxes merged "
                             "by NMS) instead of by Ultralytics -- required for TensorRT .engine detectors, "
                             "which silently ignore augment=True")
    parser.add_argument("--all-frames", action="store_true",
                        help="process every raw frame instead of odd frames only (fast motion); off by default")
    parser.add_argument("--iou-threshold", type=float, default=0.08)
    parser.add_argument("--max-age", type=int, default=20)
    parser.add_argument("--appearance-model", default="osnet_x0_25")
    parser.add_argument(
        "--appearance-checkpoint",
        type=Path,
        default=Path(os.environ.get("BARKTRACE_MODELS", Path(__file__).resolve().parents[2] / "models")) / "osnet" / "osnet_x0_25_dog_finetuned.pth.tar",
        help="Fine-tuned OSNet dog-identity weights bundled with this "
        "checkpoint (see CHECKPOINT_NOTES.md). Pass --appearance-checkpoint "
        "none to fall back to the original ImageNet-pretrained OSNet.",
    )
    parser.add_argument("--appearance-threshold", type=float, default=0.70)
    parser.add_argument(
        "--identity-verification-threshold", type=float, default=None,
        help=(
            "Minimum appearance similarity required for Stage 1's cheap-path "
            "track continuation to be trusted (fixes ID swaps when two dogs "
            "pass close together). Defaults to --appearance-threshold."
        ),
    )
    parser.add_argument("--appearance-ema", type=float, default=0.30)
    parser.add_argument("--iou-weight", type=float, default=0.60)
    parser.add_argument(
        "--gmc-method",
        default="none",  # off: no gain on the 21-video benchmark
        help=(
            "Global motion compensation for StableTracker's geometry: "
            "sparseOptFlow|orb|sift|ecc|none. StableTracker replaces "
            "BoT-SORT's association with its own Kalman filter, which "
            "discards BoT-SORT's own camera-motion estimate -- this puts it "
            "back, so track positions are compared in the current frame's "
            "coordinates instead of the previous frame's. Default 'none' "
            "(camera-motion-blind); pass e.g. 'sparseOptFlow' to enable."
        ),
    )
    parser.add_argument(
        "--gmc-downscale",
        type=int,
        default=2,
        help="Frame downscale factor for the GMC estimate (speed vs accuracy).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing tracks.csv.",
    )
    appearance_backends.add_backend_cli_args(parser)
    return parser.parse_args()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def read_video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if fps <= 0 or width <= 0 or height <= 0 or frames <= 0:
        raise RuntimeError(
            f"Invalid video metadata: fps={fps}, width={width}, "
            f"height={height}, frames={frames}"
        )

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames,
        "duration_seconds": frames / fps,
    }


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.array(
        [
            [(x1 + x2) / 2.0],
            [(y1 + y2) / 2.0],
            [x2 - x1],
            [y2 - y1],
        ],
        dtype=np.float32,
    )


def affine_scale_factors(rotation: np.ndarray) -> tuple[float, float]:
    """Extract per-axis scale from an affine's 2x2 linear part.

    Column norms, not the raw diagonal: for a similarity transform the
    columns of R are the images of the unit x/y axes, so their lengths are
    exactly the x/y scale factors and stay correct under rotation (where the
    diagonal entries alone would read as cos(theta)-shrunk).
    """
    scale_x = float(np.linalg.norm(rotation[:, 0]))
    scale_y = float(np.linalg.norm(rotation[:, 1]))
    # A degenerate estimate (near-zero column) would collapse boxes to
    # nothing; fall back to "no scale change" rather than destroying state.
    if not np.isfinite(scale_x) or scale_x <= 1e-6:
        scale_x = 1.0
    if not np.isfinite(scale_y) or scale_y <= 1e-6:
        scale_y = 1.0
    return scale_x, scale_y


def warp_box(box: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Map an axis-aligned box through a 2x3 affine, re-axis-aligning it.

    All four corners are transformed and the enclosing axis-aligned box is
    taken, rather than transforming two corners: under any rotation the
    two-corner shortcut silently shrinks or inverts the box.
    """
    x1, y1, x2, y2 = map(float, box)
    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float64,
    )
    rotation = warp[:2, :2].astype(np.float64)
    translation = warp[:2, 2].astype(np.float64)
    moved = corners @ rotation.T + translation

    return np.array(
        [
            moved[:, 0].min(),
            moved[:, 1].min(),
            moved[:, 0].max(),
            moved[:, 1].max(),
        ],
        dtype=np.float32,
    )


def warp_kalman_filter(kf: cv2.KalmanFilter, warp: np.ndarray) -> None:
    """Re-express a 6-state (cx, cy, w, h, vx, vy) filter in the new frame.

    The three state pairs are NOT interchangeable, so Ultralytics' own
    `multi_gmc` (which block-diagonally rotates four position/velocity pairs)
    cannot be reused here -- this filter's middle pair is a SIZE, and
    rotating (w, h) as if it were a point would bleed width into height.
    Each pair therefore gets the treatment its physical meaning demands:

      (cx, cy)  full affine -- it is a point, so it both rotates and shifts.
      (w,  h)   scale only  -- an extent has no position, so translation must
                              not apply; it rescales with the axis factors.
      (vx, vy)  rotation only -- a velocity is a difference of two points, so
                              the translation cancels out of it exactly.

    The covariance is pushed through the same linear map (J P J^T) so it stays
    consistent with the mean it describes; warping one without the other would
    leave the filter's uncertainty pointing along the old frame's axes.
    """
    rotation = warp[:2, :2].astype(np.float32)
    translation = warp[:2, 2].astype(np.float32)
    scale_x, scale_y = affine_scale_factors(rotation)

    jacobian = np.zeros((6, 6), dtype=np.float32)
    jacobian[0:2, 0:2] = rotation
    jacobian[2, 2] = scale_x
    jacobian[3, 3] = scale_y
    jacobian[4:6, 4:6] = rotation

    for attribute in ("statePost", "statePre"):
        state = getattr(kf, attribute, None)
        if state is None or state.size != 6:
            continue
        warped = jacobian @ state.reshape(6, 1)
        warped[0, 0] += translation[0]
        warped[1, 0] += translation[1]
        setattr(kf, attribute, warped.astype(np.float32))

    for attribute in ("errorCovPost", "errorCovPre"):
        covariance = getattr(kf, attribute, None)
        if covariance is None or covariance.shape != (6, 6):
            continue
        setattr(
            kf,
            attribute,
            (jacobian @ covariance @ jacobian.T).astype(np.float32),
        )


def create_kalman_filter() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(6, 4)
    kf.transitionMatrix = np.array(
        [
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.eye(4, 6, dtype=np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.5
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf


class AppearanceExtractor:
    _transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def __init__(
        self,
        model_name: str,
        device: str,
        checkpoint: str | None = None,
        backend=None,
    ) -> None:
        # `backend` is the opt-in alternative embedding model (DINOv3 or the
        # OSNet+DINOv3 concat). When it is None -- the default -- this class
        # behaves exactly as it always has: build OSNet here and run it below.
        # See appearance_backends.py.
        self.device = torch.device(device)
        self.backend = backend
        if backend is not None:
            self.model = None
            self.dim = backend.dim
            return

        self.dim = 512
        self.model = torchreid.models.build_model(
            model_name,
            num_classes=1,
            pretrained=True,
        )
        if checkpoint is not None:
            torchreid.utils.load_pretrained_weights(self.model, str(checkpoint))
        self.model.eval().to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        frame_bgr: np.ndarray,
        boxes: list[np.ndarray],
    ) -> np.ndarray:
        if not boxes:
            return np.empty((0, self.dim), dtype=np.float32)

        if self.backend is not None:
            return self._call_backend(frame_bgr, boxes)

        height, width = frame_bgr.shape[:2]
        tensors: list[torch.Tensor] = []
        valid: list[bool] = []

        for box in boxes:
            x1, y1, x2, y2 = map(float, box)
            ix1 = int(max(0, min(width, math.floor(x1))))
            iy1 = int(max(0, min(height, math.floor(y1))))
            ix2 = int(max(0, min(width, math.ceil(x2))))
            iy2 = int(max(0, min(height, math.ceil(y2))))

            if ix2 <= ix1 or iy2 <= iy1:
                tensors.append(torch.zeros((3, 256, 128), dtype=torch.float32))
                valid.append(False)
                continue

            crop = frame_bgr[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                tensors.append(torch.zeros((3, 256, 128), dtype=torch.float32))
                valid.append(False)
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self._transform(crop_rgb))
            valid.append(True)

        batch = torch.stack(tensors).to(self.device)
        embeddings = self.model(batch)
        norms = torch.linalg.vector_norm(
            embeddings,
            dim=1,
            keepdim=True,
        ).clamp(min=1e-6)
        output = (embeddings / norms).cpu().numpy().astype(np.float32)

        for index, is_valid in enumerate(valid):
            if not is_valid:
                output[index] = 0.0

        return output

    @staticmethod
    def backend_crops(frame_bgr: np.ndarray, boxes: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
        """The crops the backend encodes (degenerate boxes skipped) and their box indices."""
        height, width = frame_bgr.shape[:2]
        crops: list[np.ndarray] = []
        crop_index: list[int] = []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = map(float, box)
            ix1 = int(max(0, min(width, math.floor(x1))))
            iy1 = int(max(0, min(height, math.floor(y1))))
            ix2 = int(max(0, min(width, math.ceil(x2))))
            iy2 = int(max(0, min(height, math.ceil(y2))))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            crop = frame_bgr[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue
            crops.append(crop)
            crop_index.append(index)
        return crops, crop_index

    def _call_backend(
        self,
        frame_bgr: np.ndarray,
        boxes: list[np.ndarray],
    ) -> np.ndarray:
        """Same crop-extraction contract as the OSNet path, different encoder.

        Degenerate boxes still yield an all-zero row rather than a garbage
        embedding: StableTracker treats a zero vector as "no appearance
        evidence" (it guards every comparison with np.any), so preserving that
        convention is what keeps the matching logic backend-agnostic.
        """
        crops, crop_index = self.backend_crops(frame_bgr, boxes)
        output = np.zeros((len(boxes), self.dim), dtype=np.float32)
        if not crops:
            return output

        encoded = self.backend.encode(crops)
        for row, index in enumerate(crop_index):
            output[index] = encoded[row]
        return output


class ManualTTATracker:
    """detect (with box-level TTA) + BoT-SORT, mirroring what YOLO.track(persist=True) does.

    Ultralytics' augment=True only works for PyTorch weights; exported engines ignore it. Here the
    detector is run on the original frame, a 0.83x horizontally flipped copy and a 0.67x copy (the
    scales/flips Ultralytics TTA uses), boxes are mapped back, merged with NMS, and handed to the
    same BoT-SORT tracker (botsort.yaml) that YOLO.track would use."""

    VARIANTS = ((1.0, False), (0.83, True), (0.67, False))

    def __init__(self, detector, conf: float, device: str, imgsz: int, nms_iou: float = 0.7):
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml("botsort.yaml")))
        cfg.device = device
        self.tracker = BOTSORT(args=cfg)  # exactly how ultralytics.trackers.track builds it
        self.detector, self.conf, self.device, self.imgsz, self.nms_iou = detector, conf, device, imgsz, nms_iou
        self.dog_classes = [k for k, v in detector.names.items() if str(v).lower() == "dog"]

    def __call__(self, frame: np.ndarray):
        from types import SimpleNamespace

        import torchvision
        from ultralytics.engine.results import Boxes

        h, w = frame.shape[:2]
        parts = []
        for scale, flip in self.VARIANTS:
            img = frame if scale == 1.0 else cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
            if flip:
                img = np.ascontiguousarray(img[:, ::-1])
            r = self.detector.predict(img, conf=self.conf, classes=self.dog_classes, verbose=False,
                                      device=self.device, imgsz=self.imgsz)[0]
            b = r.boxes.data.detach().cpu().numpy().astype(np.float32)  # x1 y1 x2 y2 conf cls
            if len(b):
                if flip:
                    b[:, [0, 2]] = img.shape[1] - b[:, [2, 0]]
                b[:, :4] /= scale
                parts.append(b)
        merged = np.concatenate(parts) if parts else np.zeros((0, 6), np.float32)
        if len(merged):
            keep = torchvision.ops.nms(torch.from_numpy(merged[:, :4]), torch.from_numpy(merged[:, 4]), self.nms_iou).numpy()
            merged = merged[keep]
        tracks = self.tracker.update(Boxes(torch.from_numpy(merged), (h, w)).cpu().numpy(), frame)
        if len(tracks) == 0:
            return SimpleNamespace(boxes=None)
        t = torch.as_tensor(np.asarray(tracks, dtype=np.float32))
        return SimpleNamespace(boxes=SimpleNamespace(xyxy=t[:, :4], id=t[:, 4], conf=t[:, 5], cls=t[:, 6]))


class StableTracker:
    def __init__(
        self,
        iou_threshold: float,
        max_age: int,
        appearance_extractor: AppearanceExtractor,
        appearance_threshold: float,
        appearance_ema: float,
        iou_weight: float,
        identity_verification_threshold: float | None = None,
        gmc_method: str | None = None,
        gmc_downscale: int = 2,
    ) -> None:
        self.next_stable_id = 0
        self.active: dict[int, dict[str, Any]] = {}
        self.lost: dict[int, dict[str, Any]] = {}
        self.yolo_to_stable: dict[int, int] = {}

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.extractor = appearance_extractor
        self.appearance_threshold = appearance_threshold
        self.appearance_ema = appearance_ema
        self.iou_weight = iou_weight
        # Fixes an ID-swap bug: Stage 1's "cheap path" below used to trust a
        # continuing BoT-SORT ID purely on IoU-vs-Kalman-prediction, with NO
        # appearance check at all. When two dogs pass close together, both
        # dogs' predicted boxes can satisfy that IoU check even if BoT-SORT's
        # own ID has silently flipped onto the wrong physical dog -- Stage 1
        # had no way to notice. This threshold gates that cheap path on
        # "does this detection still look like the dog we think it is";
        # below it, the detection is demoted to Stage 2, which properly
        # weighs both motion AND appearance via a global Hungarian match
        # instead of blindly trusting a single geometric check.
        self.identity_verification_threshold = (
            identity_verification_threshold
            if identity_verification_threshold is not None
            else appearance_threshold
        )

        # Global motion compensation. BoT-SORT already estimates camera
        # motion (botsort.yaml ships gmc_method: sparseOptFlow), but this
        # class throws that away: it re-does association from scratch with
        # its own Kalman filter, so every IoU below was being measured
        # between a detection in THIS camera frame and a track position
        # left over from the PREVIOUS one. On a moving camera those two
        # frames do not share a coordinate system, so a stationary dog can
        # fail the geometry checks purely because the camera panned.
        self.gmc = None
        if gmc_method and str(gmc_method).lower() not in {"none", "off"}:
            from ultralytics.trackers.utils.gmc import GMC

            self.gmc = GMC(method=str(gmc_method), downscale=int(gmc_downscale))

    def compensate_camera_motion(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Re-express every stored track in the current frame's coordinates.

        Called once per PROCESSED frame, before any matching. The pipeline
        samples odd frames only, so successive processed frames are two raw
        frames apart -- that is handled implicitly and must stay that way:
        GMC measures motion between consecutive `apply()` calls, so it is
        only ever fed frames that are actually processed. Calling it on
        skipped frames too would make it report half the true motion.
        """
        if self.gmc is None:
            return None

        warp = self.gmc.apply(frame_bgr)
        if warp is None:
            return None

        warp = np.asarray(warp, dtype=np.float32)
        if warp.shape != (2, 3) or not np.all(np.isfinite(warp)):
            return None

        # The first frame (and any frame GMC cannot solve) yields identity;
        # skip the work rather than warping everything by a no-op.
        if np.allclose(warp, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)):
            return warp

        for store in (self.active, self.lost):
            for data in store.values():
                data["box"] = warp_box(data["box"], warp)
                warp_kalman_filter(data["kf"], warp)

        return warp

    @staticmethod
    def cosine(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.dot(first, second))

    def update_embedding(
        self,
        stable_id: int,
        new_embedding: np.ndarray,
        store: dict[int, dict[str, Any]],
    ) -> None:
        old_embedding = store[stable_id].get("embedding")

        if old_embedding is None or not np.any(old_embedding):
            store[stable_id]["embedding"] = new_embedding
            return

        blended = (
            self.appearance_ema * new_embedding
            + (1.0 - self.appearance_ema) * old_embedding
        )
        norm = np.linalg.norm(blended)
        store[stable_id]["embedding"] = (
            blended / norm if norm > 0 else blended
        )

    def update(
        self,
        detections: list[tuple[np.ndarray, int]],
        frame_id: int,
        frame_bgr: np.ndarray,
    ) -> list[tuple[np.ndarray, int]]:
        # Bring every stored track into this frame's coordinate system first,
        # so both Stage 1's IoU check and Stage 2's Hungarian cost compare
        # like with like. No-op when GMC is disabled.
        self.compensate_camera_motion(frame_bgr)

        if detections:
            detection_embeddings = self.extractor(
                frame_bgr,
                [box for box, _ in detections],
            )
        else:
            detection_embeddings = np.empty(
                (0, getattr(self.extractor, "dim", 512)), dtype=np.float32
            )

        updated_this_frame: set[int] = set()
        matched_stable_ids: set[int] = set()
        unmatched_detections: list[tuple[np.ndarray, int]] = []
        unmatched_embeddings: list[np.ndarray | None] = []

        # Stage 1: retain BoT-SORT IDs when geometry remains plausible.
        for index, (box, yolo_id) in enumerate(detections):
            embedding = (
                detection_embeddings[index]
                if len(detection_embeddings)
                else None
            )
            stable_id = self.yolo_to_stable.get(yolo_id)

            if stable_id is None:
                unmatched_detections.append((box, yolo_id))
                unmatched_embeddings.append(embedding)
                continue

            existing = self.active.get(stable_id)
            if existing is not None and box_iou(box, existing["box"]) < 0.5:
                self.yolo_to_stable.pop(yolo_id, None)
                unmatched_detections.append((box, yolo_id))
                unmatched_embeddings.append(embedding)
                continue

            # Identity verification: geometry alone isn't enough when two
            # dogs are close together (see class docstring note above). If
            # this detection's appearance no longer matches what we know
            # this identity looks like, don't blindly continue it -- send
            # it to Stage 2 instead, where appearance is actually weighed.
            if existing is not None:
                existing_embedding = existing.get("embedding")
                if (
                    embedding is not None
                    and np.any(embedding)
                    and existing_embedding is not None
                    and np.any(existing_embedding)
                    and self.cosine(embedding, existing_embedding)
                    < self.identity_verification_threshold
                ):
                    self.yolo_to_stable.pop(yolo_id, None)
                    unmatched_detections.append((box, yolo_id))
                    unmatched_embeddings.append(embedding)
                    continue

            if stable_id in self.lost:
                self.active[stable_id] = self.lost.pop(stable_id)

            if stable_id in self.active:
                self.active[stable_id]["kf"].correct(xyxy_to_xywh(box))
                self.active[stable_id]["box"] = np.asarray(
                    box, dtype=np.float32
                )
                self.active[stable_id]["last_seen"] = frame_id
                self.active[stable_id]["yolo_id"] = yolo_id

                if embedding is not None and np.any(embedding):
                    self.update_embedding(stable_id, embedding, self.active)

                matched_stable_ids.add(stable_id)
                updated_this_frame.add(stable_id)

        # Stage 2: Hungarian IoU + appearance matching.
        if unmatched_detections:
            predicted_lost_boxes: dict[int, np.ndarray] = {}

            for stable_id, data in self.lost.items():
                prediction = data["kf"].predict()
                center_x, center_y, width, height = prediction[:4].flatten()
                predicted_lost_boxes[stable_id] = np.array(
                    [
                        center_x - width / 2.0,
                        center_y - height / 2.0,
                        center_x + width / 2.0,
                        center_y + height / 2.0,
                    ],
                    dtype=np.float32,
                )

            candidate_ids = [
                stable_id
                for stable_id in self.active
                if stable_id not in matched_stable_ids
            ] + list(self.lost.keys())

            candidate_store = {**self.active, **self.lost}
            candidate_boxes = [
                (
                    predicted_lost_boxes[stable_id]
                    if stable_id in predicted_lost_boxes
                    else np.asarray(
                        self.active[stable_id]["box"],
                        dtype=np.float32,
                    )
                )
                for stable_id in candidate_ids
            ]

            if candidate_ids:
                cost_matrix = np.ones(
                    (len(unmatched_detections), len(candidate_ids)),
                    dtype=np.float32,
                )

                for detection_index, ((box, _), embedding) in enumerate(
                    zip(unmatched_detections, unmatched_embeddings)
                ):
                    for candidate_index, (
                        stable_id,
                        candidate_box,
                    ) in enumerate(zip(candidate_ids, candidate_boxes)):
                        iou_cost = 1.0 - box_iou(box, candidate_box)
                        candidate_embedding = candidate_store[
                            stable_id
                        ].get("embedding")

                        if (
                            embedding is not None
                            and candidate_embedding is not None
                            and np.any(embedding)
                            and np.any(candidate_embedding)
                        ):
                            appearance_cost = 1.0 - self.cosine(
                                embedding,
                                candidate_embedding,
                            )
                            cost_matrix[
                                detection_index, candidate_index
                            ] = (
                                self.iou_weight * iou_cost
                                + (1.0 - self.iou_weight) * appearance_cost
                            )
                        else:
                            cost_matrix[
                                detection_index, candidate_index
                            ] = iou_cost

                detection_indices, candidate_indices = linear_sum_assignment(
                    cost_matrix
                )
                assignment = dict(
                    zip(
                        detection_indices.tolist(),
                        candidate_indices.tolist(),
                    )
                )

                remaining_detections: list[tuple[np.ndarray, int]] = []
                remaining_embeddings: list[np.ndarray | None] = []

                for detection_index, ((box, yolo_id), embedding) in enumerate(
                    zip(unmatched_detections, unmatched_embeddings)
                ):
                    if detection_index not in assignment:
                        remaining_detections.append((box, yolo_id))
                        remaining_embeddings.append(embedding)
                        continue

                    candidate_index = assignment[detection_index]
                    stable_id = candidate_ids[candidate_index]
                    candidate_box = candidate_boxes[candidate_index]
                    iou_score = box_iou(box, candidate_box)
                    candidate_embedding = candidate_store[
                        stable_id
                    ].get("embedding")

                    appearance_score = None
                    if (
                        embedding is not None
                        and candidate_embedding is not None
                        and np.any(embedding)
                        and np.any(candidate_embedding)
                    ):
                        appearance_score = self.cosine(
                            embedding,
                            candidate_embedding,
                        )

                    accepted = (
                        iou_score >= self.iou_threshold
                        or (
                            appearance_score is not None
                            and appearance_score
                            >= self.appearance_threshold
                        )
                    )

                    if not accepted:
                        remaining_detections.append((box, yolo_id))
                        remaining_embeddings.append(embedding)
                        continue

                    if stable_id in self.lost:
                        self.active[stable_id] = self.lost.pop(stable_id)

                    previous_yolo_id = self.active[stable_id].get("yolo_id")
                    if previous_yolo_id is not None:
                        self.yolo_to_stable.pop(
                            int(previous_yolo_id),
                            None,
                        )

                    self.active[stable_id]["kf"].correct(xyxy_to_xywh(box))
                    self.active[stable_id]["box"] = np.asarray(
                        box, dtype=np.float32
                    )
                    self.active[stable_id]["last_seen"] = frame_id
                    self.active[stable_id]["yolo_id"] = yolo_id
                    self.yolo_to_stable[yolo_id] = stable_id

                    if embedding is not None and np.any(embedding):
                        self.update_embedding(
                            stable_id,
                            embedding,
                            self.active,
                        )

                    matched_stable_ids.add(stable_id)
                    updated_this_frame.add(stable_id)

                unmatched_detections = remaining_detections
                unmatched_embeddings = remaining_embeddings

        # Stage 3: unmatched detections become new stable IDs.
        for (box, yolo_id), embedding in zip(
            unmatched_detections,
            unmatched_embeddings,
        ):
            stable_id = self.next_stable_id
            self.next_stable_id += 1

            kf = create_kalman_filter()
            center_x, center_y, width, height = xyxy_to_xywh(box).flatten()
            state = np.array(
                [
                    [center_x],
                    [center_y],
                    [width],
                    [height],
                    [0],
                    [0],
                ],
                dtype=np.float32,
            )
            kf.statePre = state.copy()
            kf.statePost = state.copy()

            self.yolo_to_stable[yolo_id] = stable_id
            self.active[stable_id] = {
                "kf": kf,
                "box": np.asarray(box, dtype=np.float32),
                "last_seen": frame_id,
                "yolo_id": yolo_id,
                "embedding": embedding,
            }
            updated_this_frame.add(stable_id)

        for stable_id, data in list(self.active.items()):
            if frame_id - int(data["last_seen"]) > self.max_age:
                self.lost[stable_id] = self.active.pop(stable_id)

        return [
            (
                np.asarray(
                    self.active[stable_id]["box"],
                    dtype=np.float32,
                ),
                stable_id,
            )
            for stable_id in updated_this_frame
            if stable_id in self.active
        ]


def main() -> None:
    args = parse_args()

    video = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tracks_path = output_dir / "tracks.csv"
    metadata_path = output_dir / "tracking_metadata.json"

    if not video.is_file():
        raise FileNotFoundError(video)

    if tracks_path.exists() and not args.force:
        raise FileExistsError(
            f"{tracks_path} already exists. Use --force to replace it."
        )

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    if not args.detector_weights.is_file():
        raise FileNotFoundError(args.detector_weights)

    if args.tracking_fps <= 0:
        raise ValueError("--tracking-fps must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    info = read_video_metadata(video)

    source_fps = float(info["fps"])
    frame_step = max(
        1,
        int(round(source_fps / args.tracking_fps)),
    )
    effective_tracking_fps = source_fps / frame_step
    if frame_step % 2 == 0:
        raise ValueError(
            "frame_step (from --tracking-fps) resolved to an even number "
            f"({frame_step}); combined with the odd-frames-only sampling "
            "below, no frame would ever be selected. Pick a --tracking-fps "
            "that resolves to an odd frame_step (1 processes every odd "
            "frame; the current default)."
        )

    print("=" * 78)
    print("DOG TRACKING")
    print("=" * 78)
    print("Video:", video)
    print(
        f"Metadata: {info['width']}x{info['height']}, "
        f"{source_fps:.3f} FPS, {info['frames']} frames"
    )
    print("Tracking frame step:", frame_step)
    print("Effective tracking FPS:", effective_tracking_fps)
    print("Detector:", args.detector_weights)
    print("Device:", args.device)

    started = time.perf_counter()

    print("Loading YOLO detector (fine-tuned dog model)...")
    detector = YOLO(str(args.detector_weights))
    manual_tta = (ManualTTATracker(detector, args.conf_threshold, args.device, args.imgsz)
                  if args.manual_tta else None)

    appearance_checkpoint = args.appearance_checkpoint
    if appearance_checkpoint is not None and str(appearance_checkpoint).lower() == "none":
        appearance_checkpoint = None

    backend = appearance_backends.build_backend(
        args.appearance_backend,
        args.device,
        osnet_model=args.appearance_model,
        osnet_checkpoint=appearance_checkpoint,
        dinov3_variant=args.dinov3_variant,
        dinov3_weight=args.dinov3_weight,
        dinov3_aspect=args.dinov3_aspect,
        dinov3_pooling=args.dinov3_pooling,
    )
    print(f"Loading appearance model ({args.appearance_backend})...")
    appearance_extractor = AppearanceExtractor(
        args.appearance_model,
        args.device,
        checkpoint=appearance_checkpoint,
        backend=backend,
    )

    if str(args.device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    stable_tracker = StableTracker(
        iou_threshold=args.iou_threshold,
        max_age=args.max_age,
        appearance_extractor=appearance_extractor,
        appearance_threshold=args.appearance_threshold,
        appearance_ema=args.appearance_ema,
        iou_weight=args.iou_weight,
        identity_verification_threshold=args.identity_verification_threshold,
        gmc_method=args.gmc_method,
        gmc_downscale=args.gmc_downscale,
    )

    # Prevent tracker state from a previous invocation from leaking in.
    try:
        detector.predictor = None
    except Exception:
        pass

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    rows: list[dict[str, Any]] = []
    sampled_frames = 0
    source_frame = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            # Only process odd-numbered raw frames (1, 3, 5, ...), not every
            # frame -- combined with frame_step (still respected, in case
            # --tracking-fps is ever lowered again) for the sampling cadence.
            if source_frame % frame_step == 0 and (args.all_frames or source_frame % 2 == 1):
                sampled_frames += 1

                if manual_tta is not None:
                    result = manual_tta(frame)
                else:
                    results = detector.track(
                        frame,
                        persist=True,
                        tracker="botsort.yaml",
                        conf=args.conf_threshold,
                        verbose=False,
                        device=args.device,
                        augment=args.augment,
                        imgsz=args.imgsz,
                    )
                    result = results[0]

                detections: list[tuple[np.ndarray, int]] = []
                confidence_lookup: dict[int, float] = {}

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    class_ids = result.boxes.cls.detach().cpu().numpy()
                    confidences = result.boxes.conf.detach().cpu().numpy()
                    detector_ids = result.boxes.id.detach().cpu().numpy()

                    for (
                        box,
                        class_id_value,
                        confidence_value,
                        detector_id_value,
                    ) in zip(
                        boxes,
                        class_ids,
                        confidences,
                        detector_ids,
                    ):
                        class_id = int(class_id_value)
                        class_name = (
                            detector.names.get(class_id, str(class_id))
                            if isinstance(detector.names, dict)
                            else detector.names[class_id]
                        )
                        confidence = float(confidence_value)

                        if (
                            str(class_name).lower() != "dog"
                            or confidence < args.conf_threshold
                        ):
                            continue

                        detector_id = int(detector_id_value)
                        detections.append(
                            (
                                np.asarray(box, dtype=np.float32),
                                detector_id,
                            )
                        )
                        confidence_lookup[detector_id] = confidence

                stable_results = stable_tracker.update(
                    detections,
                    source_frame,
                    frame,
                )

                stable_confidence: dict[int, float] = {}
                for detector_id, stable_id in (
                    stable_tracker.yolo_to_stable.items()
                ):
                    if detector_id in confidence_lookup:
                        stable_confidence[int(stable_id)] = (
                            confidence_lookup[detector_id]
                        )

                for box, stable_id in stable_results:
                    x1, y1, x2, y2 = map(float, box)
                    ix1 = int(
                        max(0, min(info["width"], math.floor(x1)))
                    )
                    iy1 = int(
                        max(0, min(info["height"], math.floor(y1)))
                    )
                    ix2 = int(
                        max(0, min(info["width"], math.ceil(x2)))
                    )
                    iy2 = int(
                        max(0, min(info["height"], math.ceil(y2)))
                    )

                    if ix2 <= ix1 or iy2 <= iy1:
                        continue

                    rows.append(
                        {
                            "frame": source_frame,
                            "time_sec": source_frame / source_fps,
                            "track_id": int(stable_id),
                            "x1": ix1,
                            "y1": iy1,
                            "x2": ix2,
                            "y2": iy2,
                            "confidence": stable_confidence.get(
                                int(stable_id), 0.0
                            ),
                            "class_name": "dog",
                        }
                    )

            source_frame += 1

            if source_frame % 500 == 0:
                print(
                    f"Read {source_frame}/{info['frames']} source frames "
                    f"({source_frame / source_fps:.1f}s)"
                )
    finally:
        capture.release()

    tracks = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    temporary_tracks = tracks_path.with_suffix(".csv.tmp")
    tracks.to_csv(temporary_tracks, index=False)
    temporary_tracks.replace(tracks_path)

    elapsed = time.perf_counter() - started
    metadata = {
        "status": "complete",
        "video": file_fingerprint(video),
        "detector_weights": str(args.detector_weights.resolve()),
        "device": args.device,
        "source_fps": source_fps,
        "source_width": int(info["width"]),
        "source_height": int(info["height"]),
        "source_frames_read": int(source_frame),
        "source_duration_seconds": source_frame / source_fps,
        "requested_tracking_fps": float(args.tracking_fps),
        "frame_step": int(frame_step),
        "effective_tracking_fps": float(effective_tracking_fps),
        "sampled_frames": int(sampled_frames),
        "track_rows": int(len(tracks)),
        "unique_track_ids": int(tracks["track_id"].nunique())
        if len(tracks)
        else 0,
        "parameters": {
            "conf_threshold": float(args.conf_threshold),
            "iou_threshold": float(args.iou_threshold),
            "max_age": int(args.max_age),
            "appearance_model": args.appearance_model,
            "appearance_backend": str(args.appearance_backend),
            "appearance_dim": int(appearance_extractor.dim),
            "dinov3_variant": str(args.dinov3_variant),
            "dinov3_weight": float(args.dinov3_weight),
            "dinov3_aspect": str(args.dinov3_aspect),
            "dinov3_pooling": str(args.dinov3_pooling),
            "appearance_threshold": float(args.appearance_threshold),
            "gmc_method": str(args.gmc_method),
            "gmc_downscale": int(args.gmc_downscale),
            "identity_verification_threshold": float(stable_tracker.identity_verification_threshold),
            "appearance_ema": float(args.appearance_ema),
            "iou_weight": float(args.iou_weight),
        },
        "elapsed_seconds": float(elapsed),
        "tracks_csv": str(tracks_path),
    }

    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)

    print()
    print("=" * 78)
    print("TRACKING COMPLETE")
    print("=" * 78)
    print("Track rows:", len(tracks))
    print(
        "Unique stable IDs:",
        tracks["track_id"].nunique() if len(tracks) else 0,
    )
    print("Tracks:", tracks_path)
    print("Metadata:", metadata_path)
    print("Elapsed minutes:", round(elapsed / 60.0, 2))


if __name__ == "__main__":
    main()
