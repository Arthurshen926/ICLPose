"""ALIKE score-head inference without producing a descriptor map."""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.real_image_observation_features import (
    spatially_diverse_detection_indices,
)


@dataclass(frozen=True)
class DetectedImagePoints:
    xy: np.ndarray
    scores: np.ndarray
    dispersions: np.ndarray
    image_sha256: str

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float32)
        scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        dispersions = np.asarray(self.dispersions, dtype=np.float32).reshape(-1)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2)")
        if scores.shape != (xy.shape[0],) or dispersions.shape != scores.shape:
            raise ValueError("scores and dispersions must have one value per point")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "dispersions", dispersions)


class AlikeDetectorOnly:
    """Run ALIKE's detector score channel and never materialize descriptors."""

    def __init__(
        self,
        *,
        device: str,
        matcha_repo: Path = Path("/root/matcha"),
        model_name: str = "alike-t",
    ) -> None:
        repo = Path(matcha_repo)
        if not repo.exists():
            raise FileNotFoundError(f"MATCHA repo not found: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from third_party.alike.alike import ALike, configs  # type: ignore

        if str(model_name) not in configs:
            raise ValueError(f"unknown ALIKE model: {model_name}")
        config = dict(configs[str(model_name)])
        config.update(device=str(device), top_k=0, scores_th=0.0, n_limit=0)
        self.device = torch.device(str(device))
        self.model_name = str(model_name)
        self.model_path = Path(str(config["model_path"]))
        self.model = ALike(**config).to(self.device).eval()

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "feature_type": "alike_detector_score_only",
            "model_name": self.model_name,
            "model_checkpoint": str(self.model_path),
            "model_checkpoint_sha256": file_sha256_short(self.model_path),
            "computes_descriptor_map": False,
            "returns_descriptors": False,
        }

    @torch.no_grad()
    def _score_map(self, image: torch.Tensor) -> torch.Tensor:
        """Mirror ALNet.forward while evaluating only convhead2's score row."""

        model = self.model
        _batch, _channels, height, width = image.shape
        padded_height = int(math.ceil(height / 32) * 32)
        padded_width = int(math.ceil(width / 32) * 32)
        if padded_height != height or padded_width != width:
            image = F.pad(
                image,
                (0, padded_width - width, 0, padded_height - height),
                mode="constant",
                value=0.0,
            )
        x1 = model.block1(image)
        x2 = model.block2(model.pool2(x1))
        x3 = model.block3(model.pool4(x2))
        x4 = model.block4(model.pool4(x3))
        fused = torch.cat(
            [
                model.gate(model.conv1(x1)),
                model.upsample2(model.gate(model.conv2(x2))),
                model.upsample8(model.gate(model.conv3(x3))),
                model.upsample32(model.gate(model.conv4(x4))),
            ],
            dim=1,
        )
        if not bool(model.single_head):
            fused = model.gate(model.convhead1(fused))
        head = model.convhead2
        score_logit = F.conv2d(
            fused,
            head.weight[-1:],
            None if head.bias is None else head.bias[-1:],
            stride=head.stride,
            padding=head.padding,
            dilation=head.dilation,
            groups=1,
        )
        return torch.sigmoid(score_logit[:, :, :height, :width])

    @torch.no_grad()
    def detect(
        self,
        image_path: Path,
        *,
        image_width: int,
        image_height: int,
        top_k: int = 512,
        candidate_top_k: int = 4096,
        nms_radius_px: float = 4.0,
        grid_rows: int = 4,
        grid_cols: int = 4,
        min_score: float | None = None,
        sub_pixel: bool = True,
    ) -> DetectedImagePoints:
        import cv2

        if int(top_k) <= 0 or int(candidate_top_k) < int(top_k):
            raise ValueError("candidate_top_k must be at least top_k > 0")
        payload = Path(image_path).read_bytes()
        image_hash = hashlib.sha256(payload).hexdigest()
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to decode image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (int(image_height), int(image_width)):
            rgb = cv2.resize(
                rgb,
                (int(image_width), int(image_height)),
                interpolation=cv2.INTER_AREA,
            )
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(
            self.device, dtype=torch.float32
        )
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        score_map = self._score_map(image)
        previous_top_k = int(self.model.dkd.top_k)
        try:
            self.model.dkd.top_k = min(
                int(candidate_top_k), int(image_height) * int(image_width)
            )
            normalized_xy, dispersions, scores = self.model.dkd.detect_keypoints(
                score_map, sub_pixel=bool(sub_pixel)
            )
        finally:
            self.model.dkd.top_k = previous_top_k
        points = normalized_xy[0]
        points = (points + 1.0) * 0.5 * points.new_tensor(
            [max(int(image_width) - 1, 1), max(int(image_height) - 1, 1)]
        )
        point_values = points.cpu().numpy().astype(np.float32, copy=False)
        score_values = scores[0].cpu().numpy().astype(np.float32, copy=False)
        if dispersions[0] is None:
            dispersion_values = np.full((len(point_values),), np.nan, dtype=np.float32)
        else:
            dispersion_values = (
                dispersions[0].cpu().numpy().astype(np.float32, copy=False)
            )
        selected = spatially_diverse_detection_indices(
            point_values,
            score_values,
            top_k=int(top_k),
            nms_radius_px=float(nms_radius_px),
            image_width=int(image_width),
            image_height=int(image_height),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            min_score=min_score,
        )
        return DetectedImagePoints(
            xy=point_values[selected],
            scores=score_values[selected],
            dispersions=dispersion_values[selected],
            image_sha256=image_hash,
        )
