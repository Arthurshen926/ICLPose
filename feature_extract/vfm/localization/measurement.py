"""Real-image RGB patch measurement adapters."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.localization.schemas import CoarseProposal, MeasurementResult
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import crop_rgb_window


def _rgb_chw_tensor(image: np.ndarray, *, name: str, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 3:
        raise ValueError(f"{name} must have shape (3, H, W)")
    return torch.as_tensor(arr[None], dtype=torch.float32, device=device)


def _proposal_xy_tensor(proposals: Sequence[CoarseProposal], *, side: str, device: torch.device) -> torch.Tensor:
    if side == "query":
        values = [item.query_xy for item in proposals]
    elif side == "reference":
        values = [item.reference_xy for item in proposals]
    else:
        raise ValueError("side must be query or reference")
    return torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32, device=device)


def _prediction_delta(prediction, *, head: str) -> torch.Tensor:
    name = str(head)
    if name == "gated" and getattr(prediction, "gated_mean_offset_xy", None) is not None:
        return prediction.gated_mean_offset_xy
    if name == "direct" and getattr(prediction, "direct_mean_offset_xy", None) is not None:
        return prediction.direct_mean_offset_xy
    if name in {"likelihood_mean", "mean", "gated"} and getattr(prediction, "mean_offset_xy", None) is not None:
        return prediction.mean_offset_xy
    raise ValueError(f"prediction does not provide offset head '{head}'")


def _prediction_uncertainty(
    prediction,
    *,
    count: int,
    uncertainty_scale: float = 1.0,
    uncertainty_floor_px: float = 0.0,
) -> list[float | None]:
    sigma = getattr(prediction, "direct_log_sigma_xy", None)
    if sigma is None:
        return [None for _ in range(int(count))]
    values = torch.exp(sigma.detach()).mean(dim=1).cpu().numpy().astype(np.float32, copy=False)
    values = values * float(uncertainty_scale) + float(uncertainty_floor_px)
    return [float(item) for item in values.tolist()]


@dataclass
class RGBPatchMeasurementAdapter:
    """Adapter from coarse proposals to real query/reference RGB measurements."""

    branch: object
    device: str = "cpu"
    prediction_head: str = "gated"
    batch_size: int = 128
    confidence_temperature: float = 1.0
    confidence_bias: float = 0.0
    uncertainty_scale: float = 1.0
    uncertainty_floor_px: float = 0.0
    use_amp: bool = False
    amp_dtype: str = "float16"
    tensor_cache_size: int = 0
    _reference_tensor_cache: OrderedDict[tuple[str, str], torch.Tensor] = field(default_factory=OrderedDict, init=False)

    def measure(
        self,
        query_rgb: np.ndarray,
        reference_rgb: np.ndarray,
        proposals: Sequence[CoarseProposal],
        **_kwargs,
    ) -> list[MeasurementResult]:
        grouped = self.measure_by_reference(
            query_rgb,
            {"reference": reference_rgb},
            {"reference": proposals},
        )
        return list(grouped.get("reference", []))

    def measure_by_reference(
        self,
        query_rgb: np.ndarray,
        reference_rgb_by_id: Mapping[str, np.ndarray],
        proposals_by_reference: Mapping[str, Sequence[CoarseProposal]],
    ) -> dict[str, list[MeasurementResult]]:
        """Measure proposals against multiple reference images with one batched branch schedule."""

        grouped = self.measure_many_by_reference(
            {"query": query_rgb},
            reference_rgb_by_id,
            {("query", str(reference_id)): proposals for reference_id, proposals in proposals_by_reference.items()},
        )
        return {
            str(reference_id): list(grouped.get(("query", str(reference_id)), []))
            for reference_id in proposals_by_reference
        }

    def measure_many_by_reference(
        self,
        query_rgb_by_id: Mapping[str, np.ndarray],
        reference_rgb_by_id: Mapping[str, np.ndarray],
        proposals_by_query_reference: Mapping[tuple[str, str], Sequence[CoarseProposal]],
    ) -> dict[tuple[str, str], list[MeasurementResult]]:
        """Measure proposals from multiple queries and reference images in larger branch batches."""

        grouped = {
            (str(key[0]), str(key[1])): list(values)
            for key, values in proposals_by_query_reference.items()
        }
        out: dict[tuple[str, str], list[MeasurementResult]] = {key: [] for key in grouped}
        ordered_pairs: list[tuple[str, str]] = []
        ordered_proposals: list[CoarseProposal] = []
        for pair, proposals in grouped.items():
            for proposal in proposals:
                ordered_pairs.append(pair)
                ordered_proposals.append(proposal)
        values = ordered_proposals
        if not values:
            return out
        torch_device = torch.device(self.device if torch.cuda.is_available() or not str(self.device).startswith("cuda") else "cpu")
        branch = self.branch.to(torch_device).eval()
        radius = float(getattr(branch, "crop_radius_px"))
        step = float(getattr(branch, "step_px"))
        batch_size = max(1, int(self.batch_size))
        with torch.no_grad():
            query_patch = self._crop_query_patches(
                values,
                ordered_pairs,
                query_rgb_by_id=query_rgb_by_id,
                device=torch_device,
                radius=radius,
                step=step,
            )
            reference_patch = self._crop_reference_patches(
                values,
                ordered_pairs,
                reference_rgb_by_id=reference_rgb_by_id,
                device=torch_device,
                radius=radius,
                step=step,
            )
            for start in range(0, len(values), batch_size):
                end = min(start + batch_size, len(values))
                batch = values[start:end]
                with self._amp_context(torch_device):
                    prediction = branch.forward_from_patches(
                        query_patch[start:end],
                        reference_patch[start:end],
                        prior_scale_px=None,
                    )
                delta = _prediction_delta(prediction, head=str(self.prediction_head)).detach().cpu().numpy().astype(np.float32, copy=False)
                confidence = self._calibrated_confidence(getattr(prediction, "dustbin_logit", None), count=len(batch))
                uncertainty = _prediction_uncertainty(
                    prediction,
                    count=len(batch),
                    uncertainty_scale=float(self.uncertainty_scale),
                    uncertainty_floor_px=float(self.uncertainty_floor_px),
                )
                for index, proposal in enumerate(batch):
                    global_index = int(start + index)
                    pair = ordered_pairs[global_index]
                    out[pair].append(
                        MeasurementResult(
                            proposal=proposal,
                            measured_query_xy=proposal.query_xy + delta[index],
                            measured_reference_xy=proposal.reference_xy,
                            confidence=float(confidence[index]),
                            uncertainty_px=uncertainty[index],
                        )
                    )
        return out

    def _calibrated_confidence(self, dustbin_logit: torch.Tensor | None, *, count: int) -> np.ndarray:
        if dustbin_logit is None:
            return np.ones((int(count),), dtype=np.float32)
        temperature = max(float(self.confidence_temperature), 1e-6)
        valid_logit = (-dustbin_logit.detach() + float(self.confidence_bias)) / temperature
        return torch.sigmoid(valid_logit).cpu().numpy().astype(np.float32, copy=False)

    def _amp_context(self, device: torch.device):
        if not bool(self.use_amp) or device.type != "cuda":
            return nullcontext()
        dtype = torch.float16 if str(self.amp_dtype) == "float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _reference_tensor(self, reference_id: str, image: np.ndarray, *, device: torch.device) -> torch.Tensor:
        key = (str(device), str(reference_id))
        cached = self._reference_tensor_cache.get(key)
        if cached is not None:
            self._reference_tensor_cache.move_to_end(key)
            return cached
        tensor = _rgb_chw_tensor(image, name=f"reference_rgb[{reference_id}]", device=device)
        if int(self.tensor_cache_size) > 0:
            self._reference_tensor_cache[key] = tensor
            self._reference_tensor_cache.move_to_end(key)
            while len(self._reference_tensor_cache) > int(self.tensor_cache_size):
                self._reference_tensor_cache.popitem(last=False)
        return tensor

    def _crop_query_patches(
        self,
        proposals: Sequence[CoarseProposal],
        ordered_pairs: Sequence[tuple[str, str]],
        *,
        query_rgb_by_id: Mapping[str, np.ndarray],
        device: torch.device,
        radius: float,
        step: float,
    ) -> torch.Tensor:
        query_indices: dict[str, list[int]] = {}
        for index, pair in enumerate(ordered_pairs):
            query_indices.setdefault(str(pair[0]), []).append(int(index))
        slots: list[torch.Tensor | None] = [None for _ in proposals]
        for query_id, indices in query_indices.items():
            if query_id not in query_rgb_by_id:
                raise KeyError(f"missing query RGB for '{query_id}'")
            query = _rgb_chw_tensor(query_rgb_by_id[query_id], name=f"query_rgb[{query_id}]", device=device)
            local = [proposals[index] for index in indices]
            qxy = _proposal_xy_tensor(local, side="query", device=device)
            patch, _grid = crop_rgb_window(
                query.expand(len(local), -1, -1, -1),
                qxy,
                radius_px=float(radius),
                step_px=float(step),
                image_width=int(query.shape[3]),
                image_height=int(query.shape[2]),
            )
            for local_index, global_index in enumerate(indices):
                slots[int(global_index)] = patch[local_index : local_index + 1]
        if any(slot is None for slot in slots):
            raise RuntimeError("internal measurement batching error: missing query patch")
        return torch.cat([slot for slot in slots if slot is not None], dim=0)

    def _crop_reference_patches(
        self,
        proposals: Sequence[CoarseProposal],
        ordered_pairs: Sequence[tuple[str, str]],
        *,
        reference_rgb_by_id: Mapping[str, np.ndarray],
        device: torch.device,
        radius: float,
        step: float,
    ) -> torch.Tensor:
        reference_indices: dict[str, list[int]] = {}
        for index, pair in enumerate(ordered_pairs):
            reference_indices.setdefault(str(pair[1]), []).append(int(index))
        slots: list[torch.Tensor | None] = [None for _ in proposals]
        for reference_id, indices in reference_indices.items():
            if reference_id not in reference_rgb_by_id:
                raise KeyError(f"missing reference RGB for '{reference_id}'")
            reference = self._reference_tensor(reference_id, reference_rgb_by_id[reference_id], device=device)
            local = [proposals[index] for index in indices]
            rxy = _proposal_xy_tensor(local, side="reference", device=device)
            patch, _grid = crop_rgb_window(
                reference.expand(len(local), -1, -1, -1),
                rxy,
                radius_px=float(radius),
                step_px=float(step),
                image_width=int(reference.shape[3]),
                image_height=int(reference.shape[2]),
            )
            for local_index, global_index in enumerate(indices):
                slots[int(global_index)] = patch[local_index : local_index + 1]
        if any(slot is None for slot in slots):
            raise RuntimeError("internal measurement batching error: missing reference patch")
        return torch.cat([slot for slot in slots if slot is not None], dim=0)
