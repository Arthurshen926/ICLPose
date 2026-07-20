"""Compact, pose-free RADIO-final image context used by candidate evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


RADIO_FINAL_CONTEXT_PCA_FORMAT = "radio_final_context_pca_v1"


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-8)


@dataclass(frozen=True)
class RadioFinalContextPcaCache:
    image_ids: np.ndarray
    image_sizes: np.ndarray
    summary_descriptors: np.ndarray
    global_descriptors: np.ndarray
    grid4_descriptors: np.ndarray
    metadata: dict[str, object]
    grid8_descriptors: np.ndarray | None = None
    grid16_descriptors: np.ndarray | None = None

    def __post_init__(self) -> None:
        image_ids = np.asarray(self.image_ids).astype(str).reshape(-1)
        sizes = np.asarray(self.image_sizes, dtype=np.int64).reshape(-1, 2)
        summary = np.asarray(self.summary_descriptors, dtype=np.float32)
        global_descriptors = np.asarray(self.global_descriptors, dtype=np.float32)
        grid4 = np.asarray(self.grid4_descriptors, dtype=np.float32)
        grid8 = (
            None
            if self.grid8_descriptors is None
            else np.asarray(self.grid8_descriptors, dtype=np.float32)
        )
        grid16 = (
            None
            if self.grid16_descriptors is None
            else np.asarray(self.grid16_descriptors, dtype=np.float32)
        )
        count = len(image_ids)
        if len(set(image_ids.tolist())) != count:
            raise ValueError("RADIO final context cache contains duplicate image IDs")
        if np.any(sizes <= 0) or sizes.shape != (count, 2):
            raise ValueError("RADIO final context image sizes are invalid")
        if summary.ndim != 2 or global_descriptors.shape != summary.shape:
            raise ValueError("RADIO final summary/global descriptors are not aligned")
        if grid4.shape != (count, 16, int(summary.shape[1])):
            raise ValueError("RADIO final grid4 descriptors are not aligned")
        if grid8 is not None and grid8.shape != (count, 64, int(summary.shape[1])):
            raise ValueError("RADIO final grid8 descriptors are not aligned")
        if grid16 is not None and grid16.shape != (count, 256, int(summary.shape[1])):
            raise ValueError("RADIO final grid16 descriptors are not aligned")
        for name, values in (
            ("summary", summary),
            ("global", global_descriptors),
            ("grid4", grid4),
            *(() if grid8 is None else (("grid8", grid8),)),
            *(() if grid16 is None else (("grid16", grid16),)),
        ):
            if np.any(~np.isfinite(values)):
                raise ValueError(f"RADIO final {name} descriptors are not finite")
            norms = np.linalg.norm(values, axis=-1)
            if np.max(np.abs(norms - 1.0)) > 5e-3:
                raise ValueError(f"RADIO final {name} descriptors are not L2 normalized")
        metadata = dict(self.metadata)
        if metadata.get("format") != RADIO_FINAL_CONTEXT_PCA_FORMAT:
            raise ValueError("unsupported RADIO final context PCA cache")
        if bool(metadata.get("pose_or_ground_truth_used", True)):
            raise ValueError("RADIO final context PCA cache must be pose/GT free")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "image_sizes", sizes)
        object.__setattr__(self, "summary_descriptors", summary)
        object.__setattr__(self, "global_descriptors", global_descriptors)
        object.__setattr__(self, "grid4_descriptors", grid4)
        object.__setattr__(self, "grid8_descriptors", grid8)
        object.__setattr__(self, "grid16_descriptors", grid16)
        object.__setattr__(self, "metadata", metadata)

    @property
    def descriptor_dim(self) -> int:
        return int(self.summary_descriptors.shape[1])

    @property
    def node_feature_dim(self) -> int:
        return 3 * self.descriptor_dim

    def _image_row_and_cells(
        self, image_id: str, xy: np.ndarray
    ) -> tuple[int, np.ndarray]:
        positions = getattr(self, "_position_by_image", None)
        if positions is None:
            positions = {value: row for row, value in enumerate(self.image_ids.tolist())}
            object.__setattr__(self, "_position_by_image", positions)
        row = positions.get(str(image_id))
        if row is None:
            raise KeyError(f"RADIO final context cache has no image {image_id!r}")
        coordinates = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        width, height = self.image_sizes[int(row)]
        columns = np.clip(
            np.floor(coordinates[:, 0] / max(float(width), 1.0) * 4.0).astype(np.int64),
            0,
            3,
        )
        rows = np.clip(
            np.floor(coordinates[:, 1] / max(float(height), 1.0) * 4.0).astype(np.int64),
            0,
            3,
        )
        return int(row), rows * 4 + columns

    def node_grid4_descriptors(self, image_id: str, xy: np.ndarray) -> np.ndarray:
        """Return only candidate-local grid4 descriptors.

        This is the production S1 accessor.  It intentionally cannot expose
        whole-image summary/global terms to a local candidate probe.
        """

        row, cells = self._image_row_and_cells(image_id, xy)
        return self.grid4_descriptors[int(row), cells].astype(np.float32)

    def image_grid_descriptors(self, image_id: str, *, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one image's pose-free spatial grid and its pixel dimensions.

        Callers must crop this grid around a query/support anchor.  Returning
        the grid rather than the legacy summary/global terms makes accidental
        whole-image retrieval use impossible in the structured-context path.
        """

        row, _ = self._image_row_and_cells(str(image_id), np.zeros((1, 2), dtype=np.float32))
        grids = {
            4: self.grid4_descriptors,
            8: self.grid8_descriptors,
            16: self.grid16_descriptors,
        }
        values = grids.get(int(grid_size))
        if values is None:
            raise ValueError(f"RADIO final context cache has no grid{int(grid_size)} descriptors")
        return (
            values[int(row)]
            .reshape(int(grid_size), int(grid_size), self.descriptor_dim)
            .astype(np.float32),
            self.image_sizes[int(row)].astype(np.int64),
        )

    def node_descriptors(self, image_id: str, xy: np.ndarray) -> np.ndarray:
        """Return the legacy concatenated context representation."""

        row, cells = self._image_row_and_cells(image_id, xy)
        count = len(cells)
        return np.concatenate(
            [
                np.repeat(self.summary_descriptors[int(row)][None], count, axis=0),
                np.repeat(self.global_descriptors[int(row)][None], count, axis=0),
                self.grid4_descriptors[int(row), cells],
            ],
            axis=1,
        ).astype(np.float32)


def load_radio_final_context_pca_cache(
    path: Path,
    *,
    expected_metadata: Mapping[str, object] | None = None,
) -> RadioFinalContextPcaCache:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if expected_metadata:
            mismatches = {
                key: {"expected": expected, "actual": metadata.get(key)}
                for key, expected in expected_metadata.items()
                if metadata.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"stale RADIO final context PCA cache: {json.dumps(mismatches, sort_keys=True)}"
                )
        return RadioFinalContextPcaCache(
            image_ids=np.asarray(data["image_ids"]),
            image_sizes=np.asarray(data["image_sizes"]),
            summary_descriptors=np.asarray(data["summary_descriptors"]),
            global_descriptors=np.asarray(data["global_descriptors"]),
            grid4_descriptors=np.asarray(data["grid4_descriptors"]),
            metadata=metadata,
            grid8_descriptors=(
                None
                if "grid8_descriptors" not in data.files
                else np.asarray(data["grid8_descriptors"])
            ),
            grid16_descriptors=(
                None
                if "grid16_descriptors" not in data.files
                else np.asarray(data["grid16_descriptors"])
            ),
        )
