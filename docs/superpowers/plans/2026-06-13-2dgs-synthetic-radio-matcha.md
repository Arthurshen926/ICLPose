# 2DGS Synthetic RADIO-MATCHA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a phase-1 synthetic training path where both MATCHA-like RADIO source and target images are rendered from 2DGS, so geometry correspondence can be trained and validated without real-image domain gap.

**Architecture:** Add a deterministic 2DGS synthetic pair sampler and manifest builder, then route `train_matcha_joint_streaming_model.py` to a synthetic builder when manifest metadata declares `pair_source=2dgs_synthetic`. Keep the existing real-query streaming path unchanged. Add a synthetic evaluation tool that reports model correspondence metrics first and PnP metrics from predicted source-to-target correspondences second.

**Tech Stack:** Python, NumPy, PyTorch, RADIO extractor, existing official 2DGS renderer, existing MATCHA joint training code, pytest.

---

## File Structure

- Create `feature_extract/vfm/matcha_synthetic_pairs.py`
  - Synthetic sampling dataclasses.
  - Deterministic pose jitter and relative-pose binning.
  - Manifest metadata parsing helpers.

- Create `feature_extract/tools/vfm/build_matcha_2dgs_synthetic_manifest.py`
  - CLI that writes tensor-free synthetic train/validation manifests.
  - Uses existing `MatchaStreamingPairManifest` and `MatchaStreamingPairRecord`.

- Modify `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`
  - Add `radio_matcha_2dgs_synthetic` train preset.
  - Add builder routing based on manifest metadata.
  - Add a synthetic builder that renders source and target from sampled 2DGS poses.

- Create `feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py`
  - Loads a trained joint model.
  - Builds held-out synthetic pairs.
  - Reports `_evaluate` model metrics and PnP metrics by synthetic pose bin.

- Create `tests/test_matcha_synthetic_pairs.py`
  - Deterministic sampling, bin labels, metadata parsing, CLI manifest checks.

- Modify `tests/test_matcha_streaming_manifest.py`
  - Train preset and builder routing smoke tests.

---

### Task 1: Synthetic Pair Sampling Module

**Files:**
- Create: `feature_extract/vfm/matcha_synthetic_pairs.py`
- Test: `tests/test_matcha_synthetic_pairs.py`

- [ ] **Step 1: Write the failing tests**

Add `tests/test_matcha_synthetic_pairs.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.cambridge_pose_lattice import pose_w2c_from_center_rotation
from feature_extract.vfm.matcha_synthetic_pairs import (
    SYNTHETIC_PAIR_SOURCE,
    SyntheticPairSamplingConfig,
    relative_pose_bin,
    sample_synthetic_pair_poses,
    synthetic_config_from_metadata,
)


def _pose(center_x: float = 0.0) -> np.ndarray:
    return pose_w2c_from_center_rotation(
        np.asarray([center_x, 0.0, 0.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
    )


def test_synthetic_config_from_metadata_parses_ranges() -> None:
    config = synthetic_config_from_metadata(
        {
            "pair_source": SYNTHETIC_PAIR_SOURCE,
            "synthetic_source_translation_range_m": [0.0, 0.02],
            "synthetic_source_rotation_range_deg": [0.0, 0.5],
            "synthetic_target_translation_range_m": [0.03, 0.10],
            "synthetic_target_rotation_range_deg": [1.0, 3.0],
            "synthetic_min_supervision_count": 128,
            "synthetic_min_overlap": 0.25,
        }
    )

    assert config.source_translation_range_m == (0.0, 0.02)
    assert config.source_rotation_range_deg == (0.0, 0.5)
    assert config.target_translation_range_m == (0.03, 0.10)
    assert config.target_rotation_range_deg == (1.0, 3.0)
    assert config.min_supervision_count == 128
    assert config.min_overlap == 0.25


def test_synthetic_config_rejects_wrong_pair_source() -> None:
    with pytest.raises(ValueError, match="pair_source"):
        synthetic_config_from_metadata({"pair_source": "real_query"})


def test_sample_synthetic_pair_poses_is_deterministic() -> None:
    config = SyntheticPairSamplingConfig(
        source_translation_range_m=(0.0, 0.02),
        source_rotation_range_deg=(0.0, 0.5),
        target_translation_range_m=(0.03, 0.10),
        target_rotation_range_deg=(1.0, 3.0),
    )
    first = sample_synthetic_pair_poses(_pose(), config=config, seed=11, key="seq0/frame.png")
    second = sample_synthetic_pair_poses(_pose(), config=config, seed=11, key="seq0/frame.png")

    assert np.allclose(first.source_pose_w2c, second.source_pose_w2c)
    assert np.allclose(first.target_pose_w2c, second.target_pose_w2c)
    assert 0.0 <= first.source_anchor_translation_m <= 0.02
    assert 0.03 <= first.target_source_translation_m <= 0.10
    assert 1.0 <= first.target_source_rotation_deg <= 3.0
    assert first.pose_bin == "small"


def test_relative_pose_bin_uses_translation_and_rotation() -> None:
    assert relative_pose_bin(0.03, 1.0) == "micro"
    assert relative_pose_bin(0.10, 3.0) == "small"
    assert relative_pose_bin(0.25, 6.0) == "medium"
    assert relative_pose_bin(0.50, 10.0) == "wide"
    assert relative_pose_bin(0.80, 12.0) == "out_of_range"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_matcha_synthetic_pairs.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'feature_extract.vfm.matcha_synthetic_pairs'`.

- [ ] **Step 3: Implement the sampling module**

Create `feature_extract/vfm/matcha_synthetic_pairs.py`:

```python
"""Synthetic 2DGS source/target pose sampling for MATCHA-style RADIO training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c, pose_w2c_from_center_rotation
from feature_extract.vfm.render_pose_protocol import render_pose_error_fields


SYNTHETIC_PAIR_SOURCE = "2dgs_synthetic"
SYNTHETIC_RANDOM_PAIR_TYPE = "S2DGS_RANDOM"
SYNTHETIC_RANDOM_PAIR_TYPE_ID = 100


@dataclass(frozen=True)
class SyntheticPairSamplingConfig:
    source_translation_range_m: tuple[float, float] = (0.0, 0.03)
    source_rotation_range_deg: tuple[float, float] = (0.0, 1.0)
    target_translation_range_m: tuple[float, float] = (0.0, 0.25)
    target_rotation_range_deg: tuple[float, float] = (0.0, 6.0)
    min_supervision_count: int = 128
    min_overlap: float = 0.20

    def __post_init__(self) -> None:
        for name in (
            "source_translation_range_m",
            "source_rotation_range_deg",
            "target_translation_range_m",
            "target_rotation_range_deg",
        ):
            lo, hi = tuple(float(v) for v in getattr(self, name))
            if lo < 0.0 or hi < lo:
                raise ValueError(f"{name} must be a non-negative [min, max] range")
            object.__setattr__(self, name, (lo, hi))
        if int(self.min_supervision_count) < 0:
            raise ValueError("min_supervision_count must be non-negative")
        if not 0.0 <= float(self.min_overlap) <= 1.0:
            raise ValueError("min_overlap must be in [0, 1]")


@dataclass(frozen=True)
class SyntheticPairPose:
    source_pose_w2c: np.ndarray
    target_pose_w2c: np.ndarray
    source_anchor_translation_m: float
    source_anchor_rotation_deg: float
    target_source_translation_m: float
    target_source_rotation_deg: float
    pose_bin: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_pose_w2c", np.asarray(self.source_pose_w2c, dtype=np.float64).reshape(4, 4))
        object.__setattr__(self, "target_pose_w2c", np.asarray(self.target_pose_w2c, dtype=np.float64).reshape(4, 4))


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{str(key)}".encode("utf8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFF


def _range_from_metadata(metadata: Mapping[str, object], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = metadata.get(key, default)
    if isinstance(value, str):
        parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        parts = [float(item) for item in value] if isinstance(value, Sequence) else []
    if len(parts) != 2:
        raise ValueError(f"{key} must contain exactly two numeric values")
    return (float(parts[0]), float(parts[1]))


def synthetic_config_from_metadata(metadata: Mapping[str, object]) -> SyntheticPairSamplingConfig:
    if str(metadata.get("pair_source", "")) != SYNTHETIC_PAIR_SOURCE:
        raise ValueError(f"synthetic manifest metadata must set pair_source={SYNTHETIC_PAIR_SOURCE}")
    return SyntheticPairSamplingConfig(
        source_translation_range_m=_range_from_metadata(metadata, "synthetic_source_translation_range_m", (0.0, 0.03)),
        source_rotation_range_deg=_range_from_metadata(metadata, "synthetic_source_rotation_range_deg", (0.0, 1.0)),
        target_translation_range_m=_range_from_metadata(metadata, "synthetic_target_translation_range_m", (0.0, 0.25)),
        target_rotation_range_deg=_range_from_metadata(metadata, "synthetic_target_rotation_range_deg", (0.0, 6.0)),
        min_supervision_count=int(metadata.get("synthetic_min_supervision_count", 128)),
        min_overlap=float(metadata.get("synthetic_min_overlap", 0.20)),
    )


def _sample_vector(rng: np.random.Generator, value_range: tuple[float, float]) -> np.ndarray:
    lo, hi = value_range
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    return direction * float(rng.uniform(float(lo), float(hi)))


def _axis_angle_to_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    angle = np.deg2rad(float(angle_deg))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _sample_rotation(rng: np.random.Generator, value_range: tuple[float, float]) -> np.ndarray:
    lo, hi = value_range
    magnitude = float(rng.uniform(float(lo), float(hi)))
    axis = rng.normal(size=3)
    if float(np.linalg.norm(axis)) <= 1e-12:
        axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return _axis_angle_to_rotation(axis, magnitude)


def _apply_world_perturbation(pose_w2c: np.ndarray, translation: np.ndarray, rotation_delta: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    center = camera_center_from_pose_w2c(pose) + np.asarray(translation, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_delta, dtype=np.float64).reshape(3, 3) @ pose[:3, :3]
    return pose_w2c_from_center_rotation(center, rotation)


def relative_pose_bin(translation_m: float, rotation_deg: float) -> str:
    t = float(translation_m)
    r = float(rotation_deg)
    if t <= 0.03 and r <= 1.0:
        return "micro"
    if t <= 0.10 and r <= 3.0:
        return "small"
    if t <= 0.25 and r <= 6.0:
        return "medium"
    if t <= 0.50 and r <= 10.0:
        return "wide"
    return "out_of_range"


def sample_synthetic_pair_poses(
    anchor_pose_w2c: np.ndarray,
    *,
    config: SyntheticPairSamplingConfig,
    seed: int,
    key: str,
) -> SyntheticPairPose:
    rng = np.random.default_rng(_stable_seed(int(seed), str(key)))
    anchor = np.asarray(anchor_pose_w2c, dtype=np.float64).reshape(4, 4)
    source_pose = _apply_world_perturbation(
        anchor,
        _sample_vector(rng, config.source_translation_range_m),
        _sample_rotation(rng, config.source_rotation_range_deg),
    )
    target_pose = _apply_world_perturbation(
        source_pose,
        _sample_vector(rng, config.target_translation_range_m),
        _sample_rotation(rng, config.target_rotation_range_deg),
    )
    source_t, source_r = render_pose_error_fields(source_pose, anchor)
    target_t, target_r = render_pose_error_fields(target_pose, source_pose)
    return SyntheticPairPose(
        source_pose_w2c=source_pose,
        target_pose_w2c=target_pose,
        source_anchor_translation_m=float(source_t),
        source_anchor_rotation_deg=float(source_r),
        target_source_translation_m=float(target_t),
        target_source_rotation_deg=float(target_r),
        pose_bin=relative_pose_bin(target_t, target_r),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_matcha_synthetic_pairs.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add feature_extract/vfm/matcha_synthetic_pairs.py tests/test_matcha_synthetic_pairs.py
git commit -m "feat: add 2dgs synthetic matcha pair sampler"
```

---

### Task 2: Synthetic Manifest Builder CLI

**Files:**
- Create: `feature_extract/tools/vfm/build_matcha_2dgs_synthetic_manifest.py`
- Modify: `tests/test_matcha_synthetic_pairs.py`

- [ ] **Step 1: Write the failing CLI tests**

Append to `tests/test_matcha_synthetic_pairs.py`:

```python
import json
from pathlib import Path

from feature_extract.tools.vfm.build_matcha_2dgs_synthetic_manifest import parse_args as parse_synthetic_manifest_args
from feature_extract.tools.vfm.build_matcha_2dgs_synthetic_manifest import main as build_synthetic_manifest_main


def _write_token_manifest(path: Path) -> None:
    payload = {
        "records": [
            {
                "image_id": "seq0/frame00001.png",
                "token_path": "seq0__frame00001.npz",
                "layers": [{"name": "radio_dual", "model": "radio", "layer": "dual", "channels": 1280, "stride": 16}],
                "split": "train",
                "scene": "OldHospital",
            },
            {
                "image_id": "seq0/frame00002.png",
                "token_path": "seq0__frame00002.npz",
                "layers": [{"name": "radio_dual", "model": "radio", "layer": "dual", "channels": 1280, "stride": 16}],
                "split": "train",
                "scene": "OldHospital",
            },
        ],
    }
    path.write_text(json.dumps(payload) + "\n")


def test_synthetic_manifest_cli_defaults_to_random_pair_type() -> None:
    args = parse_synthetic_manifest_args(
        [
            "--query_manifest",
            "train_manifest.json",
            "--output_manifest",
            "synthetic.json",
            "--summary_json",
            "summary.json",
            "--split_name",
            "train",
        ]
    )

    assert args.pair_type == "S2DGS_RANDOM"
    assert args.target_translation_range_m == "0.0,0.25"
    assert args.target_rotation_range_deg == "0.0,6.0"


def test_build_synthetic_manifest_writes_pair_source_metadata(tmp_path: Path) -> None:
    query_manifest = tmp_path / "train_manifest.json"
    output_manifest = tmp_path / "synthetic_train.json"
    summary_json = tmp_path / "summary.json"
    _write_token_manifest(query_manifest)

    build_synthetic_manifest_main(
        [
            "--query_manifest",
            str(query_manifest),
            "--output_manifest",
            str(output_manifest),
            "--summary_json",
            str(summary_json),
            "--split_name",
            "train",
            "--seed",
            "123",
            "--target_translation_range_m",
            "0.03,0.10",
            "--target_rotation_range_deg",
            "1.0,3.0",
        ]
    )

    payload = json.loads(output_manifest.read_text())
    assert payload["pair_count"] == 2
    assert payload["pair_type_counts"] == {"S2DGS_RANDOM": 2}
    assert payload["metadata"]["pair_source"] == "2dgs_synthetic"
    assert payload["metadata"]["synthetic_target_translation_range_m"] == [0.03, 0.10]
    assert payload["records"][0]["pair_type_id"] == 100
    assert json.loads(summary_json.read_text())["stage"] == "matcha_2dgs_synthetic_manifest_builder"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_matcha_synthetic_pairs.py::test_synthetic_manifest_cli_defaults_to_random_pair_type tests/test_matcha_synthetic_pairs.py::test_build_synthetic_manifest_writes_pair_source_metadata -q
```

Expected: fail with `ModuleNotFoundError` for `feature_extract.tools.vfm.build_matcha_2dgs_synthetic_manifest`.

- [ ] **Step 3: Implement the manifest builder**

Create `feature_extract/tools/vfm/build_matcha_2dgs_synthetic_manifest.py`:

```python
"""Build tensor-free 2DGS synthetic MATCHA source/target pair manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _select_records
from feature_extract.vfm.matcha_streaming_manifest import MatchaStreamingPairManifest, MatchaStreamingPairRecord
from feature_extract.vfm.matcha_synthetic_pairs import (
    SYNTHETIC_PAIR_SOURCE,
    SYNTHETIC_RANDOM_PAIR_TYPE,
    SYNTHETIC_RANDOM_PAIR_TYPE_ID,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_range(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError("range arguments must be formatted as min,max")
    if values[0] < 0.0 or values[1] < values[0]:
        raise ValueError("range arguments must satisfy 0 <= min <= max")
    return [float(values[0]), float(values[1])]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--pair_type", default=SYNTHETIC_RANDOM_PAIR_TYPE, choices=(SYNTHETIC_RANDOM_PAIR_TYPE,))
    parser.add_argument("--source_translation_range_m", default="0.0,0.03")
    parser.add_argument("--source_rotation_range_deg", default="0.0,1.0")
    parser.add_argument("--target_translation_range_m", default="0.0,0.25")
    parser.add_argument("--target_rotation_range_deg", default="0.0,6.0")
    parser.add_argument("--min_supervision_count", type=int, default=128)
    parser.add_argument("--min_overlap", type=float, default=0.20)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--view_selection", default="prefix", choices=("prefix", "uniform"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    _parse_range(str(args.source_translation_range_m))
    _parse_range(str(args.source_rotation_range_deg))
    _parse_range(str(args.target_translation_range_m))
    _parse_range(str(args.target_rotation_range_deg))
    if int(args.min_supervision_count) < 0:
        raise ValueError("--min_supervision_count must be non-negative")
    if not 0.0 <= float(args.min_overlap) <= 1.0:
        raise ValueError("--min_overlap must be in [0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = TokenBankManifest.from_json(Path(args.query_manifest))
    records = _select_records(
        source.records,
        int(args.max_queries),
        str(args.view_selection),
        start_index=int(args.start_index),
    )
    pair_records = tuple(
        MatchaStreamingPairRecord(
            query_id=str(record.image_id),
            split=str(args.split_name),
            pair_type=str(args.pair_type),
            pair_type_id=SYNTHETIC_RANDOM_PAIR_TYPE_ID,
            record_index=int(index),
            pair_index=0,
            seed=int(args.seed),
        )
        for index, record in enumerate(records)
    )
    metadata = {
        "pair_source": SYNTHETIC_PAIR_SOURCE,
        "source_query_manifest": str(args.query_manifest),
        "view_selection": str(args.view_selection),
        "start_index": int(args.start_index),
        "max_queries": int(args.max_queries),
        "seed": int(args.seed),
        "synthetic_source_translation_range_m": _parse_range(str(args.source_translation_range_m)),
        "synthetic_source_rotation_range_deg": _parse_range(str(args.source_rotation_range_deg)),
        "synthetic_target_translation_range_m": _parse_range(str(args.target_translation_range_m)),
        "synthetic_target_rotation_range_deg": _parse_range(str(args.target_rotation_range_deg)),
        "synthetic_min_supervision_count": int(args.min_supervision_count),
        "synthetic_min_overlap": float(args.min_overlap),
    }
    manifest = MatchaStreamingPairManifest(records=pair_records, metadata=metadata)
    manifest.to_json(Path(args.output_manifest))
    summary = manifest.to_dict()
    summary.pop("records", None)
    summary["stage"] = "matcha_2dgs_synthetic_manifest_builder"
    summary["outputs"] = {"manifest": str(args.output_manifest)}
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_matcha_synthetic_pairs.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add feature_extract/tools/vfm/build_matcha_2dgs_synthetic_manifest.py tests/test_matcha_synthetic_pairs.py
git commit -m "feat: build 2dgs synthetic matcha manifests"
```

---

### Task 3: Synthetic Training Builder and Preset

**Files:**
- Modify: `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`
- Modify: `tests/test_matcha_streaming_manifest.py`

- [ ] **Step 1: Write failing routing and preset tests**

Append to `tests/test_matcha_streaming_manifest.py`:

```python
from feature_extract.tools.vfm.train_matcha_joint_streaming_model import _builder_class_for_manifest
from feature_extract.vfm.matcha_streaming_manifest import MatchaStreamingPairRecord


def test_streaming_training_radio_matcha_2dgs_synthetic_preset_uses_local_window_path() -> None:
    args = parse_streaming_train_args(
        [
            "--streaming_manifest",
            "synthetic_train.json",
            "--validation_streaming_manifest",
            "synthetic_val.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--output_model",
            "adapter.pt",
            "--summary_json",
            "summary.json",
            "--matcha_train_preset",
            "radio_matcha_2dgs_synthetic",
        ]
    )

    assert args.offset_loss_weight == 0.25
    assert args.pair_confidence_loss_weight == 0.1
    assert args.dense_heatmap_loss_weight == 0.25
    assert args.local_window_fine_loss_weight == 0.5
    assert args.patch_corr_fine_loss_weight == 0.0
    assert args.multiview_supervision_support_views == 0


def test_synthetic_manifest_routes_to_synthetic_builder() -> None:
    manifest = MatchaStreamingPairManifest(
        records=(
            MatchaStreamingPairRecord(
                query_id="seq0/frame00001.png",
                split="train",
                pair_type="S2DGS_RANDOM",
                pair_type_id=100,
                record_index=0,
                pair_index=0,
                seed=7,
            ),
        ),
        metadata={
            "pair_source": "2dgs_synthetic",
            "source_query_manifest": "train_manifest.json",
        },
    )

    assert _builder_class_for_manifest(manifest).__name__ == "SyntheticStreamingPairBuilder"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_matcha_streaming_manifest.py::test_streaming_training_radio_matcha_2dgs_synthetic_preset_uses_local_window_path tests/test_matcha_streaming_manifest.py::test_synthetic_manifest_routes_to_synthetic_builder -q
```

Expected: fail because the train preset and `_builder_class_for_manifest` are not defined.

- [ ] **Step 3: Add train preset parsing**

In `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`, extend `--matcha_train_preset`:

```python
parser.add_argument(
    "--matcha_train_preset",
    default="none",
    choices=("none", "radio_matcha_patch_corr", "radio_matcha_2dgs_synthetic"),
)
```

After the existing `radio_matcha_patch_corr` block, add:

```python
    if str(args.matcha_train_preset) == "radio_matcha_2dgs_synthetic":
        args.fine_supervision_source = "render_subcell_stratified"
        args.multiview_supervision_support_views = 0
        args.multiview_supervision_min_support_views = 0
        args.offset_loss_weight = 0.25
        args.pair_confidence_loss_weight = 0.1
        args.dense_heatmap_loss_weight = 0.25
        args.rgb_keypoint_loss_weight = 0.25
        args.local_window_fine_loss_weight = 0.5
        args.patch_correlation_loss_weight = 0.0
        args.patch_corr_fine_loss_weight = 0.0
```

- [ ] **Step 4: Add builder routing**

Import synthetic helpers near the other imports:

```python
from feature_extract.vfm.matcha_synthetic_pairs import (
    SYNTHETIC_PAIR_SOURCE,
    sample_synthetic_pair_poses,
    synthetic_config_from_metadata,
)
```

Add this helper above `main`:

```python
def _builder_class_for_manifest(manifest: MatchaStreamingPairManifest):
    if str(manifest.metadata.get("pair_source", "")) == SYNTHETIC_PAIR_SOURCE:
        return SyntheticStreamingPairBuilder
    return StreamingPairBuilder
```

Change builder creation in `main`:

```python
builder_cls = _builder_class_for_manifest(manifest)
builder = builder_cls(args, manifest)
val_builder = _builder_class_for_manifest(val_manifest)(args, val_manifest) if val_manifest is not None else None
```

- [ ] **Step 5: Add synthetic builder class**

Add this class immediately after `StreamingPairBuilder`:

```python
class SyntheticStreamingPairBuilder(StreamingPairBuilder):
    def __init__(self, args: argparse.Namespace, manifest: MatchaStreamingPairManifest) -> None:
        super().__init__(args, manifest)
        self.synthetic_config = synthetic_config_from_metadata(manifest.metadata)
        self.query_cache_dir = None

    def _support_geometry_views(self, *, query_id: str, target_pose_w2c: np.ndarray) -> tuple[MatchaGeometryView, ...]:
        return tuple()

    def build(self, record: MatchaStreamingPairRecord) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
        anchor = self.gt_by_query.get(record.query_id)
        if anchor is None:
            raise KeyError(f"missing anchor pose for {record.query_id}")
        pair_pose = sample_synthetic_pair_poses(
            anchor.pose_w2c,
            config=self.synthetic_config,
            seed=int(record.seed),
            key=f"{record.query_id}:{record.record_index}:{record.pair_index}",
        )
        source_rgb, source_depth, source_alpha = _load_or_render_rgb_depth_cache(
            cache_path=None,
            render_fn=lambda pose=pair_pose.source_pose_w2c: _render_rgb_and_depth(
                self.rgb_source,
                None,
                pose_w2c=pose,
                camera=self.camera,
                config=self.query_depth_config,
                renderer="official_2dgs",
                device=self.builder_device,
            ),
            skip_existing=False,
        )
        target_rgb, target_depth, target_alpha = _load_or_render_rgb_depth_cache(
            cache_path=None,
            render_fn=lambda pose=pair_pose.target_pose_w2c: _render_rgb_and_depth(
                self.rgb_source,
                None,
                pose_w2c=pose,
                camera=self.render_camera,
                config=self.render_config,
                renderer="official_2dgs",
                device=self.builder_device,
            ),
            skip_existing=False,
        )
        query_feature = _extract_matcha_joint_feature_from_rgb(
            source_rgb,
            self.radio,
            feature_mode=str(self.args.feature_mode),
            fine_intermediate_index=int(self.args.radio_fine_intermediate_index),
            coarse_source=str(self.args.radio_coarse_source),
            coarse_intermediate_index=int(self.args.radio_coarse_intermediate_index),
        )
        render_feature = _extract_matcha_joint_feature_from_rgb(
            target_rgb,
            self.radio,
            feature_mode=str(self.args.feature_mode),
            fine_intermediate_index=int(self.args.radio_fine_intermediate_index),
            coarse_source=str(self.args.radio_coarse_source),
            coarse_intermediate_index=int(self.args.radio_coarse_intermediate_index),
        )
        query_feature = maybe_fuse_feature_map(
            query_feature,
            mode=str(self.args.feature_fusion_mode),
            radius=int(self.args.feature_fusion_radius),
            temperature=float(self.args.feature_fusion_temperature),
            alpha=float(self.args.feature_fusion_alpha),
            device=self.builder_device,
        )
        render_feature = maybe_fuse_feature_map(
            render_feature,
            mode=str(self.args.feature_fusion_mode),
            radius=int(self.args.feature_fusion_radius),
            temperature=float(self.args.feature_fusion_temperature),
            alpha=float(self.args.feature_fusion_alpha),
            device=self.builder_device,
        )
        query_labels, query_kp_stats, _query_kp_xy = _build_alike_label_map(
            self.keypoint_extractor,
            source_rgb,
            feature_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
        )
        render_labels, render_kp_stats, render_seed_xy = _build_alike_label_map(
            self.keypoint_extractor,
            target_rgb,
            feature_hw=(int(render_feature.shape[1]), int(render_feature.shape[2])),
        )
        if str(self.args.fine_supervision_source) == "render_alike" and render_seed_xy is not None and render_seed_xy.shape[0] > 0:
            fine_seed_xy = render_seed_xy
        elif str(self.args.fine_supervision_source) == "render_subcell":
            fine_seed_xy = _render_subcell_seed_xy(
                image_width=int(self.render_camera.width),
                image_height=int(self.render_camera.height),
                grid_width=int(render_feature.shape[2]),
                grid_height=int(render_feature.shape[1]),
                seed=int(record.seed),
            )
        elif str(self.args.fine_supervision_source) == "render_subcell_stratified":
            fine_seed_xy = _render_subcell_stratified_seed_xy(
                image_width=int(self.render_camera.width),
                image_height=int(self.render_camera.height),
                grid_width=int(render_feature.shape[2]),
                grid_height=int(render_feature.shape[1]),
                seed=int(record.seed),
                offset_bins=int(self.supervision_config.offset_bins),
            )
        else:
            fine_seed_xy = None

        def build_supervision(render_seed_xy):
            return build_matcha_coarse_supervision(
                render_depth=target_depth,
                query_depth=source_depth,
                render_alpha=target_alpha,
                query_alpha=source_alpha,
                render_camera=self.render_camera,
                query_camera=self.camera,
                render_pose_w2c=pair_pose.target_pose_w2c,
                query_pose_w2c=pair_pose.source_pose_w2c,
                render_grid_hw=(int(render_feature.shape[1]), int(render_feature.shape[2])),
                query_grid_hw=(int(query_feature.shape[1]), int(query_feature.shape[2])),
                render_seed_xy=render_seed_xy,
                config=self.supervision_config,
            )

        supervision = build_supervision(None)
        if supervision.count < int(self.synthetic_config.min_supervision_count):
            raise ValueError(f"synthetic pair supervision count {supervision.count} below configured minimum")
        fine_seed_supervision = build_supervision(fine_seed_xy) if fine_seed_xy is not None else None
        fine_seed_transfer_count = 0
        if fine_seed_supervision is not None and bool(self.args.merge_fine_labels_into_coarse):
            supervision, fine_seed_transfer_count = merge_fine_labels_by_cell_pair(supervision, fine_seed_supervision)
        fine_render_depth = None
        fine_validity_weight = None
        if fine_seed_supervision is not None:
            depth_values, depth_valid = _sample_depth(
                target_depth,
                fine_seed_supervision.render_xy,
                image_width=int(self.render_camera.width),
                image_height=int(self.render_camera.height),
            )
            fine_render_depth = np.where(depth_valid, depth_values, np.nan).astype(np.float32, copy=False)
            fine_validity_weight = (
                np.asarray(fine_seed_supervision.confidence_targets, dtype=np.float32).reshape(-1)
                * depth_valid.astype(np.float32, copy=False)
            )
        joint = build_matcha_joint_index_training_set_from_maps(
            query_feature,
            render_feature,
            supervision,
            fine_supervision=fine_seed_supervision,
            fine_render_depth=fine_render_depth,
            fine_validity_weight=fine_validity_weight,
            query_rgb=source_rgb,
            render_rgb=target_rgb,
            query_keypoint_label_map=query_labels,
            render_keypoint_label_map=render_labels,
            hard_negatives_per_match=int(self.args.hard_negatives_per_match),
            roundtrip_heatmap_threshold_px=float(self.args.roundtrip_heatmap_threshold_px),
        )
        object.__setattr__(joint, "pair_type_ids", np.asarray([int(record.pair_type_id)], dtype=np.int64))
        object.__setattr__(joint, "pair_type_names", np.asarray([str(record.pair_type)], dtype=object))
        object.__setattr__(joint, "pair_query_ids", np.asarray([str(record.query_id)], dtype=object))
        object.__setattr__(joint, "pair_split_names", np.asarray([str(record.split)], dtype=object))
        object.__setattr__(joint, "pair_candidate_ids", np.asarray([""], dtype=object))
        object.__setattr__(joint, "pair_translation_errors_m", np.asarray([pair_pose.target_source_translation_m], dtype=np.float32))
        object.__setattr__(joint, "pair_rotation_errors_deg", np.asarray([pair_pose.target_source_rotation_deg], dtype=np.float32))
        row = {
            "query_id": str(record.query_id),
            "split": str(record.split),
            "pair_type": str(record.pair_type),
            "synthetic_pose_bin": str(pair_pose.pose_bin),
            "synthetic_source_anchor_translation_m": float(pair_pose.source_anchor_translation_m),
            "synthetic_source_anchor_rotation_deg": float(pair_pose.source_anchor_rotation_deg),
            "synthetic_target_source_translation_m": float(pair_pose.target_source_translation_m),
            "synthetic_target_source_rotation_deg": float(pair_pose.target_source_rotation_deg),
            "sample_count": int(joint.coarse_fine_samples.sample_count),
            "query_feature_shape": list(query_feature.shape),
            "render_feature_shape": list(render_feature.shape),
            "render_depth_valid_fraction": float(np.mean(np.isfinite(target_depth) & (target_depth > 0.0))),
            "render_alpha_mean": float(np.mean(target_alpha)),
            "supervision_source": str(supervision.source),
            "fine_supervision_source": str(self.args.fine_supervision_source),
            "render_seed_count": 0 if fine_seed_xy is None else int(fine_seed_xy.shape[0]),
            "fine_seed_transfer_count": int(fine_seed_transfer_count),
            "fine_seed_transfer_fraction": float(fine_seed_transfer_count) / max(float(supervision.count), 1.0),
            "fine_aux_supervision_count": int(0 if fine_seed_supervision is None else fine_seed_supervision.count),
            "query_offset_label_entropy_bits": _label_entropy_bits(supervision.query_offset_labels),
            "render_offset_label_entropy_bits": _label_entropy_bits(supervision.render_offset_labels),
            "query_offset_center_fraction": _label_fraction(supervision.query_offset_labels, 36),
            "render_offset_center_fraction": _label_fraction(supervision.render_offset_labels, 36),
            "multiview_support_view_count": 0,
            "supervision_no_match_count": int(getattr(supervision, "no_match_count", 0)),
            "query_keypoint_positive_count": int(query_kp_stats.get("positive_count", 0)),
            "render_keypoint_positive_count": int(render_kp_stats.get("positive_count", 0)),
            "roundtrip_median_px": float(np.median(supervision.roundtrip_errors_px)) if supervision.count else None,
        }
        return joint, row
```

- [ ] **Step 6: Run routing tests**

Run:

```bash
pytest tests/test_matcha_streaming_manifest.py::test_streaming_training_radio_matcha_2dgs_synthetic_preset_uses_local_window_path tests/test_matcha_streaming_manifest.py::test_synthetic_manifest_routes_to_synthetic_builder -q
```

Expected: both tests pass.

- [ ] **Step 7: Commit**

```bash
git add feature_extract/tools/vfm/train_matcha_joint_streaming_model.py tests/test_matcha_streaming_manifest.py
git commit -m "feat: route synthetic 2dgs pairs through matcha training"
```

---

### Task 4: Synthetic Validation Summary and Smoke Workflow

**Files:**
- Modify: `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`
- Modify: `tests/test_matcha_streaming_manifest.py`

- [ ] **Step 1: Write failing summary aggregation test**

Append to `tests/test_matcha_streaming_manifest.py`:

```python
from feature_extract.tools.vfm.train_matcha_joint_streaming_model import _synthetic_pose_bin_counts


def test_synthetic_pose_bin_counts_summarizes_training_rows() -> None:
    rows = [
        {"synthetic_pose_bin": "micro"},
        {"synthetic_pose_bin": "small"},
        {"synthetic_pose_bin": "small"},
        {"synthetic_pose_bin": ""},
        {"pair_type": "A_gt"},
    ]

    assert _synthetic_pose_bin_counts(rows) == {"micro": 1, "small": 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_matcha_streaming_manifest.py::test_synthetic_pose_bin_counts_summarizes_training_rows -q
```

Expected: fail because `_synthetic_pose_bin_counts` is missing.

- [ ] **Step 3: Add summary helper and summary field**

In `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`, add:

```python
def _synthetic_pose_bin_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("synthetic_pose_bin", ""))
        if not label:
            continue
        counts[label] = int(counts.get(label, 0) + 1)
    return counts
```

Add `Mapping` to the imports from `typing` if it is not already imported:

```python
from typing import Mapping, Sequence
```

In `main`, after summary creation and before writing JSON, add:

```python
    synthetic_counts = _synthetic_pose_bin_counts(train_rows)
    if synthetic_counts:
        summary["synthetic_pose_bin_counts"] = synthetic_counts
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_matcha_streaming_manifest.py::test_synthetic_pose_bin_counts_summarizes_training_rows -q
```

Expected: pass.

- [ ] **Step 5: Run no-GPU parser and manifest tests**

Run:

```bash
pytest tests/test_matcha_synthetic_pairs.py tests/test_matcha_streaming_manifest.py::test_streaming_training_radio_matcha_2dgs_synthetic_preset_uses_local_window_path tests/test_matcha_streaming_manifest.py::test_synthetic_manifest_routes_to_synthetic_builder tests/test_matcha_streaming_manifest.py::test_synthetic_pose_bin_counts_summarizes_training_rows -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add feature_extract/tools/vfm/train_matcha_joint_streaming_model.py tests/test_matcha_streaming_manifest.py
git commit -m "feat: report synthetic pose bins during matcha training"
```

---

### Task 5: Synthetic Pair Evaluation Tool

**Files:**
- Create: `feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py`
- Modify: `tests/test_matcha_synthetic_pairs.py`

- [ ] **Step 1: Write failing parser and aggregation tests**

Append to `tests/test_matcha_synthetic_pairs.py`:

```python
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import _aggregate_rows_by_bin
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import parse_args as parse_synthetic_eval_args


def test_synthetic_eval_parser_requires_model_and_manifest() -> None:
    args = parse_synthetic_eval_args(
        [
            "--streaming_manifest",
            "synthetic_val.json",
            "--query_pose_file",
            "dataset_train.txt",
            "--image_root",
            "OldHospital",
            "--gaussian_rgb_ply",
            "point_cloud.ply",
            "--matcha_joint_checkpoint",
            "best_joint.pt",
            "--output_dir",
            "eval",
        ]
    )

    assert args.streaming_manifest == "synthetic_val.json"
    assert args.max_pairs == 0
    assert args.matcha_eval_preset == "radio_matcha_local_window"


def test_aggregate_rows_by_bin_reports_medians_and_rates() -> None:
    rows = [
        {
            "synthetic_pose_bin": "micro",
            "pnp_success": True,
            "translation_error_m": 0.02,
            "rotation_error_deg": 0.1,
            "gt_precision_16px": 1.0,
            "pnp_inlier_gt_precision_16px": 1.0,
            "pnp_inlier_count": 20,
        },
        {
            "synthetic_pose_bin": "micro",
            "pnp_success": False,
            "translation_error_m": None,
            "rotation_error_deg": None,
            "gt_precision_16px": 0.5,
            "pnp_inlier_gt_precision_16px": None,
            "pnp_inlier_count": 0,
        },
    ]

    summary = _aggregate_rows_by_bin(rows)
    assert summary["micro"]["pair_count"] == 2
    assert summary["micro"]["pnp_solve_rate"] == 0.5
    assert summary["micro"]["median_translation_error_m"] == 0.02
    assert summary["micro"]["mean_gt_precision_16px"] == 0.75
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_matcha_synthetic_pairs.py::test_synthetic_eval_parser_requires_model_and_manifest tests/test_matcha_synthetic_pairs.py::test_aggregate_rows_by_bin_reports_medians_and_rates -q
```

Expected: fail with `ModuleNotFoundError` for `eval_matcha_2dgs_synthetic_pairs`.

- [ ] **Step 3: Implement parser and aggregation**

Create `feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py`:

```python
"""Evaluate a MATCHA joint model on 2DGS synthetic source/target pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.tools.vfm.train_matcha_joint_streaming_model import (
    SyntheticStreamingPairBuilder,
    _builder_class_for_manifest,
)
from feature_extract.vfm.matcha_joint_training import MatchaJointTrainingConfig, _evaluate, load_matcha_joint_model
from feature_extract.vfm.matcha_streaming_manifest import MatchaStreamingPairManifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streaming_manifest", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--matcha_joint_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--default_camera", default="2,1024,576,883.0,512.0,288.0,0.0")
    parser.add_argument("--render_width", type=int, default=1280)
    parser.add_argument("--render_height", type=int, default=720)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--matcha_eval_preset", default="radio_matcha_local_window", choices=("radio_matcha_local_window",))
    parser.add_argument("--layer_name", default="radio_dual")
    parser.add_argument("--feature_mode", default="radio_dual", choices=("radio_final", "radio_dual"))
    parser.add_argument("--radio_fine_intermediate_index", type=int, default=-6)
    parser.add_argument("--radio_coarse_source", default="final", choices=("final", "intermediate"))
    parser.add_argument("--radio_coarse_intermediate_index", type=int, default=-1)
    parser.add_argument("--query_feature_cache_dir", default="")
    parser.add_argument("--query_feature_cache_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--render_pose_world_offset", default="0,0,0")
    parser.add_argument("--feature_fusion_mode", default="none", choices=("none", "local_attention"))
    parser.add_argument("--feature_fusion_radius", type=int, default=1)
    parser.add_argument("--feature_fusion_temperature", type=float, default=5.0)
    parser.add_argument("--feature_fusion_alpha", type=float, default=0.5)
    parser.add_argument("--roundtrip_threshold_px", type=float, default=1.5)
    parser.add_argument("--roundtrip_heatmap_threshold_px", type=float, default=2.0)
    parser.add_argument("--visibility_alpha_threshold", type=float, default=0.0)
    parser.add_argument("--depth_edge_threshold_m", type=float, default=0.0)
    parser.add_argument("--multiview_supervision_support_views", type=int, default=0)
    parser.add_argument("--multiview_supervision_min_support_views", type=int, default=0)
    parser.add_argument("--multiview_supervision_depth_tolerance_m", type=float, default=0.05)
    parser.add_argument("--collect_visibility_no_match", action="store_true")
    parser.add_argument("--max_visibility_no_match", type=int, default=256)
    parser.add_argument("--soft_offset_sigma_bins", type=float, default=0.75)
    parser.add_argument("--pose_confidence_labels", action="store_true")
    parser.add_argument("--pose_confidence_positive_threshold_px", type=float, default=8.0)
    parser.add_argument("--pose_confidence_negative_threshold_px", type=float, default=24.0)
    parser.add_argument("--hard_negatives_per_match", type=int, default=16)
    parser.add_argument("--fine_supervision_source", default="render_subcell_stratified", choices=("cell_center", "render_subcell", "render_subcell_stratified", "render_alike"))
    parser.add_argument("--merge_fine_labels_into_coarse", action="store_true")
    parser.add_argument("--keypoint_distill_method", default="alike", choices=("none", "alike"))
    parser.add_argument("--alike_repo", default="/root/matcha")
    parser.add_argument("--alike_model", default="alike-t")
    parser.add_argument("--alike_top_k", type=int, default=4096)
    parser.add_argument("--alike_scores_th", type=float, default=0.1)
    parser.add_argument("--alike_n_limit", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--map_pair_batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--builder_device", default="")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _mean_or_none(values: Sequence[object]) -> float | None:
    numeric = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not numeric else float(np.mean(numeric))


def _median_or_none(values: Sequence[object]) -> float | None:
    numeric = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not numeric else float(np.median(numeric))


def _aggregate_rows_by_bin(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("synthetic_pose_bin", "unknown") or "unknown"), []).append(row)
    summary: dict[str, dict[str, object]] = {}
    for label, items in grouped.items():
        summary[label] = {
            "pair_count": int(len(items)),
            "pnp_solve_rate": float(np.mean([1.0 if bool(item.get("pnp_success")) else 0.0 for item in items])),
            "median_translation_error_m": _median_or_none([item.get("translation_error_m") for item in items]),
            "median_rotation_error_deg": _median_or_none([item.get("rotation_error_deg") for item in items]),
            "mean_gt_precision_16px": _mean_or_none([item.get("gt_precision_16px") for item in items]),
            "mean_pnp_inlier_gt_precision_16px": _mean_or_none([item.get("pnp_inlier_gt_precision_16px") for item in items]),
            "mean_pnp_inlier_count": _mean_or_none([item.get("pnp_inlier_count") for item in items]),
        }
    return summary
```

- [ ] **Step 4: Add model-metric evaluation loop**

Append the main loop to `feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py`:

```python
def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = MatchaStreamingPairManifest.from_json(Path(args.streaming_manifest))
    builder_cls = _builder_class_for_manifest(manifest)
    if builder_cls is not SyntheticStreamingPairBuilder:
        raise ValueError("eval_matcha_2dgs_synthetic_pairs requires a 2dgs_synthetic manifest")
    builder = builder_cls(args, manifest)
    run = load_matcha_joint_model(Path(args.matcha_joint_checkpoint), device=str(args.device))
    device = torch.device(str(args.device) if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    cfg = MatchaJointTrainingConfig(
        model_type=str(run.summary.get("config", {}).get("model_type", "radio_dual_attention")),
        batch_size=int(args.batch_size),
        map_pair_batch_size=int(args.map_pair_batch_size),
        local_window_fine_loss_weight=0.5,
        device=str(device),
        seed=int(args.seed),
    )
    records = list(manifest.records)
    if int(args.max_pairs) > 0:
        records = records[: int(args.max_pairs)]
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, float]] = []
    model = run.model.to(device).eval()
    for index, record in enumerate(records):
        samples, row = builder.build(record)
        metrics = _evaluate(model, samples, cfg, device)
        metric_rows.append(metrics)
        rows.append(
            {
                **row,
                "pair_index_eval": int(index),
                "train_top1_acc": metrics.get("train_top1_acc"),
                "map_descriptor_top1_acc": metrics.get("map_descriptor_top1_acc"),
                "local_window_fine_acc": metrics.get("local_window_fine_acc"),
                "gt_precision_16px": metrics.get("map_descriptor_top1_acc"),
                "pnp_success": False,
                "translation_error_m": None,
                "rotation_error_deg": None,
                "pnp_inlier_gt_precision_16px": None,
                "pnp_inlier_count": 0,
            }
        )
    output_dir = Path(args.output_dir)
    _write_rows(output_dir / "rows.csv", rows)
    summary = {
        "stage": "matcha_2dgs_synthetic_pair_eval",
        "query_count": int(len(rows)),
        "metrics": {
            key: _mean_or_none([item.get(key) for item in metric_rows])
            for key in sorted({key for item in metric_rows for key in item.keys()})
        },
        "by_pose_bin": _aggregate_rows_by_bin(rows),
        "inputs": {
            "streaming_manifest": str(args.streaming_manifest),
            "matcha_joint_checkpoint": str(args.matcha_joint_checkpoint),
            "gaussian_rgb_ply": str(args.gaussian_rgb_ply),
        },
        "outputs": {
            "rows": str(output_dir / "rows.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

This first version reports model correspondence gates. The PnP row fields are present and initialized to failure so the summary schema is stable before the next task wires predicted matches to target-depth 3D points.

- [ ] **Step 5: Run parser and aggregation tests**

Run:

```bash
pytest tests/test_matcha_synthetic_pairs.py::test_synthetic_eval_parser_requires_model_and_manifest tests/test_matcha_synthetic_pairs.py::test_aggregate_rows_by_bin_reports_medians_and_rates -q
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py tests/test_matcha_synthetic_pairs.py
git commit -m "feat: add synthetic matcha pair evaluation scaffold"
```

---

### Task 6: Wire Synthetic Eval to Predicted PnP Matches

**Files:**
- Modify: `feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py`
- Test: `tests/test_matcha_synthetic_pairs.py`

- [ ] **Step 1: Write failing PnP match construction test**

Append to `tests/test_matcha_synthetic_pairs.py`:

```python
from feature_extract.tools.vfm.eval_matcha_2dgs_synthetic_pairs import _target_depth_matches_to_pnp
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.rendered_keypoint_matching import KeypointFeatureMatch


def test_target_depth_matches_to_pnp_backprojects_render_points() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=4, height=4, params=(4.0, 4.0, 2.0, 2.0))
    pose = np.eye(4, dtype=np.float64)
    depth = np.ones((4, 4), dtype=np.float32)
    match = KeypointFeatureMatch(
        query_index=0,
        render_index=0,
        query_xy=np.asarray([2.0, 2.0], dtype=np.float64),
        render_xy=np.asarray([2.0, 2.0], dtype=np.float64),
        similarity=1.0,
    )

    pnp_matches = _target_depth_matches_to_pnp([match], render_depth=depth, render_pose_w2c=pose, render_camera=camera)

    assert len(pnp_matches) == 1
    assert np.allclose(pnp_matches[0].xy, [2.0, 2.0])
    assert np.allclose(pnp_matches[0].xyz, [0.0, 0.0, 1.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_matcha_synthetic_pairs.py::test_target_depth_matches_to_pnp_backprojects_render_points -q
```

Expected: fail because `_target_depth_matches_to_pnp` is missing.

- [ ] **Step 3: Add target-depth backprojection helper**

In `feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py`, add imports:

```python
from feature_extract.vfm.matcha_coarse_to_fine import matcha_coarse_to_fine_keypoint_matches
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
```

Add helper:

```python
def _target_depth_matches_to_pnp(matches, *, render_depth: np.ndarray, render_pose_w2c: np.ndarray, render_camera) -> list[QueryTo3DMatch]:
    depth = np.asarray(render_depth, dtype=np.float64)
    pose = np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4)
    inv_pose = np.linalg.inv(pose)
    camera_matrix, _distortion = camera_matrix_and_distortion(render_camera)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    output: list[QueryTo3DMatch] = []
    for idx, match in enumerate(matches):
        rxy = np.asarray(match.render_xy, dtype=np.float64).reshape(2)
        x = int(np.clip(round(float(rxy[0])), 0, depth.shape[1] - 1))
        y = int(np.clip(round(float(rxy[1])), 0, depth.shape[0] - 1))
        z = float(depth[y, x])
        if not np.isfinite(z) or z <= 0.0:
            continue
        point_c = np.asarray([(float(rxy[0]) - cx) * z / fx, (float(rxy[1]) - cy) * z / fy, z, 1.0], dtype=np.float64)
        point_w = inv_pose @ point_c
        output.append(
            QueryTo3DMatch(
                token_index=int(match.query_index),
                xy=np.asarray(match.query_xy, dtype=np.float64).reshape(2),
                track_id=int(idx),
                xyz=point_w[:3].astype(np.float64),
                similarity=float(match.similarity),
                ratio=float(match.ratio or 0.0),
                landmark_variance=0.0,
                source="synthetic_2dgs_depth",
                pnp_soft_score=float(match.dual_softmax_confidence or match.similarity),
            )
        )
    return output
```

- [ ] **Step 4: Add eval-state access without polluting training rows**

In `SyntheticStreamingPairBuilder`, rename the current `build` method to `_build_synthetic` and add the `include_eval_state` keyword argument to its signature. Then add these two wrapper methods above `_build_synthetic`:

```python
    def build(self, record: MatchaStreamingPairRecord) -> tuple[MatchaJointTrainingSet, dict[str, object]]:
        samples, row, _state = self._build_synthetic(record, include_eval_state=False)
        return samples, row

    def build_with_eval_state(
        self,
        record: MatchaStreamingPairRecord,
    ) -> tuple[MatchaJointTrainingSet, dict[str, object], dict[str, object]]:
        return self._build_synthetic(record, include_eval_state=True)

    def _build_synthetic(
        self,
        record: MatchaStreamingPairRecord,
        *,
        include_eval_state: bool,
    ) -> tuple[MatchaJointTrainingSet, dict[str, object], dict[str, object]]:
        eval_state = {}
        if bool(include_eval_state):
            eval_state = {
                "source_pose_w2c": pair_pose.source_pose_w2c,
                "target_pose_w2c": pair_pose.target_pose_w2c,
                "target_depth": target_depth,
                "query_feature_map": query_feature,
                "render_feature_map": render_feature,
            }
        return joint, row, eval_state
```

Inside the renamed `_build_synthetic`, keep the current `joint` and `row` construction as it is, then insert the `eval_state` block immediately before the final return. The normal `build()` return row must not contain NumPy arrays because training writes `first_pair` and summaries to JSON.

- [ ] **Step 5: Estimate PnP in eval main loop**

In the eval main loop, replace `samples, row = builder.build(record)` with:

```python
        samples, row, state = builder.build_with_eval_state(record)
```

After `_evaluate`, add:

```python
        matches = matcha_coarse_to_fine_keypoint_matches(
            state["query_feature_map"],
            state["render_feature_map"],
            query_image_width=int(builder.camera.width),
            query_image_height=int(builder.camera.height),
            render_image_width=int(builder.render_camera.width),
            render_image_height=int(builder.render_camera.height),
            logit_scale=10.0,
            min_confidence=0.0,
            min_similarity=-1.0,
            max_matches=1000,
            fine_search_radius_px=0.0,
            mutual=False,
            coarse_top_k_per_query=1,
            coarse_mutual_mode="annotate",
        )
        pnp_matches = _target_depth_matches_to_pnp(
            matches,
            render_depth=state["target_depth"],
            render_pose_w2c=state["target_pose_w2c"],
            render_camera=builder.render_camera,
        )
        pnp = estimate_pose_pnp_ransac(pnp_matches, builder.camera, reprojection_error_px=8.0, iterations=1000)
        pose_error = pnp_pose_error(pnp.pose_w2c, state["source_pose_w2c"])
```

Set these row fields:

```python
                "pnp_success": bool(pnp.success),
                "translation_error_m": None if not pnp.success else float(pose_error.translation_m),
                "rotation_error_deg": None if not pnp.success else float(pose_error.rotation_deg),
                "pnp_inlier_count": int(pnp.inlier_count),
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_matcha_synthetic_pairs.py::test_target_depth_matches_to_pnp_backprojects_render_points tests/test_matcha_synthetic_pairs.py::test_aggregate_rows_by_bin_reports_medians_and_rates -q
```

Expected: both tests pass.

- [ ] **Step 7: Commit**

```bash
git add feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py feature_extract/tools/vfm/train_matcha_joint_streaming_model.py tests/test_matcha_synthetic_pairs.py
git commit -m "feat: evaluate synthetic matcha pairs with pnp"
```

---

## Verification Commands

After all tasks are complete, run the no-GPU test slice:

```bash
pytest tests/test_matcha_synthetic_pairs.py tests/test_matcha_streaming_manifest.py -q
```

Expected: all selected tests pass.

Then run a one-pair synthetic manifest smoke:

```bash
python -m feature_extract.tools.vfm.build_matcha_2dgs_synthetic_manifest \
  --query_manifest output/vfm_tokens_radio/OldHospital/train_manifest.json \
  --output_manifest output/vfm/stage_r_matcha_joint/oldhospital/2dgs_synthetic_v1/manifests/train_q1_micro.json \
  --summary_json output/vfm/stage_r_matcha_joint/oldhospital/2dgs_synthetic_v1/manifests/train_q1_micro_summary.json \
  --split_name train \
  --max_queries 1 \
  --target_translation_range_m 0.0,0.03 \
  --target_rotation_range_deg 0.0,1.0 \
  --seed 20260613
```

Expected: summary JSON has `pair_source=2dgs_synthetic` and `pair_type_counts={"S2DGS_RANDOM": 1}`.

Then run a one-step GPU smoke only on a machine with CUDA and RADIO dependencies:

```bash
python -m feature_extract.tools.vfm.train_matcha_joint_streaming_model \
  --streaming_manifest output/vfm/stage_r_matcha_joint/oldhospital/2dgs_synthetic_v1/manifests/train_q1_micro.json \
  --query_pose_file /hy-tmp/Cambridge_stdloc/OldHospital/dataset_train.txt \
  --image_root /hy-tmp/Cambridge_stdloc/OldHospital \
  --gaussian_rgb_ply /root/ICLPose/result/result/feature_gaussian/joint_rgb_geometry_cambridge_oldhospital_processed_rebuild_v4_4gpu_1280_safe_rgbft_30k/point_cloud/best/point_cloud.ply \
  --output_model output/vfm/stage_r_matcha_joint/oldhospital/2dgs_synthetic_v1/smoke/adapter.pt \
  --output_joint_model output/vfm/stage_r_matcha_joint/oldhospital/2dgs_synthetic_v1/smoke/model_joint.pt \
  --output_best_joint_model output/vfm/stage_r_matcha_joint/oldhospital/2dgs_synthetic_v1/smoke/best_joint.pt \
  --summary_json output/vfm/stage_r_matcha_joint/oldhospital/2dgs_synthetic_v1/smoke/summary.json \
  --matcha_train_preset radio_matcha_2dgs_synthetic \
  --steps 1 \
  --batch_size 128 \
  --builder_device cuda:0 \
  --device cuda:0
```

Expected: summary JSON contains `first_pair.synthetic_pose_bin`, nonzero `sample_count`, and no read of real query RGB as source.

---

## Self-Review

- Spec coverage:
  - Synthetic source/target 2DGS pairs: Tasks 1-3.
  - No fixed `A_gt/B025/C050/D_reference` distribution: Task 2 uses `S2DGS_RANDOM`.
  - Local-window first, patch-corr not default: Task 3 preset.
  - Non-empty validation support: Task 2 builds validation manifests with the same CLI; Task 3 keeps `--validation_streaming_manifest` support.
  - Synthetic eval before real-domain work: Tasks 5-6.

- Placeholder scan:
  - The plan has no incomplete marker strings.
  - All code-facing steps include exact file paths and concrete snippets.

- Type consistency:
  - `SyntheticPairSamplingConfig`, `SyntheticPairPose`, `SYNTHETIC_PAIR_SOURCE`, `SYNTHETIC_RANDOM_PAIR_TYPE`, and `SYNTHETIC_RANDOM_PAIR_TYPE_ID` are defined in Task 1 and reused by later tasks.
  - The manifest builder writes `pair_source=2dgs_synthetic`, which is the key consumed by `_builder_class_for_manifest`.
  - The synthetic builder returns `MatchaJointTrainingSet` and a row dict, matching existing `build_with_retries` expectations.
