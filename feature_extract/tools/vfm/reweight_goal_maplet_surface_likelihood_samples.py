"""Apply role-directed offline-teacher weights to cached G18 render evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_surface_likelihood_samples import (
    _teacher_cues,
    _typed_teacher_weight,
)
from feature_extract.vfm.localization_goal_maplet.surface_pose_likelihood import (
    SAMPLE_SCHEMA,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_samples", required=True)
    parser.add_argument("--teacher_cache", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source, output = Path(args.input_samples), Path(args.output_npz)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite G18 teacher-reweighted samples")
    with np.load(source, allow_pickle=False) as data:
        values = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if metadata.get("artifact_type") != SAMPLE_SCHEMA:
        raise ValueError("not a G18 surface-likelihood sample artifact")
    token_count = int(args.height) * int(args.width)
    if values["typed_target"].shape[-1] != token_count:
        raise ValueError("height/width do not match the cached token grid")
    weights, available = [], []
    for image_id, target in zip(values["image_ids"].astype(str), values["typed_target"]):
        teacher_path = Path(args.teacher_cache) / (image_id.replace("/", "__") + ".npz")
        if teacher_path.exists():
            cues, _audit = _teacher_cues(teacher_path, int(args.height), int(args.width))
            weights.append(_typed_teacher_weight(target, cues).astype(np.float16))
            available.append(True)
        else:
            weights.append(np.ones(target.shape, dtype=np.float16))
            available.append(False)
    values["teacher_weight"] = np.stack(weights)
    metadata.update({
        "teacher_available_fraction": float(np.mean(available)),
        "teacher_embeddings_stored": False,
        "teacher_supervision": "event_conditioned_spatial_weights_not_runtime_features",
        "teacher_reweighted_from_render_cache": str(source),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **values,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({
        "output": str(output),
        "query_count": int(values["image_ids"].shape[0]),
        "teacher_available_fraction": metadata["teacher_available_fraction"],
        "teacher_weight_shape": list(values["teacher_weight"].shape),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
