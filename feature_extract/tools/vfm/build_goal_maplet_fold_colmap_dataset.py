"""Build a held-route-clean posed COLMAP input for strict fold 2DGS rebuilds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image as PILImage

from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.official_oof_protocol import file_sha256, ordered_id_sha256


def _proportional_subsample(
    image_ids: list[str], maximum_images: int,
) -> list[str]:
    if maximum_images <= 0 or len(image_ids) <= maximum_images:
        return sorted(image_ids)
    grouped: dict[str, list[str]] = {}
    for image_id in sorted(image_ids):
        grouped.setdefault(image_id.split("/", 1)[0], []).append(image_id)
    if maximum_images < len(grouped):
        raise ValueError(
            "maximum_images must retain at least one view per mapping trajectory"
        )
    exact = {
        route: maximum_images * len(values) / len(image_ids)
        for route, values in grouped.items()
    }
    counts = {route: max(1, int(np.floor(value))) for route, value in exact.items()}
    while sum(counts.values()) < maximum_images:
        route = max(
            counts,
            key=lambda value: (exact[value] - counts[value], len(grouped[value]), value),
        )
        counts[route] += 1
    while sum(counts.values()) > maximum_images:
        route = min(
            (value for value in counts if counts[value] > 1),
            key=lambda value: (exact[value] - counts[value], -len(grouped[value]), value),
        )
        counts[route] -= 1
    selected = []
    for route, values in sorted(grouped.items()):
        count = min(counts[route], len(values))
        indices = np.linspace(0, len(values) - 1, count).round().astype(np.int64)
        selected.extend(values[int(index)] for index in indices)
    if len(selected) != maximum_images or len(selected) != len(set(selected)):
        raise AssertionError("route-proportional subsampling is not exact")
    return sorted(selected)


def _flattened_name(image_id: str) -> str:
    return image_id.replace("/", "__")


def _chart_indices(
    dataset_image_ids: list[str], selected_chart_image_ids: list[str]
) -> list[int]:
    """Return MAtCha's zero-based ``--image_idx`` for selected chart views."""

    if len(dataset_image_ids) != len(set(dataset_image_ids)):
        raise ValueError("COLMAP dataset image IDs must be unique")
    by_id = {image_id: index for index, image_id in enumerate(dataset_image_ids)}
    missing = sorted(set(selected_chart_image_ids) - set(by_id))
    if missing:
        raise ValueError(f"chart views are absent from dense dataset: {missing[:8]}")
    return [by_id[image_id] for image_id in selected_chart_image_ids]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol_json", required=True)
    parser.add_argument("--fold_id", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--camera_manifest", required=True)
    parser.add_argument("--maximum_images", type=int, default=512)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    summary_path = output / "fold_colmap_dataset.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite fold COLMAP dataset")
    protocol_path = Path(args.protocol_json)
    protocol = json.loads(protocol_path.read_text())
    if str(args.fold_id) == "final_alltrain":
        fold = {
            "fold_id": "final_alltrain",
            "held_query_trajectories": [],
            "mapping_count": int(protocol["official_train"]["count"]),
            "mapping_image_ids_sha256": protocol["official_train"]["image_ids_sha256"],
        }
    else:
        matches = [
            fold for fold in protocol["development"]["folds"]
            if str(fold["fold_id"]) == str(args.fold_id)
        ]
        if len(matches) != 1:
            raise ValueError("fold_id is not unique in protocol")
        fold = matches[0]
    held = set(str(value) for value in fold["held_query_trajectories"])
    pose_file = Path(protocol["official_train"]["pose_file"])
    pose_records = parse_cambridge_pose_file(pose_file)
    pose_by_id = {value.image_id: value.pose_w2c for value in pose_records}
    mapping_ids = [
        value.image_id for value in pose_records
        if value.image_id.split("/", 1)[0] not in held
    ]
    if (
        len(mapping_ids) != int(fold["mapping_count"])
        or ordered_id_sha256(mapping_ids) != str(fold["mapping_image_ids_sha256"])
    ):
        raise ValueError("protocol fold mapping set changed")
    # Every mapping image remains in the posed COLMAP model and is therefore
    # available to MAtCha's dense RGB supervision.  ``maximum_images`` limits
    # only the expensive DepthAnything chart initialization, not the training
    # data.  The strict runner passes the recorded zero-based indices through
    # MAtCha's --image_idx option.
    dataset_image_ids = sorted(mapping_ids)
    selected = _proportional_subsample(dataset_image_ids, int(args.maximum_images))
    chart_indices = _chart_indices(dataset_image_ids, selected)
    camera_payload = json.loads(Path(args.camera_manifest).read_text())
    camera_by_id = camera_payload["cameras"]

    # Import the repository-bundled COLMAP writer without requiring pycolmap's
    # mutable database API.  The written model contains calibrated cameras and
    # poses but deliberately no SfM points or query observations.
    hloc_root = Path(__file__).resolve().parents[3] / "third_party" / "Hierarchical-Localization"
    if str(hloc_root) not in sys.path:
        sys.path.insert(0, str(hloc_root))
    from hloc.utils.read_write_model import (  # type: ignore
        Camera,
        Image,
        rotmat2qvec,
        write_cameras_binary,
        write_images_binary,
        write_points3D_binary,
    )

    image_dir = output / "images"
    sparse_dir = output / "sparse" / "0"
    image_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    cameras = {}
    images = {}
    source_root = Path(args.image_root)
    for index, image_id in enumerate(dataset_image_ids, start=1):
        source = source_root / image_id
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = image_dir / _flattened_name(image_id)
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"existing fold image link differs: {destination}")
        else:
            destination.symlink_to(source.resolve())
        with PILImage.open(source) as image:
            native_width, native_height = image.size
        camera = camera_by_id[image_id]
        width, height = int(camera["width"]), int(camera["height"])
        params = np.asarray(camera["params"], dtype=np.float64)
        model_id = int(camera["model_id"])
        if model_id == 0:
            focal, cx, cy = params[:3]
            fx = fy = focal
        elif model_id in (1, 2):
            if model_id == 1:
                fx, fy, cx, cy = params[:4]
            else:
                focal, cx, cy = params[:3]
                fx = fy = focal
        else:
            raise ValueError(f"unsupported Cambridge camera model: {model_id}")
        scale_x = native_width / width
        scale_y = native_height / height
        cameras[index] = Camera(
            id=index,
            model="PINHOLE",
            width=native_width,
            height=native_height,
            params=np.asarray([
                fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y
            ], dtype=np.float64),
        )
        pose = np.asarray(pose_by_id[image_id], dtype=np.float64)
        images[index] = Image(
            id=index,
            qvec=rotmat2qvec(pose[:3, :3]),
            tvec=pose[:3, 3],
            camera_id=index,
            name=destination.name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images, sparse_dir / "images.bin")
    write_points3D_binary({}, sparse_dir / "points3D.bin")
    payload = {
        "artifact_type": "goal_maplet_fold_clean_posed_colmap_dataset_v2",
        "fold_id": str(args.fold_id),
        "split_role": (
            "final_official_train_map_for_locked_test"
            if str(args.fold_id) == "final_alltrain"
            else "route_grouped_oof_mapping_fold"
        ),
        "held_query_trajectories": sorted(held),
        "mapping_image_count": len(mapping_ids),
        "dense_supervision_image_count": len(dataset_image_ids),
        "dense_supervision_uses_all_mapping_images": True,
        "dense_supervision_image_ids_sha256": ordered_id_sha256(dataset_image_ids),
        "selected_geometry_image_count": len(selected),
        "selected_geometry_image_ids": selected,
        "selected_geometry_image_ids_sha256": ordered_id_sha256(selected),
        "selected_chart_image_indices_zero_based": chart_indices,
        "selected_contains_held_query": False,
        "selection": (
            "all_mapping_images_for_dense_supervision; "
            "route_proportional_uniform_spacing_fixed_maximum_for_initial_charts"
        ),
        "maximum_images": int(args.maximum_images),
        "camera_conversion": (
            "native_resolution_PINHOLE_from_declared_camera_K; radial term omitted "
            "because official MAtCha posed loader accepts PINHOLE only"
        ),
        "stores_sfm_points": False,
        "protocol_json": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "camera_manifest": str(args.camera_manifest),
        "camera_manifest_sha256": file_sha256(Path(args.camera_manifest)),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "selected_geometry_image_ids"}, indent=2))


if __name__ == "__main__":
    main()
