"""Cache exact stride-grid 2DGS primitive contributors for V6 training/baking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMFeatureView,
    load_gaussian_vfm_source_from_ply,
)
from feature_extract.vfm.localization_v6.primitive_contributors import (
    clean_primitive_surface_elements,
    render_primitive_contributors,
)
from feature_extract.vfm.surface_maplet_bank import load_2dgs_primitive_quality
from feature_extract.vfm.tokens import TokenBankManifest
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument(
        "--clean_gaussian_ply",
        default="",
        help=(
            "Optional source-indexed clean PLY mask. Geometry is loaded from "
            "--gaussian_ply so contributor IDs remain canonical."
        ),
    )
    parser.add_argument(
        "--clean_surface_elements",
        default="",
        help=(
            "Optional surface-element declaration of the clean source mask. "
            "Its parent_gaussian_indices select source PLY disks; this is the "
            "strict MAtCha path and is mutually exclusive with --clean_gaussian_ply."
        ),
    )
    parser.add_argument("--mapping_manifest", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--mapping_camera_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--trajectory_ids", nargs="+", default=[])
    parser.add_argument("--image_ids_file", default="")
    parser.add_argument("--first_frame", type=int, default=0)
    parser.add_argument("--views_per_trajectory", type=int, default=6)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _declared_clean_source_indices(
    source_count: int,
    *,
    clean_gaussian_ply: str = "",
    clean_surface_elements: str = "",
) -> tuple[np.ndarray, str, Path | None]:
    """Resolve one auditable clean-geometry declaration in source-ID space."""

    if clean_gaussian_ply and clean_surface_elements:
        raise ValueError(
            "clean_gaussian_ply and clean_surface_elements are mutually exclusive"
        )
    declaration: Path | None = None
    if clean_gaussian_ply:
        declaration = Path(clean_gaussian_ply)
        quality = load_2dgs_primitive_quality(declaration)
        if not bool((quality.metadata or {}).get("source_indexed", False)):
            raise ValueError("clean_gaussian_ply must carry canonical source_index")
        if len(quality) > int(source_count):
            raise ValueError("clean PLY source indices exceed full 2DGS")
        indices = np.flatnonzero(quality.geometry_confidence > 0.0).astype(np.int64)
        policy = "declared_clean_source_index_mask"
    elif clean_surface_elements:
        declaration = Path(clean_surface_elements)
        surface = SurfaceElementMap.load_npz(declaration)
        indices = np.unique(
            np.asarray(surface.parent_gaussian_indices, dtype=np.int64)
        )
        policy = "declared_surface_element_parent_source_mask"
    else:
        indices = np.arange(int(source_count), dtype=np.int64)
        policy = "complete_input_2dgs"
    if (
        indices.size == 0
        or np.any(indices < 0)
        or np.any(indices >= int(source_count))
    ):
        raise ValueError("declared clean geometry retained invalid source primitives")
    return indices, policy, declaration


def _frame_number(image_id: str) -> int:
    stem = Path(image_id).stem
    return int(stem.replace("frame", ""))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_index_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_json)
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite contributor cache")
    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply))
    clean_source_indices, occlusion_policy, clean_declaration = (
        _declared_clean_source_indices(
            int(source.xyz.shape[0]),
            clean_gaussian_ply=str(args.clean_gaussian_ply),
            clean_surface_elements=str(args.clean_surface_elements),
        )
    )
    # Occlusion and contributor identity must come from the same declared clean
    # 2DGS prior as the canonical atlas.  The full source is retained only to
    # preserve stable source-index identity.
    elements = clean_primitive_surface_elements(
        source, clean_source_indices
    )
    geometry_source_sha256 = _file_sha256(Path(args.gaussian_ply))
    clean_geometry_source_sha256 = (
        _file_sha256(clean_declaration)
        if clean_declaration is not None
        else geometry_source_sha256
    )
    clean_source_index_sha256 = _source_index_sha256(clean_source_indices)
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.mapping_pose_file))
    }
    camera_by_image, camera_audit = _load_query_camera_manifest(
        Path(args.mapping_camera_manifest)
    )
    manifest = TokenBankManifest.from_json(Path(args.mapping_manifest))
    wanted = set(str(value) for value in args.trajectory_ids)
    explicit_ids = (
        {
            line.strip()
            for line in Path(args.image_ids_file).read_text().splitlines()
            if line.strip()
        }
        if str(args.image_ids_file)
        else set()
    )
    if not wanted and not explicit_ids:
        raise ValueError("provide trajectory_ids or image_ids_file")
    grouped: dict[str, list[object]] = {value: [] for value in wanted}
    for record in manifest.records:
        trajectory = record.image_id.split("/", 1)[0]
        if (
            (trajectory in wanted or record.image_id in explicit_ids)
            and record.image_id in pose_by_image
            and record.image_id in camera_by_image
        ):
            grouped.setdefault(trajectory, []).append(record)
    selected = [
        record
        for records in grouped.values()
        for record in records
        if record.image_id in explicit_ids
    ]
    for trajectory in sorted(grouped):
        if trajectory not in wanted:
            continue
        records = sorted(grouped[trajectory], key=lambda item: _frame_number(item.image_id))
        if explicit_ids:
            records = [record for record in records if record.image_id not in explicit_ids]
        records = records[int(args.first_frame) :]
        selected.extend(records[: int(args.views_per_trajectory)])
    if not selected:
        raise ValueError("no requested mapping records were found")
    shard_count = int(args.shard_count)
    shard_index = int(args.shard_index)
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard_index/shard_count")
    selected = [
        record
        for index, record in enumerate(selected)
        if index % shard_count == shard_index
    ]
    if not selected:
        raise ValueError("requested shard contains no mapping records")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(selected):
        destination = output_dir / (record.image_id.replace("/", "__") + ".npz")
        if destination.exists() and not bool(args.force):
            with np.load(destination, allow_pickle=False) as data:
                rows.append(json.loads(str(data["metadata_json"].item())))
            continue
        camera = camera_by_image[record.image_id]
        view = GaussianVFMFeatureView(
            image_id=record.image_id,
            feature_map=np.zeros(
                (1, int(args.height), int(args.width)), dtype=np.float32
            ),
            pose_w2c=pose_by_image[record.image_id],
            camera=camera,
        )
        buffer = render_primitive_contributors(
            elements,
            view,
            width=int(args.width),
            height=int(args.height),
            top_k=int(args.top_k),
            device=str(args.device),
        )
        metadata = {
            "artifact_type": "v6_training_contributor_cache",
            "image_id": record.image_id,
            "trajectory_id": record.image_id.split("/", 1)[0],
            "token_path": str(record.token_path),
            "width": int(args.width),
            "height": int(args.height),
            "top_k": int(args.top_k),
            "assignment": "gsplat_topk_contributor_source_index",
            "uses_complete_2dgs_for_occlusion": clean_declaration is None,
            "uses_declared_clean_2dgs_for_occlusion": clean_declaration is not None,
            "occlusion_primitive_policy": occlusion_policy,
            "clean_geometry_declaration": (
                "" if clean_declaration is None else str(clean_declaration)
            ),
            "geometry_source_sha256": geometry_source_sha256,
            "clean_geometry_source_sha256": clean_geometry_source_sha256,
            "clean_source_index_sha256": clean_source_index_sha256,
            "stores_rgb": False,
            "stores_rgb_path": False,
            "uses_kdtree_fallback": False,
        }
        np.savez_compressed(
            destination,
            topk_ids=buffer.topk_ids.astype(np.int32),
            topk_weights=buffer.topk_weights.astype(np.float16),
            dominant_depth=buffer.primitive_depth.astype(np.float32),
            pose_w2c=np.asarray(view.pose_w2c, dtype=np.float64),
            camera_model_id=np.asarray(camera.model_id, dtype=np.int32),
            camera_width=np.asarray(camera.width, dtype=np.int32),
            camera_height=np.asarray(camera.height, dtype=np.int32),
            camera_params=np.asarray(camera.params, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        rows.append(metadata)
        print(f"[{index + 1}/{len(selected)}] {record.image_id}", flush=True)
    report = {
        "stage": "v6_exact_primitive_contributor_cache",
        "view_count": len(rows),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "trajectory_ids": sorted(
            {row["trajectory_id"] for row in rows}
        ),
        "trajectory_disjoint_role": "caller_declared_split",
        "camera_audit": camera_audit,
        "assignment": "gsplat_topk_contributor_source_index",
        "uses_complete_2dgs_for_occlusion": clean_declaration is None,
        "uses_declared_clean_2dgs_for_occlusion": clean_declaration is not None,
        "occlusion_primitive_policy": occlusion_policy,
        "clean_geometry_declaration": (
            "" if clean_declaration is None else str(clean_declaration)
        ),
        "full_primitive_count": int(source.xyz.shape[0]),
        "retained_primitive_count": int(clean_source_indices.size),
        "geometry_source_sha256": geometry_source_sha256,
        "clean_geometry_source_sha256": clean_geometry_source_sha256,
        "clean_source_index_sha256": clean_source_index_sha256,
        "stores_rgb": False,
        "uses_kdtree_fallback": False,
        "records": rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
