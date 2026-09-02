"""Materialize physically isolated source/held COLMAP inputs for chart SfM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _materialize_role(
    source: Path, destination: Path, indices: list[int], declared_routes: set[str]
) -> dict:
    source_names = sorted(path.name for path in (source / "images").iterdir() if path.is_file())
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("role image indices must be nonempty and unique")
    if min(indices) < 0 or max(indices) >= len(source_names):
        raise ValueError("role image index outside source inventory")
    names = [source_names[index] for index in indices]
    routes = {name.split("__", 1)[0] for name in names}
    if routes != declared_routes:
        raise ValueError("selected names differ from declared role routes")
    source_cameras = read_colmap_cameras_binary(source / "sparse" / "0" / "cameras.bin")
    source_images = read_colmap_images_binary(source / "sparse" / "0" / "images.bin")
    image_by_name = {image.image_name: image for image in source_images.values()}

    hloc_root = Path(__file__).resolve().parents[3] / "third_party" / "Hierarchical-Localization"
    if str(hloc_root) not in sys.path:
        sys.path.insert(0, str(hloc_root))
    from hloc.utils.read_write_model import (  # type: ignore
        Camera,
        Image,
        write_cameras_binary,
        write_images_binary,
        write_points3D_binary,
    )

    images_dir = destination / "images"
    sparse_dir = destination / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    cameras = {}
    images = {}
    rows = []
    for new_id, name in enumerate(names, start=1):
        source_image_path = source / "images" / name
        destination_image_path = images_dir / name
        shutil.copy2(source_image_path, destination_image_path)
        image = image_by_name[name]
        camera = source_cameras[image.camera_id]
        if int(camera.model_id) != 1:
            raise ValueError("isolated chart inputs require frozen PINHOLE cameras")
        cameras[new_id] = Camera(
            id=new_id,
            model="PINHOLE",
            width=camera.width,
            height=camera.height,
            params=np.asarray(camera.params, np.float64),
        )
        images[new_id] = Image(
            id=new_id,
            qvec=np.asarray(image.qvec, np.float64),
            tvec=np.asarray(image.tvec, np.float64),
            camera_id=new_id,
            name=name,
            xys=np.zeros((0, 2), np.float64),
            point3D_ids=np.zeros((0,), np.int64),
        )
        rows.append(
            {
                "name": name,
                "source_image_file_sha256": _sha(source_image_path),
                "isolated_image_file_sha256": _sha(destination_image_path),
            }
        )
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images, sparse_dir / "images.bin")
    write_points3D_binary({}, sparse_dir / "points3D.bin")
    return {
        "root": str(destination.resolve()),
        "indices_in_original_lexical_inventory": indices,
        "routes": sorted(routes),
        "ordered_names": names,
        "image_count": len(names),
        "rows": rows,
        "cameras_file_sha256": _sha(sparse_dir / "cameras.bin"),
        "images_file_sha256": _sha(sparse_dir / "images.bin"),
        "points3D_file_sha256": _sha(sparse_dir / "points3D.bin"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posed_colmap", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--source_indices", type=int, nargs="+", required=True)
    parser.add_argument("--held_indices", type=int, nargs="+", required=True)
    parser.add_argument("--source_routes", nargs="+", required=True)
    parser.add_argument("--held_routes", nargs="+", required=True)
    parser.add_argument("--forbidden_routes", nargs="+", default=["seq12", "seq14"])
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("refusing to reuse isolated chart SfM input root")
    source_routes = set(args.source_routes)
    held_routes = set(args.held_routes)
    forbidden = set(args.forbidden_routes)
    if source_routes & held_routes or (source_routes | held_routes) & forbidden:
        raise ValueError("source/held/forbidden route sets overlap")
    args.output_root.mkdir(parents=True)
    source = _materialize_role(
        args.posed_colmap,
        args.output_root / "source_posed_colmap",
        args.source_indices,
        source_routes,
    )
    held = _materialize_role(
        args.posed_colmap,
        args.output_root / "held_posed_colmap",
        args.held_indices,
        held_routes,
    )
    if set(source["ordered_names"]) & set(held["ordered_names"]):
        raise ValueError("isolated source and held image sets overlap")
    manifest = {
        "artifact_type": "goal_maplet_isolated_chart_sfm_inputs_v1",
        "original_posed_colmap_root": str(args.posed_colmap.resolve()),
        "original_cameras_file_sha256": _sha(
            args.posed_colmap / "sparse" / "0" / "cameras.bin"
        ),
        "original_images_file_sha256": _sha(
            args.posed_colmap / "sparse" / "0" / "images.bin"
        ),
        "source": source,
        "held": held,
        "forbidden_routes": sorted(forbidden),
        "physical_image_input_roots_disjoint": True,
        "source_held_image_disjoint": True,
        "uses_mapping_camera_pose": True,
        "uses_query_or_ground_truth": False,
    }
    manifest["content_sha256"] = _canonical(manifest)
    temporary = args.output_root / "manifest.json.temporary"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, args.output_root / "manifest.json")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"source", "held"}}, indent=2))


if __name__ == "__main__":
    main()
