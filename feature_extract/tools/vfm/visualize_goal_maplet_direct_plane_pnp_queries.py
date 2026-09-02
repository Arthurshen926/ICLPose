"""Visualize query planes and frozen 2D-3D PnP correspondence residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--correspondences", type=Path, required=True)
    parser.add_argument("--frozen_poses", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite visualization")
    with np.load(args.correspondences, allow_pickle=False) as data:
        names = np.asarray(data["names"]).astype(str)
        offsets = np.asarray(data["correspondence_offsets"], np.int64)
        points = np.asarray(data["world_points"], np.float64)
        tokens = np.asarray(data["query_tokens"], np.int64)
        matrices = np.asarray(data["camera_matrices"], np.float64)
        radial = np.asarray(data["radial_k1"], np.float64)
        metadata = json.loads(str(data["metadata_json"].item()))
    with np.load(args.frozen_poses, allow_pickle=False) as data:
        pose_names = np.asarray(data["names"]).astype(str)
        poses = np.asarray(data["pose_w2c"], np.float64)
        usable = np.asarray(data["usable"], bool)
    if not np.array_equal(names, pose_names):
        raise ValueError("pose and correspondence query orders differ")
    grid_h, grid_w = map(int, metadata.get("token_grid", (36, 64)))
    report = json.loads(args.report.read_text())
    error_by_name = {
        row["name"]: float(row["translation_error_m"])
        for row in report["rows"] if row.get("translation_error_m") is not None
    }
    # Uniform sequence indices: visual evidence is not selected by outcome.
    selected = np.rint(np.linspace(0, len(names) - 1, min(int(args.count), len(names)))).astype(int)
    figure, axes = plt.subplots(2, len(selected) // 2, figsize=(4 * (len(selected) // 2), 7), squeeze=False)
    for axis, row in zip(axes.ravel(), selected.tolist()):
        name = names[row]
        image_id = name[:-4].replace("__", "/")
        image = cv2.cvtColor(cv2.imread(str(args.image_root / image_id)), cv2.COLOR_BGR2RGB)
        planes, _ = QueryPlaneRegions.load_npz(args.query_plane_dir / name)
        boundary = cv2.morphologyEx((planes.labels >= 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        boundary = cv2.resize(boundary, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        shown = image.copy(); shown[boundary] = np.asarray([255, 220, 0], np.uint8)
        lo, hi = map(int, offsets[row:row + 2])
        local_token = tokens[lo:hi]; world = points[lo:hi]
        pixel = np.c_[(local_token % grid_w + .5) * 256 / grid_w - .5,
                      (local_token // grid_w + .5) * 144 / grid_h - .5]
        axis.imshow(shown)
        if bool(usable[row]) and len(pixel):
            rotation, translation = poses[row, :3, :3], poses[row, :3, 3]
            projected, _ = cv2.projectPoints(
                world, cv2.Rodrigues(rotation)[0], translation, matrices[row],
                np.asarray([radial[row], 0, 0, 0, 0], np.float64),
            )
            residual = np.linalg.norm(projected.reshape(-1, 2) - pixel, axis=1)
            keep = np.arange(len(pixel))[::max(1, len(pixel) // 300)]
            xy = (pixel[keep] + .5) * np.asarray([image.shape[1] / 256, image.shape[0] / 144]) - .5
            colors = np.where((residual[keep] <= 4)[:, None], [[0.1, .9, .2]], [[1., .1, .1]])
            axis.scatter(xy[:, 0], xy[:, 1], s=5, c=colors, alpha=.65)
            title = f"{image_id}  t={error_by_name[name]:.3f}m\n{hi-lo} frozen matches"
        else:
            title = f"{image_id}  unusable\n{hi-lo} frozen matches"
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{args.image_root.name}: MoGe3 plane boundaries (yellow), "
        "frozen matches (green residual <=4px)"
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
