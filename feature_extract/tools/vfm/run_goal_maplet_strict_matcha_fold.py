"""Run one route-disjoint MAtCha geometry fold with checked subprocesses.

The upstream ``train.py`` and its helper wrappers use ``os.system`` without
propagating failures.  This runner invokes the three underlying stages
directly, requires the pinned source commit, validates the fold input contract
and records a hash-bound manifest for every produced geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


PINNED_MATCHA_COMMIT = "b119fd96e484fc81eb40623c1ea92ad3dbd3c21e"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MATCHA_PATCH_CONTRACT = (
    PROJECT_ROOT / "configs" / "vfm"
    / "matcha_b119fd96_rtx3090_cuda116.patch"
)


def _fold_contract(dataset: Path) -> dict[str, object]:
    dataset = Path(dataset).resolve()
    metadata_path = dataset / "fold_colmap_dataset.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("artifact_type") != "goal_maplet_fold_clean_posed_colmap_dataset_v2":
        raise ValueError("strict MAtCha input must use the v2 fold COLMAP contract")
    mapping_count = int(metadata["mapping_image_count"])
    chart_indices = [
        int(value) for value in metadata["selected_chart_image_indices_zero_based"]
    ]
    if not bool(metadata.get("dense_supervision_uses_all_mapping_images", False)):
        raise ValueError("strict MAtCha input truncated dense mapping supervision")
    if int(metadata.get("dense_supervision_image_count", -1)) != mapping_count:
        raise ValueError("dense supervision count differs from mapping image count")
    if bool(metadata.get("selected_contains_held_query", True)):
        raise ValueError("strict MAtCha chart set contains a held query")
    if not chart_indices or len(chart_indices) != len(set(chart_indices)):
        raise ValueError("strict MAtCha chart indices are empty or duplicated")
    if min(chart_indices) < 0 or max(chart_indices) >= mapping_count:
        raise ValueError("strict MAtCha chart index is outside the mapping dataset")
    images = sorted(path for path in (dataset / "images").iterdir() if path.is_file())
    if len(images) != mapping_count:
        raise ValueError("posed COLMAP image directory does not match mapping count")
    required_sparse = [
        dataset / "sparse" / "0" / name
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    ]
    if any(not path.is_file() for path in required_sparse):
        raise ValueError("posed COLMAP input is incomplete")
    if (dataset / "sparse" / "0" / "points3D.bin").stat().st_size != 8:
        raise ValueError("posed COLMAP input unexpectedly contains SfM points")
    return {
        "metadata_path": metadata_path,
        "metadata": metadata,
        "mapping_image_count": mapping_count,
        "chart_indices": chart_indices,
    }


def _commands(
    *,
    conda_env: str,
    dataset: Path,
    output: Path,
    chart_indices: Sequence[int],
    gaussian_iterations: int,
) -> dict[str, list[str]]:
    python = ["conda", "run", "--no-capture-output", "-n", str(conda_env), "python"]
    sfm = output / "mast3r_sfm"
    gaussians = output / "free_gaussians"
    return {
        "sfm": python + [
            "mast3r/run_mast3r.py",
            "--scene_path", str(dataset),
            "--output_dir", str(sfm),
            "--weights_path", "./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            "--retrieval_model", "./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
            "--min_conf_thr", "0", "--matching_conf_thr", "0",
            "--n_coarse_iterations", "1000", "--n_refinement_iterations", "1000",
            "--TSDF_thresh", "0", "--fix_focal", "--fix_principal_point",
            "--fix_rotation", "--fix_translation", "--image_idx",
            *[str(value) for value in chart_indices],
            "--image_size", "512", "--max_window_size", "20", "--max_refid", "10",
            "--use_calibrated_poses", "--output_conf_thr", "0.1",
            "--align_camera_locations",
        ],
        "alignment": python + [
            "scripts/align_charts.py",
            "--source_path", str(sfm), "--mast3r_scene", str(sfm),
            "--output_path", str(sfm), "--config", "default",
            "--depth_model", "depthanythingv2",
            "--depthanythingv2_checkpoint_dir", "./Depth-Anything-V2/checkpoints/",
            "--depthanything_encoder", "vitl",
        ],
        "gaussians": python + [
            "2d-gaussian-splatting/train_with_charts.py",
            "-s", str(sfm), "-m", str(gaussians),
            "--iterations", str(int(gaussian_iterations)),
            "--densify_until_iter", "15000", "--opacity_reset_interval", "3000",
            "--depth_ratio", "0.5", "--use_mip_filter",
            "--dense_data_path", str(dataset),
            "--normal_consistency_from", "7000", "--distortion_from", "3000",
            "--depthanythingv2_checkpoint_dir", "./Depth-Anything-V2/checkpoints/",
            "--depthanything_encoder", "vitl", "--dense_regul", "default", "--quiet",
        ],
    }


def _stage_outputs(output: Path, chart_count: int, iterations: int) -> dict[str, list[Path]]:
    sfm = output / "mast3r_sfm"
    return {
        "sfm": [
            sfm / "points.ply", sfm / "cameras.json",
            sfm / "sparse" / "0" / "cameras.bin",
            sfm / "sparse" / "0" / "images.bin",
            sfm / "sparse" / "0" / "points3D.bin",
        ] + [sfm / "pointmaps" / f"__count_{chart_count}__"],
        "alignment": [sfm / "charts_data.npz"],
        "gaussians": [
            output / "free_gaussians" / "point_cloud"
            / f"iteration_{int(iterations)}" / "point_cloud.ply"
        ],
    }


def _validate_stage(stage: str, outputs: list[Path]) -> list[Path]:
    real_outputs = []
    for path in outputs:
        if path.name.startswith("__count_"):
            expected = int(path.name.removeprefix("__count_").removesuffix("__"))
            pointmaps = sorted(path.parent.glob("*.json"))
            if len(pointmaps) != expected:
                raise RuntimeError(
                    f"MAtCha SfM produced {len(pointmaps)} pointmaps, expected {expected}"
                )
            real_outputs.extend(pointmaps)
        elif not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"MAtCha {stage} output is missing or empty: {path}")
        else:
            real_outputs.append(path)
    return real_outputs


def _git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE
    ).stdout


def _validate_tracked_patch(tracked_patch: bytes, contract_path: Path) -> dict[str, str]:
    expected = Path(contract_path).read_bytes()
    actual_sha = hashlib.sha256(tracked_patch).hexdigest()
    expected_sha = hashlib.sha256(expected).hexdigest()
    if tracked_patch != expected:
        raise ValueError(
            "MAtCha tracked source differs from the versioned RTX3090/CUDA11.6 patch"
        )
    return {
        "path": str(Path(contract_path).resolve()),
        "sha256": expected_sha,
        "applied_diff_sha256": actual_sha,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--matcha_repo", default="/tmp/matcha-gaussians-official")
    parser.add_argument("--conda_env", default="matcha")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gaussian_iterations", type=int, default=30000)
    parser.add_argument(
        "--stages", nargs="+", choices=("sfm", "alignment", "gaussians"),
        default=("sfm", "alignment", "gaussians"),
    )
    parser.add_argument("--manifest", default="")
    args = parser.parse_args(argv)
    if int(args.gaussian_iterations) < 1:
        raise ValueError("gaussian_iterations must be positive")
    canonical_stage_order = ["sfm", "alignment", "gaussians"]
    requested_stage_order = [
        stage for stage in canonical_stage_order if stage in set(args.stages)
    ]
    if list(args.stages) != requested_stage_order:
        raise ValueError("MAtCha stages must be unique and follow sfm/alignment/gaussians")
    dataset, output = Path(args.fold_dataset).resolve(), Path(args.output_dir).resolve()
    repo = Path(args.matcha_repo).resolve()
    commit = _git_bytes(repo, "rev-parse", "HEAD").decode().strip()
    if commit != PINNED_MATCHA_COMMIT:
        raise ValueError(f"MAtCha commit drift: expected {PINNED_MATCHA_COMMIT}, got {commit}")
    tracked_patch = _git_bytes(repo, "diff", "--binary", "--no-ext-diff")
    patch_contract = _validate_tracked_patch(tracked_patch, MATCHA_PATCH_CONTRACT)
    contract = _fold_contract(dataset)
    commands = _commands(
        conda_env=str(args.conda_env), dataset=dataset, output=output,
        chart_indices=contract["chart_indices"],
        gaussian_iterations=int(args.gaussian_iterations),
    )
    expected = _stage_outputs(
        output, len(contract["chart_indices"]), int(args.gaussian_iterations)
    )
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    completed = {}
    for stage in args.stages:
        try:
            produced = _validate_stage(stage, expected[stage])
            status = "reused_complete_outputs"
        except RuntimeError:
            subprocess.run(
                commands[stage], cwd=repo, env=environment, check=True
            )
            produced = _validate_stage(stage, expected[stage])
            status = "executed_and_verified"
        completed[stage] = {
            "status": status,
            "command": commands[stage],
            "outputs": [
                {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
                for path in produced
            ],
        }
    source_inputs = [
        repo / "mast3r" / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        repo / "mast3r" / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
        repo / "mast3r" / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
        repo / "Depth-Anything-V2" / "checkpoints" / "depth_anything_v2_vitl.pth",
        repo / "configs" / "charts_alignment" / "default.yaml",
        repo / "mast3r" / "run_mast3r.py",
        repo / "scripts" / "align_charts.py",
        repo / "2d-gaussian-splatting" / "train_with_charts.py",
    ]
    if any(not path.is_file() for path in source_inputs):
        raise RuntimeError("pinned MAtCha source or checkpoint input is missing")
    runtime_text = subprocess.run(
        [
            "conda", "run", "-n", str(args.conda_env), "python", "-c",
            (
                "import json,torch; print(json.dumps({"
                "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available()}))"
            ),
        ],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout.strip().splitlines()[-1]
    manifest = {
        "artifact_type": "goal_maplet_strict_matcha_fold_run_v1",
        "fold_id": contract["metadata"]["fold_id"],
        "fold_dataset": str(dataset),
        "fold_dataset_contract": str(contract["metadata_path"]),
        "fold_dataset_contract_sha256": file_sha256(contract["metadata_path"]),
        "mapping_image_count": contract["mapping_image_count"],
        "chart_count": len(contract["chart_indices"]),
        "held_query_trajectories": contract["metadata"]["held_query_trajectories"],
        "dense_supervision_uses_all_mapping_images": True,
        "matcha_repo": str(repo),
        "matcha_commit": commit,
        "matcha_tracked_patch_sha256": hashlib.sha256(tracked_patch).hexdigest(),
        "matcha_tracked_patch_contract": patch_contract,
        "matcha_untracked_files_are_not_runtime_inputs": True,
        "source_input_sha256": {
            str(path.relative_to(repo)): file_sha256(path) for path in source_inputs
        },
        "runtime": json.loads(runtime_text),
        "upstream_wrappers_used": False,
        "subprocess_failures_propagated": True,
        "gpu_physical_index": int(args.gpu),
        "gaussian_iterations": int(args.gaussian_iterations),
        "stages": completed,
    }
    manifest_path = (
        Path(args.manifest).resolve() if str(args.manifest)
        else output / "strict_matcha_run_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact_type": manifest["artifact_type"],
        "fold_id": manifest["fold_id"],
        "manifest": str(manifest_path),
        "stages": {key: value["status"] for key, value in completed.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
