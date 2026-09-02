"""Freeze exact source/held MASt3R commands before isolated runs start."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


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


def _command(
    matcha: Path, scene: Path, output: Path, *, image_count: int
) -> list[str]:
    if image_count < 2:
        raise ValueError("isolated MASt3R role must contain at least two images")
    command = [
        "python",
        "mast3r/run_mast3r.py",
        "--scene_path",
        str(scene.resolve()),
        "--output_dir",
        str(output.resolve()),
        "--weights_path",
        "./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "--retrieval_model",
        "./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
        "--min_conf_thr",
        "0",
        "--matching_conf_thr",
        "0",
        "--n_coarse_iterations",
        "1000",
        "--n_refinement_iterations",
        "1000",
        "--TSDF_thresh",
        "0",
        "--fix_focal",
        "--fix_principal_point",
        "--fix_rotation",
        "--fix_translation",
        "--image_size",
        "512",
        "--max_window_size",
        "20",
        "--max_refid",
        "10",
        "--use_calibrated_poses",
        "--output_conf_thr",
        "0.1",
        "--align_camera_locations",
    ]
    # run_mast3r defaults to n_images=10 and its implicit sampler also indexes
    # past the end for some inventory sizes.  An isolated input root is not a
    # sufficient all-images contract unless every lexical row is explicit.
    command.extend(["--image_idx", *[str(index) for index in range(image_count)]])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isolated_inputs", type=Path, required=True)
    parser.add_argument("--matcha_repo", type=Path, required=True)
    parser.add_argument("--source_output", type=Path, required=True)
    parser.add_argument("--held_output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.source_output.exists() or args.held_output.exists():
        raise FileExistsError("preexecution contract requires absent output targets")
    isolated = json.loads(args.isolated_inputs.read_text())
    claimed = isolated.pop("content_sha256", None)
    if claimed != _canonical(isolated):
        raise ValueError("isolated input manifest content hash differs")
    isolated["content_sha256"] = claimed
    if isolated.get("artifact_type") != "goal_maplet_isolated_chart_sfm_inputs_v1":
        raise ValueError("wrong isolated input schema")
    source_command = _command(
        args.matcha_repo,
        Path(isolated["source"]["root"]),
        args.source_output,
        image_count=int(isolated["source"]["image_count"]),
    )
    held_command = _command(
        args.matcha_repo,
        Path(isolated["held"]["root"]),
        args.held_output,
        image_count=int(isolated["held"]["image_count"]),
    )
    files = {
        "mast3r/run_mast3r.py": args.matcha_repo / "mast3r" / "run_mast3r.py",
        "mast3r/mast3r/cloud_opt/sparse_ga.py": args.matcha_repo
        / "mast3r"
        / "mast3r"
        / "cloud_opt"
        / "sparse_ga.py",
        "metric_checkpoint": args.matcha_repo
        / "mast3r"
        / "checkpoints"
        / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "retrieval_checkpoint": args.matcha_repo
        / "mast3r"
        / "checkpoints"
        / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
    }
    report = {
        "artifact_type": "goal_maplet_chart_sfm_preexecution_contract_v2",
        "isolated_inputs_file_sha256": _sha(args.isolated_inputs),
        "isolated_inputs_content_sha256": claimed,
        "matcha_repo": str(args.matcha_repo.resolve()),
        "source_command": source_command,
        "held_command": held_command,
        "role_environment": {
            "source": {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1"},
            "held": {"CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "1"},
        },
        "source_file_sha256": {name: _sha(path) for name, path in files.items()},
        "source_output_absent_at_freeze": True,
        "held_output_absent_at_freeze": True,
        "all_isolated_images_explicitly_indexed": True,
        "uses_query_or_ground_truth": False,
    }
    report["content_sha256"] = _canonical(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
