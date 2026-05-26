"""Score fixed candidates using rendered selected-map descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.colmap_tracks import (
    ColmapCamera,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.artifacts import attach_report_inputs
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.map_lifting import load_selected_track_bank_npz
from feature_extract.vfm.rendered_map_verifier import (
    build_track_observation_index,
    build_track_visibility_index,
    build_track_xyz_index,
    score_candidate_bank_by_rendered_selected_map,
    score_candidate_bank_by_projected_rendered_selected_map,
    score_candidate_bank_by_sparse_rendered_selected_map,
)
from feature_extract.vfm.reporting import build_gate_table
from feature_extract.vfm.score_table import evaluate_score_table
from feature_extract.vfm.selector_descriptor_scoring import load_selector_from_checkpoint
from feature_extract.vfm.tokens import TokenBankManifest


def _mean_present(values):
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def _evidence_summary(rows) -> dict[str, object]:
    match_counts = [row.match_count for row in rows if row.match_count is not None]
    if match_counts:
        empty_fraction = float(sum(1 for value in match_counts if int(value) <= 0) / len(match_counts))
    else:
        empty_fraction = None
    return {
        "mean_similarity": _mean_present(row.mean_similarity for row in rows),
        "mean_inlier_fraction": _mean_present(row.inlier_fraction for row in rows),
        "mean_match_count": _mean_present(row.match_count for row in rows),
        "mean_visibility_fraction": _mean_present(row.visibility_fraction for row in rows),
        "empty_evidence_fraction": empty_fraction,
    }


def _track_summary_path(track_observations: Path) -> Path:
    return track_observations.with_name(f"{track_observations.stem}_summary.json")


def _validate_camera_track_model_match(track_observations: Path, camera_model_dir: Optional[str]) -> None:
    if not camera_model_dir:
        return
    summary_path = _track_summary_path(Path(track_observations))
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    model_dir = summary.get("model_dir")
    if not model_dir:
        return
    expected = str(Path(str(model_dir)).resolve(strict=False))
    observed = str(Path(str(camera_model_dir)).resolve(strict=False))
    if expected != observed:
        raise ValueError(
            "camera_model_dir does not match track observation model_dir: "
            f"{observed} != {expected}"
        )


def _parse_default_camera(text: str) -> ColmapCamera:
    parts = [float(item) for item in text.split(",") if item.strip()]
    if len(parts) < 6:
        raise ValueError("--default_camera must be 'model_id,width,height,param0,param1,...'")
    model_id = int(parts[0])
    width = int(parts[1])
    height = int(parts[2])
    params = tuple(float(item) for item in parts[3:])
    return ColmapCamera(camera_id=-1, model_id=model_id, width=width, height=height, params=params)


def _load_cameras(model_dir: Optional[str], default_camera_text: str) -> tuple[dict[str, ColmapCamera], ColmapCamera]:
    fallback = _parse_default_camera(default_camera_text)
    if not model_dir:
        return {}, fallback
    model_path = Path(model_dir)
    cameras = read_colmap_cameras_binary(model_path / "cameras.bin")
    images = read_colmap_images_binary(model_path / "images.bin")
    camera_by_image = {
        image.image_name: cameras[image.camera_id]
        for image in images.values()
        if image.camera_id in cameras
    }
    if camera_by_image:
        sorted_cameras = sorted(camera_by_image.values(), key=lambda camera: camera.camera_id)
        fallback = sorted_cameras[len(sorted_cameras) // 2]
    return camera_by_image, fallback


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score candidates with rendered selected-map descriptor cosine similarity"
    )
    parser.add_argument("--bank", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--track_bank", required=True)
    parser.add_argument("--track_observations", required=True)
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--layer_name", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", default="rendered_selected_map_cosine")
    parser.add_argument(
        "--mode",
        default="global_descriptor",
        choices=["global_descriptor", "sparse_grid", "projected_grid"],
    )
    parser.add_argument("--local_radius", type=int, default=0)
    parser.add_argument("--inlier_threshold", type=float, default=0.5)
    parser.add_argument("--inlier_weight", type=float, default=0.0)
    parser.add_argument("--risk_from_inliers", action="store_true")
    parser.add_argument("--projected_visibility_filter", default="none", choices=("none", "reference_image"))
    parser.add_argument("--camera_model_dir", default=None)
    parser.add_argument("--default_camera", default="2,1024,576,885,512,288,0")
    parser.add_argument("--translation_threshold_m", type=float, default=0.25)
    parser.add_argument("--rotation_threshold_deg", type=float, default=5.0)
    parser.add_argument("--output_rows", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args(argv)

    if args.mode == "projected_grid":
        _validate_camera_track_model_match(Path(args.track_observations), args.camera_model_dir)

    selector = load_selector_from_checkpoint(Path(args.selector_checkpoint), device=args.device)
    bank = CandidateHypothesisBank.from_jsonl(Path(args.bank))
    query_manifest = TokenBankManifest.from_json(Path(args.query_manifest))
    track_bank = load_selected_track_bank_npz(Path(args.track_bank))
    if args.mode == "sparse_grid":
        rows = score_candidate_bank_by_sparse_rendered_selected_map(
            bank=bank,
            query_manifest=query_manifest,
            track_bank=track_bank,
            observation_index=build_track_observation_index(Path(args.track_observations)),
            selector=selector,
            layer_name=args.layer_name,
            method=args.method,
            translation_threshold_m=args.translation_threshold_m,
            rotation_threshold_deg=args.rotation_threshold_deg,
            device=args.device,
            local_radius=args.local_radius,
            inlier_threshold=args.inlier_threshold,
            inlier_weight=args.inlier_weight,
            risk_from_inliers=args.risk_from_inliers,
        )
    elif args.mode == "projected_grid":
        camera_by_image, default_camera = _load_cameras(args.camera_model_dir, args.default_camera)
        visibility_index = None
        if args.projected_visibility_filter == "reference_image":
            visibility_index = build_track_visibility_index(Path(args.track_observations))
        rows = score_candidate_bank_by_projected_rendered_selected_map(
            bank=bank,
            query_manifest=query_manifest,
            track_bank=track_bank,
            track_xyz_index=build_track_xyz_index(Path(args.track_observations)),
            camera_by_image=camera_by_image,
            default_camera=default_camera,
            selector=selector,
            layer_name=args.layer_name,
            method=args.method,
            translation_threshold_m=args.translation_threshold_m,
            rotation_threshold_deg=args.rotation_threshold_deg,
            device=args.device,
            local_radius=args.local_radius,
            inlier_threshold=args.inlier_threshold,
            inlier_weight=args.inlier_weight,
            risk_from_inliers=args.risk_from_inliers,
            visibility_index=visibility_index,
        )
    else:
        rows = score_candidate_bank_by_rendered_selected_map(
            bank=bank,
            query_manifest=query_manifest,
            track_bank=track_bank,
            visibility_index=build_track_visibility_index(Path(args.track_observations)),
            selector=selector,
            layer_name=args.layer_name,
            method=args.method,
            translation_threshold_m=args.translation_threshold_m,
            rotation_threshold_deg=args.rotation_threshold_deg,
            device=args.device,
        )

    row_payload = []
    for row in rows:
        item = dict(row.__dict__)
        item["protocol_kind"] = row.protocol_kind.value
        row_payload.append(item)
    output_rows = Path(args.output_rows)
    output_rows.parent.mkdir(parents=True, exist_ok=True)
    output_rows.write_text(json.dumps(row_payload, indent=2, sort_keys=True) + "\n")

    report = evaluate_score_table(rows).to_dict()
    report_with_inputs = attach_report_inputs(
        report,
        bank,
        Path(args.bank),
        {
            "query_manifest": args.query_manifest,
            "track_bank": args.track_bank,
            "track_observations": args.track_observations,
            "selector_checkpoint": args.selector_checkpoint,
        },
    )
    report_with_inputs["rendered_map_parameters"] = {
        "mode": args.mode,
        "method": args.method,
        "layer_name": args.layer_name,
        "device": args.device,
        "local_radius": int(args.local_radius),
        "inlier_threshold": float(args.inlier_threshold),
        "inlier_weight": float(args.inlier_weight),
        "risk_from_inliers": bool(args.risk_from_inliers),
        "projected_visibility_filter": args.projected_visibility_filter,
        "camera_model_dir": args.camera_model_dir,
        "default_camera": args.default_camera,
        "translation_threshold_m": float(args.translation_threshold_m),
        "rotation_threshold_deg": float(args.rotation_threshold_deg),
    }
    report_with_inputs["evidence_summary"] = _evidence_summary(rows)
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report_with_inputs, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_gate_table("Rendered Selected Map Cosine", [report]) + "\n")


if __name__ == "__main__":
    main()
