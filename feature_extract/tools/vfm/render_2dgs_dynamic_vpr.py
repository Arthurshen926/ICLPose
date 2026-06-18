"""Render init-centered 2DGS pose lattices and rerank them with VPR descriptors."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor
from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _load_camera_with_source,
    _parse_default_camera,
    _scale_camera,
)
from feature_extract.tools.vfm.render_2dgs_virtual_tokens import render_virtual_pose_token_manifest
from feature_extract.vfm.cambridge_pose_lattice import (
    build_init_pose_lattice_bank,
    fine_lattice_world_offsets,
    parse_world_offsets,
    q_level_world_offsets,
)
from feature_extract.vfm.dynamic_rendered_vpr import (
    candidate_bank_to_render_records,
    rerank_dynamic_rendered_vpr_candidates,
)
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.official_2dgs_renderer import (
    load_official_2dgs_source_from_ply,
    render_official_2dgs_rgb_depth,
)
from feature_extract.vfm.retrieval_benchmark import summarize_reference_pose_retrieval
from feature_extract.vfm.token_descriptor_bank import (
    TokenDescriptorBank,
    apply_pca_whitening_transform,
    build_token_descriptor_bank,
    load_pca_whitening_transform_npz,
    load_vlad_codebook_npz,
)


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _build_offsets(args: argparse.Namespace) -> tuple[tuple[float, float, float], ...]:
    if args.fine_radius_m is not None:
        return fine_lattice_world_offsets(
            radius_m=float(args.fine_radius_m),
            step_m=float(args.fine_step_m),
            height_offsets_m=_parse_float_list(str(args.fine_height_offsets_m)),
        )
    if args.offsets:
        return parse_world_offsets(str(args.offsets))
    if args.q_level:
        return q_level_world_offsets(str(args.q_level))
    return parse_world_offsets("0,0,0")


def _limit_candidate_bank_queries(bank: CandidateHypothesisBank, max_queries: int) -> CandidateHypothesisBank:
    if int(max_queries) <= 0:
        return bank
    selected: list[str] = []
    selected_set: set[str] = set()
    candidates = []
    for candidate in bank.candidates:
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.query_id not in selected_set:
            if len(selected) >= int(max_queries):
                continue
            selected.append(candidate.query_id)
            selected_set.add(candidate.query_id)
        if candidate.query_id in selected_set:
            candidates.append(candidate)
    return CandidateHypothesisBank.from_candidates(
        protocol_name=bank.protocol_name,
        protocol_kind=bank.protocol_kind,
        candidates=candidates,
        protocol_fingerprint=bank.protocol_fingerprint,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_bank", required=True)
    parser.add_argument("--gt_pose_file", required=True)
    parser.add_argument("--query_descriptors", required=True)
    parser.add_argument("--gaussian_rgb_ply", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument("--output_candidate_bank", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--token_output_root", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--image_output_root", default="")
    parser.add_argument("--scene", default="OldHospital")
    parser.add_argument("--split", default="dynamic_virtual")
    parser.add_argument("--protocol_name", default="2dgs_dynamic_vpr")
    parser.add_argument("--layer_name", default="radio_final")
    parser.add_argument("--render_width", type=int, default=320)
    parser.add_argument("--render_height", type=int, default=180)
    parser.add_argument("--default_camera", default="2,1920,1080,1400.0,960.0,540.0,0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--storage_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--min_alpha_coverage", type=float, default=0.0)
    parser.add_argument("--q_level", choices=("q10", "q25", "q50"), default=None)
    parser.add_argument("--offsets", default=None)
    parser.add_argument("--fine_radius_m", type=float, default=None)
    parser.add_argument("--fine_step_m", type=float, default=0.1)
    parser.add_argument("--fine_height_offsets_m", default="0")
    parser.add_argument("--yaw_offsets_deg", default="0")
    parser.add_argument("--pitch_offsets_deg", default="0")
    parser.add_argument("--roll_offsets_deg", default="0")
    parser.add_argument("--max_inits_per_query", type=int, default=1)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--pooling", default="mean", choices=("mean", "gem", "vlad"))
    parser.add_argument("--gem_power", type=float, default=3.0)
    parser.add_argument("--normalize_tokens", action="store_true")
    parser.add_argument("--descriptor_power", type=float, default=1.0)
    parser.add_argument("--vlad_codebook_input", default="")
    parser.add_argument("--vlad_clusters", type=int, default=32)
    parser.add_argument("--vlad_iterations", type=int, default=20)
    parser.add_argument("--vlad_max_tokens", type=int, default=200000)
    parser.add_argument("--vlad_tokens_per_image", type=int, default=0)
    parser.add_argument("--pca_whitening_input", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    offsets = _build_offsets(args)
    init_bank = _limit_candidate_bank_queries(
        CandidateHypothesisBank.from_jsonl(Path(args.init_bank)),
        int(args.max_queries),
    )
    lattice_bank = build_init_pose_lattice_bank(
        init_bank=init_bank,
        gt_pose_file=Path(args.gt_pose_file),
        protocol_name=f"{args.protocol_name}:lattice",
        offsets=offsets,
        max_inits_per_query=int(args.max_inits_per_query),
        yaw_offsets_deg=_parse_float_list(str(args.yaw_offsets_deg)),
        pitch_offsets_deg=_parse_float_list(str(args.pitch_offsets_deg)),
        roll_offsets_deg=_parse_float_list(str(args.roll_offsets_deg)),
    )
    render_records, candidate_to_image = candidate_bank_to_render_records(lattice_bank)

    camera, camera_source = _load_camera_with_source(
        Path(args.camera_model_dir),
        _parse_default_camera(args.default_camera),
    )
    render_camera = _scale_camera(camera, int(args.render_width), int(args.render_height))
    source = load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
    radio = RADIOFeatureExtractor(version=args.radio_version, device=args.device, radio_repo=args.radio_repo)

    def render_fn(**kwargs):
        return render_official_2dgs_rgb_depth(
            kwargs["source"],
            pose_w2c=kwargs["pose_w2c"],
            camera=kwargs["camera"],
            width=int(kwargs["width"]),
            height=int(kwargs["height"]),
            device=str(args.device),
        )

    manifest = render_virtual_pose_token_manifest(
        records=render_records,
        source=source,
        camera=render_camera,
        width=int(args.render_width),
        height=int(args.render_height),
        extractor=radio,
        render_rgb_depth_fn=render_fn,
        image_output_root=Path(args.image_output_root) if args.image_output_root else None,
        token_output_root=Path(args.token_output_root),
        scene=str(args.scene),
        split=str(args.split),
        layer_name=str(args.layer_name),
        model_name=str(args.radio_version),
        storage_dtype=str(args.storage_dtype),
        renderer_name="official_2dgs",
        source_path=str(args.gaussian_rgb_ply),
        min_alpha_coverage=float(args.min_alpha_coverage),
    )
    manifest_path = Path(args.output_manifest)
    manifest.to_json(manifest_path)

    codebook = None
    if args.vlad_codebook_input:
        codebook, _metadata = load_vlad_codebook_npz(Path(args.vlad_codebook_input))
    rendered_descriptors = build_token_descriptor_bank(
        manifest=manifest,
        layer_name=str(args.layer_name),
        pooling=str(args.pooling),
        gem_power=float(args.gem_power),
        normalize_tokens=bool(args.normalize_tokens),
        normalize=True,
        metadata={
            "token_manifest": str(manifest_path),
            "descriptor_source": "2dgs_dynamic_init_lattice_render",
        },
        vlad_clusters=int(args.vlad_clusters),
        vlad_iterations=int(args.vlad_iterations),
        vlad_max_tokens=int(args.vlad_max_tokens),
        seed=int(args.seed),
        vlad_codebook=codebook,
        vlad_tokens_per_image=int(args.vlad_tokens_per_image),
        descriptor_power=float(args.descriptor_power),
    )
    if args.pca_whitening_input:
        transform = load_pca_whitening_transform_npz(Path(args.pca_whitening_input))
        rendered_descriptors = TokenDescriptorBank(
            image_ids=rendered_descriptors.image_ids,
            descriptors=apply_pca_whitening_transform(rendered_descriptors.descriptors, transform, normalize=True),
            layer_name=rendered_descriptors.layer_name,
            pooling=f"{rendered_descriptors.pooling}+pca_whiten",
            normalized=True,
            metadata={
                **dict(rendered_descriptors.metadata or {}),
                "pca_whitening_input": str(args.pca_whitening_input),
            },
        )

    query_descriptors = TokenDescriptorBank.from_npz(Path(args.query_descriptors))
    output_bank = rerank_dynamic_rendered_vpr_candidates(
        lattice_bank=lattice_bank,
        query_descriptors=query_descriptors,
        rendered_descriptors=rendered_descriptors,
        candidate_to_image=candidate_to_image,
        protocol_name=str(args.protocol_name),
        top_k=int(args.top_k),
    )
    output_candidate_bank = Path(args.output_candidate_bank)
    output_bank.to_jsonl(output_candidate_bank)

    summary = {
        "init_bank": str(args.init_bank),
        "gt_pose_file": str(args.gt_pose_file),
        "query_descriptors": str(args.query_descriptors),
        "gaussian_rgb_ply": str(args.gaussian_rgb_ply),
        "camera_source": str(camera_source),
        "output_candidate_bank": str(output_candidate_bank),
        "output_manifest": str(manifest_path),
        "token_output_root": str(args.token_output_root),
        "image_output_root": str(args.image_output_root),
        "scene": str(args.scene),
        "split": str(args.split),
        "render_width": int(args.render_width),
        "render_height": int(args.render_height),
        "lattice_candidate_count": len(lattice_bank.candidates),
        "init_candidate_count": len(init_bank.candidates),
        "rendered_pose_count": len(manifest.records),
        "reranked_candidate_count": len(output_bank.candidates),
        "max_inits_per_query": int(args.max_inits_per_query),
        "max_queries": int(args.max_queries),
        "top_k": int(args.top_k),
        "pooling": rendered_descriptors.pooling,
        "layer_name": rendered_descriptors.layer_name,
        "min_alpha_coverage": float(args.min_alpha_coverage),
        "offset_count": len(offsets),
        "yaw_offsets_deg": list(_parse_float_list(str(args.yaw_offsets_deg))),
        "pitch_offsets_deg": list(_parse_float_list(str(args.pitch_offsets_deg))),
        "roll_offsets_deg": list(_parse_float_list(str(args.roll_offsets_deg))),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    summary["metrics_25cm_5deg"] = summarize_reference_pose_retrieval(
        output_bank,
        top_ks=(1, 5, 10, min(20, int(args.top_k))),
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        rot_cost_weight=0.1,
    )
    summary["metrics_50cm_10deg"] = summarize_reference_pose_retrieval(
        output_bank,
        top_ks=(1, 5, 10, min(20, int(args.top_k))),
        translation_threshold_m=0.5,
        rotation_threshold_deg=10.0,
        rot_cost_weight=0.1,
    )
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
