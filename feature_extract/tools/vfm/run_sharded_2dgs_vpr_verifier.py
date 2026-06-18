"""Stream a large 2DGS virtual-pose DB through RADIO VPR top-k verification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_rendered_feature_keypoint_pose import (
    _load_camera_with_source,
    _parse_default_camera,
    _scale_camera,
)
from feature_extract.tools.vfm.render_2dgs_virtual_tokens import _alpha_stats, _rgb_to_uint8, _write_rgb
from feature_extract.vfm.retrieval_benchmark import summarize_reference_pose_retrieval
from feature_extract.vfm.sharded_streaming_vpr import (
    StreamingVPRTopK,
    estimate_virtual_reference_grid_record_count,
    iter_virtual_reference_grid_records,
)
from feature_extract.vfm.token_descriptor_bank import (
    TokenDescriptorBank,
    apply_pca_whitening_transform,
    load_pca_whitening_transform_npz,
    load_vlad_codebook_npz,
    _normalize_rows,
    _normalize_tokens,
    _pool_feature,
    _power_normalize_rows,
    _vlad_pool_feature,
)


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _descriptor_from_feature(
    feature: np.ndarray,
    *,
    pooling: str,
    gem_power: float,
    normalize_tokens: bool,
    descriptor_power: float,
    vlad_codebook: np.ndarray | None,
    vlad_tokens_per_image: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(feature, dtype=np.float32)
    if str(pooling) == "vlad":
        if vlad_codebook is None:
            raise ValueError("--vlad_codebook_input is required when --pooling=vlad")
        descriptor = _vlad_pool_feature(
            values,
            vlad_codebook,
            normalize_tokens=bool(normalize_tokens),
            max_tokens_per_image=int(vlad_tokens_per_image),
            rng=rng,
        )
    else:
        if normalize_tokens:
            values = _normalize_tokens(values)
        descriptor = _pool_feature(values, str(pooling), gem_power=float(gem_power))
    descriptor = _power_normalize_rows(np.asarray(descriptor, dtype=np.float32).reshape(1, -1), float(descriptor_power))
    return _normalize_rows(descriptor)[0].astype(np.float32, copy=False)


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _grid_summary(args: argparse.Namespace, estimated_pose_count: int) -> dict[str, object]:
    height_offsets = _parse_float_list(str(args.grid_height_offsets_m))
    yaw_offsets = _parse_float_list(str(args.yaw_offsets_deg))
    return {
        "reference_pose_file": str(args.reference_pose_file),
        "query_pose_file": str(args.query_pose_file),
        "estimated_pose_count": int(estimated_pose_count),
        "grid_step_m": float(args.grid_step_m),
        "grid_margin_m": float(args.grid_margin_m),
        "grid_height_mode": str(args.grid_height_mode),
        "grid_height_knn": int(args.grid_height_knn),
        "grid_height_offsets_m": list(height_offsets),
        "grid_orientation_knn": int(args.grid_orientation_knn),
        "yaw_offsets_deg": list(yaw_offsets),
        "image_prefix": str(args.image_prefix),
        "start_ordinal": int(args.start_ordinal),
        "max_pose_records": int(args.max_pose_records),
    }


def _validate_real_run_args(args: argparse.Namespace) -> None:
    missing = []
    for name in ("query_descriptors", "gaussian_rgb_ply", "camera_model_dir", "output_candidate_bank"):
        if not getattr(args, name):
            missing.append(f"--{name}")
    if missing:
        raise ValueError(f"real verifier run requires: {', '.join(missing)}")
    if int(args.records_per_shard) <= 0:
        raise ValueError("--records_per_shard must be positive")
    if int(args.render_width) <= 0 or int(args.render_height) <= 0:
        raise ValueError("--render_width and --render_height must be positive")
    if float(args.min_alpha_coverage) < 0.0 or float(args.min_alpha_coverage) > 1.0:
        raise ValueError("--min_alpha_coverage must be in [0, 1]")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--query_descriptors", default="")
    parser.add_argument("--gaussian_rgb_ply", default="")
    parser.add_argument("--camera_model_dir", default="")
    parser.add_argument("--output_candidate_bank", default="")
    parser.add_argument("--protocol_name", default="sharded_2dgs_radio_vpr")
    parser.add_argument("--grid_step_m", type=float, default=0.1)
    parser.add_argument("--grid_margin_m", type=float, default=1.0)
    parser.add_argument("--grid_height_mode", default="idw", choices=("nearest", "idw"))
    parser.add_argument("--grid_height_knn", type=int, default=5)
    parser.add_argument("--grid_height_offsets_m", default="-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1")
    parser.add_argument("--grid_orientation_knn", type=int, default=8)
    parser.add_argument("--yaw_offsets_deg", default="-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30")
    parser.add_argument("--image_prefix", default="virtual_grid010_o8_yaw5_h9_margin1")
    parser.add_argument("--start_ordinal", type=int, default=0)
    parser.add_argument("--max_pose_records", type=int, default=0)
    parser.add_argument("--records_per_shard", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--render_width", type=int, default=320)
    parser.add_argument("--render_height", type=int, default=180)
    parser.add_argument("--default_camera", default="2,1920,1080,1400.0,960.0,540.0,0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radio_version", default="c-radio_v4-h")
    parser.add_argument("--radio_repo", default="feature_extract/checkpoints/RADIO")
    parser.add_argument("--pooling", default="mean", choices=("mean", "gem", "vlad"))
    parser.add_argument("--gem_power", type=float, default=3.0)
    parser.add_argument("--normalize_tokens", action="store_true")
    parser.add_argument("--descriptor_power", type=float, default=1.0)
    parser.add_argument("--vlad_codebook_input", default="")
    parser.add_argument("--vlad_tokens_per_image", type=int, default=0)
    parser.add_argument("--pca_whitening_input", default="")
    parser.add_argument("--min_alpha_coverage", type=float, default=0.0)
    parser.add_argument("--debug_image_output_root", default="")
    parser.add_argument("--debug_image_limit", type=int, default=0)
    parser.add_argument("--dry_run_count_only", action="store_true")
    parser.add_argument("--progress_every_shards", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    height_offsets = _parse_float_list(str(args.grid_height_offsets_m))
    yaw_offsets = _parse_float_list(str(args.yaw_offsets_deg))
    estimated_pose_count = estimate_virtual_reference_grid_record_count(
        reference_pose_file=Path(args.reference_pose_file),
        grid_step_m=float(args.grid_step_m),
        grid_margin_m=float(args.grid_margin_m),
        height_offsets_m=height_offsets,
        orientation_knn=int(args.grid_orientation_knn),
        yaw_offsets_deg=yaw_offsets,
    )
    summary = _grid_summary(args, estimated_pose_count)
    if args.dry_run_count_only:
        summary.update(
            {
                "dry_run_count_only": True,
                "elapsed_sec": float(time.perf_counter() - started),
            }
        )
        _write_summary(Path(args.summary_json), summary)
        return

    _validate_real_run_args(args)

    from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor
    from feature_extract.tools.vfm.build_render_rgb_keypoint_adapter_samples import _extract_radio_feature_from_rgb
    from feature_extract.vfm.official_2dgs_renderer import (
        load_official_2dgs_source_from_ply,
        render_official_2dgs_rgb_depth,
    )
    from feature_extract.vfm.sharded_streaming_vpr import build_streaming_vpr_candidate_bank

    query_descriptors = TokenDescriptorBank.from_npz(Path(args.query_descriptors))
    vlad_codebook = None
    if args.vlad_codebook_input:
        vlad_codebook, _metadata = load_vlad_codebook_npz(Path(args.vlad_codebook_input))
    pca_transform = None
    if args.pca_whitening_input:
        pca_transform = load_pca_whitening_transform_npz(Path(args.pca_whitening_input))

    camera, camera_source = _load_camera_with_source(
        Path(args.camera_model_dir),
        _parse_default_camera(str(args.default_camera)),
    )
    render_camera = _scale_camera(camera, int(args.render_width), int(args.render_height))
    source = load_official_2dgs_source_from_ply(Path(args.gaussian_rgb_ply))
    radio = RADIOFeatureExtractor(version=str(args.radio_version), device=str(args.device), radio_repo=str(args.radio_repo))
    topk = StreamingVPRTopK(
        query_ids=query_descriptors.image_ids,
        query_descriptors=query_descriptors.descriptors,
        top_k=int(args.top_k),
    )
    rng = np.random.default_rng(int(args.seed))
    iterator = iter_virtual_reference_grid_records(
        reference_pose_file=Path(args.reference_pose_file),
        grid_step_m=float(args.grid_step_m),
        grid_margin_m=float(args.grid_margin_m),
        height_mode=str(args.grid_height_mode),
        height_knn=int(args.grid_height_knn),
        height_offsets_m=height_offsets,
        orientation_knn=int(args.grid_orientation_knn),
        yaw_offsets_deg=yaw_offsets,
        image_prefix=str(args.image_prefix),
        start_ordinal=int(args.start_ordinal),
        max_records=int(args.max_pose_records),
    )

    batch_records = []
    batch_ordinals = []
    rendered_count = 0
    accepted_count = 0
    skipped_low_alpha = 0
    shard_count = 0
    debug_image_count = 0
    descriptor_dim = None

    def flush_batch() -> None:
        nonlocal batch_records
        nonlocal batch_ordinals
        nonlocal rendered_count
        nonlocal accepted_count
        nonlocal skipped_low_alpha
        nonlocal shard_count
        nonlocal debug_image_count
        nonlocal descriptor_dim
        if not batch_records:
            return
        shard_count += 1
        accepted_records = []
        accepted_ordinals = []
        descriptors = []
        for record_idx, record in enumerate(batch_records):
            rgb, _depth, alpha = render_official_2dgs_rgb_depth(
                source,
                pose_w2c=record.pose_w2c,
                camera=render_camera,
                width=int(args.render_width),
                height=int(args.render_height),
                device=str(args.device),
            )
            rendered_count += 1
            rgb_u8 = _rgb_to_uint8(rgb)
            _alpha_mean, alpha_coverage = _alpha_stats(alpha)
            if alpha_coverage < float(args.min_alpha_coverage):
                skipped_low_alpha += 1
                continue
            if args.debug_image_output_root and (
                int(args.debug_image_limit) <= 0 or debug_image_count < int(args.debug_image_limit)
            ):
                _write_rgb(Path(args.debug_image_output_root) / record.image_id, rgb_u8)
                debug_image_count += 1
            feature = _extract_radio_feature_from_rgb(rgb_u8, radio)
            descriptor = _descriptor_from_feature(
                feature,
                pooling=str(args.pooling),
                gem_power=float(args.gem_power),
                normalize_tokens=bool(args.normalize_tokens),
                descriptor_power=float(args.descriptor_power),
                vlad_codebook=vlad_codebook,
                vlad_tokens_per_image=int(args.vlad_tokens_per_image),
                rng=rng,
            )
            accepted_records.append(record)
            accepted_ordinals.append(batch_ordinals[record_idx])
            descriptors.append(descriptor)
        if descriptors:
            descriptor_array = np.stack(descriptors, axis=0).astype(np.float32, copy=False)
            if pca_transform is not None:
                descriptor_array = apply_pca_whitening_transform(
                    descriptor_array,
                    pca_transform,
                    descriptor_power=1.0,
                    normalize=True,
                )
            if descriptor_array.shape[1] != query_descriptors.descriptors.shape[1]:
                raise ValueError(
                    "rendered descriptor dimension does not match query descriptor dimension: "
                    f"{descriptor_array.shape[1]} vs {query_descriptors.descriptors.shape[1]}"
                )
            descriptor_dim = int(descriptor_array.shape[1])
            topk.update(
                records=accepted_records,
                descriptors=descriptor_array,
                start_ordinal=int(accepted_ordinals[0]),
                ordinals=accepted_ordinals,
            )
            accepted_count += len(accepted_records)
        if int(args.progress_every_shards) > 0 and shard_count % int(args.progress_every_shards) == 0:
            print(
                json.dumps(
                    {
                        "shard": shard_count,
                        "rendered": rendered_count,
                        "accepted": accepted_count,
                        "skipped_low_alpha": skipped_low_alpha,
                        "elapsed_sec": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        batch_records = []
        batch_ordinals = []

    for streamed in iterator:
        batch_records.append(streamed.record)
        batch_ordinals.append(int(streamed.ordinal))
        if len(batch_records) >= int(args.records_per_shard):
            flush_batch()
    flush_batch()

    if accepted_count <= 0:
        raise ValueError(f"no rendered pose passed filtering; skipped_low_alpha={skipped_low_alpha}")

    output_bank = build_streaming_vpr_candidate_bank(
        topk,
        query_pose_file=Path(args.query_pose_file),
        protocol_name=str(args.protocol_name),
        descriptor_pooling=str(args.pooling),
    )
    output_candidate_bank = Path(args.output_candidate_bank)
    output_bank.to_jsonl(output_candidate_bank)
    top_ks = tuple(sorted({1, 5, 10, min(20, int(args.top_k)), int(args.top_k)}))
    summary.update(
        {
            "dry_run_count_only": False,
            "query_descriptors": str(args.query_descriptors),
            "gaussian_rgb_ply": str(args.gaussian_rgb_ply),
            "camera_model_dir": str(args.camera_model_dir),
            "camera_source": str(camera_source),
            "output_candidate_bank": str(output_candidate_bank),
            "records_per_shard": int(args.records_per_shard),
            "top_k": int(args.top_k),
            "render_width": int(args.render_width),
            "render_height": int(args.render_height),
            "radio_version": str(args.radio_version),
            "pooling": str(args.pooling),
            "gem_power": float(args.gem_power),
            "normalize_tokens": bool(args.normalize_tokens),
            "descriptor_power": float(args.descriptor_power),
            "vlad_codebook_input": str(args.vlad_codebook_input),
            "vlad_tokens_per_image": int(args.vlad_tokens_per_image),
            "pca_whitening_input": str(args.pca_whitening_input),
            "min_alpha_coverage": float(args.min_alpha_coverage),
            "shard_count": int(shard_count),
            "rendered_pose_count": int(rendered_count),
            "accepted_pose_count": int(accepted_count),
            "skipped_low_alpha": int(skipped_low_alpha),
            "processed_candidate_count": int(topk.processed_candidate_count),
            "descriptor_dim": None if descriptor_dim is None else int(descriptor_dim),
            "query_count": int(len(query_descriptors.image_ids)),
            "candidate_count": int(len(output_bank.candidates)),
            "elapsed_sec": float(time.perf_counter() - started),
        }
    )
    summary["metrics_25cm_5deg"] = summarize_reference_pose_retrieval(
        output_bank,
        top_ks=top_ks,
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        rot_cost_weight=0.1,
    )
    summary["metrics_50cm_10deg"] = summarize_reference_pose_retrieval(
        output_bank,
        top_ks=top_ks,
        translation_threshold_m=0.5,
        rotation_threshold_deg=10.0,
        rot_cost_weight=0.1,
    )
    _write_summary(Path(args.summary_json), summary)


if __name__ == "__main__":
    main()
