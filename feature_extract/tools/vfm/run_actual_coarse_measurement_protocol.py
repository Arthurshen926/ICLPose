"""Train/evaluate RGB measurement branch from actual MATCHA coarse proposals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from feature_extract.tools.vfm.apply_rgb_patch_measurement_to_match_table import (
    apply_rgb_patch_measurements_to_match_table,
)
from feature_extract.tools.vfm.build_rgb_patch_measurement_rows_from_match_table import (
    build_rgb_patch_measurement_rows_from_match_table,
)
from feature_extract.tools.vfm.eval_dense_depth_measurement_fusion import (
    _load_camera_from_model_dir,
    _query_pose_lookup,
    evaluate_dense_depth_measurement_fusion,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import train_rgb_patch_measurement_branch


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(base_dir) / value


def _validate_render_cache_manifest(
    manifest_csv: Path,
    *,
    base_dir: Path,
    required_query_ids: set[str],
) -> dict[str, Any]:
    rows = _read_csv(Path(manifest_csv))
    available: dict[str, Path] = {}
    blank_path_count = 0
    missing_path_count = 0
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        cache_path = str(row.get("rgb_depth_cache_path", "")).strip()
        if not query_id:
            continue
        if not cache_path:
            blank_path_count += 1
            continue
        resolved = _resolve_path(cache_path, base_dir=base_dir)
        if not resolved.exists():
            missing_path_count += 1
            continue
        available[query_id] = resolved
    missing_queries = sorted(query_id for query_id in required_query_ids if query_id not in available)
    return {
        "manifest_csv": str(manifest_csv),
        "manifest_row_count": int(len(rows)),
        "available_cache_count": int(len(available)),
        "blank_path_count": int(blank_path_count),
        "missing_path_count": int(missing_path_count),
        "required_query_count": int(len(required_query_ids)),
        "missing_required_query_count": int(len(missing_queries)),
        "missing_required_query_examples": missing_queries[:10],
        "ok": bool(not missing_queries),
    }


def _query_ids_from_match_table(path: Path, *, max_rows: int | None = None) -> set[str]:
    out: set[str] = set()
    with Path(path).open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            query_id = str(row.get("query_id", "")).strip()
            if query_id:
                out.add(query_id)
            if max_rows is not None and index + 1 >= int(max_rows):
                break
    return out


def run_actual_coarse_measurement_protocol(
    *,
    train_match_table_csv: Path,
    train_render_cache_manifest_csv: Path,
    val_match_table_csv: Path,
    val_render_cache_manifest_csv: Path,
    query_pose_file: Path,
    camera_model_dir: Path,
    image_root: Path,
    output_dir: Path,
    query_image_width: int,
    query_image_height: int,
    render_image_width: int,
    render_image_height: int,
    match_table_query_image_width: int | None = None,
    match_table_query_image_height: int | None = None,
    search_radius_px: float = 2.0,
    context_radius_px: float = 8.0,
    step_px: float = 0.25,
    coarse_search_radius_px: float | None = None,
    coarse_step_px: float | None = None,
    steps: int = 1000,
    batch_size: int = 32,
    eval_batch_size: int | None = None,
    feature_dim: int = 32,
    hidden_dim: int | None = None,
    input_mode: str = "rgb",
    encoder_arch: str = "fpn",
    template_scale_factors: Sequence[float] = (1.0,),
    lr: float = 1e-3,
    epe_weight: float = 0.25,
    likelihood_loss_weight: float = 1.0,
    coarse_likelihood_loss_weight: float = 0.0,
    delta_loss_weight: float = 0.25,
    gated_delta_loss_weight: float = 0.25,
    gate_supervision_loss_weight: float = 0.1,
    gate_center_radius_px: float = 0.5,
    gate_full_radius_px: float = 2.0,
    gate_target_mode: str = "utility",
    gate_utility_temperature_px: float = 0.25,
    dustbin_bce_weight: float = 0.2,
    dustbin_positive_weight: float = 2.0,
    target_heatmap_sigma_px: float = 0.5,
    train_max_rows: int | None = None,
    val_max_rows: int | None = None,
    max_eval_rows: int | None = 2048,
    hard_negative_fraction: float = 0.25,
    render_patch_augmentation: str = "realistic",
    residual_balanced_sampling: bool = False,
    residual_sampling_bins_px: Sequence[float] = (),
    condition_on_prior_scale: bool = False,
    prior_scale_key: str = "",
    prior_scale_expert_centers_px: Sequence[float] = (),
    prior_scale_expert_projection: bool = False,
    prior_scale_expert_gate: str = "soft",
    prediction_heads: Sequence[str] = ("center", "likelihood_mode", "gated", "direct"),
    run_pose_eval: bool = True,
    pose_variants: Sequence[str] = ("center", "measurement", "oracle"),
    pose_solvers: Sequence[str] = ("ransac", "weighted", "covariance"),
    reprojection_error_px: float = 8.0,
    geometry_source: str = "prefer_world_xyz",
    device: str = "cuda",
    data_parallel_device_ids: Sequence[int] = (),
    base_dir: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    base = Path.cwd() if base_dir is None else Path(base_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    measurement_search_radius = float(search_radius_px)
    if coarse_search_radius_px is not None:
        measurement_search_radius += float(coarse_search_radius_px)
    train_query_ids = _query_ids_from_match_table(Path(train_match_table_csv), max_rows=train_max_rows)
    val_query_ids = _query_ids_from_match_table(Path(val_match_table_csv), max_rows=val_max_rows)
    train_cache_audit = _validate_render_cache_manifest(
        Path(train_render_cache_manifest_csv),
        base_dir=base,
        required_query_ids=train_query_ids,
    )
    val_cache_audit = _validate_render_cache_manifest(
        Path(val_render_cache_manifest_csv),
        base_dir=base,
        required_query_ids=val_query_ids,
    )
    if not train_cache_audit["ok"] or not val_cache_audit["ok"]:
        summary = {
            "stage": "actual_coarse_measurement_protocol",
            "status": "blocked_missing_render_cache",
            "train_render_cache_audit": train_cache_audit,
            "val_render_cache_audit": val_cache_audit,
            "outputs": {"summary": str(output / "summary.json")},
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        raise ValueError(
            "render cache manifest does not cover required query ids; "
            f"train_missing={train_cache_audit['missing_required_query_count']}, "
            f"val_missing={val_cache_audit['missing_required_query_count']}"
        )

    rows_train_dir = output / "rows_train"
    rows_val_dir = output / "rows_val"
    train_rows_summary = build_rgb_patch_measurement_rows_from_match_table(
        match_table_csv=Path(train_match_table_csv),
        query_pose_file=Path(query_pose_file),
        camera_model_dir=Path(camera_model_dir),
        output_dir=rows_train_dir,
        max_rows=train_max_rows,
        measurement_search_radius_px=measurement_search_radius,
        requested_residual_bin_px=1.0,
        proposal_source="actual_matcha_coarse_train",
        match_table_query_image_width=match_table_query_image_width,
        match_table_query_image_height=match_table_query_image_height,
        measurement_query_image_width=int(query_image_width) if match_table_query_image_width is not None else None,
        measurement_query_image_height=int(query_image_height) if match_table_query_image_height is not None else None,
    )
    val_rows_summary = build_rgb_patch_measurement_rows_from_match_table(
        match_table_csv=Path(val_match_table_csv),
        query_pose_file=Path(query_pose_file),
        camera_model_dir=Path(camera_model_dir),
        output_dir=rows_val_dir,
        max_rows=val_max_rows,
        measurement_search_radius_px=measurement_search_radius,
        requested_residual_bin_px=1.0,
        proposal_source="actual_matcha_coarse_val",
        match_table_query_image_width=match_table_query_image_width,
        match_table_query_image_height=match_table_query_image_height,
        measurement_query_image_width=int(query_image_width) if match_table_query_image_width is not None else None,
        measurement_query_image_height=int(query_image_height) if match_table_query_image_height is not None else None,
    )

    train_dir = output / "train_measurement"
    train_summary = train_rgb_patch_measurement_branch(
        rows_csv=rows_train_dir / "measurement_rows.csv",
        val_rows_csv=rows_val_dir / "measurement_rows.csv",
        render_cache_manifest_csv=Path(train_render_cache_manifest_csv),
        val_render_cache_manifest_csv=Path(val_render_cache_manifest_csv),
        image_root=Path(image_root),
        output_dir=train_dir,
        image_width=int(query_image_width),
        image_height=int(query_image_height),
        query_image_width=int(query_image_width),
        query_image_height=int(query_image_height),
        render_image_width=int(render_image_width),
        render_image_height=int(render_image_height),
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        step_px=float(step_px),
        coarse_search_radius_px=None if coarse_search_radius_px is None else float(coarse_search_radius_px),
        coarse_step_px=None if coarse_step_px is None else float(coarse_step_px),
        steps=int(steps),
        batch_size=int(batch_size),
        eval_batch_size=eval_batch_size,
        feature_dim=int(feature_dim),
        hidden_dim=hidden_dim,
        input_mode=str(input_mode),
        encoder_arch=str(encoder_arch),
        template_scale_factors=tuple(float(value) for value in template_scale_factors),
        lr=float(lr),
        epe_weight=float(epe_weight),
        likelihood_loss_weight=float(likelihood_loss_weight),
        coarse_likelihood_loss_weight=float(coarse_likelihood_loss_weight),
        delta_loss_weight=float(delta_loss_weight),
        gated_delta_loss_weight=float(gated_delta_loss_weight),
        gate_supervision_loss_weight=float(gate_supervision_loss_weight),
        gate_center_radius_px=float(gate_center_radius_px),
        gate_full_radius_px=float(gate_full_radius_px),
        gate_target_mode=str(gate_target_mode),
        gate_utility_temperature_px=float(gate_utility_temperature_px),
        dustbin_bce_weight=float(dustbin_bce_weight),
        dustbin_positive_weight=float(dustbin_positive_weight),
        target_heatmap_sigma_px=float(target_heatmap_sigma_px),
        max_eval_rows=max_eval_rows,
        seed=int(seed),
        device=str(device),
        base_dir=base,
        query_source="real",
        render_patch_augmentation=str(render_patch_augmentation),
        hard_negative_fraction=float(hard_negative_fraction),
        residual_balanced_sampling=bool(residual_balanced_sampling),
        residual_sampling_bins_px=tuple(float(value) for value in residual_sampling_bins_px),
        condition_on_prior_scale=bool(condition_on_prior_scale),
        prior_scale_key=str(prior_scale_key),
        prior_scale_expert_centers_px=tuple(float(value) for value in prior_scale_expert_centers_px),
        prior_scale_expert_projection=bool(prior_scale_expert_projection),
        prior_scale_expert_gate=str(prior_scale_expert_gate),
        target_dustbin_filter="all",
        data_parallel_device_ids=[int(value) for value in data_parallel_device_ids],
    )
    checkpoint = Path(train_summary["outputs"]["checkpoint"])
    apply_summaries: dict[str, Any] = {}
    pose_summaries: dict[str, Any] = {}
    camera = _load_camera_from_model_dir(Path(camera_model_dir))
    query_pose_lookup = _query_pose_lookup(Path(query_pose_file))
    for head in prediction_heads:
        head_name = str(head)
        head_dir = output / f"apply_{head_name}"
        apply_summary = apply_rgb_patch_measurements_to_match_table(
            match_table_csv=rows_val_dir / "measurement_rows.csv",
            render_cache_manifest_csv=Path(val_render_cache_manifest_csv),
            image_root=Path(image_root),
            checkpoint=checkpoint,
            output_dir=head_dir,
            query_image_width=int(query_image_width),
            query_image_height=int(query_image_height),
            render_image_width=int(render_image_width),
            render_image_height=int(render_image_height),
            batch_size=max(1, int(eval_batch_size or batch_size)),
            device=str(device),
            base_dir=base,
            max_rows=val_max_rows,
            prediction_head=head_name,
            data_parallel_device_ids=[int(value) for value in data_parallel_device_ids],
        )
        apply_summaries[head_name] = apply_summary
        if run_pose_eval:
            pose_summaries[head_name] = evaluate_dense_depth_measurement_fusion(
                match_table_csv=head_dir / "match_table.csv",
                output_dir=output / f"pose_{head_name}",
                camera=camera,
                query_pose_w2c_by_id=query_pose_lookup,
                variants=tuple(str(value) for value in pose_variants),
                solvers=tuple(str(value) for value in pose_solvers),
                reprojection_error_px=float(reprojection_error_px),
                geometry_source=str(geometry_source),
                strict_measurement_schema=True,
            )

    summary = {
        "stage": "actual_coarse_measurement_protocol",
        "status": "complete",
        "train_match_table_csv": str(train_match_table_csv),
        "val_match_table_csv": str(val_match_table_csv),
        "query_pose_file": str(query_pose_file),
        "camera_model_dir": str(camera_model_dir),
        "image_root": str(image_root),
        "measurement_search_radius_px": float(measurement_search_radius),
        "match_table_query_image_width": None if match_table_query_image_width is None else int(match_table_query_image_width),
        "match_table_query_image_height": None if match_table_query_image_height is None else int(match_table_query_image_height),
        "geometry_source": str(geometry_source),
        "train_render_cache_audit": train_cache_audit,
        "val_render_cache_audit": val_cache_audit,
        "train_rows_summary": train_rows_summary,
        "val_rows_summary": val_rows_summary,
        "train_summary": train_summary,
        "apply_summaries": apply_summaries,
        "pose_summaries": pose_summaries,
        "outputs": {
            "rows_train": str(rows_train_dir / "measurement_rows.csv"),
            "rows_val": str(rows_val_dir / "measurement_rows.csv"),
            "checkpoint": str(checkpoint),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_match_table_csv", required=True)
    parser.add_argument("--train_render_cache_manifest_csv", required=True)
    parser.add_argument("--val_match_table_csv", default="")
    parser.add_argument("--val_render_cache_manifest_csv", default="")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--camera_model_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_image_width", type=int, required=True)
    parser.add_argument("--query_image_height", type=int, required=True)
    parser.add_argument("--render_image_width", type=int, required=True)
    parser.add_argument("--render_image_height", type=int, required=True)
    parser.add_argument("--match_table_query_image_width", type=int, default=0)
    parser.add_argument("--match_table_query_image_height", type=int, default=0)
    parser.add_argument("--search_radius_px", type=float, default=2.0)
    parser.add_argument("--context_radius_px", type=float, default=8.0)
    parser.add_argument("--step_px", type=float, default=0.25)
    parser.add_argument("--coarse_search_radius_px", type=float, default=-1.0)
    parser.add_argument("--coarse_step_px", type=float, default=-1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--input_mode", default="rgb", choices=("rgb", "rgb_graygrad", "norm_graygrad"))
    parser.add_argument("--encoder_arch", default="fpn", choices=("simple", "fpn"))
    parser.add_argument("--template_scale_factors", nargs="+", type=float, default=[1.0])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epe_weight", type=float, default=0.25)
    parser.add_argument("--likelihood_loss_weight", type=float, default=1.0)
    parser.add_argument("--coarse_likelihood_loss_weight", type=float, default=0.0)
    parser.add_argument("--delta_loss_weight", type=float, default=0.25)
    parser.add_argument("--gated_delta_loss_weight", type=float, default=0.25)
    parser.add_argument("--gate_supervision_loss_weight", type=float, default=0.1)
    parser.add_argument("--gate_center_radius_px", type=float, default=0.5)
    parser.add_argument("--gate_full_radius_px", type=float, default=2.0)
    parser.add_argument("--gate_target_mode", default="utility", choices=("residual", "utility"))
    parser.add_argument("--gate_utility_temperature_px", type=float, default=0.25)
    parser.add_argument("--dustbin_bce_weight", type=float, default=0.2)
    parser.add_argument("--dustbin_positive_weight", type=float, default=2.0)
    parser.add_argument("--target_heatmap_sigma_px", type=float, default=0.5)
    parser.add_argument("--train_max_rows", type=int, default=0)
    parser.add_argument("--val_max_rows", type=int, default=0)
    parser.add_argument("--max_eval_rows", type=int, default=2048)
    parser.add_argument("--hard_negative_fraction", type=float, default=0.25)
    parser.add_argument("--render_patch_augmentation", default="realistic", choices=("none", "realistic"))
    parser.add_argument("--residual_balanced_sampling", action="store_true")
    parser.add_argument("--residual_sampling_bins_px", nargs="+", type=float, default=[])
    parser.add_argument("--condition_on_prior_scale", action="store_true")
    parser.add_argument("--prior_scale_key", default="")
    parser.add_argument("--prior_scale_expert_centers_px", nargs="*", type=float, default=[])
    parser.add_argument("--prior_scale_expert_projection", action="store_true")
    parser.add_argument("--prior_scale_expert_gate", default="soft", choices=("soft", "hard"))
    parser.add_argument("--prediction_heads", nargs="+", default=["center", "likelihood_mode", "gated", "direct"])
    parser.add_argument("--no_pose_eval", action="store_true")
    parser.add_argument("--pose_variants", nargs="+", default=["center", "measurement", "oracle"])
    parser.add_argument("--pose_solvers", nargs="+", default=["ransac", "weighted", "covariance"])
    parser.add_argument("--reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--geometry_source", default="prefer_world_xyz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_parallel_device_ids", nargs="*", type=int, default=[])
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    val_match_table = Path(args.val_match_table_csv) if str(args.val_match_table_csv).strip() else Path(args.train_match_table_csv)
    val_manifest = (
        Path(args.val_render_cache_manifest_csv)
        if str(args.val_render_cache_manifest_csv).strip()
        else Path(args.train_render_cache_manifest_csv)
    )
    summary = run_actual_coarse_measurement_protocol(
        train_match_table_csv=Path(args.train_match_table_csv),
        train_render_cache_manifest_csv=Path(args.train_render_cache_manifest_csv),
        val_match_table_csv=val_match_table,
        val_render_cache_manifest_csv=val_manifest,
        query_pose_file=Path(args.query_pose_file),
        camera_model_dir=Path(args.camera_model_dir),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        query_image_width=int(args.query_image_width),
        query_image_height=int(args.query_image_height),
        render_image_width=int(args.render_image_width),
        render_image_height=int(args.render_image_height),
        match_table_query_image_width=int(args.match_table_query_image_width) if int(args.match_table_query_image_width) > 0 else None,
        match_table_query_image_height=int(args.match_table_query_image_height) if int(args.match_table_query_image_height) > 0 else None,
        search_radius_px=float(args.search_radius_px),
        context_radius_px=float(args.context_radius_px),
        step_px=float(args.step_px),
        coarse_search_radius_px=float(args.coarse_search_radius_px) if float(args.coarse_search_radius_px) >= 0.0 else None,
        coarse_step_px=float(args.coarse_step_px) if float(args.coarse_step_px) > 0.0 else None,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else None,
        feature_dim=int(args.feature_dim),
        hidden_dim=int(args.hidden_dim) if int(args.hidden_dim) > 0 else None,
        input_mode=str(args.input_mode),
        encoder_arch=str(args.encoder_arch),
        template_scale_factors=[float(value) for value in args.template_scale_factors],
        lr=float(args.lr),
        epe_weight=float(args.epe_weight),
        likelihood_loss_weight=float(args.likelihood_loss_weight),
        coarse_likelihood_loss_weight=float(args.coarse_likelihood_loss_weight),
        delta_loss_weight=float(args.delta_loss_weight),
        gated_delta_loss_weight=float(args.gated_delta_loss_weight),
        gate_supervision_loss_weight=float(args.gate_supervision_loss_weight),
        gate_center_radius_px=float(args.gate_center_radius_px),
        gate_full_radius_px=float(args.gate_full_radius_px),
        gate_target_mode=str(args.gate_target_mode),
        gate_utility_temperature_px=float(args.gate_utility_temperature_px),
        dustbin_bce_weight=float(args.dustbin_bce_weight),
        dustbin_positive_weight=float(args.dustbin_positive_weight),
        target_heatmap_sigma_px=float(args.target_heatmap_sigma_px),
        train_max_rows=int(args.train_max_rows) if int(args.train_max_rows) > 0 else None,
        val_max_rows=int(args.val_max_rows) if int(args.val_max_rows) > 0 else None,
        max_eval_rows=int(args.max_eval_rows) if int(args.max_eval_rows) > 0 else None,
        hard_negative_fraction=float(args.hard_negative_fraction),
        render_patch_augmentation=str(args.render_patch_augmentation),
        residual_balanced_sampling=bool(args.residual_balanced_sampling),
        residual_sampling_bins_px=[float(value) for value in args.residual_sampling_bins_px],
        condition_on_prior_scale=bool(args.condition_on_prior_scale),
        prior_scale_key=str(args.prior_scale_key),
        prior_scale_expert_centers_px=[float(value) for value in args.prior_scale_expert_centers_px],
        prior_scale_expert_projection=bool(args.prior_scale_expert_projection),
        prior_scale_expert_gate=str(args.prior_scale_expert_gate),
        prediction_heads=[str(value) for value in args.prediction_heads],
        run_pose_eval=not bool(args.no_pose_eval),
        pose_variants=[str(value) for value in args.pose_variants],
        pose_solvers=[str(value) for value in args.pose_solvers],
        reprojection_error_px=float(args.reprojection_error_px),
        geometry_source=str(args.geometry_source),
        device=str(args.device),
        data_parallel_device_ids=[int(value) for value in args.data_parallel_device_ids],
        base_dir=Path(args.base_dir),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
