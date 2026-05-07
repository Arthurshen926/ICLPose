#!/usr/bin/env python3
"""Evaluate query-student descriptors for retrieval quality."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_retrieval_dataset import (  # noqa: E402
    _find_cls_feature_path,
    _load_cls_descriptor,
    list_colmap_split_samples,
)
from feature_extract import (  # noqa: E402
    JointRADIOQueryDataset,
    RadioQueryStudent,
    TeacherFeatureStore,
    build_all_records,
    load_config,
    safe_torch_load,
    split_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate query-student retrieval descriptors.")
    parser.add_argument("--config", required=True, help="Feature-track YAML config")
    parser.add_argument("--checkpoint", required=True, help="Student checkpoint path")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--colmap-dir", default=None, help="Optional COLMAP sparse directory override")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path. Defaults to /root/result/loc/<exp_name>_retrieval_probe.json",
    )
    return parser.parse_args()


def _camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    R = pose_w2c[:3, :3]
    t = pose_w2c[:3, 3]
    return -R.T @ t


def _rotation_error_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    R_rel = pose_a[:3, :3].T @ pose_b[:3, :3]
    cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    return float(np.degrees(np.arccos(cos_angle)))


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-8, None)


@torch.no_grad()
def infer_descriptors(
    model: RadioQueryStudent,
    records: List[Dict],
    teacher_store: TeacherFeatureStore,
    cfg: Dict,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, np.ndarray]:
    dataset = JointRADIOQueryDataset(
        records,
        teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    outputs: Dict[str, List[torch.Tensor] | List[str]] = {
        "teacher_fine_avg": [],
        "teacher_coarse_avg": [],
        "teacher_concat_avg": [],
        "student_fine_avg": [],
        "student_coarse_avg": [],
        "student_concat_avg": [],
        "sample_name": [],
    }
    has_retrieval_head = False

    use_amp = device.type == "cuda"
    model.eval()
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if bool(cfg["model"].get("teacher_fine_condition", False)):
                pred = model(rgb, teacher_fine=batch.get("teacher_fine").to(device, non_blocking=True))
            else:
                pred = model(rgb)

        teacher_fine = batch["teacher_fine"].float().mean(dim=(-1, -2))
        teacher_coarse = batch["teacher_coarse"].float().mean(dim=(-1, -2))
        student_fine = pred["fine"].float().mean(dim=(-1, -2)).cpu()
        student_coarse = pred["coarse"].float().mean(dim=(-1, -2)).cpu()

        teacher_fine = torch.nn.functional.normalize(teacher_fine, dim=1)
        teacher_coarse = torch.nn.functional.normalize(teacher_coarse, dim=1)
        student_fine = torch.nn.functional.normalize(student_fine, dim=1)
        student_coarse = torch.nn.functional.normalize(student_coarse, dim=1)

        outputs["teacher_fine_avg"].append(teacher_fine)
        outputs["teacher_coarse_avg"].append(teacher_coarse)
        outputs["teacher_concat_avg"].append(
            torch.nn.functional.normalize(torch.cat([teacher_fine, teacher_coarse], dim=1), dim=1)
        )
        outputs["student_fine_avg"].append(student_fine)
        outputs["student_coarse_avg"].append(student_coarse)
        outputs["student_concat_avg"].append(
            torch.nn.functional.normalize(torch.cat([student_fine, student_coarse], dim=1), dim=1)
        )

        if "retrieval" in pred:
            has_retrieval_head = True
            outputs.setdefault("student_retrieval", [])
            outputs["student_retrieval"].append(pred["retrieval"].float().cpu())

        outputs["sample_name"].extend(batch["sample_name"])

    result: Dict[str, np.ndarray] = {"sample_name": outputs["sample_name"]}  # type: ignore[assignment]
    tensor_keys = [
        "teacher_fine_avg",
        "teacher_coarse_avg",
        "teacher_concat_avg",
        "student_fine_avg",
        "student_coarse_avg",
        "student_concat_avg",
    ]
    if has_retrieval_head:
        tensor_keys.append("student_retrieval")
    for key in tensor_keys:
        result[key] = torch.cat(outputs[key], dim=0).numpy().astype(np.float32)  # type: ignore[arg-type]
    return result


def attach_pose_and_cls(
    descs: Dict[str, np.ndarray],
    records: List[Dict],
    split_file: str,
    retrieval_feature_dir: str | None,
    colmap_dir: str,
) -> None:
    split_samples = list_colmap_split_samples(colmap_dir, split_file)
    name_to_pose = {sample["image_name"].replace("\\", "/"): sample["pose_w2c"] for sample in split_samples}
    descs["pose_w2c"] = np.stack([name_to_pose[record["normalized_name"]] for record in records]).astype(np.float32)

    if retrieval_feature_dir and os.path.isdir(os.path.join(retrieval_feature_dir, "cls")):
        cls_descs = []
        for record in records:
            stem = Path(record["normalized_name"]).with_suffix("").as_posix().replace("/", "_")
            cls_path = _find_cls_feature_path(retrieval_feature_dir, stem)
            if cls_path is None:
                raise FileNotFoundError(f"Missing CLS descriptor for {record['normalized_name']}")
            cls_descs.append(_load_cls_descriptor(cls_path))
        descs["cls"] = np.stack(cls_descs).astype(np.float32)


def summarize_retrieval(
    train_descs: Dict[str, np.ndarray],
    val_descs: Dict[str, np.ndarray],
    keys: List[str],
    topk: List[int],
) -> Dict[str, Dict]:
    train_centers = np.stack([_camera_center_from_w2c(pose) for pose in train_descs["pose_w2c"]]).astype(np.float32)
    val_centers = np.stack([_camera_center_from_w2c(pose) for pose in val_descs["pose_w2c"]]).astype(np.float32)

    results = {}
    for key in keys:
        db = _normalize_rows(train_descs[key])
        query = _normalize_rows(val_descs[key])
        sims = query @ db.T
        order = np.argsort(-sims, axis=1)

        entry = {}
        for k in topk:
            best_trans = []
            best_rot = []
            hit_1m = 0
            hit_2m = 0
            hit_5deg_1m = 0

            for idx in range(order.shape[0]):
                cand_indices = order[idx, :k]
                errs = []
                for db_idx in cand_indices:
                    trans_mm = float(np.linalg.norm(train_centers[db_idx] - val_centers[idx]) * 1000.0)
                    rot_deg = _rotation_error_deg(train_descs["pose_w2c"][db_idx], val_descs["pose_w2c"][idx])
                    errs.append((trans_mm, rot_deg))
                best_trans.append(min(t for t, _ in errs))
                best_rot.append(min(r for _, r in errs))
                hit_1m += any(t < 1000.0 for t, _ in errs)
                hit_2m += any(t < 2000.0 for t, _ in errs)
                hit_5deg_1m += any((t < 1000.0 and r < 5.0) for t, r in errs)

            n = max(1, len(best_trans))
            entry[f"top{k}"] = {
                "trans_median_mm": float(np.median(best_trans)),
                "rot_median_deg": float(np.median(best_rot)),
                "hit_1m_pct": float(hit_1m / n * 100.0),
                "hit_2m_pct": float(hit_2m / n * 100.0),
                "hit_5deg_1m_pct": float(hit_5deg_1m / n * 100.0),
            }
        results[key] = entry
    return results


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_train_samples is not None:
        cfg["dataset"]["max_train_samples"] = None if args.max_train_samples < 0 else args.max_train_samples
    if args.max_val_samples is not None:
        cfg["dataset"]["max_val_samples"] = None if args.max_val_samples < 0 else args.max_val_samples

    requested_device = args.device
    device = torch.device(
        requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
    )
    feature_dim = int(cfg["model"]["feature_dim"])
    cfg["model"]["fine_feature_dim"] = int(cfg["model"].get("fine_feature_dim") or teacher_store.fine_feature_dim)
    cfg["model"]["coarse_feature_dim"] = int(cfg["model"].get("coarse_feature_dim") or teacher_store.coarse_feature_dim)
    cfg["dataset"]["feature_hw"] = list(teacher_store.feature_hw)
    cfg["dataset"]["coarse_feature_hw"] = list(teacher_store.coarse_feature_hw)
    colmap_dir = args.colmap_dir or str(Path(cfg["dataset"]["source_dir"]) / "sparse" / "0")
    all_records = build_all_records(cfg["dataset"], teacher_store, allow_synthetic=False)
    train_records, val_records = split_records(all_records, cfg["dataset"])

    checkpoint = safe_torch_load(args.checkpoint)
    retrieval_cfg = cfg.get("retrieval", {})
    retrieval_dim = None
    retrieval_hidden_dim = None
    retrieval_dropout = 0.0
    retrieval_l2_normalize = True
    if retrieval_cfg.get("enabled", False):
        retrieval_dim = int(retrieval_cfg.get("student_dim", 0) or 0)
        retrieval_hidden_dim = int(retrieval_cfg.get("hidden_dim", 0) or 0)
        retrieval_dropout = float(retrieval_cfg.get("dropout", 0.0))
        retrieval_l2_normalize = bool(retrieval_cfg.get("l2_normalize", True))

    model = RadioQueryStudent(
        in_channels=3,
        feature_dim=feature_dim,
        fine_feature_dim=int(cfg["model"].get("fine_feature_dim") or feature_dim),
        coarse_feature_dim=int(cfg["model"].get("coarse_feature_dim") or feature_dim),
        base_channels=int(cfg["model"]["base_channels"]),
        stage_dims=tuple(cfg["model"]["stage_dims"]),
        output_hw=tuple(cfg["dataset"]["feature_hw"]),
        coarse_output_hw=tuple(cfg["dataset"].get("coarse_feature_hw") or cfg["dataset"]["feature_hw"]),
        input_hw=tuple(cfg["dataset"]["input_hw"]),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        l2_normalize=bool(cfg["model"].get("l2_normalize", True)),
        predict_magnitude=bool(cfg["model"].get("predict_magnitude", False)),
        fine_init_norm=float(cfg["model"].get("fine_init_norm", 1.0)),
        coarse_init_norm=float(cfg["model"].get("coarse_init_norm", 1.0)),
        magnitude_min=float(cfg["model"].get("magnitude_min", 1e-4)),
        retrieval_dim=retrieval_dim,
        retrieval_hidden_dim=retrieval_hidden_dim,
        retrieval_dropout=retrieval_dropout,
        retrieval_l2_normalize=retrieval_l2_normalize,
        fine_low_level_skip=bool(cfg["model"].get("fine_low_level_skip", False)),
        fine_low_level_init=float(cfg["model"].get("fine_low_level_init", 0.0)),
        fine_highres_skip=bool(cfg["model"].get("fine_highres_skip", False)),
        fine_highres_source=str(cfg["model"].get("fine_highres_source", "stage2")),
        fine_highres_init=float(cfg["model"].get("fine_highres_init", 0.0)),
        fine_highres_zero_init=bool(cfg["model"].get("fine_highres_zero_init", False)),
        global_context_enabled=bool(cfg["model"].get("global_context_enabled", False)),
        global_context_zero_init=bool(cfg["model"].get("global_context_zero_init", True)),
        window_attention_layers=int(cfg["model"].get("window_attention_layers", 0)),
        window_attention_heads=int(cfg["model"].get("window_attention_heads", 8)),
        window_attention_size=int(cfg["model"].get("window_attention_size", 16)),
        window_attention_mlp_ratio=float(cfg["model"].get("window_attention_mlp_ratio", 2.0)),
        window_attention_dropout=float(cfg["model"].get("window_attention_dropout", 0.0)),
        window_attention_shift=bool(cfg["model"].get("window_attention_shift", False)),
        window_attention_zero_init=bool(cfg["model"].get("window_attention_zero_init", True)),
        fine_loc_head=bool(cfg["model"].get("fine_loc_head", False)),
        fine_loc_init=float(cfg["model"].get("fine_loc_init", 1.0)),
        fine_loc_zero_init=bool(cfg["model"].get("fine_loc_zero_init", True)),
        fine_loc_detach_base=bool(cfg["model"].get("fine_loc_detach_base", False)),
        fine_loc_highres_source=cfg["model"].get("fine_loc_highres_source"),
        fine_loc_highres_init=float(cfg["model"].get("fine_loc_highres_init", 1.0)),
        fine_loc_highres_zero_init=bool(cfg["model"].get("fine_loc_highres_zero_init", True)),
        fine_loc_highres_detach=bool(cfg["model"].get("fine_loc_highres_detach", True)),
        teacher_fine_condition=bool(cfg["model"].get("teacher_fine_condition", False)),
        teacher_fine_init=float(cfg["model"].get("teacher_fine_init", 1.0)),
        teacher_fine_zero_init=bool(cfg["model"].get("teacher_fine_zero_init", True)),
        teacher_fine_detach=bool(cfg["model"].get("teacher_fine_detach", True)),
        scene_coord_head=bool(cfg["model"].get("scene_coord_head", False)),
        scene_coord_zero_init=bool(cfg["model"].get("scene_coord_zero_init", True)),
        scene_coord_detach_base=bool(cfg["model"].get("scene_coord_detach_base", False)),
        scene_coord_use_pixel_grid=bool(cfg["model"].get("scene_coord_use_pixel_grid", False)),
        scene_coord_global_context=bool(cfg["model"].get("scene_coord_global_context", False)),
        local_matcher_enabled=bool(cfg["model"].get("local_matcher_enabled", False)),
        local_matcher_radius=int(cfg["model"].get("local_matcher_radius", 4)),
        local_matcher_hidden_dim=int(cfg["model"].get("local_matcher_hidden_dim", 64)),
        local_matcher_zero_init=bool(cfg["model"].get("local_matcher_zero_init", True)),
        local_matcher_residual_scale=float(cfg["model"].get("local_matcher_residual_scale", 1.0)),
        local_flow_head_enabled=bool(cfg["model"].get("local_flow_head_enabled", False)),
        local_flow_head_radius=int(cfg["model"].get("local_flow_head_radius", cfg["model"].get("local_matcher_radius", 4))),
        local_flow_head_hidden_dim=int(cfg["model"].get("local_flow_head_hidden_dim", 64)),
        local_flow_head_zero_init=bool(cfg["model"].get("local_flow_head_zero_init", True)),
        local_flow_head_max_flow=cfg["model"].get("local_flow_head_max_flow"),
        local_flow_head_base_flow_mode=str(cfg["model"].get("local_flow_head_base_flow_mode", "none")),
        local_flow_head_base_temperature=float(cfg["model"].get("local_flow_head_base_temperature", 0.05)),
        query_channel_gate_enabled=bool(cfg["model"].get("query_channel_gate_enabled", False)),
        query_channel_gate_hidden_dim=cfg["model"].get("query_channel_gate_hidden_dim"),
        query_channel_gate_zero_init=bool(cfg["model"].get("query_channel_gate_zero_init", True)),
        apply_query_channel_gate=bool(cfg["model"].get("apply_query_channel_gate", False)),
        pose_init_head=bool(cfg["model"].get("pose_init_head", False)),
        pose_init_hypotheses=int(cfg["model"].get("pose_init_hypotheses", 3)),
        pose_init_hidden_dim=cfg["model"].get("pose_init_hidden_dim"),
        pose_init_mode=str(cfg["model"].get("pose_init_mode", "direct")),
        pose_init_anchor_centers=cfg["model"].get("pose_init_anchor_centers"),
        pose_init_anchor_rotmats=cfg["model"].get("pose_init_anchor_rotmats"),
        pose_init_anchor_descriptors=cfg["model"].get("pose_init_anchor_descriptors"),
        pose_init_residual_scale=float(cfg["model"].get("pose_init_residual_scale", 1.5)),
        pose_init_temperature=float(cfg["model"].get("pose_init_temperature", 0.05)),
        pose_init_feature_bank_projector=str(cfg["model"].get("pose_init_feature_bank_projector", "mlp")),
        pose_init_feature_source=str(cfg["model"].get("pose_init_feature_source", "fine_coarse")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_descs = infer_descriptors(
        model,
        train_records,
        teacher_store,
        cfg,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_descs = infer_descriptors(
        model,
        val_records,
        teacher_store,
        cfg,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    retrieval_feature_dir = retrieval_cfg.get("feature_dir")
    attach_pose_and_cls(
        train_descs,
        train_records,
        cfg["dataset"]["train_split"],
        retrieval_feature_dir,
        colmap_dir,
    )
    attach_pose_and_cls(
        val_descs,
        val_records,
        cfg["dataset"]["val_split"],
        retrieval_feature_dir,
        colmap_dir,
    )

    keys = [key for key in train_descs.keys() if key not in {"sample_name", "pose_w2c"}]
    results = summarize_retrieval(train_descs, val_descs, keys, sorted(set(args.topk)))
    results["_meta"] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "num_train_samples": len(train_records),
        "num_val_samples": len(val_records),
        "device": str(device),
    }

    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path("/root/result/loc") / f"{Path(args.checkpoint).parent.parent.name}_retrieval_probe.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(json.dumps(results, indent=2))
    print(f"Saved retrieval probe to {output_json}")


if __name__ == "__main__":
    main()
