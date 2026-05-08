#!/usr/bin/env python3
"""Export learned query-student fine/coarse features in RADIO dual-feature format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from feature_extract import (
    JointRADIOQueryDataset,
    RadioQueryStudent,
    RetrievalTeacherStore,
    TeacherFeatureStore,
    build_all_records,
    load_config,
    safe_torch_load,
    sample_name_to_feature_stem,
    split_records,
)
from data.radio_loc_dataset import read_colmap_images
from feature_field.utils.project_config import load_yaml_config
from feature_field.utils.project_paths import resolve_repo_path


DEFAULT_JOINT_OVERRIDE_COMPONENTS = [
    "fine_decoder",
    "feat_sharp",
    "hash_grid_mlp",
    "gaussian_latent",
    "fsm",
]

DCFF_MANIFEST_BASE_KEYS = (
    "checkpoint",
    "ply_path",
    "feature_dim",
    "fine_feature_dim",
    "coarse_feature_dim",
    "latent_dim",
    "fine_latent_dim",
    "coarse_latent_dim",
    "render_width",
    "render_height",
    "coarse_smoothing_kernel",
    "fine_decoder_override",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export learned query-student dual features.")
    parser.add_argument("--config", required=True, help="Feature-track YAML config")
    parser.add_argument("--checkpoint", required=True, help="Joint query-student checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output feature directory")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of records to export")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-retrieval-head", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--name-mode",
        choices=["teacher_idx", "colmap_image_id"],
        default="teacher_idx",
        help="Filename stem mode for exported features",
    )
    parser.add_argument(
        "--colmap-dir",
        default=None,
        help="COLMAP sparse model directory used to resolve image_id filenames",
    )
    parser.add_argument("--train-init-cache", default=None, help="Optional fixed train init cache for manifest")
    parser.add_argument("--val-init-cache", default=None, help="Optional fixed val/test init cache for manifest")
    parser.add_argument(
        "--joint-override-components",
        nargs="*",
        default=DEFAULT_JOINT_OVERRIDE_COMPONENTS,
        help="DCFF components overridden by this joint checkpoint in pose_refine",
    )
    parser.add_argument(
        "--skip-localization-manifest",
        action="store_true",
        help="Do not write localization_manifest.json",
    )
    return parser.parse_args()


def _normalize_name(name: str) -> str:
    return Path(name).as_posix().replace("\\", "/")


def sha256_file_or_none(path: str | os.PathLike | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with open(candidate, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_entry(path: str | None) -> dict | None:
    if not path:
        return None
    resolved = str(Path(path).resolve())
    entry = {"path": resolved}
    digest = sha256_file_or_none(resolved)
    if digest is not None:
        entry["sha256"] = digest
    return entry


def _base_dcff_manifest_from_map_config(cfg: dict) -> dict:
    map_cfg_path = cfg.get("map_supervision", {}).get("config_path")
    if not map_cfg_path:
        return {}
    map_cfg = load_yaml_config(str(map_cfg_path))
    dcff_cfg = map_cfg.get("dcff", {})
    if not isinstance(dcff_cfg, dict):
        return {}
    base_dcff = {}
    for key in DCFF_MANIFEST_BASE_KEYS:
        if key not in dcff_cfg or dcff_cfg[key] is None:
            continue
        if key in {"checkpoint", "ply_path"}:
            resolved = resolve_repo_path(dcff_cfg[key], enforce_local=False)
            base_dcff[key] = str(resolved) if resolved is not None else str(dcff_cfg[key])
        else:
            base_dcff[key] = dcff_cfg[key]
    return base_dcff


def build_localization_manifest(
    *,
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    cfg: dict,
    train_init_cache: str | None = None,
    val_init_cache: str | None = None,
    joint_override_components: list[str] | tuple[str, ...] | None = None,
) -> dict:
    fine_key = (
        cfg.get("export", {}).get("fine_key")
        or cfg.get("model", {}).get("export_fine_key")
        or cfg.get("map_supervision", {}).get("query_fine_key")
        or "fine"
    )
    init_caches = {}
    train_cache_entry = _cache_entry(train_init_cache)
    val_cache_entry = _cache_entry(val_init_cache)
    if train_cache_entry is not None:
        init_caches["train"] = train_cache_entry
    if val_cache_entry is not None:
        init_caches["val"] = val_cache_entry
    teacher_corr_entry = _cache_entry(
        cfg.get("dataset", {}).get("teacher_correspondence_path")
        or cfg.get("map_supervision", {}).get("teacher_correspondence_path")
    )
    teacher_corr_train_entry = _cache_entry(cfg.get("dataset", {}).get("teacher_correspondence_train_path"))
    teacher_corr_val_entry = _cache_entry(cfg.get("dataset", {}).get("teacher_correspondence_val_path"))
    resolved_config = str(Path(config_path).resolve())
    resolved_checkpoint = str(Path(checkpoint_path).resolve())
    base_dcff = _base_dcff_manifest_from_map_config(cfg)
    return {
        "schema_version": 1,
        "method_boundary": "radio_student_dcff_multiscale_featuremetric_corr_wls",
        "teacher_only": {
            "netvlad_render_loftr_pnp": "init_cache_and_pseudo_label_only",
            "loftr_pnp_sparse_correspondences": "training_supervision_only",
        },
        "dataset": {
            "feature_dir": str(Path(output_dir).resolve()),
            **({"train_init_poses_path": init_caches["train"]["path"]} if "train" in init_caches else {}),
            **({"val_init_poses_path": init_caches["val"]["path"]} if "val" in init_caches else {}),
            **(
                {"teacher_correspondence_path": teacher_corr_entry["path"]}
                if teacher_corr_entry is not None
                else {}
            ),
            **(
                {"teacher_correspondence_train_path": teacher_corr_train_entry["path"]}
                if teacher_corr_train_entry is not None
                else {}
            ),
            **(
                {"teacher_correspondence_val_path": teacher_corr_val_entry["path"]}
                if teacher_corr_val_entry is not None
                else {}
            ),
        },
        "dcff": {
            **base_dcff,
            "joint_checkpoint": resolved_checkpoint,
            "joint_override_components": list(joint_override_components or DEFAULT_JOINT_OVERRIDE_COMPONENTS),
        },
        "export": {
            "fine_key": str(fine_key),
        },
        "init_caches": init_caches,
        "source": {
            "config_path": resolved_config,
            "config_sha256": sha256_file_or_none(resolved_config),
            "checkpoint_path": resolved_checkpoint,
            "checkpoint_sha256": sha256_file_or_none(resolved_checkpoint),
        },
    }


def build_colmap_name_to_id(colmap_dir: str | None, records: list[dict]) -> dict[str, int]:
    if not colmap_dir:
        return {}
    images = read_colmap_images(str(Path(colmap_dir) / "images.bin"))
    mapping: dict[str, int] = {}
    record_names = {_normalize_name(r["sample_name"]) for r in records}
    for image_id, meta in images.items():
        norm = _normalize_name(meta.name)
        keys = {norm, os.path.basename(norm)}
        if norm.startswith("images/"):
            keys.add(norm[len("images/"):])
        if not record_names.intersection(keys):
            continue
        for key in keys:
            mapping.setdefault(key, int(image_id))
    return mapping


def apply_export_feature_dims(cfg: dict, teacher_store: TeacherFeatureStore) -> None:
    cfg.setdefault("model", {})
    cfg.setdefault("dataset", {})
    cfg["model"]["fine_feature_dim"] = int(cfg["model"].get("fine_feature_dim") or teacher_store.fine_feature_dim)
    cfg["model"]["coarse_feature_dim"] = int(cfg["model"].get("coarse_feature_dim") or teacher_store.coarse_feature_dim)
    cfg["dataset"]["teacher_feature_hw"] = list(teacher_store.feature_hw)
    cfg["dataset"]["teacher_coarse_feature_hw"] = list(teacher_store.coarse_feature_hw)
    cfg["dataset"]["feature_hw"] = list(cfg["dataset"].get("student_feature_hw") or teacher_store.feature_hw)
    cfg["dataset"]["coarse_feature_hw"] = list(
        cfg["dataset"].get("student_coarse_feature_hw") or teacher_store.coarse_feature_hw
    )


def build_model(cfg: dict, checkpoint: dict, device: torch.device) -> RadioQueryStudent:
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

    feature_dim = int(cfg["model"]["feature_dim"])
    fine_feature_dim = int(cfg["model"].get("fine_feature_dim") or feature_dim)
    coarse_feature_dim = int(cfg["model"].get("coarse_feature_dim") or feature_dim)
    model = RadioQueryStudent(
        feature_dim=feature_dim,
        fine_feature_dim=fine_feature_dim,
        coarse_feature_dim=coarse_feature_dim,
        base_channels=int(cfg["model"].get("base_channels", 32)),
        stage_dims=tuple(cfg["model"].get("stage_dims", [32, 64, 96, 128])),
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
        local_matcher_context_mode=str(cfg["model"].get("local_matcher_context_mode", "basic")),
        local_flow_head_enabled=bool(cfg["model"].get("local_flow_head_enabled", False)),
        local_flow_head_radius=int(cfg["model"].get("local_flow_head_radius", cfg["model"].get("local_matcher_radius", 4))),
        local_flow_head_hidden_dim=int(cfg["model"].get("local_flow_head_hidden_dim", 64)),
        local_flow_head_zero_init=bool(cfg["model"].get("local_flow_head_zero_init", True)),
        local_flow_head_max_flow=cfg["model"].get("local_flow_head_max_flow"),
        local_flow_head_base_flow_mode=str(cfg["model"].get("local_flow_head_base_flow_mode", "none")),
        local_flow_head_base_temperature=float(cfg["model"].get("local_flow_head_base_temperature", 0.05)),
        local_flow_head_context_mode=str(cfg["model"].get("local_flow_head_context_mode", "basic")),
        local_corr_projector_enabled=bool(cfg["model"].get("local_corr_projector_enabled", False)),
        local_corr_projector_hidden_dim=int(cfg["model"].get("local_corr_projector_hidden_dim", 96)),
        local_corr_projector_output_dim=cfg["model"].get("local_corr_projector_output_dim"),
        local_corr_projector_zero_init=bool(cfg["model"].get("local_corr_projector_zero_init", True)),
        local_corr_projector_l2_normalize=bool(cfg["model"].get("local_corr_projector_l2_normalize", True)),
        local_corr_projector_domain_adapter=bool(cfg["model"].get("local_corr_projector_domain_adapter", False)),
        query_channel_gate_enabled=bool(cfg["model"].get("query_channel_gate_enabled", False)),
        query_channel_gate_hidden_dim=cfg["model"].get("query_channel_gate_hidden_dim"),
        query_channel_gate_zero_init=bool(cfg["model"].get("query_channel_gate_zero_init", True)),
        apply_query_channel_gate=bool(cfg["model"].get("apply_query_channel_gate", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def select_records(cfg: dict, teacher_store: TeacherFeatureStore, split: str, limit: int | None) -> list[dict]:
    all_records = build_all_records(cfg["dataset"], teacher_store, allow_synthetic=False)
    if split == "all":
        records = all_records
    else:
        train_records, val_records = split_records(all_records, cfg["dataset"])
        records = train_records if split == "train" else val_records
    if limit is not None:
        records = records[:limit]
    return records


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
    )
    apply_export_feature_dims(cfg, teacher_store)
    records = select_records(cfg, teacher_store, args.split, args.limit)
    if not records:
        raise RuntimeError("No records selected for export")

    retrieval_teacher_store = None
    if args.save_retrieval_head and cfg.get("retrieval", {}).get("enabled", False):
        retrieval_feature_dir = cfg["retrieval"].get("feature_dir")
        teacher_subdir = cfg["retrieval"].get("teacher_subdir", "cls")
        if retrieval_feature_dir:
            retrieval_teacher_store = RetrievalTeacherStore(
                retrieval_feature_dir,
                subdir=teacher_subdir,
                cache_in_memory=False,
            )

    colmap_name_to_id = build_colmap_name_to_id(args.colmap_dir or cfg["dataset"].get("colmap_dir"), records)

    dataset = JointRADIOQueryDataset(
        records,
        teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        retrieval_teacher_store=retrieval_teacher_store,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = safe_torch_load(args.checkpoint)
    model = build_model(cfg, checkpoint, device)
    fine_key = str(
        cfg.get("export", {}).get(
            "fine_key",
            cfg.get("model", {}).get("export_fine_key", "fine"),
        )
    )

    output_dir = Path(args.output_dir)
    fine_dir = output_dir / "fine_geo"
    coarse_dir = output_dir / "coarse_sem"
    fine_dir.mkdir(parents=True, exist_ok=True)
    coarse_dir.mkdir(parents=True, exist_ok=True)
    cls_dir = output_dir / "cls"
    if args.save_retrieval_head:
        cls_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    exported = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        teacher_indices = batch["teacher_idx"]
        sample_names = batch["sample_name"]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if bool(cfg["model"].get("teacher_fine_condition", False)):
                pred = model(rgb, teacher_fine=batch.get("teacher_fine").to(device, non_blocking=True))
            else:
                pred = model(rgb)

        if fine_key not in pred:
            raise KeyError(f"export fine_key={fine_key!r} not found in model outputs")
        fine = pred[fine_key].detach().float().cpu()
        coarse = pred["coarse"].detach().float().cpu()
        retrieval = pred.get("retrieval")
        if retrieval is not None:
            retrieval = retrieval.detach().float().cpu()

        for i in range(fine.shape[0]):
            teacher_idx = int(teacher_indices[i])
            sample_name = sample_names[i]
            export_idx = teacher_idx
            colmap_id = None
            if args.name_mode == "colmap_image_id":
                normalized = _normalize_name(sample_name)
                for key in (normalized, os.path.basename(normalized)):
                    if key in colmap_name_to_id:
                        colmap_id = colmap_name_to_id[key]
                        break
                if colmap_id is None:
                    raise KeyError(f"Could not resolve COLMAP image_id for sample_name={sample_name}")
                export_idx = colmap_id
            fine_i = fine[i].half()
            coarse_i = coarse[i].half()
            d_f, h_f, w_f = fine_i.shape
            d_c, h_c, w_c = coarse_i.shape
            fine_path = fine_dir / f"rgb_{export_idx}_fine_geo_{d_f}x{h_f}x{w_f}.pt"
            coarse_path = coarse_dir / f"rgb_{export_idx}_coarse_sem_{d_c}x{h_c}x{w_c}.pt"

            if not (args.skip_existing and fine_path.exists() and coarse_path.exists()):
                torch.save(fine_i, fine_path)
                torch.save(coarse_i, coarse_path)

            if args.save_retrieval_head and retrieval is not None:
                sample_stem = sample_name_to_feature_stem(sample_name)
                retrieval_i = retrieval[i].half().view(-1)
                cls_path = cls_dir / f"{sample_stem}_cls_{retrieval_i.numel()}.pt"
                if not (args.skip_existing and cls_path.exists()):
                    torch.save(retrieval_i, cls_path)
            exported += 1

    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "split": args.split,
        "limit": args.limit,
        "num_records": len(records),
        "batch_size": args.batch_size,
        "save_retrieval_head": bool(args.save_retrieval_head),
        "name_mode": args.name_mode,
        "colmap_dir": str(Path(args.colmap_dir).resolve()) if args.colmap_dir else cfg["dataset"].get("colmap_dir"),
        "feature_hw": list(cfg["dataset"]["feature_hw"]),
        "coarse_feature_hw": list(cfg["dataset"].get("coarse_feature_hw") or cfg["dataset"]["feature_hw"]),
        "teacher_feature_hw": list(cfg["dataset"].get("teacher_feature_hw") or cfg["dataset"]["feature_hw"]),
        "teacher_coarse_feature_hw": list(
            cfg["dataset"].get("teacher_coarse_feature_hw")
            or cfg["dataset"].get("coarse_feature_hw")
            or cfg["dataset"]["feature_hw"]
        ),
        "feature_dim": int(cfg["model"]["feature_dim"]),
        "fine_feature_dim": int(cfg["model"].get("fine_feature_dim") or cfg["model"]["feature_dim"]),
        "coarse_feature_dim": int(cfg["model"].get("coarse_feature_dim") or cfg["model"]["feature_dim"]),
    }
    export_index = [
        {
            "teacher_idx": int(record["teacher_idx"]),
            "sample_name": record["sample_name"],
            "colmap_image_id": colmap_name_to_id.get(_normalize_name(record["sample_name"])),
        }
        for record in records
    ]
    (output_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "export_index.json").write_text(json.dumps(export_index, indent=2) + "\n", encoding="utf-8")
    if not args.skip_localization_manifest:
        manifest = build_localization_manifest(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=str(output_dir),
            cfg=cfg,
            train_init_cache=args.train_init_cache,
            val_init_cache=args.val_init_cache,
            joint_override_components=args.joint_override_components,
        )
        (output_dir / "localization_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        f"Exported {exported} samples to {output_dir} "
        f"(split={args.split}, feature_hw={tuple(cfg['dataset']['feature_hw'])}, "
        f"coarse_hw={tuple(cfg['dataset'].get('coarse_feature_hw') or cfg['dataset']['feature_hw'])}, "
        f"dims={cfg['model'].get('fine_feature_dim')}/{cfg['model'].get('coarse_feature_dim')})"
    )


if __name__ == "__main__":
    main()
