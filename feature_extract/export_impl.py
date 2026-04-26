#!/usr/bin/env python3
"""Export learned query-student fine/coarse features in RADIO dual-feature format."""

from __future__ import annotations

import argparse
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
    return parser.parse_args()


def _normalize_name(name: str) -> str:
    return Path(name).as_posix().replace("\\", "/")


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

    model = RadioQueryStudent(
        feature_dim=int(cfg["model"]["feature_dim"]),
        base_channels=int(cfg["model"].get("base_channels", 32)),
        stage_dims=tuple(cfg["model"].get("stage_dims", [32, 64, 96, 128])),
        output_hw=tuple(cfg["dataset"]["feature_hw"]),
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
            pred = model(rgb)

        fine = pred["fine"].detach().float().cpu()
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
        "feature_dim": int(cfg["model"]["feature_dim"]),
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

    print(
        f"Exported {exported} samples to {output_dir} "
        f"(split={args.split}, feature_hw={tuple(cfg['dataset']['feature_hw'])}, dim={cfg['model']['feature_dim']})"
    )


if __name__ == "__main__":
    main()
