"""Distill RADIO DINO/SAM/SigLIP readouts into one localization adaptor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from feature_extract.tools.vfm.evaluate_v6_retrieval_pose_basin import _region_geometry
from feature_extract.tools.vfm.train_v6_metric_encoder import _load_views
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_atlas import MapletFeatureAtlasBank
from feature_extract.vfm.localization_v8.multi_teacher_student import (
    MapletRetrievalAdaptor,
    MapletRetrievalAdaptorConfig,
    save_maplet_retrieval_adaptor,
    transform_canonical_maplet_bank,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    encode_radio_final_regions,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--teacher_cache", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--atlas_geometry", required=True)
    parser.add_argument("--identity_bank", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--output_adaptor", required=True)
    parser.add_argument("--output_bank", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--distillation_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _compress_teacher(value: np.ndarray, dimensions: int = 64) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    width = int(array.shape[1])
    if width % int(dimensions):
        raise ValueError("teacher dimension is not divisible by compact width")
    compact = array.reshape(array.shape[0], dimensions, width // dimensions).mean(axis=2)
    return compact / np.maximum(np.linalg.norm(compact, axis=1, keepdims=True), 1e-8)


def _metrics(model, features, labels, prototypes, device: str) -> dict[str, float]:
    valid = labels >= 0
    if not np.any(valid):
        return {"count": 0, "recall_at_1": 0.0, "recall_at_5": 0.0}
    with torch.inference_mode():
        query = model(torch.from_numpy(features[valid]).to(device))
        proto = model(torch.from_numpy(prototypes).to(device))
        scores = query @ proto.T
        ranked = torch.argsort(scores, dim=1, descending=True)[:, :5].cpu().numpy()
    truth = labels[valid, None]
    return {
        "count": int(np.sum(valid)),
        "recall_at_1": float(np.mean(ranked[:, :1] == truth)),
        "recall_at_5": float(np.mean(np.any(ranked == truth, axis=1))),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = [Path(args.output_adaptor), Path(args.output_bank), Path(args.summary_json)]
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError("refusing to overwrite V8 student artifacts")
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    atlas = MapletFeatureAtlasBank.load_npz(Path(args.atlas_geometry))
    bank = SurfaceRetrievalMapletBank.load_npz(Path(args.identity_bank))
    mapper, mapper_metadata = load_surface_maplet_mapper(
        Path(args.surface_mapper), device=str(args.device)
    )
    views = _load_views(Path(args.contributors), atlas, Path(args.image_root))
    teacher_dir = Path(args.teacher_cache)
    config = RadioFinalRegionConfig(
        pool_sizes=tuple(mapper_metadata.get("pool_sizes", (1, 3, 5, 9))),
        pool_weights=tuple(mapper_metadata.get("pool_weights", (0.4, 0.3, 0.2, 0.1))),
        global_context_weight=float(mapper_metadata.get("global_context_weight", 0.0)),
    )
    row_by_id = {int(value): row for row, value in enumerate(bank.maplet_ids.tolist())}
    flat_cells = atlas.height * atlas.width
    all_base, all_teacher, all_labels, all_trajectories = [], [], [], []
    teacher_hash = hashlib.sha256()
    for view in views:
        teacher_path = teacher_dir / (view.image_id.replace("/", "__") + ".npz")
        if not teacher_path.exists():
            continue
        teacher_hash.update(teacher_path.read_bytes())
        with np.load(teacher_path, allow_pickle=False) as teacher:
            token_xy = np.asarray(teacher["token_xy"], dtype=np.float32)
            dino = np.asarray(teacher["dino_v3_7b"], dtype=np.float32)
            sam = np.asarray(teacher["sam3"], dtype=np.float32)
            siglip = np.asarray(teacher["siglip2-g"], dtype=np.float32)
            summary = np.broadcast_to(
                np.asarray(teacher["siglip2-g_summary"], dtype=np.float32)[None],
                (token_xy.shape[0], 1536),
            ).copy()
        base_map = mapper.project(view.radio.numpy()).measurement_context
        base = encode_radio_final_regions(base_map, token_xy, config).astype(np.float32)
        xy, extent = _region_geometry(
            token_xy,
            token_width=int(view.radio.shape[2]), token_height=int(view.radio.shape[1]),
            image_width=int(view.camera.width), image_height=int(view.camera.height), config=config,
        )
        surface_maplet_ids = atlas.maplet_ids[np.asarray(view.visible_rows) // flat_cells]
        labels = np.full((token_xy.shape[0],), -1, dtype=np.int64)
        for region in range(token_xy.shape[0]):
            inside = np.max(
                np.abs(np.asarray(view.image_xy) - xy[region][None])
                / np.maximum(extent[region][None], 1.0), axis=1
            ) <= 1.0
            if not np.any(inside):
                continue
            ids, counts = np.unique(surface_maplet_ids[inside], return_counts=True)
            for position in np.argsort(-counts, kind="stable").tolist():
                row = row_by_id.get(int(ids[position]))
                if row is not None:
                    labels[region] = int(row)
                    break
        teacher_target = np.concatenate([
            np.sqrt(0.40) * _compress_teacher(dino),
            np.sqrt(0.25) * _compress_teacher(sam),
            np.sqrt(0.20) * _compress_teacher(siglip),
            np.sqrt(0.15) * _compress_teacher(summary),
        ], axis=1)
        teacher_target /= np.maximum(np.linalg.norm(teacher_target, axis=1, keepdims=True), 1e-8)
        all_base.append(base)
        all_teacher.append(teacher_target.astype(np.float32))
        all_labels.append(labels)
        all_trajectories.extend([view.trajectory_id] * token_xy.shape[0])
    base = np.concatenate(all_base)
    teacher = np.concatenate(all_teacher)
    labels = np.concatenate(all_labels)
    trajectories = np.asarray(all_trajectories)
    map_trajectories = set(str(x) for x in (bank.metadata or {}).get("mapping_trajectory_ids", []))
    train = np.asarray([value in map_trajectories for value in trajectories]) & (labels >= 0)
    validation = (~np.asarray([value in map_trajectories or value == "seq11" for value in trajectories])) & (labels >= 0)
    prototypes = []
    for row in range(len(bank)):
        begin, end = int(bank.descriptor_offsets[row]), int(bank.descriptor_offsets[row + 1])
        value = np.sum(bank.descriptors[begin:end] * bank.descriptor_weights[begin:end, None], axis=0)
        prototypes.append(value / max(float(np.linalg.norm(value)), 1e-8))
    prototypes = np.asarray(prototypes, dtype=np.float32)
    model = MapletRetrievalAdaptor(MapletRetrievalAdaptorConfig(feature_dim=base.shape[1])).to(str(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    history = []
    train_rows = np.flatnonzero(train)
    for step in range(max(int(args.steps), 1)):
        selected = rng.choice(train_rows, size=min(int(args.batch_size), train_rows.size), replace=train_rows.size < int(args.batch_size))
        feature = torch.from_numpy(base[selected]).to(str(args.device))
        target = torch.from_numpy(teacher[selected]).to(str(args.device))
        label = torch.from_numpy(labels[selected]).to(str(args.device))
        student = model(feature)
        proto = model(torch.from_numpy(prototypes).to(str(args.device)))
        identity_loss = F.cross_entropy(student @ proto.T / 0.07, label)
        teacher_similarity = target @ target.T
        student_similarity = student @ student.T
        mask = ~torch.eye(student.shape[0], dtype=torch.bool, device=student.device)
        distillation_loss = F.smooth_l1_loss(student_similarity[mask], teacher_similarity[mask])
        loss = identity_loss + float(args.distillation_weight) * distillation_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step + 1 == int(args.steps):
            row = {"step": step, "loss": float(loss.item()), "identity_loss": float(identity_loss.item()), "distillation_loss": float(distillation_loss.item())}
            history.append(row); print(json.dumps(row), flush=True)
    model.eval()
    baseline = MapletRetrievalAdaptor(MapletRetrievalAdaptorConfig(feature_dim=base.shape[1])).to(str(args.device)).eval()
    baseline_validation = _metrics(baseline, base[validation], labels[validation], prototypes, str(args.device))
    student_validation = _metrics(model, base[validation], labels[validation], prototypes, str(args.device))
    save_maplet_retrieval_adaptor(model, Path(args.output_adaptor), metadata={
        "teacher_names": ["dino_v3_7b", "sam3", "siglip2-g"],
        "teacher_cache_sha256": teacher_hash.hexdigest(),
        "training_trajectory_ids": sorted(map_trajectories),
        "validation_trajectory_ids": sorted(set(trajectories[validation].tolist())),
        "identity_bank_sha256": hashlib.sha256(Path(args.identity_bank).read_bytes()).hexdigest(),
    })
    transformed = transform_canonical_maplet_bank(bank, model, adaptor_path=Path(args.output_adaptor), device=str(args.device))
    transformed.save_npz(Path(args.output_bank))
    report = {
        "stage": "v8_multi_teacher_single_student_distillation",
        "sample_count": int(base.shape[0]), "labeled_train_count": int(np.sum(train)),
        "labeled_validation_count": int(np.sum(validation)),
        "baseline_validation": baseline_validation, "student_validation": student_validation,
        "teacher_names": ["dino_v3_7b", "sam3", "siglip2-g"],
        "runtime_teacher_count": 0, "map_feature_type_count": 1,
        "map_feature_dimension": int(transformed.descriptors.shape[1]),
        "map_descriptor_component_count": int(transformed.descriptors.shape[0]),
        "history": history,
    }
    Path(args.summary_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
