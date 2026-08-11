"""Train G17 physical-instance readouts with offline RADIO adaptor teachers.

SigLIP supervises the parent/context role, DINO supervises child/local physical
identity, and SAM supervises within-image support boundaries.  All teachers
are discarded after fitting.  Deployment stores one canonical RADIO code per
2DGS primitive plus this small regenerable readout; it never stores mapping
images or downstream teacher embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.canonical_field import (
    CanonicalSurfaceField,
    readout_canonical_field,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.oracle_pose import token_oracle_evidence
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.physical_instance_readout import (
    PhysicalInstanceReadout,
    PhysicalInstanceReadoutConfig,
    _region_token_sets,
    save_physical_instance_readout,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--teacher_cache", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--output_readout", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--train_trajectories", nargs="+", default=["seq1", "seq2", "seq4", "seq6", "seq7", "seq8"])
    parser.add_argument("--selection_trajectories", nargs="+", default=["seq9", "seq10"])
    parser.add_argument("--validation_trajectories", nargs="+", default=["seq12", "seq14"])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--primitive_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher_compression",
        default="legacy_block_mean",
        choices=("legacy_block_mean", "signed_block_sketch"),
    )
    parser.add_argument("--teacher_compact_dimensions", type=int, default=64)
    parser.add_argument(
        "--selection_metric",
        default="joint_parent32_child16_primitive8",
        choices=("joint_parent32_child16", "joint_parent32_child16_primitive8"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _compress(
    value: np.ndarray,
    dimensions: int = 64,
    *,
    method: str = "legacy_block_mean",
    seed: int = 0,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    width = int(array.shape[-1])
    if width % int(dimensions):
        raise ValueError("teacher width is not divisible by the compact width")
    grouped = array.reshape(
        *array.shape[:-1], int(dimensions), width // int(dimensions),
    )
    if str(method) == "legacy_block_mean":
        compact = grouped.mean(axis=-1)
    elif str(method) == "signed_block_sketch":
        # A deterministic signed sketch avoids the common-mode bias of
        # averaging every teacher channel with the same positive sign.  It is
        # used only while distilling pairwise teacher affinity; no sketch or
        # downstream embedding is serialized into the deployment map.
        signs = np.random.default_rng(int(seed)).choice(
            (-1.0, 1.0),
            size=(int(dimensions), width // int(dimensions)),
        ).astype(np.float32)
        compact = np.sum(grouped * signs, axis=-1)
    else:
        raise ValueError(f"unknown teacher compression: {method}")
    return compact / np.maximum(np.linalg.norm(compact, axis=-1, keepdims=True), 1e-8)


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def _build_dataset(
    args,
    mapper,
    physical: GoalMapletPhysicalMap,
    field: CanonicalSurfaceField,
) -> dict[str, np.ndarray]:
    collections: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "context_tokens", "context_xy", "context_weight", "context_mask",
            "local_tokens", "local_xy", "local_weight", "local_mask",
            "parent", "child", "primitive_field_row", "image_row", "token_xy",
            "siglip", "dino", "sam",
            "trajectory",
        )
    }
    field_row_by_primitive = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    field_row_by_primitive[np.asarray(field.primitive_rows, dtype=np.int64)] = np.arange(
        field.primitive_rows.size, dtype=np.int64,
    )
    requested = set(args.train_trajectories) | set(args.selection_trajectories) | set(args.validation_trajectories)
    teacher_root = Path(args.teacher_cache)
    image_row = 0
    for path in sorted(Path(args.contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        trajectory = str(metadata["trajectory_id"])
        if trajectory not in requested:
            continue
        teacher_path = teacher_root / (str(metadata["image_id"]).replace("/", "__") + ".npz")
        if not teacher_path.exists():
            continue
        with np.load(teacher_path, allow_pickle=False) as teacher:
            token_xy = np.asarray(teacher["token_xy"], dtype=np.float32)
            compression = {
                "dimensions": int(args.teacher_compact_dimensions),
                "method": str(args.teacher_compression),
            }
            dino = _compress(
                np.asarray(teacher["dino_v3_7b"], dtype=np.float32),
                **compression, seed=17011,
            )
            sam = _compress(
                np.asarray(teacher["sam3"], dtype=np.float32),
                **compression, seed=17013,
            )
            siglip_spatial = _compress(
                np.asarray(teacher["siglip2-g"], dtype=np.float32),
                **compression, seed=17017,
            )
            siglip_summary = _compress(
                np.asarray(teacher["siglip2-g_summary"], dtype=np.float32)[None],
                **compression, seed=17017,
            )
        siglip = _normalize(np.concatenate([
            siglip_spatial,
            np.broadcast_to(siglip_summary, siglip_spatial.shape).copy(),
        ], axis=1))
        labels = ContributorLabels.load_npz(path)
        with np.load(Path(str(metadata["token_path"])), allow_pickle=False) as data:
            raw = np.asarray(data["radio_final"], dtype=np.float32)
        mapped = mapper.project(raw).measurement_context
        oracle = token_oracle_evidence(
            labels, physical, token_xy,
            token_height=int(raw.shape[1]), token_width=int(raw.shape[2]),
            image_height=int(metadata["height"]), image_width=int(metadata["width"]),
        )
        primitive_field_row = np.full(oracle.child_rows.shape, -1, dtype=np.int64)
        for token in np.flatnonzero(oracle.child_rows >= 0).tolist():
            child = int(oracle.child_rows[token])
            start = int(physical.child_member_offsets[child])
            end = int(physical.child_member_offsets[child + 1])
            members = np.asarray(physical.child_member_primitive_rows[start:end], dtype=np.int64)
            observed = members[field_row_by_primitive[members] >= 0]
            if observed.size:
                primitive = int(observed[np.argmin(np.linalg.norm(
                    physical.primitive_centers[observed] - oracle.child_local_xyz[token], axis=1,
                ))])
                primitive_field_row[token] = int(field_row_by_primitive[primitive])
        valid = (
            (oracle.parent_rows >= 0) & (oracle.child_rows >= 0)
            & (primitive_field_row >= 0)
        )
        if not np.any(valid):
            continue
        context = _region_token_sets(mapped, token_xy, role="context", config=PhysicalInstanceReadoutConfig())
        local = _region_token_sets(mapped, token_xy, role="local", config=PhysicalInstanceReadoutConfig())
        for name, value in zip(
            ("context_tokens", "context_xy", "context_weight", "context_mask"), context,
        ):
            collections[name].append(np.asarray(value)[valid])
        for name, value in zip(
            ("local_tokens", "local_xy", "local_weight", "local_mask"), local,
        ):
            collections[name].append(np.asarray(value)[valid])
        count = int(np.sum(valid))
        collections["parent"].append(oracle.parent_rows[valid])
        collections["child"].append(oracle.child_rows[valid])
        collections["primitive_field_row"].append(primitive_field_row[valid])
        collections["image_row"].append(np.full((count,), image_row, dtype=np.int64))
        collections["token_xy"].append(token_xy[valid])
        collections["siglip"].append(siglip[valid])
        collections["dino"].append(dino[valid])
        collections["sam"].append(sam[valid])
        collections["trajectory"].append(np.asarray([trajectory] * count))
        print(json.dumps({
            "image_id": str(metadata["image_id"]), "trajectory": trajectory,
            "physical_instance_samples": count,
        }), flush=True)
        image_row += 1
    if not collections["parent"]:
        raise ValueError("no aligned teacher/contributor physical-instance samples")
    result: dict[str, np.ndarray] = {}
    for name, values in collections.items():
        result[name] = np.concatenate(values, axis=0)
    # Feature-token tensors dominate memory but need no float32 precision while
    # held in the CPU dataset.  They are promoted immediately before training.
    result["context_tokens"] = result["context_tokens"].astype(np.float16)
    result["local_tokens"] = result["local_tokens"].astype(np.float16)
    return result


def _subset_metrics(
    model: PhysicalInstanceReadout,
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    parent_prototype: torch.Tensor,
    child_prototype: torch.Tensor,
    primitive_prototype: torch.Tensor,
    field_row_by_primitive: np.ndarray,
    physical: GoalMapletPhysicalMap,
    parent_valid: np.ndarray,
    child_valid: np.ndarray,
    *,
    device: str,
    batch_size: int = 256,
) -> dict[str, float]:
    rows = np.flatnonzero(mask)
    parent_rank, child_rank, primitive_rank = [], [], []
    model.eval()
    with torch.inference_mode():
        parent_map = model.project_flat(parent_prototype, role="context")
        child_map = model.project_flat(child_prototype, role="local")
        for start in range(0, rows.size, int(batch_size)):
            selected = rows[start:start + int(batch_size)]
            context = model(
                torch.from_numpy(data["context_tokens"][selected].astype(np.float32)).to(device),
                torch.from_numpy(data["context_xy"][selected]).to(device),
                torch.from_numpy(data["context_weight"][selected]).to(device),
                torch.from_numpy(data["context_mask"][selected]).to(device), role="context",
            )
            local = model(
                torch.from_numpy(data["local_tokens"][selected].astype(np.float32)).to(device),
                torch.from_numpy(data["local_xy"][selected]).to(device),
                torch.from_numpy(data["local_weight"][selected]).to(device),
                torch.from_numpy(data["local_mask"][selected]).to(device), role="local",
            )
            parent_score = context @ parent_map.T
            child_score = local @ child_map.T
            parent_score[:, ~torch.from_numpy(parent_valid).to(device)] = -torch.inf
            child_score[:, ~torch.from_numpy(child_valid).to(device)] = -torch.inf
            parent_truth = torch.from_numpy(data["parent"][selected]).to(device)
            child_truth = torch.from_numpy(data["child"][selected]).to(device)
            parent_truth_score = parent_score.gather(1, parent_truth[:, None])
            child_truth_score = child_score.gather(1, child_truth[:, None])
            parent_rank.extend((1 + torch.sum(parent_score > parent_truth_score, dim=1)).cpu().tolist())
            child_rank.extend((1 + torch.sum(child_score > child_truth_score, dim=1)).cpu().tolist())
            primitive_truth = np.asarray(data["primitive_field_row"][selected], dtype=np.int64)
            batch_primitive_rank = np.full((selected.size,), np.iinfo(np.int32).max, dtype=np.int64)
            for child in np.unique(data["child"][selected]).tolist():
                batch_rows = np.flatnonzero(data["child"][selected] == child)
                start = int(physical.child_member_offsets[int(child)])
                end = int(physical.child_member_offsets[int(child) + 1])
                member_primitives = np.asarray(
                    physical.child_member_primitive_rows[start:end], dtype=np.int64,
                )
                candidates = field_row_by_primitive[member_primitives]
                candidates = candidates[candidates >= 0]
                if candidates.size == 0:
                    continue
                candidate_tensor = torch.from_numpy(candidates).to(device)
                primitive_map = model.project_flat(
                    primitive_prototype[candidate_tensor], role="local",
                )
                score = local[torch.from_numpy(batch_rows).to(device)] @ primitive_map.T
                candidate_lookup = {int(value): index for index, value in enumerate(candidates.tolist())}
                truth_column = torch.as_tensor(
                    [candidate_lookup[int(primitive_truth[row])] for row in batch_rows.tolist()],
                    dtype=torch.long, device=device,
                )
                truth_score = score.gather(1, truth_column[:, None])
                batch_primitive_rank[batch_rows] = np.asarray(
                    (1 + torch.sum(score > truth_score, dim=1)).cpu(), dtype=np.int64,
                )
            primitive_rank.extend(batch_primitive_rank.tolist())
    parent_rank = np.asarray(parent_rank, dtype=np.int64)
    child_rank = np.asarray(child_rank, dtype=np.int64)
    primitive_rank = np.asarray(primitive_rank, dtype=np.int64)
    return {
        "count": int(rows.size),
        "parent_recall_at_1": float(np.mean(parent_rank <= 1)),
        "parent_recall_at_8": float(np.mean(parent_rank <= 8)),
        "parent_recall_at_32": float(np.mean(parent_rank <= 32)),
        "child_recall_at_1": float(np.mean(child_rank <= 1)),
        "child_recall_at_8": float(np.mean(child_rank <= 8)),
        "child_recall_at_16": float(np.mean(child_rank <= 16)),
        "joint_parent32_child16": float(np.mean((parent_rank <= 32) & (child_rank <= 16))),
        "primitive_recall_at_1_within_truth_child": float(np.mean(primitive_rank <= 1)),
        "primitive_recall_at_8_within_truth_child": float(np.mean(primitive_rank <= 8)),
        "joint_parent32_child16_primitive8": float(np.mean(
            (parent_rank <= 32) & (child_rank <= 16) & (primitive_rank <= 8)
        )),
        "parent_median_rank": float(np.median(parent_rank)),
        "child_median_rank": float(np.median(child_rank)),
        "primitive_median_rank_within_truth_child": float(np.median(primitive_rank)),
    }


def main() -> None:
    args = _parse_args()
    output, summary = Path(args.output_readout), Path(args.summary_json)
    if not args.force and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite physical-instance readout artifacts")
    partitions = [set(args.train_trajectories), set(args.selection_trajectories), set(args.validation_trajectories)]
    if any(partitions[a] & partitions[b] for a in range(3) for b in range(a + 1, 3)):
        raise ValueError("physical-instance trajectory partitions overlap")
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    raw_readout = readout_canonical_field(field, physical)
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(args.device))
    data = _build_dataset(args, mapper, physical, field)
    trajectory = data["trajectory"].astype(str)
    train = np.isin(trajectory, list(args.train_trajectories))
    selection = np.isin(trajectory, list(args.selection_trajectories))
    validation = np.isin(trajectory, list(args.validation_trajectories))
    if not np.any(train) or not np.any(selection) or not np.any(validation):
        raise ValueError("one or more physical-instance partitions are empty")
    parent_valid = raw_readout.parent_coverage > 0.0
    child_valid = raw_readout.child_coverage > 0.0
    train &= parent_valid[data["parent"]] & child_valid[data["child"]]
    selection &= parent_valid[data["parent"]] & child_valid[data["child"]]
    validation &= parent_valid[data["parent"]] & child_valid[data["child"]]
    device = str(args.device)
    parent_prototype = torch.from_numpy(raw_readout.parent_descriptors).to(device)
    child_prototype = torch.from_numpy(raw_readout.child_descriptors).to(device)
    primitive_prototype = torch.from_numpy(field.codes).to(device)
    field_row_by_primitive = np.full((physical.primitive_ids.size,), -1, dtype=np.int64)
    field_row_by_primitive[np.asarray(field.primitive_rows, dtype=np.int64)] = np.arange(
        field.primitive_rows.size, dtype=np.int64,
    )
    model = PhysicalInstanceReadout(PhysicalInstanceReadoutConfig(feature_dim=field.feature_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    baseline = _subset_metrics(
        model, data, validation, parent_prototype, child_prototype,
        primitive_prototype, field_row_by_primitive, physical,
        parent_valid, child_valid, device=device,
    )
    train_rows = np.flatnonzero(train)
    history, best_state, best_score = [], None, -np.inf
    parent_valid_tensor = torch.from_numpy(parent_valid).to(device)
    child_valid_tensor = torch.from_numpy(child_valid).to(device)
    for step in range(max(int(args.steps), 1)):
        selected = rng.choice(
            train_rows, size=min(int(args.batch_size), train_rows.size),
            replace=train_rows.size < int(args.batch_size),
        )
        context = model(
            torch.from_numpy(data["context_tokens"][selected].astype(np.float32)).to(device),
            torch.from_numpy(data["context_xy"][selected]).to(device),
            torch.from_numpy(data["context_weight"][selected]).to(device),
            torch.from_numpy(data["context_mask"][selected]).to(device), role="context",
        )
        local = model(
            torch.from_numpy(data["local_tokens"][selected].astype(np.float32)).to(device),
            torch.from_numpy(data["local_xy"][selected]).to(device),
            torch.from_numpy(data["local_weight"][selected]).to(device),
            torch.from_numpy(data["local_mask"][selected]).to(device), role="local",
        )
        parent_map = model.project_flat(parent_prototype, role="context")
        child_map = model.project_flat(child_prototype, role="local")
        parent_logits = context @ parent_map.T / 0.07
        child_logits = local @ child_map.T / 0.07
        parent_logits[:, ~parent_valid_tensor] = -torch.inf
        child_logits[:, ~child_valid_tensor] = -torch.inf
        parent_target = torch.from_numpy(data["parent"][selected]).to(device)
        child_target = torch.from_numpy(data["child"][selected]).to(device)
        parent_loss = F.cross_entropy(parent_logits, parent_target)
        child_loss = F.cross_entropy(child_logits, child_target)

        # Exact primitive identity is supervised only inside the true physical
        # child.  Candidate rows combine the batch truths with deterministic
        # same-child negatives, so the local head learns fine surface identity
        # without turning the deployed map into a second embedding store.
        primitive_target_rows = np.asarray(data["primitive_field_row"][selected], dtype=np.int64)
        candidate_rows = set(primitive_target_rows.tolist())
        for child in np.unique(data["child"][selected]).tolist():
            start = int(physical.child_member_offsets[int(child)])
            end = int(physical.child_member_offsets[int(child) + 1])
            member_rows = field_row_by_primitive[
                physical.child_member_primitive_rows[start:end]
            ]
            member_rows = member_rows[member_rows >= 0]
            # A deterministic spread prevents the loss from depending on an
            # additional negative-mining hyperparameter or on query ordering.
            if member_rows.size > 8:
                member_rows = member_rows[np.linspace(
                    0, member_rows.size - 1, num=8, dtype=np.int64,
                )]
            candidate_rows.update(member_rows.tolist())
        primitive_candidates = np.asarray(sorted(candidate_rows), dtype=np.int64)
        primitive_candidate_tensor = torch.from_numpy(primitive_candidates).to(device)
        primitive_map = model.project_flat(
            primitive_prototype[primitive_candidate_tensor], role="local",
        )
        primitive_logits = local @ primitive_map.T / 0.07
        primitive_lookup = {int(value): index for index, value in enumerate(primitive_candidates.tolist())}
        primitive_target = torch.as_tensor(
            [primitive_lookup[int(value)] for value in primitive_target_rows.tolist()],
            dtype=torch.long, device=device,
        )
        primitive_loss = F.cross_entropy(primitive_logits, primitive_target)

        context_similarity = context @ context.T
        local_similarity = local @ local.T
        off_diagonal = ~torch.eye(selected.size, dtype=torch.bool, device=device)
        siglip = torch.from_numpy(data["siglip"][selected]).to(device)
        dino = torch.from_numpy(data["dino"][selected]).to(device)
        siglip_loss = F.smooth_l1_loss(context_similarity[off_diagonal], (siglip @ siglip.T)[off_diagonal])
        dino_loss = F.smooth_l1_loss(local_similarity[off_diagonal], (dino @ dino.T)[off_diagonal])
        image = torch.from_numpy(data["image_row"][selected]).to(device)
        xy = torch.from_numpy(data["token_xy"][selected]).to(device)
        near_same_image = (
            (image[:, None] == image[None, :])
            & (torch.sum((xy[:, None] - xy[None, :]) ** 2, dim=-1) <= 36.0)
            & off_diagonal
        )
        sam = torch.from_numpy(data["sam"][selected]).to(device)
        sam_loss = (
            F.smooth_l1_loss(local_similarity[near_same_image], (sam @ sam.T)[near_same_image])
            if torch.any(near_same_image) else torch.zeros((), device=device)
        )
        same_parent_other_child = (
            (parent_target[:, None] == parent_target[None, :])
            & (child_target[:, None] != child_target[None, :]) & off_diagonal
        )
        same_child = (child_target[:, None] == child_target[None, :]) & off_diagonal
        hard_negative_loss = torch.zeros((), device=device)
        if torch.any(same_parent_other_child) and torch.any(same_child):
            hard_negative_loss = F.relu(
                0.15 + torch.max(local_similarity[same_parent_other_child])
                - torch.mean(local_similarity[same_child])
            )
        loss = (
            parent_loss + child_loss + float(args.primitive_loss_weight) * primitive_loss
            + 0.25 * siglip_loss + 0.25 * dino_loss + 0.25 * sam_loss
            + 0.25 * hard_negative_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 100 == 0 or step + 1 == int(args.steps):
            selection_metrics = _subset_metrics(
                model, data, selection, parent_prototype, child_prototype,
                primitive_prototype, field_row_by_primitive, physical,
                parent_valid, child_valid, device=device,
            )
            score = float(selection_metrics[str(args.selection_metric)])
            row = {
                "step": int(step), "loss": float(loss.item()),
                "parent_loss": float(parent_loss.item()), "child_loss": float(child_loss.item()),
                "primitive_loss": float(primitive_loss.item()),
                "siglip_distillation": float(siglip_loss.item()),
                "dino_distillation": float(dino_loss.item()),
                "sam_boundary_distillation": float(sam_loss.item()),
                "physical_hard_negative_loss": float(hard_negative_loss.item()),
                "selection": selection_metrics,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("physical-instance selection did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    selected_validation = _subset_metrics(
        model, data, validation, parent_prototype, child_prototype,
        primitive_prototype, field_row_by_primitive, physical,
        parent_valid, child_valid, device=device,
    )
    save_physical_instance_readout(model, output, metadata={
        "teacher_roles": {
            "siglip2-g": "parent_context", "dino_v3_7b": "child_physical_instance",
            "sam3": "support_boundary_affinity",
        },
        "teacher_use": "offline_training_only_separate_role_losses",
        "teacher_compression": {
            "method": str(args.teacher_compression),
            "dimensions": int(args.teacher_compact_dimensions),
            "stored_at_deployment": False,
        },
        "exact_geometry_supervision": (
            "nearest_observed_2dgs_primitive_within_truth_child_training_only"
            if float(args.primitive_loss_weight) > 0.0 else "disabled_after_g17_v2_ablation"
        ),
        "primitive_loss_weight": float(args.primitive_loss_weight),
        "training_trajectory_ids": sorted(args.train_trajectories),
        "selection_trajectory_ids": sorted(args.selection_trajectories),
        "validation_trajectory_ids": sorted(args.validation_trajectories),
        "physical_map_sha256": physical.content_sha256,
        "canonical_field_sha256": field.content_sha256,
        "surface_mapper_sha256": file_sha256(Path(args.surface_mapper)),
        "selection_rule": (
            f"maximum_{str(args.selection_metric)}_on_predeclared_selection_trajectories"
        ),
        "selection_metric": str(args.selection_metric),
        "selected_score": float(best_score),
    })
    result = {
        "stage": "g17_physical_instance_readout",
        "sample_count": int(data["parent"].size),
        "partition_counts": {
            "train": int(np.sum(train)), "selection": int(np.sum(selection)),
            "validation": int(np.sum(validation)),
        },
        "baseline_validation": baseline,
        "selected_validation": selected_validation,
        "selection_metric": str(args.selection_metric),
        "selection_best_score": float(best_score),
        "history": history,
        "deployment_contract": {
            "map_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "runtime_teacher_count": 0,
            "stores_mapping_rgb": False,
            "position_preserving_context_token_set": True,
            "separate_regenerable_context_and_local_heads": True,
            "teacher_compression_stored_at_deployment": False,
        },
        "output_readout": str(output),
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
