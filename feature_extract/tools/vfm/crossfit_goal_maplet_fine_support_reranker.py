"""Sequence-disjoint diagnostic reranking of frozen pure-RADIO child supports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from feature_extract.tools.vfm.evaluate_goal_maplet_pure_retrieval import (
    _contributor_inventory,
    _json_without_duplicates,
    _load_contributor,
)
from feature_extract.tools.vfm.reselect_goal_maplet_pure_radio_fine_support import (
    _atomic_save,
)
from feature_extract.vfm.localization_goal_maplet.connected_fine_support import (
    connected_fine_support_components,
)
from feature_extract.vfm.localization_goal_maplet.fine_support_reranker import (
    FEATURE_NAMES,
    FEATURE_SEMANTICS,
    ChildRerankingFeatures,
    extract_child_reranking_features,
)
from feature_extract.vfm.localization_goal_maplet.fine_support_selection import (
    CHILD_PROBABILITY_SEMANTICS,
    child_surface_area_m2,
    total_map_surface_area_m2,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    PhysicalIncidence,
    remap_pinhole_contributors_to_raw_grid,
    token_primitive_visibility,
)
from feature_extract.vfm.tokens import compute_file_sha256


MODEL_SEMANTICS = "sequence_crossfit_hist_gradient_boosted_truth_density_v1"
MODEL_CONFIG = {
    "learning_rate": 0.05,
    "max_iter": 64,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 128,
    "l2_regularization": 1.0,
    "max_bins": 64,
    "early_stopping": False,
    "random_state": 20260816,
}


@dataclass(frozen=True)
class QueryExample:
    image_id: str
    sequence: str
    source_path: Path
    source_file_sha256: str
    source_content_sha256: str
    features: ChildRerankingFeatures
    truth_density: np.ndarray
    normalized_truth_mass: np.ndarray


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval_summary", nargs="+", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--maximum_area_fraction", type=float, default=0.10)
    parser.add_argument("--maximum_children", type=int, default=2048)
    parser.add_argument("--expected_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _model_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _feature_source_sha256() -> str:
    import feature_extract.vfm.localization_goal_maplet.fine_support_reranker as module

    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def _fit_model(examples: list[QueryExample]) -> tuple[HistGradientBoostingRegressor, float]:
    positive = np.concatenate(
        [value.truth_density[value.truth_density > 0.0] for value in examples]
    )
    if positive.size == 0:
        raise ValueError("crossfit training has no positive child truth")
    scale = float(np.median(positive))
    x = np.concatenate([value.features.values for value in examples], axis=0)
    raw_y = np.concatenate([value.truth_density for value in examples])
    y = np.log1p(raw_y / max(scale, 1e-12))
    weights: list[np.ndarray] = []
    for value in examples:
        mass = np.asarray(value.normalized_truth_mass, dtype=np.float64)
        positive_mass = mass[mass > 0.0]
        reference = float(np.mean(positive_mass)) if positive_mass.size else 1.0
        emphasis = 0.25 + 0.75 * np.minimum(mass / max(reference, 1e-12), 20.0)
        weights.append(emphasis / max(float(mass.size), 1.0))
    sample_weight = np.concatenate(weights)
    model = HistGradientBoostingRegressor(**MODEL_CONFIG)
    with threadpool_limits(limits=1):
        model.fit(x, y, sample_weight=sample_weight)
    return model, scale


def _select(
    rows: np.ndarray,
    predicted_density: np.ndarray,
    area: np.ndarray,
    *,
    maximum_area: float,
    maximum_children: int,
) -> np.ndarray:
    child = np.asarray(rows, dtype=np.int64)
    density = np.maximum(np.asarray(predicted_density, dtype=np.float64), 0.0)
    if density.shape != child.shape or np.any(~np.isfinite(density)):
        raise ValueError("invalid reranker predictions")
    predicted_mass = density * area[child]
    order = np.lexsort((child, -predicted_mass, -density))
    chosen: list[int] = []
    used = 0.0
    for index in order.tolist():
        if density[index] <= 0.0 or len(chosen) >= int(maximum_children):
            break
        row = int(child[index])
        cost = float(area[row])
        if used + cost <= float(maximum_area) + 1e-12:
            chosen.append(row)
            used += cost
    return np.asarray(chosen, dtype=np.int64)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary_json).resolve()
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite crossfit retrieval summary")
    physical_path = Path(args.physical_map).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    child_area = child_surface_area_m2(physical)
    total_area = total_map_surface_area_m2(physical)
    primitive_members = np.asarray(
        physical.child_member_primitive_rows, dtype=np.int64
    )
    if (
        primitive_members.size != physical.primitive_ids.size
        or np.unique(primitive_members).size != primitive_members.size
    ):
        raise ValueError("crossfit diagnostic requires a disjoint complete child partition")
    maximum_area = float(args.maximum_area_fraction) * total_area
    if not 0.0 < float(args.maximum_area_fraction) <= 1.0:
        raise ValueError("maximum area fraction must lie in (0,1]")

    source_paths = [Path(value).resolve() for value in args.retrieval_summary]
    records: list[dict[str, object]] = []
    for path in source_paths:
        summary = _json_without_duplicates(path)
        if summary.get("artifact_type") != "goal_maplet_pure_radio_retrieval_run_v1":
            raise ValueError("not a pure RADIO retrieval summary")
        if str(summary.get("physical_map_sha256", "")) != physical.content_sha256:
            raise ValueError("retrieval and physical map differ")
        for key in (
            "uses_query_pose", "uses_query_ground_truth", "uses_alike", "uses_pnp",
            "uses_sfm_points", "uses_sfm_tracks", "uses_mapping_rgb",
            "uses_image_retrieval",
        ):
            if summary.get(key) is not False:
                raise ValueError(f"retrieval summary requires {key}=false")
        records.extend(list(summary.get("rows", [])))
    records = sorted(records, key=lambda value: str(value["image_id"]))
    image_ids = [str(value["image_id"]) for value in records]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("query identities are duplicated")
    if int(args.expected_queries) > 0 and len(records) != int(args.expected_queries):
        raise ValueError("query count differs")
    contributors = _contributor_inventory(Path(args.contributors))
    incidence = PhysicalIncidence.from_physical_map(physical)

    examples: list[QueryExample] = []
    for index, record in enumerate(records):
        image_id = str(record["image_id"])
        source_path = Path(str(record["artifact"])).resolve()
        source_file_sha = compute_file_sha256(source_path)
        if source_file_sha != str(record["artifact_sha256"]):
            raise ValueError("source retrieval artifact hash differs")
        retrieval = PureRadioPhysicalRetrieval.load_npz(source_path)
        if retrieval.image_id != image_id or retrieval.content_sha256 != str(record["content_sha256"]):
            raise ValueError("source retrieval identity/content differs")
        feature = extract_child_reranking_features(retrieval, physical, child_area)
        contributor = contributors.get(image_id)
        if contributor is None:
            raise ValueError(f"missing contributor for {image_id}")
        labels, model, width, height, params, metadata = _load_contributor(contributor)
        if str(metadata.get("image_id", "")) != image_id:
            raise ValueError("contributor identity differs")
        raw, _ = remap_pinhole_contributors_to_raw_grid(
            labels,
            camera_model_id=model,
            camera_width=width,
            camera_height=height,
            camera_params=params,
        )
        token_primitive, _, _ = token_primitive_visibility(
            raw,
            physical,
            token_height=int(retrieval.metadata["token_height"]),
            token_width=int(retrieval.metadata["token_width"]),
        )
        truth = np.asarray(
            (token_primitive @ incidence.primitive_to_child).sum(axis=0)
        ).reshape(-1)
        total_truth = max(float(np.sum(truth)), 1e-12)
        normalized = truth[feature.child_rows] / total_truth
        examples.append(
            QueryExample(
                image_id=image_id,
                sequence=image_id.split("/", 1)[0],
                source_path=source_path,
                source_file_sha256=source_file_sha,
                source_content_sha256=retrieval.content_sha256,
                features=feature,
                truth_density=normalized / child_area[feature.child_rows],
                normalized_truth_mass=normalized,
            )
        )
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(json.dumps({"phase": "features", "index": index + 1, "count": len(records)}), flush=True)

    sequences = sorted({value.sequence for value in examples})
    if len(sequences) < 3:
        raise ValueError("sequence-disjoint crossfit requires at least three sequences")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    fold_audits: dict[str, object] = {}
    for held_sequence in sequences:
        training = [value for value in examples if value.sequence != held_sequence]
        held = [value for value in examples if value.sequence == held_sequence]
        model, target_scale = _fit_model(training)
        train_ids = [value.image_id for value in training]
        training_sha = hashlib.sha256(
            "".join(value + "\n" for value in train_ids).encode()
        ).hexdigest()
        fold_audits[held_sequence] = {
            "held_sequence": held_sequence,
            "held_query_count": len(held),
            "training_sequences": sorted({value.sequence for value in training}),
            "training_query_count": len(training),
            "ordered_training_image_ids_sha256": training_sha,
            "target_positive_density_median": target_scale,
        }
        for example in held:
            with threadpool_limits(limits=1):
                transformed = np.maximum(
                    np.asarray(model.predict(example.features.values), dtype=np.float64),
                    0.0,
                )
            predicted_density = np.expm1(transformed) * target_scale
            selected = _select(
                example.features.child_rows,
                predicted_density,
                child_area,
                maximum_area=maximum_area,
                maximum_children=int(args.maximum_children),
            )
            retrieval = PureRadioPhysicalRetrieval.load_npz(example.source_path)
            candidate_location = {
                int(row): index for index, row in enumerate(example.features.child_rows.tolist())
            }
            selected_density = np.asarray(
                [predicted_density[candidate_location[int(row)]] for row in selected],
                dtype=np.float64,
            )
            predicted_mass = selected_density * child_area[selected]
            score = predicted_mass / max(float(np.max(predicted_mass, initial=0.0)), 1e-12)
            components = connected_fine_support_components(selected, physical)
            metadata_out = dict(retrieval.metadata)
            metadata_out.pop("content_sha256", None)
            metadata_out.update(
                {
                    "fine_support_selection_semantics": MODEL_SEMANTICS,
                    "fine_support_reranker_feature_semantics": FEATURE_SEMANTICS,
                    "fine_support_reranker_feature_names": list(FEATURE_NAMES),
                    "fine_support_reranker_model_config": MODEL_CONFIG,
                    "fine_support_reranker_threadpool_limit": 1,
                    "fine_support_reranker_source_sha256": _model_source_sha256(),
                    "fine_support_reranker_feature_source_sha256": _feature_source_sha256(),
                    "fine_support_reranker_sklearn_version": str(sklearn.__version__),
                    "crossfit_held_sequence": held_sequence,
                    "crossfit_training_sequences": fold_audits[held_sequence]["training_sequences"],
                    "crossfit_training_query_count": len(training),
                    "crossfit_training_image_ids_sha256": training_sha,
                    "crossfit_uses_held_sequence_labels": False,
                    "reranker_prediction_is_calibrated_probability": False,
                    "maximum_fine_support_area_fraction": float(args.maximum_area_fraction),
                    "selected_fine_support_area_m2": float(np.sum(child_area[selected])),
                    "connected_fine_support_count": int(components.component_count),
                    "child_probability_semantics": CHILD_PROBABILITY_SEMANTICS,
                    "child_probability_is_calibrated_credible_mass": False,
                    "selection_uses_ground_truth": False,
                    "training_only_uses_contributor_ground_truth": True,
                    "development_530_not_untouched_test": True,
                }
            )
            result = PureRadioPhysicalRetrieval(
                **{
                    **retrieval.__dict__,
                    "scene_child_rows": selected,
                    "scene_child_scores": score.astype(np.float32),
                    "metadata": metadata_out,
                }
            )
            destination = output_dir / (example.image_id.replace("/", "__") + ".npz")
            if destination.exists() and not bool(args.force):
                raise FileExistsError(f"refusing to overwrite {destination}")
            _atomic_save(result, destination)
            output_rows.append(
                {
                    "image_id": example.image_id,
                    "artifact": str(destination),
                    "artifact_sha256": compute_file_sha256(destination),
                    "content_sha256": result.content_sha256,
                    "source_artifact_sha256": example.source_file_sha256,
                    "selected_child_count": int(selected.size),
                    "selected_surface_area_m2": float(np.sum(child_area[selected])),
                    "connected_fine_support_count": int(components.component_count),
                    "crossfit_held_sequence": held_sequence,
                }
            )

    output_rows.sort(key=lambda value: str(value["image_id"]))
    required_false = {
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_alike": False,
        "uses_pnp": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_mapping_rgb": False,
        "uses_image_retrieval": False,
    }
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_pure_radio_retrieval_run_v1",
        "query_count": len(output_rows),
        "method": "sequence_disjoint_child_truth_density_reranker_diagnostic",
        "scene_aggregation": "fixed_4x4_blocks_top4_max_sum_v1",
        "physical_map": str(physical_path),
        "physical_map_sha256": physical.content_sha256,
        "source_retrieval_summaries": [str(path) for path in source_paths],
        "source_retrieval_summary_sha256": [compute_file_sha256(path) for path in source_paths],
        "fine_support_selection_semantics": MODEL_SEMANTICS,
        "fine_support_reranker_feature_semantics": FEATURE_SEMANTICS,
        "fine_support_reranker_feature_names": list(FEATURE_NAMES),
        "fine_support_reranker_model_config": MODEL_CONFIG,
        "threadpool_limit": 1,
        "fine_support_reranker_source_sha256": _model_source_sha256(),
        "fine_support_reranker_feature_source_sha256": _feature_source_sha256(),
        "fine_support_reranker_sklearn_version": str(sklearn.__version__),
        "sequence_crossfit": fold_audits,
        "maximum_fine_support_area_fraction": float(args.maximum_area_fraction),
        "maximum_scene_children": int(args.maximum_children),
        "crossfit_query_labels_never_used_by_its_model": True,
        "development_530_not_untouched_test": True,
        "not_deployment_calibration": True,
        **required_false,
        "rows": output_rows,
    }
    _atomic_json(report, summary_path)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
