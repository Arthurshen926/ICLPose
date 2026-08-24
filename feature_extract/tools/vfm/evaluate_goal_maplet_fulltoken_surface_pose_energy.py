"""Evaluate fixed full-token RADIO/surface energy on a frozen pose dataset.

All candidate scores are computed before pose-error arrays are opened for
metrics.  The evaluator is a local-backend/oracle diagnostic: candidate zero
may be GT and controlled candidate sets may be GT-relative.  It never runs
keypoint matching, point correspondences, PnP, or absolute-pose regression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    _metrics,
)
from feature_extract.vfm.localization.surface_maplet_mapper import load_surface_maplet_mapper
from feature_extract.vfm.localization_goal_maplet.fulltoken_surface_pose_energy import (
    CHILD_GATED_CONTRASTIVE_SEMANTICS,
    CHILD_GATED_SEMANTICS,
    CONDITIONAL_PHASE_CONTROL_SEMANTICS,
    CONDITIONAL_SHIFT_PHASE_CONTROL_SEMANTICS,
    PHASE_SEMANTICS,
    MAXMIN_PHASE_SEMANTICS,
    SHIFT_TOLERANT_MAXMIN_PHASE_SEMANTICS,
    SHIFT_TOLERANT_PHASE_SEMANTICS,
    CONTRASTIVE_SEMANTICS,
    PARENT_GATED_SEMANTICS,
    PARENT_GATED_CONTRASTIVE_SEMANTICS,
    SEMANTICS,
    fulltoken_surface_pose_energy,
    child_gated_fulltoken_surface_pose_energy,
    conditional_fulltoken_phase_pose_energy_control,
    conservative_fulltoken_phase_pose_energy,
    parent_gated_fulltoken_surface_pose_energy,
)
from feature_extract.vfm.localization_goal_maplet.differentiable_pose_transport import (
    CATEGORICAL_TOKEN_CAPACITY_TRANSPORT_SEMANTICS,
    QUERY_TOKEN_CAPACITY_TRANSPORT_SEMANTICS,
    categorical_token_capacity_pose_transport,
    query_token_capacity_pose_transport,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import (
    load_pose_candidate_dataset,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import PureRadioPhysicalRetrieval


SCHEMA = "goal_maplet_fulltoken_surface_pose_energy_evaluation_v1"


def _controlled_band_metrics(
    score: np.ndarray,
    translation_m: np.ndarray,
    rotation_deg: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    """Report whether the GT anchor dominates fixed error annuli."""

    s = np.asarray(score, dtype=np.float64)
    t = np.asarray(translation_m, dtype=np.float64)
    r = np.asarray(rotation_deg, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool).copy()
    ok[:, 0] = False
    definitions = (
        ("fine_0_5m_5deg", 0.5, 5.0, 0.0, 0.0),
        ("medium_annulus_to_1m_10deg", 1.0, 10.0, 0.5, 5.0),
        ("coarse_annulus_to_2m_20deg", 2.0, 20.0, 1.0, 10.0),
    )
    output: dict[str, object] = {}
    for name, t_max, r_max, t_inner, r_inner in definitions:
        inside = ok & (t <= t_max + 1.0e-6) & (r <= r_max + 1.0e-5)
        if t_inner > 0.0 or r_inner > 0.0:
            inside &= (t > t_inner + 1.0e-6) | (r > r_inner + 1.0e-5)
        query, candidate = np.nonzero(inside)
        margin = s[query, 0] - s[query, candidate]
        output[name] = {
            "candidate_count": int(margin.size),
            "gt_strictly_higher_fraction": float(np.mean(margin > 0.0)) if margin.size else 0.0,
            "gt_not_lower_fraction": float(np.mean(margin >= 0.0)) if margin.size else 0.0,
            "median_gt_score_margin": float(np.median(margin)) if margin.size else 0.0,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--field_feature_contract", required=True)
    parser.add_argument("--physical_map")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--score_semantics",
        choices=(
            "appearance_only", "appearance_parent_product", "appearance_child_product",
            "conservative_phase", "conservative_phase_shift",
            "conservative_phase_maxmin", "conservative_phase_shift_maxmin",
            "query_token_capacity",
            "parent_layout_capacity",
            "conditional_phase_control", "conditional_phase_shift_control",
        ),
        default="appearance_only",
    )
    parser.add_argument("--local_radius_tokens", type=int, default=1)
    parser.add_argument("--minimum_cosine_evidence", type=float, default=-1.0)
    parser.add_argument("--train_queries", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite full-token pose evaluation")
    arrays, metadata = load_pose_candidate_dataset(Path(args.dataset))
    contract = json.loads(Path(args.field_feature_contract).read_text())
    mapper_sha = file_sha256(Path(args.surface_mapper))
    if (
        contract.get("artifact_type") != "goal_maplet_field_feature_contract_v1"
        or contract.get("canonical_field_sha256") != metadata.get("canonical_field_sha256")
        or contract.get("query_readout_type") != "surface_maplet_mapper"
        or contract.get("query_readout_sha256") != mapper_sha
    ):
        raise ValueError("surface mapper/canonical field lineage differs")
    device = torch.device(str(args.device))
    mapper, _ = load_surface_maplet_mapper(Path(args.surface_mapper), device=str(device))
    mapper.model.to(device).eval()
    physical = None
    if str(args.score_semantics) in (
        "appearance_parent_product", "parent_layout_capacity"
    ):
        if args.physical_map is None:
            raise ValueError("parent-aware scoring requires --physical_map")
        physical = GoalMapletPhysicalMap.load_npz(Path(args.physical_map))
        if physical.content_sha256 != metadata.get("physical_map_sha256"):
            raise ValueError("physical map lineage differs from the dataset")
    score_rows = []
    phase_component_rows = []
    with torch.no_grad():
        for query_index in range(int(arrays["image_ids"].size)):
            raw = torch.as_tensor(
                arrays["radio_final"][query_index], device=device, dtype=torch.float32
            )[None]
            query = mapper.model(raw)[0].permute(1, 2, 0).reshape(2304, 128)
            reliability = torch.as_tensor(
                arrays["query_reliability"][query_index], device=device, dtype=torch.float32
            )
            query_parent_ids = None
            query_parent_probability = None
            if physical is not None:
                retrieval_path = Path(metadata["retrieval_directory"]) / (
                    str(arrays["image_ids"][query_index]).replace("/", "__") + ".npz"
                )
                retrieval = PureRadioPhysicalRetrieval.load_npz(retrieval_path)
                query_parent_ids = torch.as_tensor(retrieval.token_parent_ids, device=device)
                query_parent_probability = torch.as_tensor(
                    retrieval.token_parent_probabilities, device=device, dtype=torch.float32
                )
            query_child_ids = torch.as_tensor(
                arrays["source_child_rows"][query_index], device=device
            )
            query_child_probability = torch.as_tensor(
                arrays["source_child_probabilities"][query_index],
                device=device, dtype=torch.float32,
            )
            candidate_scores = []
            candidate_components = []
            for candidate in range(int(arrays["candidate_valid"].shape[1])):
                if not bool(arrays["candidate_valid"][query_index, candidate]):
                    candidate_scores.append(-1.0)
                    candidate_components.append(None)
                    continue
                target_descriptor = torch.as_tensor(
                    arrays["target_canonical_features"][query_index, candidate],
                    device=device, dtype=torch.float32,
                )
                target_mass = torch.as_tensor(
                    arrays["target_child_weights"][query_index, candidate],
                    device=device, dtype=torch.float32,
                )
                target_valid = torch.as_tensor(
                    arrays["target_modality_valid"][query_index, candidate, ..., 0],
                    device=device,
                )
                if str(args.score_semantics) == "appearance_only":
                    result = fulltoken_surface_pose_energy(
                        query, reliability, arrays["token_xy"][query_index],
                        target_descriptor, target_mass, target_valid,
                        local_radius_tokens=int(args.local_radius_tokens),
                        minimum_cosine_evidence=float(args.minimum_cosine_evidence),
                    )
                elif str(args.score_semantics) == "appearance_parent_product":
                    child_rows = np.asarray(
                        arrays["target_child_rows"][query_index, candidate]
                    )
                    safe_child = np.maximum(child_rows, 0)
                    target_parent_ids = physical.maplet_ids[
                        physical.child_parent_rows[safe_child]
                    ]
                    result = parent_gated_fulltoken_surface_pose_energy(
                        query, reliability, arrays["token_xy"][query_index],
                        query_parent_ids, query_parent_probability,
                        target_descriptor, target_mass, target_valid,
                        torch.as_tensor(target_parent_ids, device=device),
                        minimum_cosine_evidence=float(args.minimum_cosine_evidence),
                    )
                elif str(args.score_semantics) == "appearance_child_product":
                    result = child_gated_fulltoken_surface_pose_energy(
                        query, reliability, arrays["token_xy"][query_index],
                        query_child_ids, query_child_probability,
                        target_descriptor, target_mass, target_valid,
                        torch.as_tensor(
                            arrays["target_child_rows"][query_index, candidate], device=device
                        ),
                        minimum_cosine_evidence=float(args.minimum_cosine_evidence),
                    )
                elif str(args.score_semantics) == "query_token_capacity":
                    result = query_token_capacity_pose_transport(
                        query,
                        query_child_probability.sum(dim=1),
                        reliability,
                        target_descriptor,
                        target_mass,
                        target_valid,
                        height=36,
                        width=64,
                        local_radius_tokens=int(args.local_radius_tokens),
                        minimum_cosine_evidence=float(args.minimum_cosine_evidence),
                    )
                elif str(args.score_semantics) == "parent_layout_capacity":
                    child_rows = np.asarray(
                        arrays["target_child_rows"][query_index, candidate]
                    )
                    safe_child = np.maximum(child_rows, 0)
                    target_parent_ids = physical.maplet_ids[
                        physical.child_parent_rows[safe_child]
                    ]
                    target_parent_ids[child_rows < 0] = -1
                    result = categorical_token_capacity_pose_transport(
                        query_parent_ids,
                        query_parent_probability,
                        reliability,
                        torch.as_tensor(target_parent_ids, device=device),
                        target_mass,
                        target_valid,
                        height=36,
                        width=64,
                        local_radius_tokens=int(args.local_radius_tokens),
                    )
                elif str(args.score_semantics) in (
                    "conservative_phase", "conservative_phase_shift",
                    "conservative_phase_maxmin", "conservative_phase_shift_maxmin",
                ):
                    result = conservative_fulltoken_phase_pose_energy(
                        query, target_descriptor, target_mass, target_valid,
                        height=36, width=64,
                        maximum_shift_tokens=(
                            int(args.local_radius_tokens)
                            if str(args.score_semantics) in (
                                "conservative_phase_shift",
                                "conservative_phase_shift_maxmin",
                            ) else 0
                        ),
                        slot_pair_reduction=(
                            "maximum_bottleneck"
                            if str(args.score_semantics) in (
                                "conservative_phase_maxmin",
                                "conservative_phase_shift_maxmin",
                            ) else "additive_product"
                        ),
                    )
                else:
                    result = conditional_fulltoken_phase_pose_energy_control(
                        query, target_descriptor, target_mass, target_valid,
                        height=36, width=64,
                        maximum_shift_tokens=(
                            int(args.local_radius_tokens)
                            if str(args.score_semantics) == "conditional_phase_shift_control"
                            else 0
                        ),
                    )
                candidate_scores.append(float(result.score.cpu()))
                content = getattr(result, "conditional_content_score", None)
                information = getattr(result, "mass_observability", None)
                candidate_components.append(
                    None
                    if content is None or information is None
                    else {
                        "conditional_content_score": float(content.cpu()),
                        "mass_observability": float(information.cpu()),
                        "horizontal_edge_observability": float(
                            result.horizontal_observability.cpu()
                        ),
                        "vertical_edge_observability": float(
                            result.vertical_observability.cpu()
                        ),
                        "selected_shift_y": int(result.selected_shift_y),
                        "selected_shift_x": int(result.selected_shift_x),
                    }
                )
            score_rows.append(candidate_scores)
            phase_component_rows.append(candidate_components)
    # Pose labels are consumed only after every score is frozen above.
    scores = np.asarray(score_rows, dtype=np.float32)
    query_count = int(scores.shape[0])
    train_count = min(max(int(args.train_queries), 0), query_count)
    report = {
        "artifact_type": SCHEMA,
        "energy_semantics": (
            (
                PARENT_GATED_CONTRASTIVE_SEMANTICS
                if float(args.minimum_cosine_evidence) > -1.0
                else PARENT_GATED_SEMANTICS
            )
            if str(args.score_semantics) == "appearance_parent_product"
            else (
                (
                    CHILD_GATED_CONTRASTIVE_SEMANTICS
                    if float(args.minimum_cosine_evidence) > -1.0
                    else CHILD_GATED_SEMANTICS
                )
                if str(args.score_semantics) == "appearance_child_product"
                else (
                    (
                        (
                            SHIFT_TOLERANT_MAXMIN_PHASE_SEMANTICS
                            if str(args.score_semantics) == "conservative_phase_shift_maxmin"
                            else MAXMIN_PHASE_SEMANTICS
                        )
                        if str(args.score_semantics) in (
                            "conservative_phase_maxmin", "conservative_phase_shift_maxmin"
                        )
                        else (
                            SHIFT_TOLERANT_PHASE_SEMANTICS
                            if str(args.score_semantics) == "conservative_phase_shift"
                            else PHASE_SEMANTICS
                        )
                    )
                    if str(args.score_semantics) in (
                        "conservative_phase", "conservative_phase_shift",
                        "conservative_phase_maxmin", "conservative_phase_shift_maxmin",
                    )
                    else (
                        (
                            CONDITIONAL_SHIFT_PHASE_CONTROL_SEMANTICS
                            if str(args.score_semantics) == "conditional_phase_shift_control"
                            else CONDITIONAL_PHASE_CONTROL_SEMANTICS
                        )
                        if str(args.score_semantics) in (
                            "conditional_phase_control", "conditional_phase_shift_control"
                        )
                        else (
                            CONTRASTIVE_SEMANTICS
                            if float(args.minimum_cosine_evidence) > -1.0
                            else SEMANTICS
                        )
                    )
                )
            ) if str(args.score_semantics) not in (
                "query_token_capacity", "parent_layout_capacity"
            ) else (
                QUERY_TOKEN_CAPACITY_TRANSPORT_SEMANTICS
                if str(args.score_semantics) == "query_token_capacity"
                else CATEGORICAL_TOKEN_CAPACITY_TRANSPORT_SEMANTICS
            )
        ),
        "score_semantics": str(args.score_semantics),
        "minimum_cosine_evidence": float(args.minimum_cosine_evidence),
        "dataset_file_sha256": file_sha256(Path(args.dataset)),
        "dataset_content_sha256": metadata["content_sha256"],
        "candidate_semantics": metadata.get("candidate_semantics", "proposal_pool_v1"),
        "controlled_candidates_are_gt_relative_oracle_diagnostic": bool(
            metadata.get("controlled_candidates_are_gt_relative_oracle_diagnostic", False)
        ),
        "surface_mapper_file_sha256": mapper_sha,
        "field_feature_contract_file_sha256": file_sha256(Path(args.field_feature_contract)),
        "local_radius_tokens": int(args.local_radius_tokens),
        "all_candidate_scores_built_before_pose_error_metrics": True,
        "candidate_score": scores.tolist(),
        "candidate_phase_evidence_components": phase_component_rows,
        "conditional_fit_and_information_are_separate": bool(
            str(args.score_semantics) in (
                "conditional_phase_control", "conditional_phase_shift_control"
            )
        ),
        "all_metrics": _metrics(
            scores, arrays["translation_m"], arrays["rotation_deg"],
            arrays["candidate_valid"], arrays["image_ids"],
        ),
        "controlled_error_band_metrics": _controlled_band_metrics(
            scores, arrays["translation_m"], arrays["rotation_deg"], arrays["candidate_valid"]
        ),
        "prefix_metrics": _metrics(
            scores[:train_count], arrays["translation_m"][:train_count],
            arrays["rotation_deg"][:train_count], arrays["candidate_valid"][:train_count],
            arrays["image_ids"][:train_count],
        ) if train_count else None,
        "suffix_metrics": _metrics(
            scores[train_count:], arrays["translation_m"][train_count:],
            arrays["rotation_deg"][train_count:], arrays["candidate_valid"][train_count:],
            arrays["image_ids"][train_count:],
        ) if train_count < query_count else None,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "claim": "map_disjoint_candidate_pose_energy_diagnostic_not_localization_success",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "all": {key: value for key, value in report["all_metrics"].items() if key != "rows"},
        "suffix": None if report["suffix_metrics"] is None else {
            key: value for key, value in report["suffix_metrics"].items() if key != "rows"
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
