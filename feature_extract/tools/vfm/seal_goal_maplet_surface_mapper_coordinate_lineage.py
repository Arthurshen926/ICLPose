"""Seal a trained surface mapper with verified coordinate-supervision lineage.

Training remains unchanged.  This post-training gate reloads the exact bank,
filtered RADIO manifest, and checkpoint split; only a fully hash-bound,
route-disjoint chain is copied into the sealed checkpoint metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
    save_surface_maplet_mapper,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    COORDINATE_CONTRACT,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank
from feature_extract.vfm.tokens import TokenBankManifest


SCHEMA = "goal_maplet_surface_mapper_coordinate_lineage_seal_v1"


def _trajectory(image_id: str) -> str:
    value = str(image_id)
    if "/" not in value:
        raise ValueError(f"image ID lacks a trajectory: {value}")
    return value.split("/", 1)[0]


def _verified_lineage(
    *,
    bank_metadata: Mapping[str, object],
    bank_file_sha256: str,
    manifest_file_sha256: str,
) -> dict[str, object]:
    raw = bank_metadata.get("supervision_coordinate_lineage")
    lineage = dict(raw) if isinstance(raw, Mapping) else {}
    if lineage.get("coordinate_correct") is not True:
        raise ValueError("surface-maplet bank does not declare coordinate-correct supervision")
    if str(lineage.get("coordinate_contract", "")) != COORDINATE_CONTRACT:
        raise ValueError("surface-maplet bank coordinate contract differs")
    if str(lineage.get("radio_final_manifest_file_sha256", "")) != str(
        manifest_file_sha256
    ):
        raise ValueError("surface-maplet bank and RADIO manifest hashes differ")
    if lineage.get("strict_holdout_present") is not False:
        raise ValueError("surface-maplet bank does not explicitly exclude strict holdouts")
    if lineage.get("route_allowlist_applied_before_opening_contributor_archives") is not True:
        raise ValueError("surface-maplet bank did not apply its route allowlist before archive open")
    return {
        **lineage,
        "coordinate_correct": True,
        "coordinate_contract": COORDINATE_CONTRACT,
        "surface_maplets_file_sha256": str(bank_file_sha256),
        "radio_final_manifest_file_sha256": str(manifest_file_sha256),
        "sealed_by": SCHEMA,
    }


def _verified_training_completion(
    training_summary: Mapping[str, object],
    *,
    checkpoint_path: Path,
    checkpoint_metadata: Mapping[str, object],
    expected_patience: int,
) -> dict[str, object]:
    """Prove that the selected checkpoint came from a naturally completed run."""

    summary = dict(training_summary)
    if summary.get("stage") != "train_surface_maplet_mapper":
        raise ValueError("mapper training summary stage differs")
    if Path(str(summary.get("output_checkpoint", ""))).resolve() != checkpoint_path.resolve():
        raise ValueError("mapper training summary checkpoint path differs")
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch != int(checkpoint_metadata.get("best_epoch", -2)):
        raise ValueError("mapper training summary/checkpoint best epoch differs")
    if summary.get("best_validation") != checkpoint_metadata.get("best_validation"):
        raise ValueError("mapper training summary/checkpoint best validation differs")
    history = summary.get("history", [])
    if not isinstance(history, list) or not history:
        raise ValueError("mapper training summary history is empty")
    history_epochs = [
        int(row.get("epoch", -1))
        for row in history
        if isinstance(row, Mapping)
    ]
    if len(history_epochs) != len(history) or history_epochs != sorted(history_epochs):
        raise ValueError("mapper training history epochs are invalid")
    config = summary.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("mapper training summary config is missing")
    configured_epochs = int(config.get("epochs", -1))
    patience = int(expected_patience)
    if patience <= 0:
        raise ValueError("expected mapper early-stop patience must be positive")
    if "patience" in config and int(config["patience"]) != patience:
        raise ValueError("mapper training summary patience differs")
    final_evaluated_epoch = int(history_epochs[-1])
    reached_epoch_budget = final_evaluated_epoch == configured_epochs
    reached_early_stop_patience = bool(
        patience > 0 and final_evaluated_epoch - best_epoch >= patience
    )
    if not (reached_epoch_budget or reached_early_stop_patience):
        raise ValueError("mapper training summary does not prove natural completion")
    return {
        "training_summary_stage": "train_surface_maplet_mapper",
        "best_epoch": best_epoch,
        "best_validation": summary.get("best_validation"),
        "final_evaluated_epoch": final_evaluated_epoch,
        "configured_epochs": configured_epochs,
        "patience": patience,
        "reached_epoch_budget": reached_epoch_budget,
        "reached_early_stop_patience": reached_early_stop_patience,
        "natural_completion_verified": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training_summary", required=True)
    parser.add_argument("--expected_patience", type=int, required=True)
    parser.add_argument("--surface_maplets", required=True)
    parser.add_argument("--radio_final_manifest", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument(
        "--reference_surface_maplets",
        default="",
        help="optional all-map control bank used only for coverage deltas",
    )
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--training_trajectories", nargs="+", required=True)
    parser.add_argument("--validation_trajectories", nargs="+", required=True)
    parser.add_argument(
        "--strict_holdout_trajectories",
        nargs="+",
        default=("seq3", "seq5", "seq12", "seq13", "seq14"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output_checkpoint)
    summary = Path(args.summary_json)
    if not bool(args.force) and (output.exists() or summary.exists()):
        raise FileExistsError("refusing to overwrite sealed surface mapper")
    training = {str(value) for value in args.training_trajectories}
    validation = {str(value) for value in args.validation_trajectories}
    holdout = {str(value) for value in args.strict_holdout_trajectories}
    if not training or not validation or training & validation:
        raise ValueError("training/validation trajectories must be nonempty/disjoint")
    if (training | validation) & holdout:
        raise ValueError("strict holdout appears in mapper split")

    checkpoint_path = Path(args.checkpoint).resolve()
    training_summary_path = Path(args.training_summary).resolve()
    bank_path = Path(args.surface_maplets).resolve()
    manifest_path = Path(args.radio_final_manifest).resolve()
    physical_path = Path(args.physical_map).resolve()
    mapper, checkpoint_metadata = load_surface_maplet_mapper(
        checkpoint_path, device="cpu"
    )
    training_summary_payload = json.loads(training_summary_path.read_text())
    if not isinstance(training_summary_payload, dict):
        raise ValueError("mapper training summary root is not an object")
    training_completion = _verified_training_completion(
        training_summary_payload,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        expected_patience=int(args.expected_patience),
    )
    bank = VfmSurfaceMapletBank.load_npz(bank_path)
    manifest = TokenBankManifest.from_json(manifest_path)
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    manifest.validate(verify_checksums=False)
    bank_hash = file_sha256(bank_path)
    manifest_hash = file_sha256(manifest_path)
    lineage = _verified_lineage(
        bank_metadata=bank.metadata,
        bank_file_sha256=bank_hash,
        manifest_file_sha256=manifest_hash,
    )
    if str(lineage.get("physical_map_sha256", "")) != physical.content_sha256:
        raise ValueError("surface-maplet bank and physical map content hashes differ")
    if str(lineage.get("physical_map_file_sha256", "")) != file_sha256(physical_path):
        raise ValueError("surface-maplet bank and physical map file hashes differ")

    bank_routes = {_trajectory(value) for value in bank.view_image_ids}
    manifest_routes = {_trajectory(record.image_id) for record in manifest.records}
    if bank_routes != training | validation or manifest_routes != training | validation:
        raise ValueError("bank/manifest route inventory differs from the declared map split")
    if bank_routes & holdout or manifest_routes & holdout:
        raise ValueError("strict holdout appears in mapper supervision inputs")
    if {
        str(value) for value in lineage.get("training_trajectory_ids", ())
    } != training:
        raise ValueError("surface-maplet lineage training routes differ")
    if {
        str(value) for value in lineage.get("validation_trajectory_ids", ())
    } != validation:
        raise ValueError("surface-maplet lineage validation routes differ")
    if {
        str(value) for value in lineage.get("strict_holdout_trajectory_ids", ())
    } != holdout:
        raise ValueError("surface-maplet lineage strict holdout routes differ")
    if int(lineage.get("selected_contributor_count", -1)) != len(manifest.records):
        raise ValueError("surface-maplet lineage contributor count differs")
    bank_images = {str(value) for value in bank.view_image_ids}
    expected_training_images = {
        value for value in bank_images if _trajectory(value) in training
    }
    expected_validation_images = {
        value for value in bank_images if _trajectory(value) in validation
    }
    if not expected_training_images or not expected_validation_images:
        raise ValueError("bank lacks a declared training or validation image split")
    for key, expected in (
        ("training_trajectory_ids", training),
        ("validation_trajectory_ids", validation),
    ):
        if {str(value) for value in checkpoint_metadata.get(key, ())} != expected:
            raise ValueError(f"checkpoint {key} differs")
    checkpoint_holdout = {
        str(value) for value in checkpoint_metadata.get(
            "strict_holdout_trajectory_ids", ()
        )
    }
    if not holdout.issubset(checkpoint_holdout):
        raise ValueError("checkpoint does not declare every strict holdout")
    for key in ("training_images", "validation_images", "prototype_images"):
        routes = {_trajectory(value) for value in checkpoint_metadata.get(key, ())}
        if routes & holdout:
            raise ValueError(f"checkpoint {key} contains a strict holdout")
    checkpoint_training_images = {
        str(value) for value in checkpoint_metadata.get("training_images", ())
    }
    checkpoint_validation_images = {
        str(value) for value in checkpoint_metadata.get("validation_images", ())
    }
    checkpoint_prototype_images = {
        str(value) for value in checkpoint_metadata.get("prototype_images", ())
    }
    if checkpoint_training_images != expected_training_images:
        raise ValueError("checkpoint training image inventory differs from the bank")
    if checkpoint_validation_images != expected_validation_images:
        raise ValueError("checkpoint validation image inventory differs from the bank")
    if checkpoint_prototype_images != expected_training_images:
        raise ValueError("checkpoint prototype inventory differs from the deployed map split")
    if any(bool(checkpoint_metadata.get(key, False)) for key in (
        "uses_radio_intermediate", "uses_sfm_points", "uses_sfm_tracks",
    )):
        raise ValueError("checkpoint uses a forbidden supervision source")

    sealed_metadata = {
        **checkpoint_metadata,
        "supervision_coordinate_lineage": lineage,
        "coordinate_supervision_sealed": True,
        "coordinate_supervision_seal_schema": SCHEMA,
        "unsealed_checkpoint_file_sha256": file_sha256(checkpoint_path),
        "training_summary_file_sha256": file_sha256(training_summary_path),
        "training_completion": training_completion,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_surface_maplet_mapper(output, mapper.model, metadata=sealed_metadata)
    _, reloaded_metadata = load_surface_maplet_mapper(output, device="cpu")
    if dict(reloaded_metadata.get("supervision_coordinate_lineage", {})) != lineage:
        raise RuntimeError("sealed mapper coordinate lineage did not round-trip")

    bank_observation_routes = Counter(
        _trajectory(value) for value in bank.view_image_ids
    )
    manifest_image_routes = Counter(
        _trajectory(record.image_id) for record in manifest.records
    )
    physical_parent_count = int(physical.maplet_ids.size)
    current_parent_count = int(bank.maplet_ids.size)
    coverage_comparison: dict[str, object] | None = None
    if str(args.reference_surface_maplets):
        reference_path = Path(args.reference_surface_maplets).resolve()
        reference = VfmSurfaceMapletBank.load_npz(reference_path)
        reference_parent_count = int(reference.maplet_ids.size)
        reference_observation_count = int(len(reference.view_image_ids))
        coverage_comparison = {
            "reference_surface_maplets": str(reference_path),
            "reference_surface_maplets_file_sha256": file_sha256(reference_path),
            "reference_parent_identity_count": reference_parent_count,
            "reference_parent_identity_fraction_of_physical": float(
                reference_parent_count / max(physical_parent_count, 1)
            ),
            "reference_view_observation_count": reference_observation_count,
            "parent_identity_count_delta": current_parent_count - reference_parent_count,
            "parent_identity_retention_fraction": float(
                current_parent_count / max(reference_parent_count, 1)
            ),
            "view_observation_count_delta": int(len(bank.view_image_ids))
            - reference_observation_count,
            "view_observation_retention_fraction": float(
                len(bank.view_image_ids) / max(reference_observation_count, 1)
            ),
        }

    report = {
        "artifact_type": SCHEMA,
        "input_checkpoint": str(checkpoint_path),
        "input_checkpoint_file_sha256": file_sha256(checkpoint_path),
        "training_summary": str(training_summary_path),
        "training_summary_file_sha256": file_sha256(training_summary_path),
        "training_completion": training_completion,
        "surface_maplets": str(bank_path),
        "surface_maplets_file_sha256": bank_hash,
        "radio_final_manifest": str(manifest_path),
        "radio_final_manifest_file_sha256": manifest_hash,
        "physical_map": str(physical_path),
        "physical_map_file_sha256": file_sha256(physical_path),
        "physical_map_sha256": physical.content_sha256,
        "output_checkpoint": str(output.resolve()),
        "output_checkpoint_file_sha256": file_sha256(output),
        "training_trajectory_ids": sorted(training),
        "validation_trajectory_ids": sorted(validation),
        "strict_holdout_trajectory_ids": sorted(holdout),
        "strict_holdout_present": False,
        "filtered_contributor_count": int(
            lineage.get("selected_contributor_count", len(manifest.records))
        ),
        "filtered_contributor_routes": sorted(bank_routes),
        "bank_parent_identity_count": current_parent_count,
        "physical_parent_count": physical_parent_count,
        "bank_parent_identity_fraction_of_physical": float(
            current_parent_count / max(physical_parent_count, 1)
        ),
        "bank_view_observation_count": int(len(bank.view_image_ids)),
        "bank_unique_image_count": len(bank_images),
        "manifest_image_count": len(manifest.records),
        "bank_view_observation_count_by_trajectory": {
            key: int(value) for key, value in sorted(bank_observation_routes.items())
        },
        "manifest_image_count_by_trajectory": {
            key: int(value) for key, value in sorted(manifest_image_routes.items())
        },
        "coverage_comparison_to_all_map_control": coverage_comparison,
        "supervision_coordinate_lineage": lineage,
        "promotion_eligible_as_mapper": True,
        "trains_or_reconstructs_3dgs": False,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
