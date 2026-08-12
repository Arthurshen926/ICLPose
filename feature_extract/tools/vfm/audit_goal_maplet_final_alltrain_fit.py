"""Audit the frozen, strict 1487-frame map before official-test inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.tools.vfm.audit_goal_maplet_strict_map_fold import audit_strict_fold
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def audit_final(
    *, protocol_path: Path, frozen_path: Path, final_dir: Path,
) -> dict[str, object]:
    protocol = json.loads(Path(protocol_path).read_text())
    frozen = json.loads(Path(frozen_path).read_text())
    final_dir = Path(final_dir)
    strict = audit_strict_fold(final_dir)
    reports = {
        "train_contributors": json.loads(
            (final_dir / "contributors_official_train_audit.json").read_text()
        ),
        "test_contributors": json.loads(
            (final_dir / "contributors_official_test_audit.json").read_text()
        ),
        "mapper": json.loads((final_dir / "surface_mapper.json").read_text()),
        "geometry": json.loads(
            (final_dir / "geometry_head" / "geometry_head_summary.json").read_text()
        ),
        "canonical": json.loads((final_dir / "canonical_field.json").read_text()),
        "readout": json.loads((final_dir / "physical_readout.json").read_text()),
        "mapping_view": json.loads((final_dir / "mapping_view_graph.json").read_text()),
        "validity": json.loads((final_dir / "validity.json").read_text()),
    }
    train = protocol["official_train"]
    test = protocol["official_test"]
    mapper_split = reports["mapper"]["split"]
    official_train_routes = set(train["trajectory_counts"])
    official_test_routes = set(test["trajectory_counts"])
    geometry = reports["geometry"]
    checks = {
        "strict_map_uses_all_official_train": (
            strict["fold_id"] == "final_alltrain"
            and strict["mapping_image_count"] == int(train["count"])
            and not strict["held_query_trajectories"]
            and bool(strict["route_clean_end_to_end"])
        ),
        "train_contributors_complete": (
            bool(reports["train_contributors"]["pass"])
            and int(reports["train_contributors"]["file_count"]) == int(train["count"])
        ),
        "test_contributors_packaged_without_evaluation": (
            bool(reports["test_contributors"]["pass"])
            and int(reports["test_contributors"]["file_count"]) == int(test["count"])
        ),
        "contributors_use_final_strict_geometry": (
            reports["train_contributors"]["geometry_source_sha256"]
            == [strict["gaussian_ply_sha256"]]
            and reports["test_contributors"]["geometry_source_sha256"]
            == [strict["gaussian_ply_sha256"]]
        ),
        "mapper_fixed_all_train": (
            reports["mapper"]["config"]["checkpoint_protocol"]
            == "fixed_epoch_no_selection"
            and reports["mapper"]["best_validation"] is None
            and set(mapper_split["training_trajectory_ids"]) == official_train_routes
            and official_test_routes.issubset(
                set(mapper_split["strict_holdout_trajectory_ids"])
            )
        ),
        "geometry_fixed_all_train": (
            int(geometry["train_count"]) == int(train["count"])
            and geometry["checkpoint_protocol"] == "fixed_epoch_no_selection"
            and not bool(geometry["production_contract"]["eval_manifest_used_for_gradient"])
            and not bool(geometry["production_contract"]["eval_manifest_used_for_checkpoint_selection"])
        ),
        "canonical_all_train": (
            int(reports["canonical"]["mapping_image_count"]) == int(train["count"])
            and set(reports["canonical"]["mapping_trajectory_ids"])
            == official_train_routes
        ),
        "readout_all_train_no_selection": (
            int(reports["readout"]["teacher_supervised_image_count"])
            == int(train["count"])
            and int(reports["readout"]["partition_counts"]["selection"]) == 0
            and int(reports["readout"]["partition_counts"]["validation"]) == 0
            and reports["readout"]["selected_validation"] is None
        ),
        "mapping_view_all_train_source": (
            int(reports["mapping_view"].get("view_node_count", -1))
            <= int(train["count"])
            and set(reports["mapping_view"]["excluded_trajectory_ids"])
            == official_test_routes
        ),
        "validity_all_train": int(reports["validity"]["image_count"])
        == int(train["count"]),
        "frozen_protocol_bound": str(frozen["protocol_sha256"])
        == file_sha256(Path(protocol_path)),
    }
    result = {
        "artifact_type": "goal_maplet_final_alltrain_fit_audit_v1",
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(Path(protocol_path)),
        "frozen_configuration": str(frozen_path),
        "frozen_configuration_sha256": file_sha256(Path(frozen_path)),
        "final_dir": str(final_dir),
        "official_train_count": int(train["count"]),
        "official_test_count_packaged_not_evaluated": int(test["count"]),
        "strict_map_audit": strict,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not result["pass"]:
        raise ValueError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--final_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite final-fit audit")
    result = audit_final(
        protocol_path=Path(args.protocol), frozen_path=Path(args.frozen),
        final_dir=Path(args.final_dir),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
