"""Phase-2 raw support for the frozen query-independent all-parent domain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c
from feature_extract.vfm.localization_goal_maplet.global_physical_pose_support import (
    load_all_parent_union_support,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


SCHEMA = "goal_maplet_global_all_parent_geometry_raw_support_v1"
SELECTION_THRESHOLD = 0.95
EXPECTED_QUERY_COUNTS = {"seq10": 88, "seq12": 188, "seq14": 36}
THRESHOLDS = (
    ("region_2m_45deg", 2.0, 45.0),
    ("loose_1m_10deg", 1.0, 10.0),
    ("strict_0_5m_5deg", 0.5, 5.0),
)
CONTRIBUTOR_MEMBERS = {
    "topk_ids.npy", "topk_weights.npy", "dominant_depth.npy", "pose_w2c.npy",
    "camera_model_id.npy", "camera_width.npy", "camera_height.npy",
    "camera_params.npy", "metadata_json.npy",
}


def _stats(values: np.ndarray) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(value)),
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p90": float(np.percentile(value, 90.0)),
        "maximum": float(np.max(value)),
    }


def _rotation_errors(codebook: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.asarray(codebook, dtype=np.float64) @ target[:3, :3].T
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0,
    )
    return np.degrees(np.arccos(cosine))


def _query_ids_from_pose_free_manifest(
    path: Path, route: str, protocol: dict,
) -> list[str]:
    official = protocol.get("official_train", {})
    if file_sha256(path) != official.get("token_manifest_sha256"):
        raise ValueError("pose-free query manifest bytes differ from protocol")
    manifest = json.loads(path.read_text())
    records = list(manifest.get("records", ()))
    all_ids = [str(record.get("image_id", "")) for record in records]
    if (
        len(all_ids) != int(official.get("count", -1))
        or len(set(all_ids)) != len(all_ids)
    ):
        raise ValueError("pose-free official-train query inventory differs")
    selected = sorted(value for value in all_ids if value.startswith(route + "/"))
    expected = int(official.get("trajectory_counts", {}).get(route, -1))
    if len(selected) != expected or expected != EXPECTED_QUERY_COUNTS[route]:
        raise ValueError("pose-free route query inventory differs")
    return selected


def _validate_protocol_and_audit(
    protocol_path: Path, audit_path: Path, contributor_dir: Path,
) -> tuple[dict, dict]:
    protocol = json.loads(protocol_path.read_text())
    official = protocol.get("official_train", {})
    if (
        protocol.get("artifact_type") != "goal_maplet_official_train_oof_protocol_v1"
        or int(official.get("count", -1)) != 1487
        or not str(official.get("pose_file_sha256", ""))
        or not str(official.get("token_manifest_sha256", ""))
    ):
        raise ValueError("official-train protocol differs")
    audit = json.loads(audit_path.read_text())
    audit_dir = Path(str(audit.get("contributors", "")))
    if not audit_dir.is_absolute():
        audit_dir = (Path.cwd() / audit_dir).resolve()
    if (
        audit.get("artifact_type") != "goal_maplet_contributor_cache_audit_v1"
        or audit.get("pass") is not True
        or int(audit.get("file_count", -1)) != int(official["count"])
        or audit.get("image_ids_sha256") != official.get("image_ids_sha256")
        or audit_dir != contributor_dir
        or len(audit.get("clean_geometry_source_sha256", ())) != 1
        or len(audit.get("clean_source_index_sha256", ())) != 1
        or len(audit.get("geometry_source_sha256", ())) != 1
    ):
        raise ValueError("clean contributor audit differs")
    return protocol, audit


def _load_route_poses(
    contributor_dir: Path,
    image_ids: list[str],
    route: str,
    audit: dict,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    clean_geometry = str(audit["clean_geometry_source_sha256"][0])
    clean_index = str(audit["clean_source_index_sha256"][0])
    geometry = str(audit["geometry_source_sha256"][0])
    poses, bindings = [], []
    for image_id in image_ids:
        artifact = contributor_dir / (image_id.replace("/", "__") + ".npz")
        try:
            with zipfile.ZipFile(artifact, "r") as archive:
                members = [entry.filename for entry in archive.infolist()]
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError("route contributor is not a valid NPZ") from error
        if len(members) != len(set(members)) or set(members) != CONTRIBUTOR_MEMBERS:
            raise ValueError("route contributor exact NPZ members differ")
        digest = file_sha256(artifact)
        with np.load(artifact, allow_pickle=False) as data:
            if set(data.files) != {name[:-4] for name in CONTRIBUTOR_MEMBERS}:
                raise ValueError("route contributor arrays differ")
            pose_raw = np.asarray(data["pose_w2c"])
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        pose = np.asarray(pose_raw, dtype=np.float64)
        rotation = pose[:3, :3] if pose.shape == (4, 4) else np.empty((0, 0))
        if (
            pose_raw.dtype != np.float64 or pose.shape != (4, 4)
            or np.any(~np.isfinite(pose))
            or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)
            or metadata.get("artifact_type") != "v6_training_contributor_cache"
            or metadata.get("image_id") != image_id
            or metadata.get("trajectory_id") != route
            or metadata.get("stores_rgb") is not False
            or metadata.get("stores_rgb_path") is not False
            or metadata.get("clean_geometry_source_sha256") != clean_geometry
            or metadata.get("clean_source_index_sha256") != clean_index
            or metadata.get("geometry_source_sha256") != geometry
        ):
            raise ValueError("route contributor pose/lineage contract differs")
        poses.append(pose)
        bindings.append({
            "image_id": image_id,
            "path": str(artifact.resolve()),
            "file_sha256": digest,
        })
    return np.stack(poses), bindings


def _validate_frozen_seq10_gate(path: Path) -> dict:
    report = json.loads(path.read_text())
    unhashed = {key: value for key, value in report.items() if key != "content_sha256"}
    gate = report.get("seq10_absolute_gate", {})
    if (
        report.get("artifact_type") != SCHEMA
        or report.get("query_route") != "seq10"
        or report.get("content_sha256") != canonical_json_sha256(unhashed)
        or gate.get("decision") != "GO"
        or int(gate.get("required_hits", -1)) != 84
        or int(gate.get("observed_hits", -1)) < 84
        or float(gate.get("threshold", -1.0)) != SELECTION_THRESHOLD
    ):
        raise ValueError("held evaluation lacks a passing frozen seq10 gate")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--query_route", choices=tuple(EXPECTED_QUERY_COUNTS), required=True)
    parser.add_argument("--token_manifest", required=True)
    parser.add_argument("--official_protocol", required=True)
    parser.add_argument("--contributors", required=True)
    parser.add_argument("--contributor_audit", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development_gate", action="store_true")
    mode.add_argument("--frozen_seq10_gate")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    route = str(args.query_route)
    if bool(args.development_gate) != (route == "seq10"):
        raise ValueError("only seq10 may create the all-parent development gate")
    output = Path(args.output_json).resolve()
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite all-parent support evaluation")
    started = time.perf_counter()
    proposal_path = Path(args.proposal).resolve()
    arrays, metadata = load_all_parent_union_support(proposal_path)
    protocol_path = Path(args.official_protocol).resolve()
    audit_path = Path(args.contributor_audit).resolve()
    contributor_dir = Path(args.contributors).resolve()
    protocol, contributor_audit = _validate_protocol_and_audit(
        protocol_path, audit_path, contributor_dir,
    )
    token_manifest = Path(args.token_manifest).resolve()
    image_ids = _query_ids_from_pose_free_manifest(
        token_manifest, route, protocol,
    )
    frozen = None
    if args.frozen_seq10_gate:
        frozen_path = Path(args.frozen_seq10_gate).resolve()
        frozen = _validate_frozen_seq10_gate(frozen_path)
        if (
            frozen.get("proposal_content_sha256") != metadata["content_sha256"]
            or frozen.get("proposal_file_sha256") != file_sha256(proposal_path)
        ):
            raise ValueError("held evaluation proposal differs from seq10 frozen gate")
    # This is the first operation that opens pose-bearing files.  The Phase-1
    # proposal and every geometry/cap validation above have already completed.
    target, contributor_bindings = _load_route_poses(
        contributor_dir, image_ids, route, contributor_audit,
    )
    origin = np.asarray(arrays["lattice_origin_world"], dtype=np.float64)
    spacing = float(arrays["lattice_spacing_m"])
    positions = origin[None] + (
        np.asarray(arrays["cell_indices_world"], dtype=np.float64) + 0.5
    ) * spacing
    centers = np.stack([camera_center_from_pose_w2c(pose) for pose in target])
    best_translation = np.asarray([
        np.sqrt(np.min(np.sum((positions - center[None]) ** 2, axis=1)))
        for center in centers
    ])
    codebook = np.asarray(arrays["orientation_rotations_w2c"], dtype=np.float64)
    best_rotation = np.asarray([
        np.min(_rotation_errors(codebook, pose)) for pose in target
    ])
    certificate_radius = float(
        metadata["orientation_cover_certificate"]["so3_covering_radius_deg"]
    )
    if np.any(best_rotation > certificate_radius + 1e-9):
        raise AssertionError("analytic60 empirical target exceeds its global certificate")
    raw_support = {}
    for name, translation_limit, rotation_limit in THRESHOLDS:
        position_hit = best_translation <= translation_limit
        orientation_hit = best_rotation <= rotation_limit
        joint = position_hit & orientation_hit
        raw_support[name] = {
            "position_hits": int(np.sum(position_hit)),
            "orientation_hits": int(np.sum(orientation_hit)),
            "joint_hits": int(np.sum(joint)),
            "joint_rate": float(np.mean(joint)),
            "position_misses": int(np.sum(~position_hit)),
            "orientation_misses": int(np.sum(~orientation_hit)),
            "joint_misses": int(np.sum(~joint)),
        }
    if raw_support["region_2m_45deg"]["orientation_hits"] != len(image_ids):
        raise AssertionError("analytic60 did not remove the 45-degree route ceiling")
    required = int(math.ceil(SELECTION_THRESHOLD * 88 - 1e-12))
    main_hits = int(raw_support["region_2m_45deg"]["joint_hits"])
    gate = None
    if route == "seq10":
        gate = {
            "metric": "absolute_region_2m_45deg_joint_query_rate",
            "threshold": SELECTION_THRESHOLD,
            "required_hits": required,
            "observed_hits": main_hits,
            "query_count": len(image_ids),
            "decision": "GO" if main_hits >= required else "KILL",
            "held_route_pose_labels_opened": False,
        }
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "query_route": route,
        "query_count": len(image_ids),
        "proposal": str(proposal_path),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_content_sha256": metadata["content_sha256"],
        "token_manifest": {
            "path": str(token_manifest), "file_sha256": file_sha256(token_manifest),
        },
        "official_protocol": {
            "path": str(protocol_path), "file_sha256": file_sha256(protocol_path),
            "declared_pose_file_sha256": protocol["official_train"]["pose_file_sha256"],
        },
        "contributor_audit": {
            "path": str(audit_path), "file_sha256": file_sha256(audit_path),
        },
        "contributor_inventory": contributor_bindings,
        "raw_support": raw_support,
        "best_translation_m": _stats(best_translation),
        "best_rotation_deg": _stats(best_rotation),
        "position_count": int(positions.shape[0]),
        "orientation_count": int(codebook.shape[0]),
        "implicit_pose_factor_count": int(arrays["implicit_pose_factor_count"]),
        "seq10_absolute_gate": gate,
        "frozen_seq10_gate": (
            {
                "path": str(Path(args.frozen_seq10_gate).resolve()),
                "file_sha256": file_sha256(Path(args.frozen_seq10_gate).resolve()),
                "content_sha256": frozen["content_sha256"],
            }
            if frozen is not None else None
        ),
        "query_rows": [
            {
                "image_id": image_id,
                "best_translation_m": float(best_translation[index]),
                "best_rotation_deg": float(best_rotation[index]),
            }
            for index, image_id in enumerate(image_ids)
        ],
        "score_before_label_separation": {
            "proposal_frozen_before_route_contributors_opened": True,
            "phase1_builder_has_no_query_retrieval_contributor_or_gt_argument": True,
            "only_selected_route_pose_files_opened_in_phase2": True,
            "held_labels_used_for_configuration_or_selection": False,
        },
        "support_claim": metadata["support_claim"],
        "raw_support_is_implicit_upper_bound_not_localization_success": True,
        "free_space_or_clearance_certificate": False,
        "collision_certificate": False,
        "ranking_performed": False,
        "phase2_elapsed_seconds": float(time.perf_counter() - started),
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output),
        "content_sha256": report["content_sha256"],
        "query_route": route,
        "query_count": len(image_ids),
        "raw_support": raw_support,
        "seq10_absolute_gate": gate,
        "phase2_elapsed_seconds": report["phase2_elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
