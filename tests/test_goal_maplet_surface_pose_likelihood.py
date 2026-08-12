import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.verify_goal_maplet_pose_modes_with_surface_field import (
    _risk_summary,
)
from feature_extract.tools.vfm.train_goal_maplet_surface_pose_likelihood import (
    _trajectory_balanced_order,
    main as train_surface_likelihood,
)
from feature_extract.tools.vfm.build_goal_maplet_surface_likelihood_samples import (
    _typed_teacher_weight,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_surface_likelihood_crossfit import (
    _threshold_oracle_choice,
)
from feature_extract.vfm.localization_goal_maplet.surface_pose_likelihood import (
    EVENT_NAMES,
    FEATURE_NAMES,
    SurfacePoseLikelihoodConfig,
    SAMPLE_SCHEMA,
    ViewGeometryConditionedSurfaceLikelihood,
    assign_pose_defined_typed_targets,
    extract_surface_likelihood_features,
    listwise_surface_pose_loss,
    load_surface_pose_likelihood,
    save_surface_pose_likelihood,
)


def _rendered(feature: np.ndarray, *, valid: bool = True):
    height, width = feature.shape[1:]
    mask = np.full((height, width), valid, dtype=bool)
    return SimpleNamespace(
        feature=feature,
        mask=mask,
        visibility=np.ones_like(mask),
        field_missing=~mask,
        depth=np.full(mask.shape, 4.0, dtype=np.float32),
        normal=np.dstack(
            [np.zeros(mask.shape), np.zeros(mask.shape), np.ones(mask.shape)]
        ).astype(np.float32),
        incidence=np.full(mask.shape, 0.8, dtype=np.float32),
        projected_scale=np.full(mask.shape, 2.0, dtype=np.float32),
        uncertainty=np.full(mask.shape, 0.1, dtype=np.float32),
        maplet_id=np.arange(height * width, dtype=np.int64).reshape(height, width),
    )


def test_typed_surface_evidence_defers_physical_phase_to_pose_labels() -> None:
    query = np.zeros((4, 3, 5), dtype=np.float32)
    query[0] = 1.0
    match, typed, summary = extract_surface_likelihood_features(
        query, _rendered(query.copy()), np.eye(4),
    )
    wrong_feature = query.copy()
    wrong_feature[0] = -1.0
    wrong, wrong_typed, _ = extract_surface_likelihood_features(
        query, _rendered(wrong_feature), np.eye(4),
    )
    missing, missing_typed, _ = extract_surface_likelihood_features(
        query, _rendered(query.copy(), valid=False), np.eye(4),
    )
    assert match.shape == wrong.shape == missing.shape == (15, len(FEATURE_NAMES))
    assert summary.shape == (4,)
    assert set(typed.tolist()) == {EVENT_NAMES.index("unresolved")}
    assert set(wrong_typed.tolist()) == {EVENT_NAMES.index("unresolved")}
    assert set(missing_typed.tolist()) == {EVENT_NAMES.index("field_missing")}
    match_target = assign_pose_defined_typed_targets(
        typed, match, translation_m=0.2, rotation_deg=2.0,
        is_listwise_target=True,
    )
    phase_target = assign_pose_defined_typed_targets(
        wrong_typed, wrong, translation_m=0.8, rotation_deg=2.0,
        is_listwise_target=False,
    )
    assert set(match_target.tolist()) == {EVENT_NAMES.index("surface_match")}
    assert set(phase_target.tolist()) == {EVENT_NAMES.index("wrong_phase")}


def test_typed_support_events_do_not_mislabel_grazing_as_occlusion() -> None:
    query = np.zeros((4, 2, 2), dtype=np.float32)
    query[0] = 1.0
    grazing = _rendered(query.copy())
    grazing.incidence[:] = 0.05
    _feature, typed, _summary = extract_surface_likelihood_features(
        query, grazing, np.eye(4),
    )
    assert set(typed.tolist()) == {EVENT_NAMES.index("grazing_surface")}

    outside = _rendered(query.copy(), valid=False)
    outside.visibility[:] = False
    outside.field_missing[:] = False
    _feature, typed, _summary = extract_surface_likelihood_features(
        query, outside, np.eye(4),
    )
    assert set(typed.tolist()) == {EVENT_NAMES.index("outside_render_support")}


def test_listwise_surface_likelihood_competes_with_typed_null() -> None:
    query = np.zeros((4, 3, 5), dtype=np.float32)
    query[0] = 1.0
    good, good_type, summary = extract_surface_likelihood_features(
        query, _rendered(query.copy()), np.eye(4),
    )
    bad_map = query.copy()
    bad_map[0] = -1.0
    bad, bad_type, _ = extract_surface_likelihood_features(
        query, _rendered(bad_map), np.eye(4),
    )
    feature = torch.from_numpy(np.stack([[good, bad]], axis=0))
    typed = torch.from_numpy(np.stack([[good_type, bad_type]], axis=0)).long()
    query_summary = torch.from_numpy(summary[None])
    model = ViewGeometryConditionedSurfaceLikelihood(SurfacePoseLikelihoodConfig(hidden_dim=16))
    validity = torch.tensor([[True, False]])
    probability, null_probability, _ = model.posterior(feature, query_summary, validity)
    assert probability[0, 0] > probability[0, 1]
    assert probability[0, 1] == 0.0
    assert torch.isfinite(null_probability).all()
    loss, report = listwise_surface_pose_loss(
        model, feature, query_summary, torch.tensor([0]), typed,
        candidate_valid=validity,
    )
    assert torch.isfinite(loss)
    assert report["listwise_nll"] > 0.0


def test_candidate_conditioned_null_handles_padded_frozen_sets() -> None:
    model = ViewGeometryConditionedSurfaceLikelihood(SurfacePoseLikelihoodConfig(
        hidden_dim=16, candidate_conditioned_null=True,
    ))
    feature = torch.zeros((2, 3, 5, len(FEATURE_NAMES)), dtype=torch.float32)
    summary = torch.zeros((2, 4), dtype=torch.float32)
    validity = torch.tensor([[True, False, False], [True, True, True]])
    candidate, null, event = model.posterior(feature, summary, validity)
    assert candidate.shape == (2, 3)
    assert event.shape == (2, 3, 5, len(EVENT_NAMES))
    assert candidate[0, 1] == 0.0
    assert candidate[0, 2] == 0.0
    assert torch.isfinite(null).all()


def test_surface_likelihood_artifact_preserves_single_field_contract(tmp_path: Path) -> None:
    model = ViewGeometryConditionedSurfaceLikelihood(SurfacePoseLikelihoodConfig(hidden_dim=16))
    path = tmp_path / "likelihood.pt"
    save_surface_pose_likelihood(
        model,
        path,
        metadata={"candidate_set_frozen_before_scoring": True},
    )
    loaded, metadata = load_surface_pose_likelihood(path)
    assert isinstance(loaded, ViewGeometryConditionedSurfaceLikelihood)
    assert metadata["stored_map_feature_type_count"] == 1
    assert metadata["stored_downstream_embedding_count"] == 0
    assert metadata["runtime_geometry_is_ephemeral"] is True
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metadata"]["stored_map_feature_type_count"] = 2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="single-field"):
        load_surface_pose_likelihood(path)


def test_risk_summary_does_not_treat_abstention_as_a_pose_catastrophe() -> None:
    rows = [{
        "mode_details": {
            "actual_parent_actual_child": [{
                "translation_m": 20.0,
                "rotation_deg": 90.0,
                "surface_alignment_score": 0.1,
            }],
        },
        "ranking_diagnostics": {
            "actual_parent_actual_child": {
                "surface_pose_null_probability": 0.9,
            },
        },
    }]
    summary = _risk_summary(rows)["actual_parent_actual_child"]
    assert summary["abstain_rate"] == 1.0
    assert summary["catastrophic_rate"] == 0.0
    assert summary["catastrophic_or_abstain_rate"] == 1.0
    assert summary["risk_coverage"]["coverage_1.0"]["accepted_count"] == 0


def test_exact_pool_oracle_prioritizes_declared_success_threshold() -> None:
    choice = _threshold_oracle_choice(
        np.asarray([0.51, 0.49, 0.1]),
        np.asarray([0.1, 4.9, 0.1]),
        np.asarray([True, True, False]),
    )
    assert choice == 1


def test_trajectory_balanced_training_does_not_follow_frame_count_skew() -> None:
    trajectory = np.asarray(["seq1"] * 6 + ["seq2"] * 2 + ["seq3"])
    order = _trajectory_balanced_order(trajectory, np.random.default_rng(3))
    selected = trajectory[order]
    counts = {name: int(np.sum(selected == name)) for name in set(selected.tolist())}
    assert counts == {"seq1": 3, "seq2": 3, "seq3": 3}


def test_teacher_cues_are_routed_to_typed_roles_instead_of_static_average() -> None:
    target = np.asarray([[
        EVENT_NAMES.index("surface_match"),
        EVENT_NAMES.index("wrong_phase"),
        EVENT_NAMES.index("field_missing"),
    ]], dtype=np.uint8)
    cue = {
        "dino_local_affinity": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "sam_boundary": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "siglip_context": np.asarray([0.5, 0.5, 0.0], dtype=np.float32),
    }
    weight = _typed_teacher_weight(target, cue)
    assert weight[0, 1] > weight[0, 0]
    assert weight[0, 2] == pytest.approx(1.5)


def test_fixed_epoch_training_never_requires_or_selects_a_validation_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "train.npz"
    model_path = tmp_path / "model.pt"
    report_path = tmp_path / "report.json"
    feature = np.zeros((1, 2, 3, len(FEATURE_NAMES)), dtype=np.float16)
    feature[0, 0, :, FEATURE_NAMES.index("cosine")] = 0.8
    feature[0, 1, :, FEATURE_NAMES.index("cosine")] = 0.2
    feature[..., FEATURE_NAMES.index("render_valid")] = 1.0
    feature[..., FEATURE_NAMES.index("surface_visible")] = 1.0
    metadata = {
        "artifact_type": SAMPLE_SCHEMA,
        "physical_map_sha256": "physical",
        "canonical_field_sha256": "field",
        "surface_mapper_sha256": "mapper",
        "physical_instance_readout_sha256": "readout",
        "candidate_count": 2,
    }
    np.savez_compressed(
        sample,
        token_feature=feature,
        typed_target=np.zeros((1, 2, 3), dtype=np.uint8),
        query_summary=np.zeros((1, 4), dtype=np.float32),
        translation_m=np.asarray([[0.2, 3.0]], dtype=np.float32),
        rotation_deg=np.asarray([[2.0, 20.0]], dtype=np.float32),
        candidate_valid=np.ones((1, 2), dtype=bool),
        target_index=np.asarray([0], dtype=np.int64),
        teacher_weight=np.ones((1, 2, 3), dtype=np.float16),
        image_ids=np.asarray(["seq12/frame.png"]),
        trajectory_ids=np.asarray(["seq12"]),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    monkeypatch.setattr("sys.argv", [
        "train_goal_maplet_surface_pose_likelihood.py",
        "--train_samples", str(sample),
        "--checkpoint_protocol", "fixed_epoch_no_selection",
        "--epochs", "1",
        "--hidden_dim", "8",
        "--output_model", str(model_path),
        "--summary_json", str(report_path),
        "--device", "cpu",
    ])
    train_surface_likelihood()
    _model, trained = load_surface_pose_likelihood(model_path)
    assert trained["checkpoint_protocol"] == "fixed_epoch_no_selection"
    assert trained["checkpoint_epoch"] == 1
    assert trained["checkpoint_selection_uses_pose_labels"] is False
    assert trained["selection_trajectory_ids"] == []
