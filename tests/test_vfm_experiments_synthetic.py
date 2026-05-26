import json

from feature_extract.vfm.experiments import (
    GateExpectation,
    VFMExperimentManifest,
    validate_gate_expectations,
)
from feature_extract.vfm.protocols import EvaluationProtocol, ProtocolKind
from feature_extract.vfm.synthetic import run_synthetic_feature_utility_validation


def test_experiment_manifest_roundtrip_and_leakage_check(tmp_path):
    protocol = EvaluationProtocol(
        name="synthetic_real_retrieval",
        kind=ProtocolKind.REAL_RETRIEVAL,
        split="test",
        candidate_generator="synthetic_fixed_top4",
        allowed_training_inputs=("query_tokens", "candidate_tokens", "candidate_prior"),
        candidate_uses_gt=False,
        solver_conditioned=False,
    )
    manifest = VFMExperimentManifest(
        experiment_id="synthetic_feature_utility",
        protocol=protocol,
        methods=("raw_vfm", "selected_feature", "metadata_only"),
        metrics=("pred_cost_m", "top1_acc", "spearman"),
        seeds=(0, 1),
        output_dir="output/vfm/synthetic_feature_utility",
    )
    path = tmp_path / "manifest.json"

    manifest.to_json(path)
    loaded = VFMExperimentManifest.from_json(path)

    assert loaded == manifest
    loaded.validate_training_inputs(("query_tokens", "candidate_tokens"))


def test_gate_expectations_detect_positive_synthetic_result():
    result = run_synthetic_feature_utility_validation(query_count=32, candidates_per_query=6, seed=3)
    expectations = [
        GateExpectation(metric="mean_top1_acc", op=">=", value=0.85),
        GateExpectation(metric="mean_pred_cost_m", op="<=", value=0.15),
    ]

    selected = result["selected_feature"]
    raw = result["raw_vfm"]

    validate_gate_expectations(selected, expectations)
    assert selected["mean_top1_acc"] > raw["mean_top1_acc"]
    assert selected["mean_pred_cost_m"] < raw["mean_pred_cost_m"]
    assert result["query_shuffle"]["mean_top1_acc"] < selected["mean_top1_acc"]


def test_synthetic_result_is_json_serializable():
    result = run_synthetic_feature_utility_validation(query_count=4, candidates_per_query=4, seed=0)
    encoded = json.dumps(result, sort_keys=True)

    assert "selected_feature" in encoded
    assert "metadata_only" in encoded
