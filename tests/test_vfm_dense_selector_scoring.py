import json

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.score_candidate_bank_dense_selector import main as dense_selector_cli_main
from feature_extract.vfm.dense_selector_scoring import score_candidate_bank_by_dense_selector
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_table import evaluate_score_table
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.selector_descriptor_scoring import load_selector_from_checkpoint
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec


def _selector_checkpoint(path):
    selector = LocalizableFeatureSelector(input_dim=2, output_dim=2, group_size=1)
    with torch.no_grad():
        selector.group_logits.fill_(8.0)
        selector.projection.weight.zero_()
        selector.projection.weight[0, 0, 0, 0] = 1.0
        selector.projection.weight[1, 1, 0, 0] = 1.0
        selector.utility_head.weight.zero_()
        selector.utility_head.bias.zero_()
        selector.uncertainty_head.weight.zero_()
        selector.uncertainty_head.bias.zero_()
    torch.save(selector.state_dict(), path)


def _token_record(tmp_path, image_id, feature, split):
    token_path = tmp_path / f"{image_id.replace('/', '__')}.npz"
    np.savez_compressed(token_path, radio_final=np.asarray(feature, dtype=np.float32))
    return TokenBankRecord(
        image_id=image_id,
        token_path=token_path,
        layers=(TokenLayerSpec("radio_final", "radio", "final", 2, 14),),
        split=split,
        scene="synthetic",
    )


def _manifests(tmp_path):
    query = _token_record(
        tmp_path,
        "query.png",
        np.asarray([[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        "test",
    )
    good = _token_record(
        tmp_path,
        "good.png",
        np.asarray([[[0.9, 0.9], [0.9, 0.9]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        "train",
    )
    bad = _token_record(
        tmp_path,
        "bad.png",
        np.asarray([[[-1.0, -1.0], [-1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        "train",
    )
    return TokenBankManifest(records=(query,)), TokenBankManifest(records=(bad, good))


def _candidate_bank():
    return CandidateHypothesisBank.from_candidates(
        protocol_name="dense_selector_scoring_smoke",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="bad",
                candidate_type="reference_pose",
                reference_image="bad.png",
                pose_error=PoseCost(1.0, 20.0),
            ),
            CandidateHypothesis(
                query_id="query.png",
                candidate_id="good",
                candidate_type="reference_pose",
                reference_image="good.png",
                pose_error=PoseCost(0.05, 1.0),
            ),
        ],
    )


def test_dense_selector_scoring_prefers_correct_reference(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)
    query_manifest, map_manifest = _manifests(tmp_path)

    rows = score_candidate_bank_by_dense_selector(
        bank=_candidate_bank(),
        query_manifest=query_manifest,
        map_manifest=map_manifest,
        selector=load_selector_from_checkpoint(checkpoint),
        layer_name="radio_final",
        method="dense_selector_cosine",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        device="cpu",
    )
    report = evaluate_score_table(rows)

    assert rows[1].score > rows[0].score
    assert report.mean_top1_acc == pytest.approx(1.0)


def test_dense_selector_scoring_cli_writes_report(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    bank_path = tmp_path / "candidates.jsonl"
    query_manifest_path = tmp_path / "query_manifest.json"
    map_manifest_path = tmp_path / "map_manifest.json"
    rows_path = tmp_path / "rows.json"
    report_path = tmp_path / "report.json"
    _selector_checkpoint(checkpoint)
    query_manifest, map_manifest = _manifests(tmp_path)
    query_manifest.to_json(query_manifest_path)
    map_manifest.to_json(map_manifest_path)
    _candidate_bank().to_jsonl(bank_path)

    dense_selector_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_manifest",
            str(query_manifest_path),
            "--map_manifest",
            str(map_manifest_path),
            "--selector_checkpoint",
            str(checkpoint),
            "--layer_name",
            "radio_final",
            "--device",
            "cpu",
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
        ]
    )

    rows = json.loads(rows_path.read_text())
    report = json.loads(report_path.read_text())
    assert rows[1]["score"] > rows[0]["score"]
    assert report["mean_top1_acc"] == pytest.approx(1.0)
    assert report["inputs"]["protocol_name"] == "dense_selector_scoring_smoke"
    assert report["inputs"]["input_files"]["selector_checkpoint"]["sha256"]
