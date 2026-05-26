import json

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.score_candidate_bank_selector_descriptors import main as score_cli_main
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_table import evaluate_score_table
from feature_extract.vfm.selector import LocalizableFeatureSelector
from feature_extract.vfm.selector_descriptor_scoring import (
    load_selector_from_checkpoint,
    score_candidate_bank_by_selector_descriptor_cosine,
)
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank


def _candidate_bank():
    return CandidateHypothesisBank.from_candidates(
        protocol_name="selector_descriptor_smoke",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="bad",
                candidate_type="reference_pose",
                reference_image="bad.png",
                pose_error=PoseCost(1.0, 20.0),
            ),
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="good",
                candidate_type="reference_pose",
                reference_image="good.png",
                pose_error=PoseCost(0.05, 1.0),
            ),
        ],
    )


def _descriptor_banks():
    query_bank = TokenDescriptorBank(
        image_ids=("q.png",),
        descriptors=np.asarray([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        layer_name="radio_final",
        pooling="mean",
        normalized=False,
    )
    map_bank = TokenDescriptorBank(
        image_ids=("bad.png", "good.png"),
        descriptors=np.asarray(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        ),
        layer_name="radio_final",
        pooling="mean",
        normalized=False,
    )
    return query_bank, map_bank


def _selector_checkpoint(path):
    selector = LocalizableFeatureSelector(input_dim=4, output_dim=2, group_size=2)
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


def test_selector_descriptor_scoring_prefers_matching_reference_and_reports_top1(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)
    selector = load_selector_from_checkpoint(checkpoint)
    query_bank, map_bank = _descriptor_banks()

    rows = score_candidate_bank_by_selector_descriptor_cosine(
        bank=_candidate_bank(),
        query_descriptors=query_bank,
        map_descriptors=map_bank,
        selector=selector,
        method="selector_radio_descriptor_cosine",
        translation_threshold_m=0.25,
        rotation_threshold_deg=5.0,
        device="cpu",
    )
    report = evaluate_score_table(rows)

    assert rows[1].score > rows[0].score
    assert report.mean_top1_acc == pytest.approx(1.0)


def test_selector_descriptor_scoring_cli_writes_rows_report_and_md(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    bank_path = tmp_path / "candidates.jsonl"
    query_path = tmp_path / "query.npz"
    map_path = tmp_path / "map.npz"
    rows_path = tmp_path / "rows.json"
    report_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    _selector_checkpoint(checkpoint)
    query_bank, map_bank = _descriptor_banks()
    _candidate_bank().to_jsonl(bank_path)
    query_bank.to_npz(query_path)
    map_bank.to_npz(map_path)

    score_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_descriptors",
            str(query_path),
            "--map_descriptors",
            str(map_path),
            "--selector_checkpoint",
            str(checkpoint),
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
            "--output_md",
            str(md_path),
            "--device",
            "cpu",
        ]
    )

    rows = json.loads(rows_path.read_text())
    report = json.loads(report_path.read_text())
    assert rows[0]["protocol_kind"] == ProtocolKind.REFERENCE_POSE.value
    assert rows[0]["method"] == "selector_radio_descriptor_cosine"
    assert report["mean_top1_acc"] == pytest.approx(1.0)
    assert report["inputs"]["input_files"]["selector_checkpoint"]["sha256"]
    assert "Selector Descriptor Cosine" in md_path.read_text()


def test_selector_descriptor_scoring_rejects_mismatched_descriptor_dimensions(tmp_path):
    checkpoint = tmp_path / "selector.pt"
    _selector_checkpoint(checkpoint)
    selector = load_selector_from_checkpoint(checkpoint)
    query_bank = TokenDescriptorBank(("q.png",), np.ones((1, 4), dtype=np.float32), "l", "mean")
    map_bank = TokenDescriptorBank(("m.png",), np.ones((1, 5), dtype=np.float32), "l", "mean")

    with pytest.raises(ValueError, match="descriptor dimensions must match"):
        score_candidate_bank_by_selector_descriptor_cosine(
            bank=CandidateHypothesisBank.from_candidates(
                protocol_name="bad",
                protocol_kind=ProtocolKind.REFERENCE_POSE,
                candidates=[
                    CandidateHypothesis(
                        query_id="q.png",
                        candidate_id="c0",
                        candidate_type="reference_pose",
                        reference_image="m.png",
                        pose_error=PoseCost(0.1, 1.0),
                    )
                ],
            ),
            query_descriptors=query_bank,
            map_descriptors=map_bank,
            selector=selector,
            method="selector_radio_descriptor_cosine",
            translation_threshold_m=0.25,
            rotation_threshold_deg=5.0,
        )
