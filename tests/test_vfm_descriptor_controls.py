import json

import numpy as np
import pytest

from feature_extract.tools.vfm.score_candidate_bank_descriptor_controls import (
    main as score_descriptor_controls_cli_main,
)
from feature_extract.vfm.descriptor_controls import apply_descriptor_control, mask_descriptor_channels_by_utility
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_table import evaluate_score_table
from feature_extract.vfm.token_descriptor_bank import (
    TokenDescriptorBank,
    score_candidate_bank_by_descriptor_cosine,
)


def _control_candidate_bank():
    return CandidateHypothesisBank.from_candidates(
        protocol_name="descriptor_control_smoke",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="q0.png",
                candidate_id="q0_bad",
                candidate_type="reference_pose",
                reference_image="m1.png",
                pose_error=PoseCost(1.0, 20.0),
            ),
            CandidateHypothesis(
                query_id="q0.png",
                candidate_id="q0_good",
                candidate_type="reference_pose",
                reference_image="m0.png",
                pose_error=PoseCost(0.1, 1.0),
            ),
            CandidateHypothesis(
                query_id="q1.png",
                candidate_id="q1_bad",
                candidate_type="reference_pose",
                reference_image="m0.png",
                pose_error=PoseCost(1.0, 20.0),
            ),
            CandidateHypothesis(
                query_id="q1.png",
                candidate_id="q1_good",
                candidate_type="reference_pose",
                reference_image="m1.png",
                pose_error=PoseCost(0.1, 1.0),
            ),
        ],
    )


def _control_descriptor_banks():
    query = TokenDescriptorBank(
        image_ids=("q0.png", "q1.png"),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        layer_name="selected64",
        pooling="mean",
        normalized=True,
    )
    maps = TokenDescriptorBank(
        image_ids=("m0.png", "m1.png"),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        layer_name="selected64",
        pooling="mean",
        normalized=True,
    )
    return query, maps


def test_descriptor_control_query_shuffle_breaks_query_map_evidence():
    query, maps = _control_descriptor_banks()
    shuffled_query = apply_descriptor_control(query, control="query_shuffle", seed=0)

    base_report = evaluate_score_table(
        score_candidate_bank_by_descriptor_cosine(
            _control_candidate_bank(),
            query,
            maps,
            method="selected",
            translation_threshold_m=0.25,
            rotation_threshold_deg=5.0,
        )
    )
    control_report = evaluate_score_table(
        score_candidate_bank_by_descriptor_cosine(
            _control_candidate_bank(),
            shuffled_query,
            maps,
            method="selected_query_shuffle",
            translation_threshold_m=0.25,
            rotation_threshold_deg=5.0,
        )
    )

    assert shuffled_query.image_ids == query.image_ids
    assert base_report.mean_top1_acc == pytest.approx(1.0)
    assert control_report.mean_top1_acc == pytest.approx(0.0)


def test_descriptor_control_wrong_scene_relabels_replacement_descriptors():
    query, _maps = _control_descriptor_banks()
    replacement = TokenDescriptorBank(
        image_ids=("other0.png", "other1.png"),
        descriptors=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        layer_name="selected64",
        pooling="mean",
        normalized=True,
    )

    wrong_scene = apply_descriptor_control(query, control="wrong_scene", seed=0, replacement_bank=replacement)

    assert wrong_scene.image_ids == query.image_ids
    np.testing.assert_array_equal(wrong_scene.descriptors, replacement.descriptors)
    assert wrong_scene.metadata["control"] == "wrong_scene"


def test_descriptor_channel_mask_zeros_high_or_low_utility_columns():
    query, _maps = _control_descriptor_banks()
    bank = TokenDescriptorBank(
        image_ids=query.image_ids,
        descriptors=np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        ),
        layer_name="selected64",
        pooling="mean",
        normalized=False,
    )
    utility = np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float32)

    high_removed = mask_descriptor_channels_by_utility(
        bank,
        utility=utility,
        fraction=0.5,
        remove="high",
        renormalize=False,
    )
    low_removed = mask_descriptor_channels_by_utility(
        bank,
        utility=utility,
        fraction=0.5,
        remove="low",
        renormalize=False,
    )

    np.testing.assert_array_equal(
        high_removed.descriptors,
        np.asarray([[0.0, 2.0, 0.0, 4.0], [0.0, 6.0, 0.0, 8.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        low_removed.descriptors,
        np.asarray([[1.0, 0.0, 3.0, 0.0], [5.0, 0.0, 7.0, 0.0]], dtype=np.float32),
    )
    assert high_removed.metadata["control"] == "high_utility_channels_removed"
    assert low_removed.metadata["control"] == "low_utility_channels_removed"


def test_descriptor_controls_cli_writes_rows_report_and_inputs(tmp_path):
    query, maps = _control_descriptor_banks()
    bank_path = tmp_path / "bank.jsonl"
    query_path = tmp_path / "query.npz"
    map_path = tmp_path / "map.npz"
    rows_path = tmp_path / "rows.json"
    report_path = tmp_path / "report.json"
    _control_candidate_bank().to_jsonl(bank_path)
    query.to_npz(query_path)
    maps.to_npz(map_path)

    score_descriptor_controls_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_descriptors",
            str(query_path),
            "--map_descriptors",
            str(map_path),
            "--control",
            "query_shuffle",
            "--seed",
            "0",
            "--method",
            "selected_query_shuffle",
            "--translation_threshold_m",
            "0.25",
            "--rotation_threshold_deg",
            "5.0",
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
        ]
    )

    rows = json.loads(rows_path.read_text())
    report = json.loads(report_path.read_text())
    assert rows[0]["method"] == "selected_query_shuffle"
    assert report["mean_top1_acc"] == pytest.approx(0.0)
    assert report["inputs"]["control"] == "query_shuffle"
    assert report["inputs"]["input_files"]["query_descriptors"]["sha256"]


def test_descriptor_controls_cli_masks_utility_channels(tmp_path):
    query, maps = _control_descriptor_banks()
    bank_path = tmp_path / "bank.jsonl"
    query_path = tmp_path / "query.npz"
    map_path = tmp_path / "map.npz"
    rows_path = tmp_path / "rows.json"
    report_path = tmp_path / "report.json"
    _control_candidate_bank().to_jsonl(bank_path)
    query.to_npz(query_path)
    maps.to_npz(map_path)

    score_descriptor_controls_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_descriptors",
            str(query_path),
            "--map_descriptors",
            str(map_path),
            "--control",
            "high_utility_channels",
            "--channel_utility",
            "1.0,0.0",
            "--mask_fraction",
            "0.5",
            "--mask_target",
            "both",
            "--seed",
            "0",
            "--method",
            "selected_high_utility_removed",
            "--translation_threshold_m",
            "0.25",
            "--rotation_threshold_deg",
            "5.0",
            "--output_rows",
            str(rows_path),
            "--output_report",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text())
    assert report["inputs"]["control"] == "high_utility_channels"
    assert report["inputs"]["mask_fraction"] == pytest.approx(0.5)
    assert report["inputs"]["mask_target"] == "both"
