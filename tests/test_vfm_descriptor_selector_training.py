import json

import numpy as np
import pytest

from feature_extract.vfm.descriptor_selector_training import (
    DescriptorSelectorTrainingConfig,
    train_descriptor_selector,
)
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank
from feature_extract.tools.vfm.train_descriptor_selector import main as train_cli_main


def _synthetic_banks(query_count=12, candidates_per_query=3, channels=8):
    rng = np.random.default_rng(7)
    query_ids = tuple(f"q{i:02d}.png" for i in range(query_count))
    query_descriptors = []
    map_ids = []
    map_descriptors = []
    candidates = []

    for query_idx, query_id in enumerate(query_ids):
        signal = rng.normal(size=(2,)).astype(np.float32)
        signal = signal / np.linalg.norm(signal)
        query_descriptor = rng.normal(scale=0.08, size=(channels,)).astype(np.float32)
        query_descriptor[:2] = signal
        query_descriptor[2:] = rng.normal(loc=1.0, scale=0.05, size=(channels - 2,))
        query_descriptors.append(query_descriptor)

        for candidate_idx in range(candidates_per_query):
            map_id = f"m{query_idx:02d}_{candidate_idx}.png"
            descriptor = rng.normal(scale=0.45, size=(channels,)).astype(np.float32)
            if candidate_idx == 0:
                descriptor[:2] = signal + rng.normal(scale=0.01, size=(2,))
                descriptor[2:] = rng.normal(loc=-1.0, scale=0.05, size=(channels - 2,))
                pose_error = PoseCost(0.03, 1.0)
            else:
                descriptor[:2] = -signal + rng.normal(scale=0.01, size=(2,))
                descriptor[2:] = query_descriptor[2:] + rng.normal(scale=0.01, size=(channels - 2,))
                pose_error = PoseCost(1.0 + candidate_idx, 20.0)
            map_ids.append(map_id)
            map_descriptors.append(descriptor)
            candidates.append(
                CandidateHypothesis(
                    query_id=query_id,
                    candidate_id=f"{query_id}:{candidate_idx}",
                    candidate_type="reference_pose",
                    reference_image=map_id,
                    pose_error=pose_error,
                )
            )

    return (
        CandidateHypothesisBank.from_candidates(
            protocol_name="synthetic_descriptor_selector",
            protocol_kind=ProtocolKind.REFERENCE_POSE,
            candidates=candidates,
        ),
        TokenDescriptorBank(
            image_ids=query_ids,
            descriptors=np.asarray(query_descriptors, dtype=np.float32),
            layer_name="radio_final",
            pooling="mean",
            normalized=False,
        ),
        TokenDescriptorBank(
            image_ids=tuple(map_ids),
            descriptors=np.asarray(map_descriptors, dtype=np.float32),
            layer_name="radio_final",
            pooling="mean",
            normalized=False,
        ),
    )


def test_descriptor_selector_training_learns_fixed_candidate_ranking():
    bank, query_bank, map_bank = _synthetic_banks()

    summary = train_descriptor_selector(
        bank,
        query_bank,
        map_bank,
        DescriptorSelectorTrainingConfig(
            steps=90,
            batch_size=4,
            output_dim=4,
            group_size=2,
            lr=0.03,
            eval_split_fraction=0.25,
            seed=3,
            device="cpu",
        ),
    )

    assert summary.query_count == 12
    assert summary.train_query_count == 9
    assert summary.eval_query_count == 3
    assert summary.initial_loss > summary.final_loss
    assert summary.raw_train_top1_acc < summary.train_top1_acc
    assert summary.raw_eval_top1_acc <= summary.eval_top1_acc
    assert summary.train_top1_acc >= 0.8
    assert summary.eval_top1_acc >= 0.8


def test_descriptor_selector_training_cli_writes_json_and_checkpoint(tmp_path):
    bank, query_bank, map_bank = _synthetic_banks()
    bank_path = tmp_path / "candidates.jsonl"
    query_path = tmp_path / "query.npz"
    map_path = tmp_path / "map.npz"
    output_path = tmp_path / "result.json"
    checkpoint_path = tmp_path / "selector.pt"
    bank.to_jsonl(bank_path)
    query_bank.to_npz(query_path)
    map_bank.to_npz(map_path)

    train_cli_main(
        [
            "--bank",
            str(bank_path),
            "--query_descriptors",
            str(query_path),
            "--map_descriptors",
            str(map_path),
            "--steps",
            "80",
            "--batch_size",
            "4",
            "--output_dim",
            "4",
            "--group_size",
            "2",
            "--lr",
            "0.03",
            "--eval_fraction",
            "0.25",
            "--seed",
            "3",
            "--device",
            "cpu",
            "--output",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )

    payload = json.loads(output_path.read_text())
    assert payload["config"]["output_dim"] == 4
    assert payload["config"]["group_size"] == 2
    assert payload["inputs"]["protocol_name"] == "synthetic_descriptor_selector"
    assert payload["inputs"]["protocol_kind"] == "reference_pose"
    assert payload["inputs"]["candidate_count"] == 36
    assert payload["inputs"]["input_files"]["candidate_bank"]["sha256"]
    assert payload["inputs"]["input_files"]["query_descriptors"]["sha256"]
    assert payload["result"]["query_count"] == 12
    assert "raw_eval_top1_acc" in payload["result"]
    assert payload["result"]["eval_top1_acc"] >= 0.8
    assert checkpoint_path.exists()


def test_descriptor_selector_training_rejects_queries_without_two_costed_candidates():
    candidate = CandidateHypothesis(
        query_id="q.png",
        candidate_id="only",
        candidate_type="reference_pose",
        reference_image="m.png",
        pose_error=PoseCost(0.1, 1.0),
    )
    bank = CandidateHypothesisBank.from_candidates(
        protocol_name="bad",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[candidate],
    )
    query_bank = TokenDescriptorBank(("q.png",), np.ones((1, 4), dtype=np.float32), "l", "mean")
    map_bank = TokenDescriptorBank(("m.png",), np.ones((1, 4), dtype=np.float32), "l", "mean")

    with pytest.raises(ValueError, match="at least 2 candidates"):
        train_descriptor_selector(
            bank,
            query_bank,
            map_bank,
            DescriptorSelectorTrainingConfig(steps=1, output_dim=2, group_size=2),
        )
