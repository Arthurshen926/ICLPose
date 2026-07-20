from __future__ import annotations

from feature_extract.tools.vfm.fit_oof_frozen_fulltrack_per_view_candidate_probe import (
    _sequence_group,
    _stable_sequence_folds,
    parse_args,
)


def test_sequence_grouped_oof_folds_are_stable_and_keep_sequences_together() -> None:
    query_ids = (
        "seq1/frame00001.png",
        "seq1/frame00002.png",
        "seq2/frame00001.png",
        "seq3/frame00001.png",
        "seq4/frame00001.png",
        "seq5/frame00001.png",
    )
    first = _stable_sequence_folds(query_ids, folds=3, seed=101)
    second = _stable_sequence_folds(tuple(reversed(query_ids)), folds=3, seed=101)
    assert first == second
    assert _sequence_group(query_ids[0]) == "seq1"
    assert first["seq1"] in {0, 1, 2}


def test_oof_cli_requires_explicit_bounded_residual_contract() -> None:
    args = parse_args(
        [
            "--appearance-artifacts",
            "appearance.npz",
            "--colmap-model-dir",
            "model",
            "--expected-identity-colmap-images-sha256",
            "hash",
            "--output-dir",
            "out",
            "--family",
            "fixedprior_fulltrack_perview_absolute_phase_intermediate_pca256_mixture",
            "--per-view-residual-architecture",
            "linear",
            "--per-view-residual-cap",
            "2.0",
            "--per-view-residual-cap-provenance",
            "sequence-grouped OOF selection",
        ]
    )
    assert args.folds == 3
    assert args.per_view_residual_architecture == "linear"
