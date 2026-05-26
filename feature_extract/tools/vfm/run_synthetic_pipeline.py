"""Run a disk-level synthetic VFM-MapLoc pipeline smoke experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.map_lifting import (
    TrackObservation,
    aggregate_selected_tracks,
    save_selected_track_bank_npz,
)
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.reporting import build_gate_table
from feature_extract.vfm.synthetic import run_synthetic_feature_utility_validation
from feature_extract.vfm.tokens import (
    TokenBankManifest,
    TokenBankRecord,
    TokenLayerSpec,
    compute_file_sha256,
    write_npz_token_record,
)


def _write_token_bank(output_dir: Path, query_count: int, seed: int) -> TokenBankManifest:
    rng = np.random.default_rng(seed)
    token_dir = output_dir / "tokens"
    layer = TokenLayerSpec(
        name="synthetic_vfm",
        model="synthetic-foundation-feature",
        layer="final",
        channels=4,
        stride=16,
    )
    records = []
    for idx in range(query_count):
        path = token_dir / f"q{idx}.npz"
        write_npz_token_record(
            path,
            {"synthetic_vfm": rng.normal(size=(4, 2, 2)).astype(np.float32)},
        )
        records.append(
            TokenBankRecord(
                image_id=f"q{idx}",
                token_path=path,
                layers=(layer,),
                split="synthetic",
                scene="SyntheticVFM",
                checksum=compute_file_sha256(path),
            )
        )
    manifest = TokenBankManifest(records=tuple(records))
    manifest.validate()
    manifest.to_json(output_dir / "token_manifest.json")
    return manifest


def _write_candidate_bank(
    output_dir: Path,
    query_count: int,
    candidates_per_query: int,
) -> None:
    candidates = []
    for query_idx in range(query_count):
        for cand_idx in range(candidates_per_query):
            cost = 0.05 if cand_idx == 0 else 0.25 + 0.05 * cand_idx
            candidates.append(
                CandidateHypothesis(
                    candidate_id=f"q{query_idx}_c{cand_idx}",
                    candidate_type="reference_pose",
                    pose_error=PoseCost(translation_m=cost, rotation_deg=2.0 + cand_idx),
                    prior_score=1.0 / (cand_idx + 1),
                    reference_image=f"ref_{cand_idx}",
                )
            )
    CandidateHypothesisBank.from_candidates(
        protocol_name="synthetic_pipeline",
        protocol_kind=ProtocolKind.REAL_RETRIEVAL,
        candidates=candidates,
    ).to_jsonl(output_dir / "candidate_bank.jsonl")


def _write_selected_track_bank(output_dir: Path) -> None:
    observations = []
    for track_id in range(4):
        base = np.zeros(4, dtype=np.float32)
        base[track_id] = 1.0
        observations.append(TrackObservation(track_id, "map0", base, True, True, utility=0.9))
        observations.append(TrackObservation(track_id, "map1", base * 0.95, True, True, utility=0.8))
    bank = aggregate_selected_tracks(observations, min_observations=2)
    save_selected_track_bank_npz(bank, output_dir / "selected_tracks.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic VFM disk pipeline")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_count", type=int, default=32)
    parser.add_argument("--candidates_per_query", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_token_bank(output_dir, query_count=args.query_count, seed=args.seed)
    _write_candidate_bank(output_dir, args.query_count, args.candidates_per_query)
    _write_selected_track_bank(output_dir)

    reports = run_synthetic_feature_utility_validation(
        query_count=args.query_count,
        candidates_per_query=args.candidates_per_query,
        seed=args.seed,
    )
    (output_dir / "score_reports.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n"
    )
    table_rows = [
        {
            "method": name,
            "top1": round(float(report["mean_top1_acc"]), 4),
            "pred_m": round(float(report["mean_pred_cost_m"]), 4),
            "spearman": round(float(report["mean_spearman"]), 4),
        }
        for name, report in reports.items()
    ]
    (output_dir / "feature_utility.md").write_text(
        build_gate_table("Synthetic Feature Utility", table_rows) + "\n"
    )


if __name__ == "__main__":
    main()
