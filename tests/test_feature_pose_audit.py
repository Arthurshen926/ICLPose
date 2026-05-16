import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.students.pose_energy_net import PairConditionedLocalMatcher
from feature_extract.tools.eval_feature_pose_audit import score_combo


def test_score_combo_supports_pair_matcher_local_mode():
    query = torch.zeros(1, 2, 5, 5)
    query[:, :, 2, 2] = 1.0
    candidate = query[:, None].repeat(1, 2, 1, 1, 1)
    candidate[:, 1] = torch.roll(candidate[:, 1], shifts=1, dims=-1)
    candidate_pose = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    valid = torch.ones(1, 2, dtype=torch.bool)
    matcher = PairConditionedLocalMatcher(
        channels=2,
        hidden_dim=8,
        offset_radius=1,
        zero_init_residual=True,
        base_dot_weight=1.0,
    )
    args = argparse.Namespace(
        score_mode="pair_matcher_local",
        score_radius=1,
        score_preprocess="spatial_center",
        score_highpass_kernel=5,
        score_map_mode="peak_offset",
        pair_matcher_radius=1,
        pair_matcher_score_stride=1,
        pair_matcher_temperature=0.05,
        pair_matcher_score_chunk_points=1024,
        pair_matcher_score_candidate_chunk_size=0,
        pair_matcher_candidate_score_mode="center_logprob_margin",
    )

    scores, combo_valid = score_combo(query, candidate, candidate_pose, valid, args, pair_matcher=matcher)

    assert scores.shape == (1, 2)
    assert combo_valid.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]
