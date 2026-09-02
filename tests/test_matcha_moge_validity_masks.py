import sys
from pathlib import Path

import pytest
import torch


MATCHA = Path("/tmp/matcha-gaussians-official")
sys.path.insert(0, str(MATCHA))
sys.path.insert(0, str(MATCHA / "Depth-Anything-V2"))

from matcha.dm_modules.matcher_3d import Matcher3D
from matcha.dm_scene.parallel_aligner import ParallelAligner


def test_depth_loss_normalizes_only_over_valid_pixels():
    aligner = ParallelAligner.__new__(ParallelAligner)
    torch.nn.Module.__init__(aligner)
    aligner.using_pts_as_reference = False
    aligner.use_learnable_confidence = False
    reference = torch.zeros((1, 1, 2))
    prediction = torch.tensor([[[2.0, 100.0]]])
    mask = torch.tensor([[[True, False]]])
    assert aligner.loss(reference, prediction, masks=mask).item() == pytest.approx(2.0)


def test_matcher_validity_changes_its_normalization_domain():
    matcher = Matcher3D.__new__(Matcher3D)
    matcher.cameras = None
    matcher._target_camera_cache = {}
    points = torch.zeros((2, 2, 3, 3))
    depths = torch.ones((2, 2, 3))
    valid = torch.tensor([
        [[True, False, False], [True, True, False]],
        [[False, False, True], [False, True, False]],
    ])
    matcher.update_references(points, depths, valid)
    assert matcher.matching_normalization_element_count == 2 * int(valid.sum())
    with pytest.raises(TypeError, match="boolean"):
        matcher.update_references(points, depths, valid.float())
