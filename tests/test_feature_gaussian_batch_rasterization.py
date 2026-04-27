#!/usr/bin/env python3
"""Static regression checks for FeatureGaussian batched 2DGS training.

These checks intentionally avoid importing CUDA/gsplat modules. They validate
that the production trainer exposes and wires a true multi-camera rasterization
path instead of silently falling back to per-camera gradient accumulation.
"""

from __future__ import annotations

import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
JOINT_RENDERER = REPO_ROOT / "feature_gaussian" / "legacy_3dgs" / "train_2dgs_joint.py"
JOINT_TRAINER = REPO_ROOT / "feature_gaussian" / "legacy_3dgs" / "train_2dgs_joint_v3.py"


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class FeatureGaussianBatchRasterizationTest(unittest.TestCase):
    def test_renderer_exports_true_batched_2dgs_rgb_render(self) -> None:
        tree = _tree(JOINT_RENDERER)
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("render_rgb_2dgs_batch", function_names)

    def test_joint_v3_trainer_uses_batch_size_for_batched_rasterization(self) -> None:
        source = JOINT_TRAINER.read_text(encoding="utf-8")
        self.assertIn("'batch_size'", source)
        self.assertIn("render_rgb_2dgs_batch", source)
        self.assertIn("batch_size = max(1", source)

    def test_geometry_only_config_skips_teacher_feature_cache(self) -> None:
        source = JOINT_TRAINER.read_text(encoding="utf-8")
        self.assertIn("feature_training_enabled", source)
        self.assertIn("Feature training disabled", source)

    def test_joint_v3_can_cache_resized_training_inputs_on_gpu(self) -> None:
        source = JOINT_TRAINER.read_text(encoding="utf-8")
        self.assertIn("cache_resized_images", source)
        self.assertIn("cache_resized_masks", source)
        self.assertIn("image_cache", source)
        self.assertIn("mask_cache", source)

    def test_joint_v3_clamps_geometry_regularizers_to_non_negative_losses(self) -> None:
        source = JOINT_TRAINER.read_text(encoding="utf-8")
        self.assertIn("normal_error = torch.nan_to_num", source)
        self.assertIn("dist_loss = torch.nan_to_num", source)
        self.assertIn("scale_penalty = torch.clamp", source)
        self.assertIn("loss_val = loss.item()", source)

    def test_joint_v3_aborts_after_too_many_bad_steps(self) -> None:
        source = JOINT_TRAINER.read_text(encoding="utf-8")
        self.assertIn("max_consecutive_bad_steps", source)
        self.assertIn("consecutive_bad_steps", source)

    def test_joint_v3_has_torchrun_distributed_sync_hooks(self) -> None:
        source = JOINT_TRAINER.read_text(encoding="utf-8")
        self.assertIn("init_process_group", source)
        self.assertIn("device_id=", source)
        self.assertIn("_all_reduce_optimizer_grads", source)
        self.assertIn("_sync_densification_stats", source)
        self.assertIn("is_main_process", source)


if __name__ == "__main__":
    unittest.main()
