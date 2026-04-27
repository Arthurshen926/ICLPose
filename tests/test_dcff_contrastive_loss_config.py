#!/usr/bin/env python3
"""Static checks for configurable DCFF contrastive supervision."""

from __future__ import annotations

import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LOSSES = REPO_ROOT / "feature_field" / "dcff" / "losses.py"
TRAIN_IMPL = REPO_ROOT / "feature_field" / "train_impl.py"


class DCFFContrastiveLossConfigTest(unittest.TestCase):
    def test_dcff_loss_constructor_exposes_nce_weights(self) -> None:
        tree = ast.parse(LOSSES.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DCFFLoss")
        init_fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        arg_names = [arg.arg for arg in init_fn.args.args]
        self.assertIn("lambda_fine_nce", arg_names)
        self.assertIn("lambda_coarse_nce", arg_names)

    def test_train_impl_passes_nce_config_to_loss(self) -> None:
        source = TRAIN_IMPL.read_text(encoding="utf-8")
        self.assertIn("lambda_fine_nce", source)
        self.assertIn("nce_temperature", source)
        self.assertIn("nce_samples", source)


if __name__ == "__main__":
    unittest.main()
