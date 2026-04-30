#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_field.utils.project_config import load_mainline_config
from pose_refine.train_impl import ConcatLocTrainer


def parse_scales(values: Iterable[str]) -> list[float]:
    """Parse pose update scales while preserving CLI order."""
    scales = [float(value) for value in values]
    if not scales:
        raise ValueError("At least one scale is required")
    return scales


def load_dcff_runtime_state(trainer: ConcatLocTrainer, checkpoint_path: str) -> list[str]:
    """Restore DCFF runtime feature modules from a separate checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=trainer.device)
    loaded: list[str] = []
    if "fine_decoder_state" in ckpt and hasattr(trainer, "dcff_renderer"):
        trainer.dcff_renderer.fine_decoder.load_state_dict(ckpt["fine_decoder_state"])
        loaded.append("fine_decoder_state")
    if (
        "coarse_fusion_state" in ckpt
        and hasattr(trainer, "dcff_renderer")
        and trainer.dcff_renderer.coarse_carrier_fusion is not None
    ):
        trainer.dcff_renderer.coarse_carrier_fusion.load_state_dict(
            ckpt["coarse_fusion_state"]
        )
        loaded.append("coarse_fusion_state")
    if "feat_sharp_state" in ckpt and hasattr(trainer, "feat_sharp_fine"):
        trainer.feat_sharp_fine.load_state_dict(ckpt["feat_sharp_state"])
        loaded.append("feat_sharp_state")
    if "fsm_state" in ckpt and getattr(trainer, "feat_select", None) is not None:
        trainer.feat_select.load_state_dict(ckpt["fsm_state"])
        loaded.append("fsm_state")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ConcatPoseNet pose_update_scale")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scales", nargs="+", default=["0", "0.25", "0.5", "0.75", "1.0"])
    parser.add_argument("--dcff_state_checkpoint", default=None)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--outer_iters_val", type=int, default=None)
    args = parser.parse_args()

    config = load_mainline_config(args.config)
    config["exp_name"] = args.exp_name or f"{Path(args.config).stem}_scale_eval"
    training_cfg = config.setdefault("training", {})
    if args.max_val_batches is not None:
        training_cfg["max_val_batches"] = int(args.max_val_batches)
    if args.outer_iters_val is not None:
        training_cfg["outer_iters_val"] = int(args.outer_iters_val)

    trainer = ConcatLocTrainer(config, gpu=args.gpu, resume_path=args.checkpoint)
    if args.dcff_state_checkpoint:
        loaded = load_dcff_runtime_state(trainer, args.dcff_state_checkpoint)
        print(f"loaded_dcff_state={','.join(loaded) if loaded else 'none'}", flush=True)

    for scale in parse_scales(args.scales):
        trainer.model.pose_update_scale = float(scale)
        metrics = trainer.validate(epoch=0)
        print(
            "scale={:.4g} init={:.4g}deg/{:.3f}mm one={:.4g}deg/{:.3f}mm "
            "final={:.4g}deg/{:.3f}mm flow_epe={:.4g} corr_epe={:.4g} fm={:.4g}deg/{:.3f}mm".format(
                scale,
                metrics.get("val_init_rot_median", float("nan")),
                metrics.get("val_init_trans_median", float("nan")),
                metrics.get("val_one_rot_median", float("nan")),
                metrics.get("val_one_trans_median", float("nan")),
                metrics.get("val_rot_median", float("nan")),
                metrics.get("val_trans_median", float("nan")),
                metrics.get("val_flow_epe", float("nan")),
                metrics.get("val_corr_flow_epe", float("nan")),
                metrics.get("val_fm_rot_median", float("nan")),
                metrics.get("val_fm_trans_median", float("nan")),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
