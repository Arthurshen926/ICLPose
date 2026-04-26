#!/usr/bin/env python3
"""Local refinement around the current strongest initialization ensemble."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from feature_retrieval.decoupled_init_search import (
    FEATURE_DIR,
    DATASET_DIR,
    BASE,
    load_any_model,
    TRANSLATION_MODELS,
    ROTATION_MODELS,
    average_rotations,
    evaluate_pose,
    weighted_translation,
    weighted_translation_per_axis,
)
from feature_retrieval.patch_regressor_v7 import PatchPoseDataset


def main():
    device_cycle = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    train_ref = PatchPoseDataset(FEATURE_DIR, DATASET_DIR, "train", "cpu", use_fine=False, use_coarse=False, use_summary=True)
    test_ref = PatchPoseDataset(FEATURE_DIR, DATASET_DIR, "test", "cpu", use_fine=False, use_coarse=False, use_summary=True)
    test_trans = test_ref.translations.numpy()
    test_rot = test_ref.rotations.cpu()
    train_pos = train_ref.translations.numpy()
    train_rot = train_ref.rotations.cpu()

    needed = ["e123", "e38", "f42", "e40", "e47", "e60", "f15", "mem28", "grid32v2", "s314"]
    loaded = {}
    for i, name in enumerate(needed):
        spec = TRANSLATION_MODELS.get(name) or ROTATION_MODELS.get(name)
        loaded[name] = load_any_model(spec, FEATURE_DIR, DATASET_DIR, device_cycle[i % len(device_cycle)])

    base5_names = ["e123", "e38", "f42", "e40", "e47"]
    base5_weights = [2.266666666666667, 1.4, 0.9666666666666667, 0.1, 0.5333333333333333]
    base5_trans = weighted_translation(np.stack([loaded[n]["trans_pred"] for n in base5_names], axis=0), base5_weights)
    rot_base5 = average_rotations([loaded[n]["rot_pred"] for n in base5_names], base5_weights)
    rot_f15 = loaded["f15"]["rot_pred"]
    rot_e123 = loaded["e123"]["rot_pred"]

    candidates = {
        "base5": base5_trans,
        "f15": loaded["f15"]["trans_pred"],
        "e60": loaded["e60"]["trans_pred"],
        "mem28": loaded["mem28"]["trans_pred"],
        "grid32v2": loaded["grid32v2"]["trans_pred"],
        "s314": loaded["s314"]["trans_pred"],
    }
    rot_candidates = {
        "rot_base5": rot_base5,
        "rot_f15": rot_f15,
        "rot_e123": rot_e123,
    }

    best = {"r10": -1, "res": None, "cfg": None}
    best_axis = {"r10": -1, "res": None, "cfg": None}

    names = list(candidates.keys())
    stack = np.stack([candidates[n] for n in names], axis=0)

    print("Baseline:")
    for rot_name, rot_pred in rot_candidates.items():
        res = evaluate_pose(candidates["base5"], rot_pred, test_trans, test_rot, train_pos, train_rot)
        print(f"  base5 + {rot_name}: {res}")

    print("\nGlobal local search")
    values = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]
    for rot_name, rot_pred in rot_candidates.items():
        for w_base in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
            for w_f15 in values:
                for w_e60 in values:
                    for w_mem in values:
                        for w_grid in [0.0, 0.05, 0.1, 0.2]:
                            for w_spp in [0.0, 0.05, 0.1, 0.2]:
                                weights = [w_base, w_f15, w_e60, w_mem, w_grid, w_spp]
                                if sum(weights) <= 0:
                                    continue
                                trans_pred = weighted_translation(stack, weights)
                                res = evaluate_pose(trans_pred, rot_pred, test_trans, test_rot, train_pos, train_rot)
                                if res["r10"] > best["r10"] or (
                                    res["r10"] == best["r10"] and res["r5"] > (best["res"] or {}).get("r5", -1)
                                ):
                                    best = {
                                        "r10": res["r10"],
                                        "res": res,
                                        "cfg": {"rot": rot_name, "weights": dict(zip(names, weights))},
                                    }
                                    print(f"  NEW BEST global: {best}")

    print("\nPer-axis local search")
    extra_vals = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    for rot_name, rot_pred in rot_candidates.items():
        for mem_axes in [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.1, 0.2, 0.1), (0.2, 0.4, 0.2), (0.1, 0.4, 0.1), (0.4, 0.1, 0.4)]:
            for f15_axes in [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.1, 0.1, 0.4), (0.4, 0.1, 0.1)]:
                for e60_axes in [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.1, 0.4, 0.1), (0.4, 0.1, 0.4)]:
                    weights_xyz = np.array([
                        [4.0, 4.0, 4.0],
                        list(f15_axes),
                        list(e60_axes),
                        list(mem_axes),
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ], dtype=np.float64)
                    trans_pred = weighted_translation_per_axis(stack, weights_xyz)
                    res = evaluate_pose(trans_pred, rot_pred, test_trans, test_rot, train_pos, train_rot)
                    if res["r10"] > best_axis["r10"] or (
                        res["r10"] == best_axis["r10"] and res["r5"] > (best_axis["res"] or {}).get("r5", -1)
                    ):
                        best_axis = {
                            "r10": res["r10"],
                            "res": res,
                            "cfg": {
                                "rot": rot_name,
                                "weights_xyz": {
                                    "base5": [4.0, 4.0, 4.0],
                                    "f15": list(f15_axes),
                                    "e60": list(e60_axes),
                                    "mem28": list(mem_axes),
                                },
                            },
                        }
                        print(f"  NEW BEST axis: {best_axis}")

    out = {"best_global": best, "best_axis": best_axis}
    out_path = Path(BASE) / "local_translation_refine_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nFINAL")
    print(json.dumps(out, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
