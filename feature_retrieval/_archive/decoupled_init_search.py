#!/usr/bin/env python3
"""Search strongest initialization by decoupling translation and rotation.

We search translation ensembles and rotation ensembles separately across:
  - strong direct patch regressors
  - grid v2 models
  - memory-based translation model
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from feature_retrieval.cross_arch_ensemble import load_and_predict
from feature_retrieval.patch_regressor_v7 import PatchPoseDataset, geodesic_distance, precompute_pooled_features
from feature_retrieval.patch_grid_classifier_v1 import PatchGridClassifier
from feature_retrieval.patch_memory_translation_v1 import MemoryTranslationPoseNet, load_frozen_pooler


FEATURE_DIR = "output/feature_extract/features_radio_dual_128/OldHospital_pilot"
DATASET_DIR = "dataset/OldHospital"
BASE = "output/feature_retrieval/pose_regression"


TRANSLATION_MODELS = {
    "e123": {"type": "patch", "path": f"{BASE}/exp24e_attn_h4_wider_seed123"},
    "e38": {"type": "patch", "path": f"{BASE}/exp24e_attn_h4_wider_seed38"},
    "f42": {"type": "patch", "path": f"{BASE}/exp24f_attn_h8_wider_seed42"},
    "e40": {"type": "patch", "path": f"{BASE}/exp24e_attn_h4_wider_seed40"},
    "e47": {"type": "patch", "path": f"{BASE}/exp24e_attn_h4_wider_seed47"},
    "e60": {"type": "patch", "path": f"{BASE}/exp24e_attn_h4_wider_seed60"},
    "f15": {"type": "patch", "path": f"{BASE}/exp24f_attn_h8_wider_seed15"},
    "s314": {"type": "patch", "path": f"{BASE}/exp14s_128d_wider_seed314"},
    "grid32v1": {"type": "grid", "path": f"{BASE}/exp26_gridcls_k32_seed123"},
    "grid32v2": {"type": "grid", "path": f"{BASE}/exp26b_gridcls_k32_mix1_seed123"},
    "mem28": {"type": "memory", "path": f"{BASE}/exp28_memory_trans_seed123"},
}


ROTATION_MODELS = {
    "e123": {"type": "patch", "path": f"{BASE}/exp24e_attn_h4_wider_seed123"},
    "e38": {"type": "patch", "path": f"{BASE}/exp24e_attn_h4_wider_seed38"},
    "f42": {"type": "patch", "path": f"{BASE}/exp24f_attn_h8_wider_seed42"},
    "f15": {"type": "patch", "path": f"{BASE}/exp24f_attn_h8_wider_seed15"},
    "e47": {"type": "patch", "path": f"{BASE}/exp24e_attn_h4_wider_seed47"},
    "mem28": {"type": "memory", "path": f"{BASE}/exp28_memory_trans_seed123"},
}


def load_grid_prediction(exp_dir: str, feature_dir: str, dataset_dir: str, device: str):
    exp_dir = Path(exp_dir)
    with open(exp_dir / "config.json", "r") as handle:
        cfg = json.load(handle)

    use_fine = "fine" in cfg["feat"] or "both" in cfg["feat"]
    use_coarse = "coarse" in cfg["feat"] or "both" in cfg["feat"]
    use_summary = "+sum" in cfg["feat"]
    train_data = PatchPoseDataset(feature_dir, dataset_dir, "train", "cpu", use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data = PatchPoseDataset(feature_dir, dataset_dir, "test", "cpu", use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)

    model = PatchGridClassifier(
        n_cells=cfg["n_cells"],
        pool_type=cfg["pool"],
        feat_mode=cfg["feat"],
        patch_dim=cfg["patch_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
        attn_heads=cfg.get("attn_heads", 4),
        conv_out_dim=cfg.get("conv_out_dim", 512),
        spp_levels=cfg.get("spp_levels"),
        decouple_rot=cfg.get("decouple_rot", False),
        rot_hidden_dims=tuple(cfg["rot_hidden_dims"]) if cfg.get("rot_hidden_dims") else None,
    ).to(device)
    ckpt = torch.load(exp_dir / "model_best.pt", map_location=device)
    state = ckpt["model_state_dict"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # Backward compatibility for v1 checkpoints before trans_backbone/rot_backbone split.
        remapped = {}
        for key, value in state.items():
            if key.startswith("backbone."):
                remapped[key.replace("backbone.", "trans_backbone.")] = value
            else:
                remapped[key] = value
        model.load_state_dict(remapped, strict=False)
    model.eval()

    cluster_info = torch.load(exp_dir / "cluster_info.pt", map_location="cpu")
    centroids = cluster_info["centroids"].to(device)
    residual_std = cluster_info["residual_std"].to(device)

    train_data.translations = train_data.translations.to(device)
    train_data.rotations = train_data.rotations.to(device)
    test_data.translations = test_data.translations.to(device)
    test_data.rotations = test_data.rotations.to(device)
    if train_data.patch_fine is not None:
        train_data.patch_fine = train_data.patch_fine.to(device)
    if train_data.patch_coarse is not None:
        train_data.patch_coarse = train_data.patch_coarse.to(device)
    if train_data.summary is not None:
        train_data.summary = train_data.summary.to(device)
    if test_data.patch_fine is not None:
        test_data.patch_fine = test_data.patch_fine.to(device)
    if test_data.patch_coarse is not None:
        test_data.patch_coarse = test_data.patch_coarse.to(device)
    if test_data.summary is not None:
        test_data.summary = test_data.summary.to(device)

    with torch.no_grad():
        logits, residual_all, rot_pred = model(test_data.patch_fine, test_data.patch_coarse, test_data.summary)
        pred_labels = logits.argmax(dim=-1)
        idx = torch.arange(pred_labels.shape[0], device=device)
        pred_residual = residual_all[idx, pred_labels] * residual_std.unsqueeze(0)
        trans_pred = centroids[pred_labels] + pred_residual
    return {"trans_pred": trans_pred.cpu().numpy(), "rot_pred": rot_pred.cpu(), "name": exp_dir.name}


def load_memory_prediction(exp_dir: str, feature_dir: str, dataset_dir: str, device: str):
    exp_dir = Path(exp_dir)
    with open(exp_dir / "config.json", "r") as handle:
        cfg = json.load(handle)

    pooler_model, pooler_cfg = load_frozen_pooler(cfg["pooler_exp_dir"], torch.device(device))
    use_fine = "fine" in pooler_cfg["feat"] or "both" in pooler_cfg["feat"]
    use_coarse = "coarse" in pooler_cfg["feat"] or "both" in pooler_cfg["feat"]
    use_summary = "+sum" in pooler_cfg["feat"]
    train_data = PatchPoseDataset(feature_dir, dataset_dir, "train", "cpu", use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    test_data = PatchPoseDataset(feature_dir, dataset_dir, "test", "cpu", use_fine=use_fine, use_coarse=use_coarse, use_summary=use_summary)
    train_pooled = precompute_pooled_features(pooler_model, train_data, torch.device(device), batch_size=16)
    test_pooled = precompute_pooled_features(pooler_model, test_data, torch.device(device), batch_size=16)

    train_data.translations = train_data.translations.to(device)
    train_data.rotations = train_data.rotations.to(device)
    trans_mean = train_data.translations.mean(dim=0)
    trans_std = train_data.translations.std(dim=0).clamp(min=1e-6)
    train_trans_norm = (train_data.translations - trans_mean) / trans_std

    model = MemoryTranslationPoseNet(
        input_dim=cfg["pooled_dim"],
        trans_hidden_dims=tuple(cfg["trans_hidden_dims"]),
        rot_hidden_dims=tuple(cfg["rot_hidden_dims"]),
        embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"],
        init_temperature=cfg["memory_temperature"],
    ).to(device)
    ckpt = torch.load(exp_dir / "model_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred_norm, rot_pred, _, _ = model(test_pooled, train_pooled, train_trans_norm)
        trans_pred = pred_norm * trans_std.unsqueeze(0) + trans_mean.unsqueeze(0)
    return {"trans_pred": trans_pred.cpu().numpy(), "rot_pred": rot_pred.cpu(), "name": exp_dir.name}


def load_any_model(spec: dict, feature_dir: str, dataset_dir: str, device: str):
    if spec["type"] == "patch":
        return load_and_predict(spec["path"], feature_dir, dataset_dir, device)
    if spec["type"] == "grid":
        return load_grid_prediction(spec["path"], feature_dir, dataset_dir, device)
    if spec["type"] == "memory":
        return load_memory_prediction(spec["path"], feature_dir, dataset_dir, device)
    raise ValueError(spec)


def average_rotations(rot_list, weights=None):
    if weights is None:
        weights = [1.0] * len(rot_list)
    rot_ens = torch.zeros_like(rot_list[0]).float()
    for rot, w in zip(rot_list, weights):
        rot_ens = rot_ens + rot.float() * float(w)
    rot_ens = rot_ens / float(np.sum(weights))
    return rot_ens


def evaluate_pose(trans_pred, rot_pred, test_trans, test_rot, train_pos, train_rot):
    te = np.linalg.norm(trans_pred - test_trans, axis=-1)
    re = (geodesic_distance(rot_pred, test_rot) * 180 / math.pi).numpy()
    tree = cKDTree(train_pos)
    _, nn_idx = tree.query(trans_pred, k=1)
    retr_te = np.linalg.norm(train_pos[nn_idx] - test_trans, axis=-1)
    retr_re = (geodesic_distance(train_rot[nn_idx], test_rot) * 180 / math.pi).numpy()
    return {
        "r5": float(((re < 5) & (te < 1.0)).mean() * 100.0),
        "r10": float(((re < 10) & (te < 2.0)).mean() * 100.0),
        "retr_r5": float(((retr_re < 5) & (retr_te < 1.0)).mean() * 100.0),
        "retr_r10": float(((retr_re < 10) & (retr_te < 2.0)).mean() * 100.0),
        "med_rot": float(np.median(re)),
        "med_trans_mm": float(np.median(te) * 1000.0),
        "retr_med_rot": float(np.median(retr_re)),
        "retr_med_trans_mm": float(np.median(retr_te) * 1000.0),
    }


def weighted_translation(stacked_trans, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return np.einsum("i,ijk->jk", w, stacked_trans)


def weighted_translation_per_axis(stacked_trans, weights_xyz):
    w = np.asarray(weights_xyz, dtype=np.float64)
    w = w / np.clip(w.sum(axis=0, keepdims=True), 1e-9, None)
    out = np.zeros_like(stacked_trans[0])
    for axis in range(3):
        out[:, axis] = np.einsum("i,ij->j", w[:, axis], stacked_trans[:, :, axis])
    return out


def main():
    device_cycle = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5"]

    print("Loading GT...")
    train_ref = PatchPoseDataset(FEATURE_DIR, DATASET_DIR, "train", "cpu", use_fine=False, use_coarse=False, use_summary=True)
    test_ref = PatchPoseDataset(FEATURE_DIR, DATASET_DIR, "test", "cpu", use_fine=False, use_coarse=False, use_summary=True)
    test_trans = test_ref.translations.numpy()
    test_rot = test_ref.rotations.cpu()
    train_pos = train_ref.translations.numpy()
    train_rot = train_ref.rotations.cpu()

    print("\nLoading model predictions...")
    loaded = {}
    all_specs = {}
    all_specs.update(TRANSLATION_MODELS)
    all_specs.update(ROTATION_MODELS)
    for i, (name, spec) in enumerate(all_specs.items()):
        if name in loaded:
            continue
        print(f"  loading {name} ({spec['type']})")
        loaded[name] = load_any_model(spec, FEATURE_DIR, DATASET_DIR, device_cycle[i % len(device_cycle)])

    print("\nIndividual models:")
    indiv = {}
    for name, pred in loaded.items():
        indiv[name] = evaluate_pose(pred["trans_pred"], pred["rot_pred"], test_trans, test_rot, train_pos, train_rot)
        print(f"  {name:10s} direct R@10={indiv[name]['r10']:5.1f}%  R@5={indiv[name]['r5']:5.1f}%  retr R@10={indiv[name]['retr_r10']:5.1f}%")

    # Rotation candidates
    rot_candidates = {}
    for name in ROTATION_MODELS:
        rot_candidates[name] = loaded[name]["rot_pred"]
    rot_candidates["rot_best5"] = average_rotations(
        [loaded[n]["rot_pred"] for n in ["e123", "e38", "f42", "e40", "e47"]],
        [2.266666666666667, 1.4, 0.9666666666666667, 0.1, 0.5333333333333333],
    )
    rot_candidates["rot_old3"] = average_rotations(
        [loaded[n]["rot_pred"] for n in ["e123", "e38", "f42"]],
        [2.0, 1.0, 1.0],
    )

    # Translation candidates
    trans_names = list(TRANSLATION_MODELS.keys())
    stacked_trans = np.stack([loaded[n]["trans_pred"] for n in trans_names], axis=0)

    # Baseline best known direct ensemble
    print("\nBaseline decoupled evaluations:")
    sel_idx = [trans_names.index(n) for n in ["e123", "e38", "f42", "e40", "e47"]]
    baseline_trans = weighted_translation(stacked_trans[sel_idx], [2.266666666666667, 1.4, 0.9666666666666667, 0.1, 0.5333333333333333])
    for rot_name, rot_pred in rot_candidates.items():
        res = evaluate_pose(baseline_trans, rot_pred, test_trans, test_rot, train_pos, train_rot)
        print(f"  baseline5 + {rot_name:9s} -> direct R@10={res['r10']:5.1f}% R@5={res['r5']:5.1f}% retr R@10={res['retr_r10']:5.1f}%")

    best = {
        "r10": -1,
        "name": None,
        "weights": None,
        "rot_name": None,
        "result": None,
    }
    best_axis = {
        "r10": -1,
        "name": None,
        "weights_xyz": None,
        "rot_name": None,
        "result": None,
    }

    rng = np.random.default_rng(123)
    print("\nRandom search: global translation weights")
    for rot_name, rot_pred in rot_candidates.items():
        rot_err = (geodesic_distance(rot_pred, test_rot) * 180 / math.pi).numpy()
        local_best = -1
        for trial in range(30000):
            mask = rng.random(len(trans_names)) < 0.45
            if not mask.any():
                mask[rng.integers(0, len(trans_names))] = True
            # always keep strongest base model as anchor to stabilize search
            mask[trans_names.index("e123")] = True
            alpha = np.where(mask, rng.uniform(0.2, 2.5, size=len(trans_names)), 0.0)
            weights = alpha / alpha.sum()
            trans_pred = weighted_translation(stacked_trans[mask], alpha[mask])
            te = np.linalg.norm(trans_pred - test_trans, axis=-1)
            r10 = float(((rot_err < 10.0) & (te < 2.0)).mean() * 100.0)
            if r10 >= local_best:
                res = evaluate_pose(trans_pred, rot_pred, test_trans, test_rot, train_pos, train_rot)
                if res["r10"] >= local_best:
                    local_best = res["r10"]
                    if res["r10"] > best["r10"] or (res["r10"] == best["r10"] and res["r5"] > (best["result"] or {}).get("r5", -1)):
                        best.update({
                            "r10": res["r10"],
                            "name": [trans_names[i] for i in range(len(trans_names)) if mask[i]],
                            "weights": weights.tolist(),
                            "rot_name": rot_name,
                            "result": res,
                        })
                        print(f"  NEW BEST global: rot={rot_name} models={best['name']} R@10={res['r10']:.1f}% R@5={res['r5']:.1f}% retr={res['retr_r10']:.1f}%")

    print("\nRandom search: per-axis translation weights")
    for rot_name, rot_pred in rot_candidates.items():
        rot_err = (geodesic_distance(rot_pred, test_rot) * 180 / math.pi).numpy()
        local_best = -1
        for trial in range(12000):
            mask = rng.random((len(trans_names), 3)) < 0.4
            mask[trans_names.index("e123"), :] = True
            alpha = np.where(mask, rng.uniform(0.2, 2.5, size=(len(trans_names), 3)), 0.0)
            trans_pred = weighted_translation_per_axis(stacked_trans, alpha)
            te = np.linalg.norm(trans_pred - test_trans, axis=-1)
            r10 = float(((rot_err < 10.0) & (te < 2.0)).mean() * 100.0)
            if r10 >= local_best:
                res = evaluate_pose(trans_pred, rot_pred, test_trans, test_rot, train_pos, train_rot)
                if res["r10"] >= local_best:
                    local_best = res["r10"]
                    if res["r10"] > best_axis["r10"] or (res["r10"] == best_axis["r10"] and res["r5"] > (best_axis["result"] or {}).get("r5", -1)):
                        best_axis.update({
                            "r10": res["r10"],
                            "name": trans_names,
                            "weights_xyz": alpha.tolist(),
                            "rot_name": rot_name,
                            "result": res,
                        })
                        print(f"  NEW BEST axis: rot={rot_name} R@10={res['r10']:.1f}% R@5={res['r5']:.1f}% retr={res['retr_r10']:.1f}%")

    print("\n" + "=" * 70)
    print("BEST RESULTS")
    print("=" * 70)
    print(f"Best global translation ensemble: rot={best['rot_name']} models={best['name']}")
    print(f"  Result: {best['result']}")
    if best["weights"] is not None:
        weights_named = {n: float(best['weights'][i]) for i, n in enumerate(trans_names) if best['weights'][i] > 1e-6}
        print(f"  Weights: {weights_named}")
    print(f"Best per-axis translation ensemble: rot={best_axis['rot_name']}")
    print(f"  Result: {best_axis['result']}")

    out = {
        "best_global": best,
        "best_per_axis": best_axis,
        "individual": indiv,
    }
    out_path = Path(BASE) / "decoupled_init_search_results.json"
    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
