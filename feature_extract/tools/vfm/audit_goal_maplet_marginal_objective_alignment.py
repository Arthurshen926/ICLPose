"""Post-label audit: does the frozen mixture objective actually prefer GT pose?"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.compare_goal_maplet_pose_upgrade import _load, _errors
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load as _load_corr
from feature_extract.tools.vfm.refine_goal_maplet_exact_marginal_surface_pose import _solve
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment_dir", type=Path, required=True)
    p.add_argument("--correspondences", type=Path, nargs=2, required=True)
    p.add_argument("--initial_poses", type=Path, required=True)
    p.add_argument("--query_contributors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite objective audit")
    previous = json.loads((args.experiment_dir / "report.json").read_text())
    initial, _ = _load(args.initial_poses)
    corrs = [_load_corr(path)[0] for path in args.correspondences]
    result = {}
    for head, corr in enumerate(corrs):
        for isotropic in (True, False):
            key = f"h{head}_" + ("isotropic" if isotropic else "anisotropic")
            path = args.experiment_dir / (key + ".npz")
            frozen, meta = _load(path)
            if (file_sha256(path) != previous["summaries"][key]["file_sha256"]
                or meta["initial_sha256"] != file_sha256(args.initial_poses)
                or meta["correspondence_sha256"] != file_sha256(args.correspondences[head])
                or meta["token_anchor_sha256"] != file_sha256(args.correspondences[0])):
                raise ValueError("frozen objective lineage differs")
            rows = []
            for i, name in enumerate(initial["names"].astype(str)):
                if not initial["usable"][i]:
                    continue
                with np.load(args.query_contributors / name) as data:
                    gt = np.asarray(data["pose_w2c"], np.float64)
                _, _, detail = _solve(initial["pose_w2c"][i], corrs[0], corr, i, isotropic, evaluation_pose=gt,
                                      token_support_policy=meta.get("token_support_policy", "initial_nearest"))
                saved = previous["diagnostics"][key][i]
                if "evaluation_loss" not in detail:
                    continue
                if not np.isclose(detail["initial_loss"], saved["initial_loss"], atol=1e-10, rtol=0):
                    raise ValueError("reconstructed objective differs")
                output_loss = saved["final_loss"] if frozen["accepted"][i] else saved["initial_loss"]
                rows.append(dict(name=name, gt_loss=detail["evaluation_loss"], initial_loss=saved["initial_loss"],
                                 output_loss=output_loss, accepted=bool(frozen["accepted"][i])))
            result[key] = dict(evaluated_count=len(rows),
                gt_better_than_initial_count=sum(r["gt_loss"] < r["initial_loss"] for r in rows),
                gt_better_than_output_count=sum(r["gt_loss"] < r["output_loss"] for r in rows),
                output_preferred_over_gt_count=sum(r["output_loss"] < r["gt_loss"] for r in rows), rows=rows)
    report = dict(artifact_type="goal_maplet_marginal_objective_gt_alignment_postlabel_v1", results=result,
                  pose_changes_performed=False, production_eligible=False,
                  original_report_sha256=file_sha256(args.experiment_dir / "report.json"))
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "rows"} for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
