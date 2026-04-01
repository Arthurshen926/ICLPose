#!/usr/bin/env python3
"""Aggregate all RADIO-GS evaluation results into comparison tables.

Parses eval logs from V11, baselines, and ablations to produce:
1. Main comparison table (LaTeX + markdown)
2. Ablation table
3. Per-scene breakdown
"""

import re
import os
import sys
from pathlib import Path
from collections import defaultdict

# --- Config: methods to aggregate ---
METHODS = {
    # method_key: (display_name, eval_log_pattern, grounding_log_pattern, notes)
    "v11_self": {
        "name": "RADIO-GS (Ours)",
        "eval_logs": [
            "output/eval_v11.log",
            "output/eval_v11_room_0.log",
        ],
        "grounding_logs": [
            "output/eval_v11_grounding.log",
            "output/eval_v11_room_0_grounding.log",
        ],
        "notes": "V11, self-guided",
    },
    "v11_gt": {
        "name": "RADIO-GS (GT guide)",
        "eval_logs": [
            "output/eval_v11_gt.log",
        ],
        "grounding_logs": [
            "output/eval_v11_gt_grounding.log",
        ],
        "notes": "V11 w/ GT RGB guide",
    },
    "v9": {
        "name": "V9 (GT guide, v6)",
        "eval_logs": [
            "output/eval_v9.log",
        ],
        "grounding_logs": [
            "output/eval_v9_grounding.log",
        ],
        "notes": "V9 baseline",
    },
    "baseline_f3dgs": {
        "name": "Feature3DGS-style",
        "eval_logs": [
            "output/eval_baseline_f3dgs.log",
        ],
        "grounding_logs": [
            "output/eval_baseline_f3dgs_grounding.log",
        ],
        "notes": "No refiner, no FeatSharp",
    },
    "ablation_no_refiner": {
        "name": "Ours w/o Refiner",
        "eval_logs": [
            "output/eval_ablation_no_refiner.log",
        ],
        "grounding_logs": [
            "output/eval_ablation_no_refiner_grounding.log",
        ],
        "notes": "FeatSharp only, no refiner",
    },
}

# Multi-scene methods
SCENES = ["room_0", "room_1", "room_2"]


def parse_eval_log(filepath):
    """Parse eval_rendered.py output log."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        content = f.read()

    results = {}

    # Feature quality
    m = re.search(r"Val decoded cosine:\s*([\d.]+)", content)
    if m:
        results["cosine"] = float(m.group(1))

    # Parse table rows
    for mode_prefix, mode_key in [
        ("Oracle", "oracle"),
        ("Rendered", "rendered"),
        ("Cross", "cross"),
    ]:
        pattern = rf"{mode_prefix}[^0-9]*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        m = re.search(pattern, content)
        if m:
            results[f"{mode_key}_absrel"] = float(m.group(1))
            results[f"{mode_key}_rmse"] = float(m.group(2))
            results[f"{mode_key}_delta1"] = float(m.group(3))
            results[f"{mode_key}_miou"] = float(m.group(4))
            results[f"{mode_key}_pixacc"] = float(m.group(5))

    # Also try individual section parsing as fallback
    if "rendered_absrel" not in results:
        m = re.search(
            r"RENDERED.*?Depth.*?AbsRel=([\d.]+)\s+RMSE=([\d.]+)\s+.*?=([\d.]+)",
            content,
            re.DOTALL,
        )
        if m:
            results["rendered_absrel"] = float(m.group(1))
            results["rendered_rmse"] = float(m.group(2))
            results["rendered_delta1"] = float(m.group(3))

        m = re.search(
            r"RENDERED.*?Seg.*?mIoU=([\d.]+)\s+PixelAcc=([\d.]+)", content, re.DOTALL
        )
        if m:
            results["rendered_miou"] = float(m.group(1))
            results["rendered_pixacc"] = float(m.group(2))

    return results if results else None


def parse_grounding_log(filepath):
    """Parse eval_grounding.py output log."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        content = f.read()

    results = {}

    m = re.search(r"Mean heatmap correlation.*?:\s*([\d.]+)", content)
    if m:
        results["hm_corr"] = float(m.group(1))

    m = re.search(r"Zero-shot argmax accuracy.*?GT:\s*([\d.]+).*?Rendered:\s*([\d.]+)", content)
    if m:
        results["gt_argmax_acc"] = float(m.group(1))
        results["rend_argmax_acc"] = float(m.group(2))

    # Parse mean row from table
    m = re.search(r"Mean\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", content)
    if m:
        results["gt_miou"] = float(m.group(1))
        results["rend_miou"] = float(m.group(2))
        results["gt_map"] = float(m.group(3))
        results["rend_map"] = float(m.group(4))

    return results if results else None


def find_first_existing(paths):
    """Return first existing file path from a list."""
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def collect_all_results():
    """Collect results from all methods."""
    all_results = {}

    for key, method in METHODS.items():
        eval_path = find_first_existing(method["eval_logs"])
        grounding_path = find_first_existing(method["grounding_logs"])

        result = {"name": method["name"], "notes": method["notes"]}

        if eval_path:
            eval_res = parse_eval_log(eval_path)
            if eval_res:
                result.update(eval_res)
                result["eval_log"] = eval_path

        if grounding_path:
            grnd_res = parse_grounding_log(grounding_path)
            if grnd_res:
                result.update(grnd_res)
                result["grounding_log"] = grounding_path

        if len(result) > 2:  # has actual metrics
            all_results[key] = result

    return all_results


def format_metric(val, best_val=None, second_val=None, lower_better=False, fmt=".4f"):
    """Format a metric value, optionally bolding best."""
    if val is None:
        return "—"
    s = f"{val:{fmt}}"
    if best_val is not None and abs(val - best_val) < 1e-6:
        s = f"**{s}**"
    elif second_val is not None and abs(val - second_val) < 1e-6:
        s = f"_{s}_"
    return s


def print_markdown_table(all_results):
    """Print a markdown comparison table."""
    print("\n## Main Results Comparison (room_0, Novel Views)\n")
    print(
        "| Method | Cosine↑ | AbsRel↓ | RMSE↓ | δ<1.25↑ | mIoU↑ | PixAcc↑ | HM Corr↑ | Grnd mAP↑ |"
    )
    print(
        "|--------|---------|---------|-------|---------|-------|---------|----------|-----------|"
    )

    for key, r in all_results.items():
        row = [
            r["name"],
            f'{r.get("cosine", 0):.4f}' if "cosine" in r else "—",
            f'{r.get("rendered_absrel", 0):.4f}' if "rendered_absrel" in r else "—",
            f'{r.get("rendered_rmse", 0):.4f}' if "rendered_rmse" in r else "—",
            f'{r.get("rendered_delta1", 0):.4f}' if "rendered_delta1" in r else "—",
            f'{r.get("rendered_miou", 0):.4f}' if "rendered_miou" in r else "—",
            f'{r.get("rendered_pixacc", 0):.4f}' if "rendered_pixacc" in r else "—",
            f'{r.get("hm_corr", 0):.4f}' if "hm_corr" in r else "—",
            f'{r.get("rend_map", 0):.4f}' if "rend_map" in r else "—",
        ]
        print("| " + " | ".join(row) + " |")

    print()


def print_latex_table(all_results):
    """Print a LaTeX comparison table."""
    print("\n% LaTeX table: Main Results")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Multi-task evaluation on Replica room\\_0 (novel views).}")
    print("\\label{tab:main_results_all}")
    print("\\resizebox{\\linewidth}{!}{")
    print("\\begin{tabular}{l ccc cc c c}")
    print("\\toprule")
    print(
        "Method & Cos$\\uparrow$ & AbsRel$\\downarrow$ & $\\delta<1.25$$\\uparrow$ & mIoU$\\uparrow$ & PixAcc$\\uparrow$ & HM Corr$\\uparrow$ & Grnd mAP$\\uparrow$ \\\\"
    )
    print("\\midrule")

    for key, r in all_results.items():
        name = r["name"].replace("_", "\\_")
        cos = f'{r.get("cosine", 0):.3f}' if "cosine" in r else "—"
        absrel = f'{r.get("rendered_absrel", 0):.3f}' if "rendered_absrel" in r else "—"
        delta1 = f'{r.get("rendered_delta1", 0):.3f}' if "rendered_delta1" in r else "—"
        miou = f'{r.get("rendered_miou", 0):.3f}' if "rendered_miou" in r else "—"
        pixacc = f'{r.get("rendered_pixacc", 0):.3f}' if "rendered_pixacc" in r else "—"
        hm = f'{r.get("hm_corr", 0):.3f}' if "hm_corr" in r else "—"
        gmap = f'{r.get("rend_map", 0):.3f}' if "rend_map" in r else "—"

        line = f"{name} & {cos} & {absrel} & {delta1} & {miou} & {pixacc} & {hm} & {gmap} \\\\"
        print(line)

    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\end{table}")
    print()


def main():
    os.chdir("/root/ICLPose")
    all_results = collect_all_results()

    if not all_results:
        print("No evaluation results found!")
        sys.exit(1)

    print("=" * 70)
    print("  RADIO-GS Results Aggregation")
    print("=" * 70)
    print(f"\nFound {len(all_results)} methods with results:\n")

    for key, r in all_results.items():
        src = []
        if "eval_log" in r:
            src.append(r["eval_log"])
        if "grounding_log" in r:
            src.append(r["grounding_log"])
        print(f"  {r['name']} ({key}): {', '.join(src)}")

    print_markdown_table(all_results)
    print_latex_table(all_results)

    # Save to file
    outdir = Path("output/aggregated_results")
    outdir.mkdir(exist_ok=True)

    with open(outdir / "comparison.md", "w") as f:
        import io
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        print_markdown_table(all_results)
        sys.stdout = old_stdout
        f.write(buffer.getvalue())

    with open(outdir / "comparison.tex", "w") as f:
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        print_latex_table(all_results)
        sys.stdout = old_stdout
        f.write(buffer.getvalue())

    print(f"Saved to {outdir}/comparison.md and comparison.tex")


if __name__ == "__main__":
    main()
