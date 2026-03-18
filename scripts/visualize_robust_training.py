#!/usr/bin/env python3
"""Comprehensive visualization for v6/v4/v3 robust training runs.

Parses train.log files and generates multi-panel charts:
  1) Overview: RGB Loss, Total Loss, PSNR, Gaussian count per scene
  2) Depth analysis: SensorD loss evolution, depth clamp events
  3) Comparative: side-by-side across scenes
  4) Detailed per-scene analysis
"""

import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
BASE = Path("/root/ICLPose")
OUT_DIR = BASE / "output" / "training_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGS = {
    "OldHospital (v6)": BASE / "output/2dgs_models/OldHospital/v6_robust/train.log",
    "stairs (v4)":      BASE / "output/2dgs_models/stairs/v4_robust/train.log",
    "room_0 (v3)":      BASE / "output/2dgs_models/room_0/v3_robust/train.log",
}

COLORS = {
    "OldHospital (v6)": "#e74c3c",
    "stairs (v4)":      "#2ecc71",
    "room_0 (v3)":      "#3498db",
}

# ── Parsing ─────────────────────────────────────────────────────────────────
def parse_log(path):
    """Parse a train.log and extract structured data."""
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", errors="replace")
    # strip ANSI codes and carriage returns
    clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw)
    clean = re.sub(r"\r", "\n", clean)
    lines = [l.strip() for l in clean.split("\n") if l.strip()]

    data = {
        "iters": [], "rgb": [], "normal": [], "dist": [],
        "scale": [], "depth": [], "sensor_d": [], "total": [],
        "n_gauss": [],
        "psnr_iters": [], "psnr_vals": [],
        "tqdm_iters": [], "tqdm_loss": [],
        "oom_count": 0, "extreme_count": 0,
    }

    # Loss breakdown pattern
    bd_pat = re.compile(
        r"\[Iter (\d+)\] Loss breakdown: "
        r"RGB=([\d.]+)\s+Normal=([\d.]+)\s+Dist=([\d.]+)\s+Scale=([\d.]+)\s+"
        r"Depth=([\d.]+)\s+SensorD=([\d.]+)\s+Total=([\d.]+)\s+N=([\d,]+)"
    )
    psnr_pat = re.compile(r"\[Iter (\d+)\] Test PSNR:\s+([\d.]+)\s+dB")
    tqdm_pat = re.compile(r"(\d+)/30000.*?Loss=([\d.]+)")

    for line in lines:
        m = bd_pat.search(line)
        if m:
            data["iters"].append(int(m.group(1)))
            data["rgb"].append(float(m.group(2)))
            data["normal"].append(float(m.group(3)))
            data["dist"].append(float(m.group(4)))
            data["scale"].append(float(m.group(5)))
            data["depth"].append(float(m.group(6)))
            data["sensor_d"].append(float(m.group(7)))
            data["total"].append(float(m.group(8)))
            data["n_gauss"].append(int(m.group(9).replace(",", "")))
            continue
        m = psnr_pat.search(line)
        if m:
            data["psnr_iters"].append(int(m.group(1)))
            data["psnr_vals"].append(float(m.group(2)))
            continue
        m = tqdm_pat.search(line)
        if m:
            data["tqdm_iters"].append(int(m.group(1)))
            data["tqdm_loss"].append(float(m.group(2)))
        if "OutOfMemoryError" in line:
            data["oom_count"] += 1
        if "EXTREME" in line:
            data["extreme_count"] += 1

    return data


# ── Plot 1: Main Overview (2x2 per scene) ──────────────────────────────────
def plot_overview(all_data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("2DGS Robust Training Overview — All Scenes", fontsize=16, fontweight="bold")

    # (0,0) RGB Loss
    ax = axes[0, 0]
    for name, d in all_data.items():
        ax.plot(d["iters"], d["rgb"], "o-", color=COLORS[name], label=name, markersize=5)
    ax.set_title("RGB Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # (0,1) Total Loss
    ax = axes[0, 1]
    for name, d in all_data.items():
        ax.plot(d["iters"], d["total"], "s-", color=COLORS[name], label=name, markersize=5)
    ax.set_title("Total Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # (1,0) PSNR
    ax = axes[1, 0]
    for name, d in all_data.items():
        if d["psnr_iters"]:
            ax.plot(d["psnr_iters"], d["psnr_vals"], "D-", color=COLORS[name],
                    label=name, markersize=8, linewidth=2)
            for xi, yi in zip(d["psnr_iters"], d["psnr_vals"]):
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                           xytext=(0, 10), fontsize=8, ha="center", color=COLORS[name])
    ax.set_title("Test PSNR (dB)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # (1,1) Gaussian Count
    ax = axes[1, 1]
    for name, d in all_data.items():
        ax.plot(d["iters"], [n / 1000 for n in d["n_gauss"]], "^-",
                color=COLORS[name], label=name, markersize=5)
    ax.set_title("Gaussian Count (×1000)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Count (K)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    out = OUT_DIR / "robust_training_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out}")


# ── Plot 2: Depth Supervision Analysis ──────────────────────────────────────
def plot_depth_analysis(all_data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Depth Supervision Analysis", fontsize=16, fontweight="bold")

    # (0,0) SensorD Loss
    ax = axes[0, 0]
    for name, d in all_data.items():
        ax.plot(d["iters"], d["sensor_d"], "o-", color=COLORS[name], label=name, markersize=5)
    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Clamp limit (0.1)")
    ax.set_title("Sensor Depth Loss (after λ weighting)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SensorD Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # (0,1) Mono Depth Loss
    ax = axes[0, 1]
    for name, d in all_data.items():
        ax.plot(d["iters"], d["depth"], "o-", color=COLORS[name], label=name, markersize=5)
    ax.set_title("Mono Depth Loss (Pearson)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Depth Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # (1,0) SensorD vs RGB for depth scenes
    ax = axes[1, 0]
    for name, d in all_data.items():
        if max(d["sensor_d"]) > 0:  # only scenes with sensor depth
            ax.plot(d["iters"], d["rgb"], "o--", color=COLORS[name], label=f"{name} RGB", alpha=0.7, markersize=4)
            ax.plot(d["iters"], d["sensor_d"], "s-", color=COLORS[name], label=f"{name} SensorD", markersize=5)
    ax.set_title("RGB vs SensorD (scenes with depth)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # (1,1) Normal Loss (once activated)
    ax = axes[1, 1]
    for name, d in all_data.items():
        if max(d["normal"]) > 0:
            ax.plot(d["iters"], d["normal"], "o-", color=COLORS[name], label=name, markersize=5)
    ax.set_title("Normal Loss (starts at iter 5000)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normal Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    out = OUT_DIR / "robust_depth_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out}")


# ── Plot 3: Per-Scene Detail ────────────────────────────────────────────────
def plot_per_scene(name, d, color):
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f"Detailed Analysis — {name}", fontsize=16, fontweight="bold")

    # Row 0: RGB, Total, PSNR
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(d["iters"], d["rgb"], "o-", color=color, markersize=5)
    ax1.set_title("RGB Loss")
    ax1.set_xlabel("Iter")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(d["iters"], d["total"], "s-", color=color, markersize=5)
    ax2.set_title("Total Loss")
    ax2.set_xlabel("Iter")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    if d["psnr_iters"]:
        ax3.plot(d["psnr_iters"], d["psnr_vals"], "D-", color=color, markersize=10, linewidth=2)
        for xi, yi in zip(d["psnr_iters"], d["psnr_vals"]):
            ax3.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 12), fontsize=10, ha="center", fontweight="bold")
    ax3.set_title("Test PSNR (dB)")
    ax3.set_xlabel("Iter")
    ax3.grid(True, alpha=0.3)

    # Row 1: SensorD, Normal, Gaussian count
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(d["iters"], d["sensor_d"], "o-", color="orange", markersize=5, label="SensorD")
    ax4.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Clamp=0.1")
    ax4.set_title("Sensor Depth Loss")
    ax4.set_xlabel("Iter")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(d["iters"], d["normal"], "o-", color="purple", markersize=5, label="Normal")
    ax5.plot(d["iters"], d["depth"], "s-", color="brown", markersize=5, label="Mono Depth")
    ax5.set_title("Normal & Mono Depth Loss")
    ax5.set_xlabel("Iter")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(d["iters"], [n / 1000 for n in d["n_gauss"]], "^-", color=color, markersize=6)
    ax6.fill_between(d["iters"], 0, [n / 1000 for n in d["n_gauss"]], color=color, alpha=0.15)
    ax6.set_title("Gaussian Count (K)")
    ax6.set_xlabel("Iter")
    ax6.grid(True, alpha=0.3)

    # Row 2: tqdm loss curve (fine-grained), loss composition stacked bar, summary text
    ax7 = fig.add_subplot(gs[2, 0])
    if d["tqdm_iters"]:
        # subsample tqdm to avoid clutter
        step = max(1, len(d["tqdm_iters"]) // 500)
        ti = d["tqdm_iters"][::step]
        tl = d["tqdm_loss"][::step]
        ax7.plot(ti, tl, "-", color=color, alpha=0.6, linewidth=0.8)
    ax7.set_title("Fine-grained Loss (tqdm)")
    ax7.set_xlabel("Iter")
    ax7.set_ylabel("Loss")
    ax7.grid(True, alpha=0.3)

    # Loss composition stacked bar
    ax8 = fig.add_subplot(gs[2, 1])
    iters = d["iters"]
    if iters:
        bar_w = max(1, (max(iters) - min(iters)) / (len(iters) * 2)) if len(iters) > 1 else 100
        bottoms = np.zeros(len(iters))
        components = [
            ("RGB", d["rgb"], "#3498db"),
            ("SensorD", d["sensor_d"], "#e67e22"),
            ("Normal", d["normal"], "#9b59b6"),
            ("Depth", d["depth"], "#8B4513"),
            ("Dist", d["dist"], "#2ecc71"),
        ]
        for lbl, vals, col in components:
            vals_arr = np.array(vals)
            if vals_arr.max() > 0:
                ax8.bar(iters, vals_arr, bottom=bottoms, width=bar_w, color=col, label=lbl, alpha=0.8)
                bottoms += vals_arr
    ax8.set_title("Loss Composition")
    ax8.set_xlabel("Iter")
    ax8.legend(fontsize=7, loc="upper right")
    ax8.grid(True, alpha=0.3, axis="y")

    # Summary text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    current_iter = d["tqdm_iters"][-1] if d["tqdm_iters"] else (d["iters"][-1] if d["iters"] else 0)
    latest_rgb = d["rgb"][-1] if d["rgb"] else 0
    latest_total = d["total"][-1] if d["total"] else 0
    latest_sensor = d["sensor_d"][-1] if d["sensor_d"] else 0
    latest_n = d["n_gauss"][-1] if d["n_gauss"] else 0
    best_psnr = max(d["psnr_vals"]) if d["psnr_vals"] else 0
    latest_psnr = d["psnr_vals"][-1] if d["psnr_vals"] else 0

    summary = (
        f"Current Iter: {current_iter:,} / 30,000\n"
        f"Progress: {current_iter/300:.1f}%\n"
        f"\n"
        f"Latest RGB Loss: {latest_rgb:.4f}\n"
        f"Latest SensorD: {latest_sensor:.4f}\n"
        f"Latest Total: {latest_total:.4f}\n"
        f"\n"
        f"Gaussians: {latest_n:,}\n"
        f"  (from 100,000 init)\n"
        f"  Pruned: {100000 - latest_n:,} ({(100000-latest_n)/1000:.1f}K)\n"
        f"\n"
        f"Best PSNR: {best_psnr:.2f} dB\n"
        f"Latest PSNR: {latest_psnr:.2f} dB\n"
        f"\n"
        f"OOM Events: {d['oom_count']}\n"
        f"Extreme Events: {d['extreme_count']}\n"
    )
    ax9.text(0.05, 0.95, summary, transform=ax9.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    scene_tag = name.split()[0].lower()
    out = OUT_DIR / f"robust_{scene_tag}_detail.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out}")


# ── Plot 4: Comparative Summary Table & Bar Chart ──────────────────────────
def plot_comparison(all_data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cross-Scene Comparison (Current Status)", fontsize=16, fontweight="bold")

    names = list(all_data.keys())
    colors = [COLORS[n] for n in names]
    x = np.arange(len(names))

    # Latest PSNR
    ax = axes[0]
    psnrs = [d["psnr_vals"][-1] if d["psnr_vals"] else 0 for d in all_data.values()]
    bars = ax.bar(x, psnrs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, psnrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               f"{val:.2f}", ha="center", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=10)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Latest Test PSNR")
    ax.grid(True, alpha=0.3, axis="y")

    # Gaussian Count
    ax = axes[1]
    gauss = [d["n_gauss"][-1]/1000 if d["n_gauss"] else 0 for d in all_data.values()]
    bars = ax.bar(x, gauss, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, gauss):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f"{val:.1f}K", ha="center", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=10)
    ax.set_ylabel("Count (K)")
    ax.set_title("Gaussians Remaining")
    ax.grid(True, alpha=0.3, axis="y")

    # RGB Loss
    ax = axes[2]
    rgb_latest = [d["rgb"][-1] if d["rgb"] else 0 for d in all_data.values()]
    bars = ax.bar(x, rgb_latest, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, rgb_latest):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
               f"{val:.4f}", ha="center", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=10)
    ax.set_ylabel("Loss")
    ax.set_title("Latest RGB Loss")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = OUT_DIR / "robust_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out}")


# ── Plot 5: Pruning & Health Analysis ──────────────────────────────────────
def plot_health(all_data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Training Health & Pruning Analysis", fontsize=16, fontweight="bold")

    # Gaussian retention rate over time
    ax = axes[0]
    for name, d in all_data.items():
        if d["iters"] and d["n_gauss"]:
            init_n = d["n_gauss"][0]
            retention = [n / init_n * 100 for n in d["n_gauss"]]
            ax.plot(d["iters"], retention, "o-", color=COLORS[name], label=name, markersize=5)
    ax.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="50% threshold")
    ax.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="20% critical")
    ax.set_title("Gaussian Retention Rate (%)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Retention (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)

    # Loss variance (smoothness)
    ax = axes[1]
    for name, d in all_data.items():
        if len(d["rgb"]) > 2:
            diffs = [abs(d["rgb"][i+1] - d["rgb"][i]) for i in range(len(d["rgb"])-1)]
            mid_iters = [(d["iters"][i] + d["iters"][i+1]) / 2 for i in range(len(d["iters"])-1)]
            ax.plot(mid_iters, diffs, "o-", color=COLORS[name], label=name, markersize=4)
    ax.set_title("RGB Loss Volatility (|Δ| between checkpoints)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|ΔLoss|")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # PSNR per 1K gaussians (efficiency)
    ax = axes[2]
    for name, d in all_data.items():
        if d["psnr_iters"] and d["n_gauss"]:
            # Find N at each PSNR eval iter
            efficiencies = []
            for pi, pv in zip(d["psnr_iters"], d["psnr_vals"]):
                # find closest iter in breakdowns
                closest_n = d["n_gauss"][0]
                for bi, bn in zip(d["iters"], d["n_gauss"]):
                    if bi <= pi:
                        closest_n = bn
                eff = pv / (closest_n / 1000) if closest_n > 0 else 0
                efficiencies.append(eff)
            ax.plot(d["psnr_iters"], efficiencies, "D-", color=COLORS[name],
                    label=name, markersize=8, linewidth=2)
            for xi, yi in zip(d["psnr_iters"], efficiencies):
                ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                           xytext=(0, 10), fontsize=8, ha="center")
    ax.set_title("PSNR Efficiency (dB per 1K Gaussians)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR / 1K Gaussians")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "robust_health_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    all_data = {}
    for name, path in LOGS.items():
        if not path.exists():
            print(f"[SKIP] {name}: {path} not found")
            continue
        print(f"[Parse] {name}: {path}")
        all_data[name] = parse_log(path)
        d = all_data[name]
        print(f"  Breakdowns: {len(d['iters'])}, PSNR evals: {len(d['psnr_iters'])}, "
              f"tqdm points: {len(d['tqdm_iters'])}, OOM: {d['oom_count']}, Extreme: {d['extreme_count']}")

    if not all_data:
        print("No data found!")
        return

    print("\n--- Generating plots ---")
    plot_overview(all_data)
    plot_depth_analysis(all_data)
    plot_comparison(all_data)
    plot_health(all_data)

    for name, d in all_data.items():
        plot_per_scene(name, d, COLORS[name])

    print(f"\nAll plots saved to: {OUT_DIR}")
    print(f"Files: {[f.name for f in sorted(OUT_DIR.glob('robust_*.png'))]}")


if __name__ == "__main__":
    main()
