#!/usr/bin/env python3#!/usr/bin/env python3








































































































































































































































































































    plot_all()if __name__ == "__main__":""")   - Fix: Increase batch_size to 16-32, max_gaussians to 500K+   - SensorD loss reasonable (0.04-0.45 range), not the main issue   - 94 extreme loss events (mostly small, -0.1 to -24)    - Likely causes: batch_size=4 too small, max_gaussians=200K too low   - PSNR 11.85 dB is very low (indicates blurry/incorrect reconstruction)3. room_0 (COMPLETED, poor quality):   - Fix: Clamp/mask invalid depths, reduce lambda to 0.01-0.1   - Every ~10 iters: extreme loss → skip backward → eventually CRITICAL reset   - lambda_sensor_depth=0.5 amplifies the problem   - Likely caused by invalid depth values (65535mm = 65.54m max uint16) not masked   - SensorD loss intermittently returns values of 10^6 to 10^72. stairs (DIVERGED):    - Only ran 2500 iterations, never activated depth loss (starts at iter 3000)   - Fix: Cast valid_mask to .bool() before indexing   - pearson_depth_loss() crashes with IndexError: valid_mask tensor type wrong1. OldHospital (CRASHED):    print("""    print("="*80)    print("ROOT CAUSE ANALYSIS")    print(f"\n{'='*80}")                print(f"           (typical good 2DGS: 25-35 dB)")            print(f"  STATUS: ✅ COMPLETED — but PSNR=11.85 dB is very poor")        elif name == "room_0":            print(f"           SensorD exploding to millions every ~500 iterations")            print(f"  STATUS: ❌ DIVERGED — stuck in death loop at iter ~18300")        elif name == "stairs":            print(f"  STATUS: ❌ CRASHED at ~iter 2500 (IndexError in pearson_depth_loss)")        if name == "OldHospital":        # Status                print(f"  Critical resets:   {n_critical}")        print(f"  Extreme events:    {n_extreme}")        n_critical = len(d["critical_iters"])        n_extreme = len(d["extreme_iters"])                    print(f"  Final PSNR:        {d['psnr'][-1]:.2f} dB (iter {d['psnr_iters'][-1]})")            print(f"  Best PSNR:         {max(d['psnr']):.2f} dB (iter {d['psnr_iters'][d['psnr'].index(max(d['psnr']))]})")        if d["psnr_iters"]:                print(f"  Gaussians:         {final_gauss:,}")        print(f"  Final Total Loss:  {final_total:.4f}")        print(f"  Final SensorD:     {final_sensor:.4f}")        print(f"  Final RGB Loss:    {final_rgb:.4f}")        print(f"  Progress:          {max_iter}/30000 ({max_iter/300:.1f}%)")                final_gauss = d["n_gauss"][-1]        final_total = d["total"][-1]        final_sensor = d["sensor_d"][-1]        final_rgb = d["rgb"][-1]        max_iter = max(d["iters"])                    continue            print("  No training data found")        if not d["iters"]:                print(f"{'─'*40}")        print(f"Scene: {name}")        print(f"\n{'─'*40}")    for name, d in all_data.items():        print("="*80)    print("TRAINING RESULTS SUMMARY")    print("\n" + "="*80)    # --- Print Summary Table ---            print(f"Saved: {save_path3}")        plt.close(fig3)        fig3.savefig(save_path3, dpi=150, bbox_inches='tight')        save_path3 = SAVE_DIR / "room_0_training_details.png"        plt.tight_layout()                    ax2.grid(True, alpha=0.3)            ax2.set_title("Test PSNR")            ax2.set_xlabel("Iteration")            ax2.set_ylabel("PSNR (dB)")                           xytext=(0, 10), fontsize=8, ha='center')                ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points",            for i, (x, y) in enumerate(zip(d["psnr_iters"], d["psnr"])):            ax2.plot(d["psnr_iters"], d["psnr"], 'o-', color='#2ecc71', linewidth=2, markersize=8)        if d["psnr_iters"]:                ax1.grid(True, alpha=0.3)        ax1.legend()        ax1.set_title("Loss Components")        ax1.set_ylabel("Loss")            ax1.plot(d["iters"], d["normal"], label="Normal", color='#1abc9c', linewidth=1, alpha=0.7)        if any(v > 0 for v in d["normal"]):        ax1.plot(d["iters"], d["total"], label="Total", color='#2ecc71', linewidth=2)        ax1.plot(d["iters"], d["sensor_d"], label="SensorD", color='#9b59b6', linewidth=1.5)        ax1.plot(d["iters"], d["rgb"], label="RGB", color='#e67e22', linewidth=1.5)                fig3.suptitle("room_0 — Training Details (Completed, PSNR=11.85 dB)", fontsize=14, fontweight='bold')        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))    if d["iters"]:    d = all_data["room_0"]    # --- room_0 detailed loss curves ---            print(f"Saved: {save_path2}")        plt.close(fig2)        fig2.savefig(save_path2, dpi=150, bbox_inches='tight')        save_path2 = SAVE_DIR / "stairs_sensor_depth_analysis.png"        plt.tight_layout()                    ax2.grid(True, alpha=0.3)            ax2.set_title(f"Extreme Loss Events Distribution (total: {len(d['extreme_iters'])})")            ax2.set_ylabel("# Extreme Loss Events per 500 iters")            ax2.set_xlabel("Iteration")            ax2.hist(extreme_iters, bins=bins, color='#e74c3c', alpha=0.7, edgecolor='black')            bins = np.arange(0, max(extreme_iters)+500, 500)            extreme_iters = np.array(d["extreme_iters"])        if d["extreme_iters"]:        # Bottom: Extreme loss events histogram                ax1.legend()        ax1.grid(True, alpha=0.3)        ax1.set_title("SensorD Loss Over Time")        ax1.set_ylabel("SensorD Loss (log scale)")        ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='threshold=1.0')        ax1.semilogy(iters[mask], sensor_d[mask], 'o-', color='#9b59b6', markersize=4)        mask = sensor_d > 0        # Top: SensorD on log scale                sensor_d = np.array(d["sensor_d"])        iters = np.array(d["iters"])                fig2.suptitle("stairs — Sensor Depth Loss Explosion Analysis", fontsize=14, fontweight='bold')        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))    if d["iters"] and any(v > 0 for v in d["sensor_d"]):    d = all_data["stairs"]    # --- Detailed SensorD plot for stairs (log scale) ---        print(f"Saved: {save_path}")    plt.close(fig)    fig.savefig(save_path, dpi=150, bbox_inches='tight')    save_path = SAVE_DIR / "depth_training_overview.png"    plt.tight_layout()                ax.set_title(f"{name} — PSNR & Gaussians", fontweight='bold')            ax.text(0.5, 0.5, "No PSNR data", transform=ax.transAxes, ha='center', va='center')        if not d["psnr_iters"] and not d["n_gauss"]:                ax.set_xlabel("Iteration")                    ax.legend(loc='upper left', fontsize=8)        if d["psnr_iters"]:                    ax2.legend(loc='lower right', fontsize=8)            ax2.set_ylabel("Gaussians (K)", color='gray')                    linewidth=1, alpha=0.6, label="Gaussians (K)")            ax2.plot(d["iters"], [n/1000 for n in d["n_gauss"]], '--', color='gray',             ax2 = ax.twinx()        if d["n_gauss"]:                    ax.grid(True, alpha=0.3)            ax.set_title(f"{name} — PSNR & Gaussians", fontweight='bold')            ax.set_ylabel("PSNR (dB)", color=colors[name])            ax.plot(d["psnr_iters"], d["psnr"], 'o-', color=colors[name], linewidth=2, markersize=6, label=f"PSNR")        if d["psnr_iters"]:                ax = axes[2, col]    for col, (name, d) in enumerate(all_data.items()):    # --- Row 2: PSNR / Gaussian count / Extreme events ---                ax.set_title(f"{name} — Loss Components", fontweight='bold')            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center')        else:            ax.grid(True, alpha=0.3)            ax.legend(fontsize=8)            ax.set_ylabel("Loss (symlog)")            ax.set_xlabel("Iteration")            ax.set_title(f"{name} — Loss Components", fontweight='bold')            ax.set_yscale("symlog", linthresh=0.1)                ax.plot(d["iters"], d["dist"], color='#34495e', linewidth=1.0, label="Dist", alpha=0.7)            if any(v > 0 for v in d["dist"]):                ax.plot(d["iters"], d["normal"], color='#1abc9c', linewidth=1.0, label="Normal", alpha=0.7)            if any(v > 0 for v in d["normal"]):                ax.plot(d["iters"], d["sensor_d"], color='#9b59b6', linewidth=1.5, label="SensorD")            if any(v > 0 for v in d["sensor_d"]):            ax.plot(d["iters"], d["rgb"], color='#e67e22', linewidth=1.5, label="RGB")        if d["iters"]:        ax = axes[1, col]    for col, (name, d) in enumerate(all_data.items()):    # --- Row 1: Component Losses (RGB + SensorD) ---                ax.set_title(f"{name} — Total Loss", fontweight='bold')            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center')        else:                ax.axvline(ci, color='red', alpha=0.3, linestyle='--', linewidth=0.8)            for ci in d["critical_iters"]:            # Mark critical events            ax.grid(True, alpha=0.3)            ax.set_ylabel("Loss (symlog)")            ax.set_xlabel("Iteration")            ax.set_title(f"{name} — Total Loss", fontweight='bold')            ax.set_yscale("symlog", linthresh=1.0)            ax.plot(d["iters"], d["total"], color=colors[name], linewidth=1.5, label="Total")        if d["iters"]:        ax = axes[0, col]    for col, (name, d) in enumerate(all_data.items()):    # --- Row 0: Total Loss ---        colors = {"OldHospital": "#e74c3c", "stairs": "#3498db", "room_0": "#2ecc71"}    fig.suptitle("2DGS Depth-Supervised Training Analysis", fontsize=16, fontweight='bold')    fig, axes = plt.subplots(3, 3, figsize=(20, 15))            all_data[name] = parse_log(path)    for name, path in LOGS.items():    all_data = {}def plot_all():    return data                    data["critical_iters"].append(int(m.group(1)))            if m:            m = CRITICAL_RE.search(line)                            continue                data["extreme_vals"].append(float(m.group(2)))                data["extreme_iters"].append(int(m.group(1)))            if m:            m = EXTREME_RE.search(line)                            continue                    data["psnr"].append(psnr)                    data["psnr_iters"].append(it)                if not data["psnr_iters"] or it != data["psnr_iters"][-1]:                it, psnr = int(m.group(1)), float(m.group(2))            if m:            m = PSNR_RE.search(line)                            continue                data["n_gauss"].append(int(m.group(9).replace(",", "")))                data["total"].append(float(m.group(8)))                data["sensor_d"].append(float(m.group(7)))                data["depth"].append(float(m.group(6)))                data["scale"].append(float(m.group(5)))                data["dist"].append(float(m.group(4)))                data["normal"].append(float(m.group(3)))                data["rgb"].append(float(m.group(2)))                data["iters"].append(it)                    continue  # skip duplicate                if data["iters"] and it == data["iters"][-1]:                it = int(m.group(1))            if m:            m = BREAKDOWN_RE.search(line)        for line in f:    with open(path) as f:            return data    if not path.exists():                "critical_iters": []}            "extreme_iters": [], "extreme_vals": [],            "psnr_iters": [], "psnr": [],            "depth": [], "sensor_d": [], "total": [], "n_gauss": [],    data = {"iters": [], "rgb": [], "normal": [], "dist": [], "scale": [],def parse_log(path):CRITICAL_RE = re.compile(r'\[Iter\s+(\d+)\]\s+CRITICAL:')EXTREME_RE = re.compile(r'\[Iter\s+(\d+)\]\s+WARNING:\s+Extreme loss=([-\d.e+]+)')PSNR_RE = re.compile(r'\[Iter\s+(\d+)\]\s+Test PSNR:\s+([\d.]+)\s+dB'))    r'Depth=([\d.]+)\s+SensorD=([\d.]+)\s+Total=([\d.]+)\s+N=([\d,]+)'    r'RGB=([\d.]+)\s+Normal=([\d.]+)\s+Dist=([\d.]+)\s+Scale=([\d.]+)\s+'    r'\[Iter\s+(\d+)\]\s+Loss breakdown:\s+'BREAKDOWN_RE = re.compile(# Pattern: [Iter X] Loss breakdown: RGB=... Normal=... Dist=... Scale=... Depth=... SensorD=... Total=... N=...}    "room_0":      OUT_DIR / "room_0/v2_depth_init/train.log",    "stairs":      OUT_DIR / "stairs/v2_depth/train.log",    "OldHospital": OUT_DIR / "OldHospital/v4_depth/train.log",LOGS = {SAVE_DIR.mkdir(parents=True, exist_ok=True)SAVE_DIR = Path("output/training_analysis")OUT_DIR = Path("output/2dgs_models")from pathlib import Pathimport numpy as npimport matplotlib.pyplot as pltmatplotlib.use('Agg')import matplotlibimport re"""Visualize 2DGS depth-supervised training results for all 3 scenes.""""""Visualize 2DGS depth-supervised training results for all 3 scenes."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path("output/2dgs_models")
SAVE_DIR = Path("output/training_analysis")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LOGS = {
    "OldHospital": OUT_DIR / "OldHospital/v4_depth/train.log",
    "stairs":      OUT_DIR / "stairs/v2_depth/train.log",
    "room_0":      OUT_DIR / "room_0/v2_depth_init/train.log",
}

BREAKDOWN_RE = re.compile(
    r'\[Iter\s+(\d+)\]\s+Loss breakdown:\s+'
    r'RGB=([\d.]+)\s+Normal=([\d.]+)\s+Dist=([\d.]+)\s+Scale=([\d.]+)\s+'
    r'Depth=([\d.]+)\s+SensorD=([\d.]+)\s+Total=([\d.]+)\s+N=([\d,]+)'
)
PSNR_RE = re.compile(r'\[Iter\s+(\d+)\]\s+Test PSNR:\s+([\d.]+)\s+dB')
EXTREME_RE = re.compile(r'\[Iter\s+(\d+)\]\s+WARNING:\s+Extreme loss=([-\d.e+]+)')
CRITICAL_RE = re.compile(r'\[Iter\s+(\d+)\]\s+CRITICAL:')


def parse_log(path):
    data = {"iters": [], "rgb": [], "normal": [], "dist": [], "scale": [],
            "depth": [], "sensor_d": [], "total": [], "n_gauss": [],
            "psnr_iters": [], "psnr": [],
            "extreme_iters": [], "extreme_vals": [],
            "critical_iters": []}
    if not path.exists():
        return data
    with open(path) as f:
        for line in f:
            m = BREAKDOWN_RE.search(line)
            if m:
                it = int(m.group(1))
                if data["iters"] and it == data["iters"][-1]:
                    continue
                data["iters"].append(it)
                data["rgb"].append(float(m.group(2)))
                data["normal"].append(float(m.group(3)))
                data["dist"].append(float(m.group(4)))
                data["scale"].append(float(m.group(5)))
                data["depth"].append(float(m.group(6)))
                data["sensor_d"].append(float(m.group(7)))
                data["total"].append(float(m.group(8)))
                data["n_gauss"].append(int(m.group(9).replace(",", "")))
                continue
            m = PSNR_RE.search(line)
            if m:
                it, psnr = int(m.group(1)), float(m.group(2))
                if not data["psnr_iters"] or it != data["psnr_iters"][-1]:
                    data["psnr_iters"].append(it)
                    data["psnr"].append(psnr)
                continue
            m = EXTREME_RE.search(line)
            if m:
                data["extreme_iters"].append(int(m.group(1)))
                data["extreme_vals"].append(float(m.group(2)))
                continue
            m = CRITICAL_RE.search(line)
            if m:
                data["critical_iters"].append(int(m.group(1)))
    return data


def plot_all():
    all_data = {}
    for name, path in LOGS.items():
        all_data[name] = parse_log(path)
        print(f"Parsed {name}: {len(all_data[name]['iters'])} iterations")

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle("2DGS Depth-Supervised Training Analysis", fontsize=16, fontweight='bold')
    colors = {"OldHospital": "#e74c3c", "stairs": "#3498db", "room_0": "#2ecc71"}

    # Row 0: Total Loss
    for col, (name, d) in enumerate(all_data.items()):
        ax = axes[0, col]
        if d["iters"]:
            ax.plot(d["iters"], d["total"], color=colors[name], linewidth=1.5)
            ax.set_yscale("symlog", linthresh=1.0)
            for ci in d["critical_iters"]:
                ax.axvline(ci, color='red', alpha=0.3, linestyle='--', linewidth=0.8)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f"{name} - Total Loss", fontweight='bold')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (symlog)")
        ax.grid(True, alpha=0.3)

    # Row 1: Component Losses
    for col, (name, d) in enumerate(all_data.items()):
        ax = axes[1, col]
        if d["iters"]:
            ax.plot(d["iters"], d["rgb"], color='#e67e22', linewidth=1.5, label="RGB")
            if any(v > 0 for v in d["sensor_d"]):
                ax.plot(d["iters"], d["sensor_d"], color='#9b59b6', linewidth=1.5, label="SensorD")
            if any(v > 0 for v in d["normal"]):
                ax.plot(d["iters"], d["normal"], color='#1abc9c', linewidth=1.0, label="Normal", alpha=0.7)
            if any(v > 0 for v in d["dist"]):
                ax.plot(d["iters"], d["dist"], color='#34495e', linewidth=1.0, label="Dist", alpha=0.7)
            ax.set_yscale("symlog", linthresh=0.1)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f"{name} - Loss Components", fontweight='bold')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (symlog)")
        ax.grid(True, alpha=0.3)

    # Row 2: PSNR + Gaussian count
    for col, (name, d) in enumerate(all_data.items()):
        ax = axes[2, col]
        if d["psnr_iters"]:
            ax.plot(d["psnr_iters"], d["psnr"], 'o-', color=colors[name], linewidth=2, markersize=6, label="PSNR")
            ax.set_ylabel("PSNR (dB)", color=colors[name])
            ax.legend(loc='upper left', fontsize=8)
        if d["n_gauss"]:
            ax2 = ax.twinx()
            ax2.plot(d["iters"], [n/1000 for n in d["n_gauss"]], '--', color='gray', linewidth=1, alpha=0.6, label="Gaussians (K)")
            ax2.set_ylabel("Gaussians (K)", color='gray')
            ax2.legend(loc='lower right', fontsize=8)
        if not d["psnr_iters"] and not d["n_gauss"]:
            ax.text(0.5, 0.5, "No PSNR data", transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f"{name} - PSNR & Gaussians", fontweight='bold')
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = SAVE_DIR / "depth_training_overview.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Stairs sensor depth analysis
    d = all_data["stairs"]
    if d["iters"] and any(v > 0 for v in d["sensor_d"]):
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        fig2.suptitle("stairs - Sensor Depth Loss Explosion Analysis", fontsize=14, fontweight='bold')
        iters = np.array(d["iters"])
        sensor_d = np.array(d["sensor_d"])
        mask = sensor_d > 0
        ax1.semilogy(iters[mask], sensor_d[mask], 'o-', color='#9b59b6', markersize=4)
        ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='threshold=1.0')
        ax1.set_ylabel("SensorD Loss (log scale)")
        ax1.set_title("SensorD Loss Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        if d["extreme_iters"]:
            extreme_iters = np.array(d["extreme_iters"])
            bins = np.arange(0, max(extreme_iters)+500, 500)
            ax2.hist(extreme_iters, bins=bins, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("# Extreme Events per 500 iters")
            ax2.set_title(f"Extreme Loss Events (total: {len(d['extreme_iters'])})")
            ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path2 = SAVE_DIR / "stairs_sensor_depth_analysis.png"
        fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {save_path2}")

    # room_0 details
    d = all_data["room_0"]
    if d["iters"]:
        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        fig3.suptitle("room_0 - Training Details (Completed, PSNR=11.85 dB)", fontsize=14, fontweight='bold')
        ax1.plot(d["iters"], d["rgb"], label="RGB", color='#e67e22', linewidth=1.5)
        ax1.plot(d["iters"], d["sensor_d"], label="SensorD", color='#9b59b6', linewidth=1.5)
        ax1.plot(d["iters"], d["total"], label="Total", color='#2ecc71', linewidth=2)
        if any(v > 0 for v in d["normal"]):
            ax1.plot(d["iters"], d["normal"], label="Normal", color='#1abc9c', linewidth=1, alpha=0.7)
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Components")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if d["psnr_iters"]:
            ax2.plot(d["psnr_iters"], d["psnr"], 'o-', color='#2ecc71', linewidth=2, markersize=8)
            for x, y in zip(d["psnr_iters"], d["psnr"]):
                ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 10), fontsize=8, ha='center')
            ax2.set_ylabel("PSNR (dB)")
            ax2.set_xlabel("Iteration")
            ax2.set_title("Test PSNR")
            ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path3 = SAVE_DIR / "room_0_training_details.png"
        fig3.savefig(save_path3, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"Saved: {save_path3}")

    # Print Summary
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    for name, d in all_data.items():
        print(f"\n--- {name} ---")
        if not d["iters"]:
            print("  No data")
            continue
        max_iter = max(d["iters"])
        print(f"  Progress:        {max_iter}/30000 ({max_iter/300:.1f}%)")
        print(f"  Final RGB:       {d['rgb'][-1]:.4f}")
        print(f"  Final SensorD:   {d['sensor_d'][-1]:.4f}")
        print(f"  Final Total:     {d['total'][-1]:.4f}")
        print(f"  Gaussians:       {d['n_gauss'][-1]:,}")
        if d["psnr_iters"]:
            print(f"  Best PSNR:       {max(d['psnr']):.2f} dB")
            print(f"  Final PSNR:      {d['psnr'][-1]:.2f} dB")
        print(f"  Extreme events:  {len(d['extreme_iters'])}")
        print(f"  Critical resets: {len(d['critical_iters'])}")


if __name__ == "__main__":
    plot_all()
