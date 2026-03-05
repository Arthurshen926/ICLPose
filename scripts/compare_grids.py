#!/usr/bin/env python3
"""Quick comparison of two visualization outputs — compose existing images side by side."""
import os
import sys
import json
import argparse
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_a", required=True, help="Visualization dir for model A")
    parser.add_argument("--dir_b", required=True, help="Visualization dir for model B")
    parser.add_argument("--label_a", default="Model A")
    parser.add_argument("--label_b", default="Model B")
    parser.add_argument("--output", default="output/comparison_grid.png")
    args = parser.parse_args()

    # Load stats
    stats_a = json.load(open(os.path.join(args.dir_a, "eval_stats.json")))
    stats_b = json.load(open(os.path.join(args.dir_b, "eval_stats.json")))

    # per_seq can be dict {seq: psnr} or list of {seq, psnr} dicts
    def parse_per_seq(stats):
        ps = stats.get("per_seq", stats.get("per_sequence", {}))
        if isinstance(ps, dict):
            return ps  # {seq_name: psnr_value}
        return {s["seq"]: s["psnr"] for s in ps}

    seqs_a = parse_per_seq(stats_a)
    seqs_b = parse_per_seq(stats_b)
    common_seqs = sorted(set(seqs_a.keys()) & set(seqs_b.keys()))

    print(f"\nComparing {args.label_a} vs {args.label_b}")
    print(f"{'Seq':8s} | {args.label_a:>12s} | {args.label_b:>12s} | Delta")
    print("-" * 55)

    for seq in common_seqs:
        pa = seqs_a[seq]
        pb = seqs_b[seq]
        diff = pb - pa
        marker = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"  {seq:6s} | {pa:10.2f} dB | {pb:10.2f} dB | {diff:+.2f} {marker}")

    ma = stats_a["overall_psnr"]
    mb = stats_b["overall_psnr"]
    print("-" * 55)
    print(f"  {'Mean':6s} | {ma:10.2f} dB | {mb:10.2f} dB | {mb-ma:+.2f}")

    # Compose overview grids side by side
    grid_a_path = os.path.join(args.dir_a, "overview_grid.png")
    grid_b_path = os.path.join(args.dir_b, "overview_grid.png")

    if os.path.exists(grid_a_path) and os.path.exists(grid_b_path):
        ga = Image.open(grid_a_path)
        gb = Image.open(grid_b_path)

        # Scale to same height
        if ga.height != gb.height:
            scale = ga.height / gb.height
            gb = gb.resize((int(gb.width * scale), ga.height), Image.LANCZOS)

        # Compose side by side with labels
        pad = 20
        label_h = 40
        total_w = ga.width + gb.width + pad
        total_h = max(ga.height, gb.height) + label_h

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        canvas.paste(ga, (0, label_h))
        canvas.paste(gb, (ga.width + pad, label_h))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()

        draw.text((10, 8), f"{args.label_a} — PSNR {ma:.2f} dB", fill=(200, 0, 0), font=font)
        draw.text((ga.width + pad + 10, 8), f"{args.label_b} — PSNR {mb:.2f} dB",
                   fill=(0, 150, 0) if mb > ma else (200, 0, 0), font=font)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        canvas.save(args.output)
        print(f"\n  Comparison grid saved: {args.output}")
    else:
        print(f"\n  Warning: overview_grid.png not found in one or both dirs")


if __name__ == "__main__":
    main()
