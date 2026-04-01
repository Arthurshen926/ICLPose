#!/usr/bin/env python3
"""Compile all RADIO-GS experiment results into LaTeX-ready tables.

Parses eval logs from output/ and generates:
  1. Feature quality table (cosine similarity)
  2. Downstream task table (depth, segmentation, grounding)
  3. Ablation table (guide type, geometry, refiner)
"""
import re
import sys
from pathlib import Path

def parse_eval_log(log_path: str) -> dict:
    """Parse eval_rendered.py output log."""
    result = {}
    text = Path(log_path).read_text()
    
    # Feature quality
    m = re.search(r'Val decoded cosine:\s*([\d.]+)', text)
    if m: result['val_cosine'] = float(m.group(1))
    
    # Rendered mode results (primary metric)
    m = re.search(r'RENDERED:.*?AbsRel=([\d.]+)\s+RMSE=([\d.]+)\s+δ<1.25=([\d.]+)', text, re.DOTALL)
    if m:
        result['depth_absrel'] = float(m.group(1))
        result['depth_rmse'] = float(m.group(2))
        result['depth_delta'] = float(m.group(3))
    
    m = re.search(r'RENDERED:.*?mIoU=([\d.]+)\s+PixelAcc=([\d.]+)', text, re.DOTALL)
    if m:
        result['seg_miou'] = float(m.group(1))
        result['seg_pixacc'] = float(m.group(2))
    
    # Oracle results
    m = re.search(r'ORACLE:.*?AbsRel=([\d.]+)', text, re.DOTALL)
    if m: result['oracle_absrel'] = float(m.group(1))
    m = re.search(r'ORACLE:.*?mIoU=([\d.]+)', text, re.DOTALL)
    if m: result['oracle_miou'] = float(m.group(1))
    
    return result

def parse_grounding_log(log_path: str) -> dict:
    """Parse eval_grounding.py output log."""
    result = {}
    text = Path(log_path).read_text()
    
    m = re.search(r'heatmap correlation.*?:\s*([\d.]+)', text, re.IGNORECASE)
    if m: result['heatmap_corr'] = float(m.group(1))
    
    m = re.search(r'^Mean\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', text, re.MULTILINE)
    if m:
        result['gt_miou'] = float(m.group(1))
        result['rend_miou'] = float(m.group(2))
        result['gt_map'] = float(m.group(3))
        result['rend_map'] = float(m.group(4))
    
    return result

def main():
    experiments = {
        'V11 (Self-RGB, v8)': {
            'eval': 'output/eval_v11.log',
            'grounding': 'output/eval_v11_grounding.log',
            'guide': 'Self-RGB',
            'geometry': 'v8 (132K)',
        },
        'V11-GT (GT RGB, v8)': {
            'eval': 'output/eval_v11_gt.log',
            'grounding': 'output/eval_v11_gt_grounding.log',
            'guide': 'GT RGB',
            'geometry': 'v8 (132K)',
        },
        'V9 (GT RGB, v6)': {
            'eval': 'output/eval_v9.log',
            'guide': 'GT RGB',
            'geometry': 'v6 (53K)',
        },
        'V10c (No guide, v6)': {
            'eval': 'output/eval_v10c.log',
            'guide': 'None',
            'geometry': 'v6 (53K)',
        },
    }
    
    # Collect results
    all_results = {}
    for name, info in experiments.items():
        r = {}
        if Path(info.get('eval', '')).exists():
            r.update(parse_eval_log(info['eval']))
        if 'grounding' in info and Path(info['grounding']).exists():
            r.update(parse_grounding_log(info['grounding']))
        r['guide'] = info['guide']
        r['geometry'] = info['geometry']
        all_results[name] = r
    
    # Print markdown table
    print("## Feature Reconstruction & Downstream Tasks (room_0, Novel Views)")
    print()
    print("| Method | Guide | Cosine↑ | AbsRel↓ | RMSE↓ | δ<1.25↑ | mIoU↑ | PixAcc↑ | HM Corr↑ | Grnd mAP↑ |")
    print("|--------|-------|---------|---------|-------|---------|-------|---------|----------|-----------|")
    
    for name, r in all_results.items():
        row = [
            name,
            r.get('guide', ''),
            f"{r.get('val_cosine', 0):.4f}" if 'val_cosine' in r else '—',
            f"{r.get('depth_absrel', 0):.4f}" if 'depth_absrel' in r else '—',
            f"{r.get('depth_rmse', 0):.4f}" if 'depth_rmse' in r else '—',
            f"{r.get('depth_delta', 0):.4f}" if 'depth_delta' in r else '—',
            f"{r.get('seg_miou', 0):.4f}" if 'seg_miou' in r else '—',
            f"{r.get('seg_pixacc', 0):.4f}" if 'seg_pixacc' in r else '—',
            f"{r.get('heatmap_corr', 0):.4f}" if 'heatmap_corr' in r else '—',
            f"{r.get('rend_map', 0):.4f}" if 'rend_map' in r else '—',
        ]
        print("| " + " | ".join(row) + " |")
    
    # LaTeX table
    print()
    print("## LaTeX Table")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Downstream task performance on Replica room\_0 (novel views).}")
    print(r"\label{tab:main_results}")
    print(r"\resizebox{\linewidth}{!}{")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Method & Guide & Cosine$\uparrow$ & AbsRel$\downarrow$ & $\delta{<}1.25\uparrow$ & mIoU$\uparrow$ & mAP$\uparrow$ \\")
    print(r"\midrule")
    
    for name, r in all_results.items():
        cols = [
            name.replace('_', r'\_'),
            r.get('guide', ''),
            f"{r.get('val_cosine', 0):.3f}" if 'val_cosine' in r else '--',
            f"{r.get('depth_absrel', 0):.3f}" if 'depth_absrel' in r else '--',
            f"{r.get('depth_delta', 0):.3f}" if 'depth_delta' in r else '--',
            f"{r.get('seg_miou', 0):.3f}" if 'seg_miou' in r else '--',
            f"{r.get('rend_map', 0):.3f}" if 'rend_map' in r else '--',
        ]
        print(" & ".join(cols) + r" \\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
