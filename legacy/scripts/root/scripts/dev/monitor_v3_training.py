#!/usr/bin/env python3
"""
Monitor 2DGS v3_improved training progress.
Shows loss, Gaussian count, speed, and estimated time.
"""
import sys, os, re, time

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else 'logs/oldhospital_v3_retrain5.log'

def parse_progress_line(line):
    """Parse tqdm progress line for metrics."""
    # Match: 2DGS Training:  18%|█▊        | 7000/40000 [09:23<43:02, 12.78it/s, Loss=0.01234, N=500,000]
    m = re.search(r'(\d+)%.*?(\d+)/(\d+)\s+\[([^\]]+)\].*?Loss=([\d.]+).*?N=([\d,]+)', line)
    if m:
        pct = int(m.group(1))
        cur = int(m.group(2))
        total = int(m.group(3))
        timing = m.group(4)
        loss = float(m.group(5))
        n_gauss = int(m.group(6).replace(',', ''))
        return {'pct': pct, 'iter': cur, 'total': total, 'time': timing,
                'loss': loss, 'n_gauss': n_gauss}
    return None

def parse_eval_line(line):
    """Parse evaluation output line."""
    # [Iter 7000] Test PSNR: 14.28 dB (182 views)
    m = re.search(r'\[Iter (\d+)\].*?PSNR:\s*([\d.]+)', line)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None

def parse_save_line(line):
    """Parse checkpoint save line."""
    m = re.search(r'\[Iter (\d+)\] Saving.*?(\d[\d,]+)\s+Gaussians', line)
    if m:
        return int(m.group(1)), int(m.group(2).replace(',', ''))
    return None

def parse_prune_line(line):
    """Parse post-densification prune line."""
    m = re.search(r'\[Iter (\d+)\] Post-densify prune: ([\d,]+)', line)
    if m:
        return int(m.group(1)), int(m.group(2).replace(',', ''))
    return None

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        sys.exit(1)

    with open(LOG_FILE) as f:
        lines = f.readlines()

    evals = []
    saves = []
    prunes = []
    last_progress = None

    for line in lines:
        line = line.strip()
        # Check for eval
        ev = parse_eval_line(line)
        if ev:
            evals.append(ev)
        # Check for save
        sv = parse_save_line(line)
        if sv:
            saves.append(sv)
        # Prune
        pr = parse_prune_line(line)
        if pr:
            prunes.append(pr)

    # Parse last progress from the last lines (tqdm writes \r)
    for line in reversed(lines[-5:]):
        # Try to get last tqdm line
        parts = line.strip().split('\r')
        for part in reversed(parts):
            p = parse_progress_line(part.strip())
            if p:
                last_progress = p
                break
        if last_progress:
            break

    print(f"\n{'='*50}")
    print(f"  v3_improved Training Monitor")
    print(f"{'='*50}")

    if last_progress:
        p = last_progress
        print(f"  Progress: {p['iter']}/{p['total']} ({p['pct']}%)")
        print(f"  Loss:     {p['loss']:.5f}")
        print(f"  Gaussians: {p['n_gauss']:,}")
        print(f"  Time:     {p['time']}")
    else:
        print("  No progress data found yet")

    if evals:
        print(f"\n  Evaluations:")
        for it, psnr in evals:
            print(f"    Iter {it:>6d}: PSNR = {psnr:.2f} dB")

    if saves:
        print(f"\n  Checkpoints:")
        for it, n in saves:
            print(f"    Iter {it:>6d}: {n:,} Gaussians")

    if prunes:
        print(f"\n  Post-densify pruning:")
        for it, n in prunes:
            print(f"    Iter {it:>6d}: pruned {n:,}")

    print(f"{'='*50}\n")

if __name__ == '__main__':
    main()
