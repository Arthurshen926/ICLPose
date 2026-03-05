#!/usr/bin/env python3
"""Analyze per-view PSNR distribution."""
import csv, sys

data = list(csv.DictReader(open('output/visualization/retrain31_15k/per_view_psnr.csv')))
psnrs = [float(d['psnr']) for d in data]
seq4 = [float(d['psnr']) for d in data if 'seq4' in d['view']]
seq8 = [float(d['psnr']) for d in data if 'seq8' in d['view']]

print(f'Total: {len(psnrs)} views, avg={sum(psnrs)/len(psnrs):.2f}')
print(f'seq4: {len(seq4)} views, avg={sum(seq4)/len(seq4):.2f}, min={min(seq4):.2f}, max={max(seq4):.2f}')
print(f'seq8: {len(seq8)} views, avg={sum(seq8)/len(seq8):.2f}, min={min(seq8):.2f}, max={max(seq8):.2f}')
print()

print('--- seq4 worst 5 ---')
s4 = sorted([(d['view'], float(d['psnr'])) for d in data if 'seq4' in d['view']], key=lambda x: x[1])
for v,p in s4[:5]: print(f'  {v}: {p:.2f}')

print('--- seq8 worst 5 ---')
s8 = sorted([(d['view'], float(d['psnr'])) for d in data if 'seq8' in d['view']], key=lambda x: x[1])
for v,p in s8[:5]: print(f'  {v}: {p:.2f}')

print('--- seq8 best 5 ---')
for v,p in s8[-5:]: print(f'  {v}: {p:.2f}')

print('\n--- PSNR histogram ---')
bins = [10,12,14,16,18,20,22,24]
for i in range(len(bins)-1):
    cnt = sum(1 for p in psnrs if bins[i]<=p<bins[i+1])
    bar = '#' * cnt
    print(f'[{bins[i]:2d},{bins[i+1]:2d}): {cnt:3d} {bar}')
