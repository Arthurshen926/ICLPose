#!/usr/bin/env python3
"""Analyze which sequences are used in COLMAP and train/test split."""
import struct, os, sys
from collections import Counter

def read_images_binary(path):
    images = {}
    with open(path, 'rb') as f:
        num = struct.unpack('Q', f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack('I', f.read(4))[0]
            qvec = struct.unpack('4d', f.read(32))
            tvec = struct.unpack('3d', f.read(24))
            cam_id = struct.unpack('I', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name += c
            name = name.decode()
            num_pts = struct.unpack('Q', f.read(8))[0]
            f.read(num_pts * 24)
            images[img_id] = name
    return images

base = "dataset/OldHospital"
imgs = read_images_binary(f"{base}/sparse/0/images.bin")
names = sorted(imgs.values())
print(f"Total COLMAP images: {len(names)}")

seqs = Counter()
for n in names:
    seq = n.split('/')[0] if '/' in n else 'root'
    seqs[seq] += 1
for s, c in sorted(seqs.items()):
    print(f"  {s}: {c}")

# Check train/test split
with open(f'{base}/dataset_train.txt') as f:
    train_lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
with open(f'{base}/dataset_test.txt') as f:
    test_lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
print(f"\ndataset_train.txt: {len(train_lines)} lines")
print(f"  First: {train_lines[:3]}")
print(f"dataset_test.txt: {len(test_lines)} lines")
print(f"  First: {test_lines[:3]}")

# Parse
train_names = set()
for l in train_lines:
    parts = l.split()
    if parts and '/' in parts[0]:
        train_names.add(parts[0])
test_names = set()
for l in test_lines:
    parts = l.split()
    if parts and '/' in parts[0]:
        test_names.add(parts[0])

print(f"\nParsed train images: {len(train_names)}")
print(f"Parsed test images: {len(test_names)}")

train_seqs = Counter()
for n in train_names:
    train_seqs[n.split('/')[0]] += 1
test_seqs = Counter()
for n in test_names:
    test_seqs[n.split('/')[0]] += 1
print(f"\nTrain sequences: {dict(sorted(train_seqs.items()))}")
print(f"Test sequences: {dict(sorted(test_seqs.items()))}")

# Check overlap
in_colmap = set(names)
train_in_colmap = train_names & in_colmap
test_in_colmap = test_names & in_colmap
print(f"\nTrain in COLMAP: {len(train_in_colmap)}/{len(train_names)}")
print(f"Test in COLMAP: {len(test_in_colmap)}/{len(test_names)}")

# Check what the training script actually used
print("\n--- What load_scene uses ---")
sparse_dir = f"{base}/sparse/0"
list_test = os.path.join(sparse_dir, "list_test.txt")
dataset_test = os.path.join(base, "dataset_test.txt")
if os.path.exists(list_test):
    print(f"Using list_test.txt from sparse/0/")
elif os.path.exists(dataset_test):
    print(f"Using dataset_test.txt from base dir")
else:
    print("No test split file found - using LLFF every-8th split")
