#!/usr/bin/env python3
"""Stack exported per-image student features into pose-regression tensors.

This converts the directory produced by `feature_extract/export_impl.py` into the
stacked tensor layout expected by `feature_retrieval/patch_regressor_v7.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from feature_extract import sample_name_to_feature_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stack exported student features for pose regression")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-summary", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def _parse_split_names(split_file: Path) -> List[str]:
    names: List[str] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Visual") or line.startswith("ImageFile"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            names.append(parts[0])
    return names


def _index_export_records(export_dir: Path) -> Dict[str, int]:
    export_index = json.loads((export_dir / "export_index.json").read_text(encoding="utf-8"))
    return {str(entry["sample_name"]): int(entry["teacher_idx"]) for entry in export_index}


def _index_feature_files(feature_dir: Path, kind: str) -> Dict[int, Path]:
    indexed: Dict[int, Path] = {}
    marker = f"_{kind}_"
    for path in sorted(feature_dir.glob(f"rgb_*_{kind}_*.pt")):
        stem = path.stem
        if not stem.startswith("rgb_") or marker not in stem:
            continue
        teacher_idx = int(stem[len("rgb_") : stem.index(marker)])
        indexed[teacher_idx] = path
    if not indexed:
        raise RuntimeError(f"No {kind} exports found under {feature_dir}")
    return indexed


def _stack_split(
    names: Sequence[str],
    sample_to_teacher_idx: Dict[str, int],
    fine_files: Dict[int, Path],
    coarse_files: Dict[int, Path],
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    fine_tensors: List[torch.Tensor] = []
    coarse_tensors: List[torch.Tensor] = []
    missing: List[str] = []

    for name in names:
        teacher_idx = sample_to_teacher_idx.get(name)
        if teacher_idx is None or teacher_idx not in fine_files or teacher_idx not in coarse_files:
            missing.append(name)
            continue
        fine_tensors.append(torch.load(fine_files[teacher_idx], map_location="cpu").to(dtype))
        coarse_tensors.append(torch.load(coarse_files[teacher_idx], map_location="cpu").to(dtype))

    if missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"Missing exported features for {len(missing)} samples: {preview}")

    return {
        "fine": torch.stack(fine_tensors, dim=0),
        "coarse": torch.stack(coarse_tensors, dim=0),
    }


def _build_summary_matrix(
    dataset_dir: Path,
    sample_to_teacher_idx: Dict[str, int],
    cls_dir: Path,
    dtype: torch.dtype,
) -> torch.Tensor:
    image_paths = sorted(dataset_dir.glob("seq*/*.png"))
    if not image_paths:
        raise RuntimeError(f"No images found under {dataset_dir}/seq*")

    summaries: List[torch.Tensor] = []
    missing: List[str] = []
    for image_path in image_paths:
        rel_name = image_path.relative_to(dataset_dir).as_posix()
        teacher_idx = sample_to_teacher_idx.get(rel_name)
        if teacher_idx is None:
            missing.append(rel_name)
            continue
        sample_stem = sample_name_to_feature_stem(rel_name)
        matches = sorted(cls_dir.glob(f"{sample_stem}_cls_*.pt"))
        if not matches:
            missing.append(rel_name)
            continue
        summaries.append(torch.load(matches[0], map_location="cpu").view(-1).to(dtype))

    if missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"Missing exported CLS descriptors for {len(missing)} samples: {preview}")

    return torch.stack(summaries, dim=0)


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    sample_to_teacher_idx = _index_export_records(export_dir)
    fine_files = _index_feature_files(export_dir / "fine_geo", "fine_geo")
    coarse_files = _index_feature_files(export_dir / "coarse_sem", "coarse_sem")

    train_names = _parse_split_names(dataset_dir / "dataset_train.txt")
    test_names = _parse_split_names(dataset_dir / "dataset_test.txt")

    train_stack = _stack_split(train_names, sample_to_teacher_idx, fine_files, coarse_files, dtype)
    test_stack = _stack_split(test_names, sample_to_teacher_idx, fine_files, coarse_files, dtype)

    torch.save(train_stack["fine"], output_dir / "fine_geo_train.pt")
    torch.save(test_stack["fine"], output_dir / "fine_geo_test.pt")
    torch.save(train_stack["coarse"], output_dir / "coarse_sem_train.pt")
    torch.save(test_stack["coarse"], output_dir / "coarse_sem_test.pt")

    summary_path = None
    if args.save_summary:
        cls_dir = export_dir / "cls"
        if not cls_dir.is_dir():
            raise RuntimeError(f"Requested --save-summary but no cls/ directory exists under {export_dir}")
        summary_matrix = _build_summary_matrix(dataset_dir, sample_to_teacher_idx, cls_dir, dtype)
        summary_path = output_dir / "summary_matrix.pt"
        torch.save(summary_matrix, summary_path)

    metadata = {
        "export_dir": str(export_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "dtype": args.dtype,
        "num_export_records": len(sample_to_teacher_idx),
        "num_train": len(train_names),
        "num_test": len(test_names),
        "fine_shape_train": list(train_stack["fine"].shape),
        "coarse_shape_train": list(train_stack["coarse"].shape),
        "fine_shape_test": list(test_stack["fine"].shape),
        "coarse_shape_test": list(test_stack["coarse"].shape),
        "summary_matrix": str(summary_path.resolve()) if summary_path is not None else None,
    }
    (output_dir / "stack_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
