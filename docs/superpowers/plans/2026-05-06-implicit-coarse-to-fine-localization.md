# Implicit Coarse-To-Fine Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the first deployable internal coarse pose-bank initializer and measure whether project coarse features can provide the large pose basin currently supplied by NetVLAD/LoFTR/PnP.

**Architecture:** Add a focused coarse pose-bank exporter under `feature_extract/` that pools exported `coarse_sem` maps into descriptors, performs top-k cosine retrieval against train/reference poses, writes the existing retrieval-init cache schema, and reports top-k pose recall. The existing refiner/evaluator remains unchanged until recall proves the coarse bank is worth wiring into the full pipeline.

**Tech Stack:** Python, PyTorch tensors, NumPy, existing COLMAP split helpers, existing retrieval-init NPZ schema, pytest.

---

### Task 1: Coarse Pose Bank Unit Tests

**Files:**
- Create: `tests/test_coarse_pose_bank_init_export.py`
- Read: `tests/test_student_feature_bank_init.py`
- Read: `feature_extract/student_feature_bank_init_export.py`

- [ ] **Step 1: Write failing tests**

Create tests covering:

```python
def test_pool_coarse_descriptor_normalizes_spatial_mean():
    feature = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]])
    desc = pool_coarse_descriptor(feature)
    assert desc.shape == (1, 2)
    assert torch.allclose(desc.norm(dim=1), torch.ones(1), atol=1e-6)
    assert desc[0, 0] > 0.99


def test_search_pose_bank_returns_sorted_matches():
    query = torch.tensor([[0.0, 1.0]])
    bank = torch.tensor([[1.0, 0.0], [0.0, 0.9], [0.0, 0.7]])
    indices, scores = search_coarse_pose_bank(query, bank, topk=2)
    assert indices.tolist() == [[1, 2]]
    assert scores[0, 0] > scores[0, 1]


def test_build_coarse_pose_bank_entries_exports_existing_schema(tmp_path):
    entries, stats = build_coarse_pose_bank_entries(...)
    save_retrieval_init_entries(entries, stats, str(tmp_path / "coarse_bank.npz"))
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(tmp_path / "coarse_bank.npz"))
    assert loaded_entries[0]["pose_init_candidates"].shape == (2, 4, 4)
    assert loaded_stats["method_requested"] == "coarse_pose_bank"


def test_topk_pose_recall_reports_any_candidate_within_threshold():
    metrics = summarize_topk_pose_recall(...)
    assert metrics["top2_joint_1deg_100mm"] == 100.0
```

- [ ] **Step 2: Run red tests**

Run:

```bash
conda run --no-capture-output -n iclpose python -m pytest tests/test_coarse_pose_bank_init_export.py -q
```

Expected: import failure for `feature_extract.coarse_pose_bank_init_export`.

### Task 2: Coarse Pose Bank Implementation

**Files:**
- Create: `feature_extract/coarse_pose_bank_init_export.py`
- Reuse: `data/radio_loc_retrieval_dataset.py`
- Reuse: `feature_retrieval/localization_mainline.py`

- [ ] **Step 1: Implement descriptor helpers**

Add:

```python
def pool_coarse_descriptor(feature: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor
def extract_cached_coarse_descriptors(feature_dir: str, samples: Sequence[Dict]) -> tuple[list[Dict], torch.Tensor]
def search_coarse_pose_bank(query_descriptors: torch.Tensor, bank_descriptors: torch.Tensor, *, topk: int)
```

Behavior:

- Accept `[C,H,W]`, `[B,C,H,W]`, or `[B,C]`.
- Spatial-average descriptors.
- L2 normalize descriptors.
- Load `coarse_sem/rgb_<img_id>_coarse_sem_*.pt` through `TeacherFeatureStore`-style discovery or direct glob.

- [ ] **Step 2: Implement retrieval cache export**

Add:

```python
def build_coarse_pose_bank_entries(...)
def export_coarse_pose_bank_init(...)
```

Behavior:

- Same output fields as `build_student_feature_bank_entries`.
- `method_requested="coarse_pose_bank"`.
- `method_used=source_name`.
- Preserve top-k candidates, frame IDs, image names, and scores.

- [ ] **Step 3: Implement top-k recall summary**

Add:

```python
def summarize_topk_pose_recall(entries: Sequence[Dict], gt_poses_by_name: Mapping[str, np.ndarray], topks=(1,5,10,20,50)) -> Dict[str, float]
```

Metrics:

- `top{k}_rot_median`
- `top{k}_trans_median`
- `top{k}_joint_1deg_100mm`
- `top{k}_joint_5deg_250mm`

For each query and k, use the best pose error among valid first-k candidates.

- [ ] **Step 4: Add CLI**

Arguments:

```bash
--cached_feature_dir
--colmap_dir
--train_split
--query_split
--save_path
--topk
--source_name
--summary_path
```

The CLI writes the NPZ cache and optional JSON summary.

- [ ] **Step 5: Run green tests**

Run:

```bash
conda run --no-capture-output -n iclpose python -m pytest tests/test_coarse_pose_bank_init_export.py -q
```

Expected: all tests pass.

### Task 3: OldHospital Coarse Bank Export And Recall

**Files:**
- Use: `feature_extract/coarse_pose_bank_init_export.py`
- Output: `/root/ICLPose/result/result/feature_extract/pose_init_exports/oldhospital_coarsebank_v68_top50_test.npz`
- Output: `/root/ICLPose/result/result/feature_extract/pose_init_exports/oldhospital_coarsebank_v68_top50_test_summary.json`

- [ ] **Step 1: Export top50 test cache**

Run:

```bash
conda run --no-capture-output -n iclpose python -m feature_extract.coarse_pose_bank_init_export \
  --cached_feature_dir /root/ICLPose/result/result/feature_extract/features_radio_dual_v68_align_stop640/cambridge_oldhospital_processed \
  --colmap_dir /hy-tmp/Cambridge_stdloc/OldHospital/sparse/0 \
  --train_split /hy-tmp/Cambridge_stdloc/OldHospital/dataset_train.txt \
  --query_split /hy-tmp/Cambridge_stdloc/OldHospital/dataset_test.txt \
  --save_path /root/ICLPose/result/result/feature_extract/pose_init_exports/oldhospital_coarsebank_v68_top50_test.npz \
  --summary_path /root/ICLPose/result/result/feature_extract/pose_init_exports/oldhospital_coarsebank_v68_top50_test_summary.json \
  --topk 50 \
  --source_name coarse_pose_bank_v68_top50
```

- [ ] **Step 2: Compare against current anchors**

Read the JSON summary and compare top-k recall to:

- best deployable top20 selected: 165.1mm final
- top50 oracle selected: 77.6mm final
- current rank0/top1: about 224.4mm initial/final depending protocol

Decision:

- If `top20_joint_1deg_100mm` and `top50_joint_1deg_100mm` are near or above current NetVLAD/LoFTR candidate recall, proceed to Task 4.
- If coarse bank recall is clearly weak, train a coarse ranking loss before refiner integration.

### Task 4: Optional Full Pipeline Smoke With Coarse Bank Cache

**Files:**
- Use: `feature_retrieval.evaluate`
- Use: `pose_refine/configs/concat_loc_cambridge_oldhospital_processed_locguided_dcff_highres_depthaware_v17_selfmap_export_corrwls_probe_extproj_scale025.yaml`

- [ ] **Step 1: Run 32-sample smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n iclpose python -m feature_retrieval.evaluate \
  --config pose_refine/configs/concat_loc_cambridge_oldhospital_processed_locguided_dcff_highres_depthaware_v17_selfmap_export_corrwls_probe_extproj_scale025.yaml \
  --checkpoint /root/ICLPose/result/pose_refine/concat_loc_cambridge_oldhospital_processed_locguided_dcff_highres_depthaware_v17_selfmap_export_corrwls_probe/checkpoints/best.pth \
  --localization_manifest /root/ICLPose/result/feature_extract/features_radio_dual_v68_align_stop640/cambridge_oldhospital_processed/localization_manifest.json \
  --init_poses_path /root/ICLPose/result/result/feature_extract/pose_init_exports/oldhospital_coarsebank_v68_top50_test.npz \
  --gpu 0 --batch_size 1 --num_workers 0 --outer_iters 1 --solver default \
  --retrieval_topk 50 --pose_fusion quality_weighted_consensus_centroid \
  --max_samples 32 \
  --output_dir /root/ICLPose/result/result/feature_retrieval/v68_coarsebank_top50_smoke32
```

Expected: a valid `summary.json`; compare final median translation to existing first-32 baselines.

### Task 5: Regression Verification

Run:

```bash
conda run --no-capture-output -n iclpose python -m pytest tests/test_coarse_pose_bank_init_export.py tests/test_student_feature_bank_init.py tests/test_localization_mainline_protocol.py -q
conda run --no-capture-output -n iclpose python -m py_compile feature_extract/coarse_pose_bank_init_export.py
git diff --check
```

Expected: tests pass, py_compile passes, diff check clean.

