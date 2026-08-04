# V8 Single-Feature Structured Maplet Localization

## Scope

V8 implements the medium-scale representation requested for St Mary's Church:

```text
query RGB
  -> RADIO-final
  -> one MapletRetrievalAdaptor
  -> complete query-region posterior graph
  -> Top-64 physical maplet induced graph
  -> pose-conditioned surface/graph likelihood
  -> multi-modal coarse SE(3)
  -> optional local refinement (not promoted in this experiment)
```

It does not store mapping images, mapping image identifiers/paths, SfM points or
tracks, ALIKE descriptors, RADIO intermediates, LoFTR matches, or point-level
correspondences. Pose estimation is region-to-surface graph alignment; PnP is
not called.

## Representation contracts

### One canonical map feature

The production-shaped student map contains 807 physical maplets and 2,859
mixture components. Every component is the same 128-dimensional canonical
localization feature. DINO, SAM and SigLIP2 embeddings are not fields of the map.

The three frozen C-RADIOv4-H readouts are used only during offline training:

- `dino_v3_7b`: physical-instance appearance;
- `sam3`: region consistency;
- `siglip2-g` spatial and summary outputs: medium/global context.

Their pairwise structure is distilled into one residual
`MapletRetrievalAdaptor`, initialized as the identity so the starting point is
exactly the verified RADIO feature. The final map has the same feature
dimension and component count as the control.

### Physical maplet graph

The graph has 807 nodes and 17,436 directed multi-scale edges:

- six local neighbours;
- six medium-range neighbours;
- four quantile-spaced long-range neighbours.

Edges store relative direction, metric distance, normal agreement, extent
ratio and edge scale. Long-range edges are necessary to make repeated facade
instances less locally isomorphic. The graph artifact contains no feature
array and is tied to the canonical feature bank by SHA-256.

### Query graph

All 128 query regions remain independent nodes. A repeated maplet candidate in
several query regions is never averaged into one pose signature. Nodes retain
their image position, support extent, Top-L physical-maplet posterior and
explicit null mass. Local, medium and long query edges retain displacement,
scale, overlap and ordering information.

The pose score marginalizes node identities and graph configurations. It uses
projected maplet surface footprints and explicit physical edge-class
compatibility, not a point-centre correspondence solver.

## Bugs and route errors corrected

1. The existing anonymous pose-vote artifact belonged to a different
   864-maplet feature bank. V8 rebuilds it against the active 807-maplet bank;
   descriptor and geometry hashes are checked at runtime.
2. The official adaptor is named `dino_v3_7b`, not `dino_v3`.
3. The official SigLIP2 adaptor unnecessarily downloads a giant text tower for
   visual-only localization. V8 reconstructs the exact visual readout from the
   C-RADIO checkpoint and does not retain the text model.
4. An early graph implementation used projected all-pairs relations but did
   not explicitly consume the stored physical edge class. The final scorer
   includes local/medium/long physical-edge compatibility; all reported final
   numbers were rerun after this fix.
5. The code no longer imports a PnP helper merely to calculate pose errors;
   camera-centre and SO(3) errors are computed directly.

## Protocol

- geometry: clean 2DGS lineage;
- map/student training trajectories: seq1, seq2, seq4, seq6, seq7, seq8;
- student validation: seq9, seq10, seq12, seq14;
- strict test: four frames each from seq3, seq5 and seq13;
- strict test has no overlap with mapping, calibration or anonymous pose
  proposal trajectories;
- A/B/C/D uses the same 128D feature dimension and 2,859 map components.

## Results

### Trajectory-disjoint student validation

| Feature | maplet R@1 | maplet R@5 |
|---|---:|---:|
| Existing canonical RADIO localization feature | 34.07% | 66.80% |
| Multi-teacher single student | **36.55%** | **72.54%** |

The teacher signal is learnable and improves the held-out training-domain
trajectories without increasing map storage.

### Strict 12-frame A/B/C/D

| Variant | region R@1 | R@5 | R@64 | median translation | P90 translation | median rotation | <=1m/10deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| A. current feature, no graph | 34.51% | 66.20% | 74.91% | 6.04 m | 9.09 m | 13.44° | 8.33% |
| B. current feature + graph | 34.51% | 66.20% | 74.91% | **1.68 m** | **5.72 m** | **5.18°** | **25.00%** |
| C. multi-teacher student, no graph | **35.69%** | **66.90%** | **76.56%** | 6.54 m | 8.38 m | 13.54° | 8.33% |
| D. multi-teacher student + graph | **35.69%** | **66.90%** | **76.56%** | 2.67 m | 7.99 m | 6.70° | 16.67% |

No variant reaches 30 cm / 3 degrees on this strict set.

## Conclusion

The graph is a real architectural gain, not a small hyperparameter change: it
reduces median translation from 6.04 m to 1.68 m and median rotation from
13.44 degrees to 5.18 degrees. It still does not restore the expected
decimetre/centimetre regime, so V8 must remain a research line rather than a
production replacement.

Multi-teacher/single-student distillation is validated at the representation
level, but D does not beat B on strict pose even after a separate seq11
recalibration tied to the student-bank hash. The current bottleneck is therefore
not map storage or the absence of another embedding. It is the interface
between identity retrieval and pose:

1. the 128 fixed RADIO samples are overlapping descriptor supports, not stable
   complete region instances;
2. anonymous historical pose modes have insufficient coarse-pose coverage;
3. the present graph optimizer can rank the ground-truth pose first in 7/12
   cases when it is inserted as an oracle, yet its available proposal set does
   not enter that basin reliably;
4. in the remaining repeated/low-coverage cases, wrong regions form a more
   self-consistent graph than the true pose.

The next justified research step is not another teacher weight or graph weight
sweep. It is a learned VFM region-support/grouping head (SAM-supervised but
student-only at runtime) plus a maplet-configuration proposal generator trained
for pose-sufficient Top-N recall. Local atlas refinement should remain optional
until those coarse hypotheses are demonstrably inside its measured basin.

## Reproducible artifacts

- final A/B/C/D report:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/v8_structured_abcd_recalibrated_strict12_v3.json`
- student training report:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/v8_multiteacher_student_d1.json`
- single student map:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/v8_identity_bank_multiteacher_student_d1.npz`
- physical graph:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/v8_physical_maplet_graph_multiteacher_d1.npz`
- student adaptor:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/v8_multiteacher_student_d1.pt`
- student probability calibration:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/v8_probability_calibration_seq11_multiteacher_d1.json`

The 195 MB teacher cache is explicitly offline-only and is not referenced by
any runtime artifact.
