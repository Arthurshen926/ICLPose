# G21 P0 Correctness Results — 2026-08-11

## Scope

These results are a seven-query development stress audit, not publication
test evidence. The queries were repeatedly inspected during G20/G21 design.
They may be used to find correctness failures and freeze a method, but not to
estimate final generalization.

Strict success is 0.5 m / 5 deg and loose success is 1 m / 10 deg. Counts below
pool seq12 frames 65, 66, 93, 144 and 155 with seq14 frames 4 and 26.

## Correctness fixes

- Mapping-view H0 is normalized before the conditional view distribution. In
  the corrected runs its null probability is about 0.88--0.95, rather than
  approximately zero and dependent on the number of retained views.
- Sparse primitive verification uses all clean physical primitives for its
  depth prepass. A featureless or unsampled foreground primitive now produces
  missing feature evidence instead of exposing a rear descriptor.
- Child-frame sampling uses row-axis frame semantics.
- Negative parent padding can no longer match an unowned rendered primitive.
- Surface rendering parses focal lengths by COLMAP camera model.
- Exact-pool selection protects structural anchors before a global fallback,
  and every replay embeds hashes, candidate counts and numeric contracts.

## Fixed-budget proposal ablation

Every row uses 128 exact full-map verification slots. "Global" is the
corrected sparse score Top-128 without structural protection.

| Exact-pool policy | Strict Top-1 | Strict Top-4 | Strict Top-16/32 | Loose Top-1 | Loose Top-4 | Loose Top-16/32 |
|---|---:|---:|---:|---:|---:|---:|
| corrected global Top-128 | 2/7 | 4/7 | 5/7 | 5/7 | 6/7 | 6/7 |
| 32 anchors × 3 + 32 global | 2/7 | 5/7 | 6/7 | 5/7 | 6/7 | 6/7 |
| 32 anchors × 4 + 0 global | 2/7 | 5/7 | 5/7 | 6/7 | 6/7 | 6/7 |
| **16 anchors × 4 + 64 global** | **2/7** | **4/7** | **6/7** | **5/7** | **6/7** | **7/7** |

The selected policy improves strict Top-16/32 from 5/7 to 6/7 and loose
Top-16/32 from 6/7 to 7/7 without increasing exact computation. It does not
improve strict Top-1, so this is a proposal-recall correction rather than a
winner-selection claim.

The 32-by-4 policy is rejected even though it makes frame155 strict Top-1: it
uses the entire budget for mapping-view anchors and removes frame26's strict
fallback basin. The 32-by-3 policy restores frame155 at strict Top-4 but removes
frame144's loose fallback basin. This cross-frame behavior is why the final
configuration reserves half of the exact budget for the global channel.

## Failure attribution

- `seq12/frame00155.png` is an S-stage failure under global Top-128. Its only
  strict state exists in the raw mapping-view beam (0.464 m / 4.392 deg), ranks
  224 after the corrected sparse screen, and is deleted before exact scoring.
  Structural protection sends it to exact verification and preserves it in the
  final Top-16/32 pool.
- `seq12/frame00139.png` is a G-stage failure. The raw 15,972-state
  mapping-view beam has no 1 m / 10 deg state; its best is 1.059 m / 3.946 deg.
  More exact slots or a different verifier cannot create the absent basin.
- `seq12/frame00144.png` has only a loose basin after proposal. It is a useful
  guard against deleting the global fallback while repairing frame155.

## Decision

The corrected G21 is retained as a hard-correspondence-free VFM-specific
proposal--surface-verification framework. Historical G21 numbers produced
before these fixes are invalid for a paper table. The present development
result is sufficient to freeze the P0 execution contract, not to claim a final
method improvement.

The next research change should target the remaining G- and V-stage limits:
a continuous visibility atlas around mapping poses and a view-conditioned,
low-rank primitive feature field. After those are frozen, evaluation must move
to map-disjoint and route-disjoint untouched data, with post-selection success
calibration reported as risk--coverage and localization success.

## Artifacts

- Canonical selected 16-by-4 replay (with tracked-diff lineage): `output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g21_p0_final_20260811/`
- Original 16-by-4 ablation: `output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g21_p0_anchor16x4_20260811/`
- Corrected global audit: `output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g21_p0_correctness_20260811/`
- 32-by-3 ablation: `output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g21_p0_anchor3_20260811/`
- 32-by-4 ablation: `output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/g21_p0_anchor4_20260811/`
