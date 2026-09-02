# G25 source-frozen continuous seam gate (2026-08-30)

## Verdict

The exact16 alignment loss is not the immediate reason that the previous seam
gate fails.  The old evaluator recomputes nearest vertices on each candidate
atlas and includes every vertex whose nearest point in the other chart lies
within 1 m.  On a partial-overlap edge this is not a seam correspondence.  It
also treats tangential pixel-lattice phase as surface thickness.  On the fresh
exact16 topology, even the isolated source-reference M0 fails all 48 old
per-edge Euclidean gates.

This round adds an independent source-only gate and does not change the formal
full-submap evaluator.  For every frozen coverage edge it computes both
directions of the exact continuous closest-point map

```text
source topology vertex -> target topology triangle + barycentric coordinates
```

on the source MASt3R geometry.  Target face identity, barycentric weights and
source vertex identity are frozen before either arm is evaluated.  The 0.30 m
and 35 degree selector thresholds only decide whether that source material
correspondence exists.  Support remains a separate per-direction statistic and
must exceed `max(5%, 0.5 * frozen_plan_overlap)`.  Euclidean distance is retained
only as an association/sampling-phase diagnostic.  Formal geometry uses
bidirectional fixed-correspondence point-to-plane and interpolated surface
normal residuals.

## M0 reachability result

The continuous construction freezes 4,490 correspondences over 48 plan
coverage edges.  It improves source-topology reachability relative to a
vertex-to-vertex diagnostic, but it still exposes a plan/topology mismatch:

- only 37/48 edges pass the pre-frozen bidirectional support rule;
- five edges have no continuous correspondence within 0.30 m / 35 degrees;
- six additional edges have some correspondences but insufficient support;
- among the 37 reachable edges, source-reference M0 passes absolute
  point-to-plane/normal integrity on 33.

Therefore this exact16 chart plan is not formally seam-gate eligible.  This is
a source-only KILL before held geometry is opened.  Rigid chart refinement
cannot repair an edge whose retained physical topology contains no supported
overlap, and moving whole charts to manufacture support would invalidate their
metric/source binding.

## Conditional arm result on the 37 reachable edges

| state | reachable-edge passes | pooled point-to-plane P50 / P90 | pooled unsigned-normal P90 |
|---|---:|---:|---:|
| source-reference M0 | 33 / 37 | 0.0395 / 0.1283 m | 20.49 deg |
| M1 DAV2 aligned | 35 / 37 | 0.0306 / 0.1012 m | 20.25 deg |
| M2 MoGe-3 aligned | **37 / 37** | **0.0263 / 0.0975 m** | **17.95 deg** |

The formal M1 and M2 decisions remain KILL because M0 does not authorize a
complete 48-edge comparison.  Conditional results are nevertheless useful:
the MoGe-3 alignment is not causing the supported seams to fail.  It improves
all three pooled surface metrics and passes every reachable edge.  DAV2 fails
two reachable edges on normal P90.  Thus another deformation/refinement loss
is not justified before the retained-topology support problem is fixed.

## Sampling-phase counterexample

The unit test uses two triangulations of the same plane whose vertex lattices
are shifted tangentially by 25 cm.  Vertex nearest-neighbour distance is 25 cm,
but the exact target-triangle closest point lies on the same plane and has zero
point-to-plane error.  The continuous gate passes.  A 20 cm displacement along
the plane normal fails the P50 gate.  This proves that the implementation
separates UV sampling phase from physical surface thickness.

## Implemented artifacts

- authority core:
  `feature_extract/vfm/localization_goal_maplet/source_seam_correspondence.py`
- source-only builder:
  `feature_extract/tools/vfm/build_goal_maplet_source_seam_correspondence_authority.py`
- paired M1/M2 evaluator:
  `feature_extract/tools/vfm/evaluate_goal_maplet_source_seam_geometry.py`
- tests:
  `tests/test_goal_maplet_source_seam_correspondence.py`
- exact16 authority (final replay, `source_seam_v2` namespace):
  `output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_exact16_fresh/source_seam_v2/source_seam_correspondence_authority_v1.npz`
  - file SHA: `5a7f3d48954e52aa4243226029064154136f72b7ec88cb03bbda88d7e9146c94`
  - content SHA: `618b06cf52bd3777181bd199f387be603666c63482935e608627bac2badb8ebf`
- exact16 report:
  `output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_exact16_fresh/source_seam_v2/source_seam_geometry_gate_v1.json`
  - file SHA: `cc60b8a893fb0bbf0d3d091aa5b8acd3cee178e09976edc7b1f9093cd36d0f65`
  - content SHA: `8b15bda81094c9cef5e551e7b595d6ee259573b824745991f8a4083039283391`

Thirty-eight tests pass across the new continuous authority and the existing
v3 topology/full-gate/strict-atlas contracts.

The earlier `source_seam_v1` run is superseded by this final replay; its
numerical geometry is identical, while `source_seam_v2` additionally seals the
core implementation hash and explicitly distinguishes reachability eligibility
from the final M0 geometry decision.

## Correct next source-only change

Do not tune the association radius, lower the support threshold or add a chart
transform using these results.  Instead, make physical topology part of chart
planning:

1. build the v3 common physical topology first;
2. construct continuous source-surface overlap on that retained topology;
3. delete unsupported edges from the planning graph before chart selection;
4. require the selected graph to remain connected and every selected edge to
   pass M0 reachability and M0 absolute surface integrity;
5. freeze the resulting plan, exact topology, continuous correspondence
   authority and choice of arm before opening a new held root.

This is not held-driven reselection: all five operations consume only isolated
mapping-source geometry and predeclared thresholds.

## New unseen-held protocol

The already inspected fresh held reconstruction is development evidence and
must not be used to choose the new topology-aware plan.  A subsequent formal
claim needs a new physically isolated held inventory whose names, byte hashes
and builder configuration are frozen before geometry access.

The formal sequence should be:

```text
source-only v3 topology
  -> topology-aware connected plan
  -> source continuous seam authority (must be M0 eligible)
  -> fixed M1/M2 atlas and arm choice
  -> freeze new held inventory/config without reading geometry
  -> build held rays after every source choice is sealed
  -> report source seam integrity and held ray coverage as separate gates
  -> composite GO only if both gates pass
```

Held geometry must never create, delete or reweight a seam correspondence, and
no arm-specific face deletion is allowed.  If M0 is not source-seam eligible,
the protocol stops before held evaluation rather than laundering a partial
conditional result into a formal pass.
