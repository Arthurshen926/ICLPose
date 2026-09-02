# G25 strict chart-alignment lineage adversarial audit (2026-08-30)

## Verdict

The previous v2 authority and initializer artifacts were individually sealed,
but the end-to-end alignment path was not fail-closed. Two independent P0
bypasses were confirmed:

1. the alignment runner ignored the frozen submap plan and resampled charts
   from `route + chart_count`;
2. a self-consistent old/alternate comparison-domain NPZ could be supplied
   without proving that it bound the current authority, plan, cameras,
   pointmaps or initializer bytes.

Both paths are now closed. Strict paired alignment requires externally pinned
plan and comparison-domain content hashes. The plan is resolved only through
`load_model_neutral_alignment_selection`, and its exact ordered names are the
only chart inventory accepted by the domain builder and alignment runner.
Strict M1/M2 runs additionally pin the content hash of the complete
`matcha/**/*.py` source inventory, preventing two arms from silently using
different mask/loss implementations.

## Replayed contract

The strict chain is now:

```text
v2 disjoint authority
  -> replay complete source tree + cameras + every source pointmap
  -> externally pinned model-neutral chart-submap plan v2
  -> exact selected_chart_names_in_order (one operational submap)
  -> sealed DAV2 and MoGe initializer manifests/files
  -> common-domain NPZ bound to authority + source tree + plan + cameras
     + pointmaps + both initializer inventories
  -> externally pinned common-domain content hash
  -> M1/M2 alignment on the exact common pixel mask
  -> explicit-atlas export on the exact common face mask
```

The runner now performs all CPU lineage/inventory checks before importing the
GPU alignment stack or creating an output directory. Manifest row counts,
duplicate names, per-file content hashes, focal/image bindings, 144x256 grid
contracts and source-tree lineage are checked explicitly.

## Pixel and face consumption

The two domains have different roles and are now reported without conflation:

- `valid` is the optimization-domain pixel mask. For either arm the runner
  first proves it is a subset of the current initializer/reference validity,
  then assigns that exact array to `PointMap.confidence`, `PointMap.masks` and
  the alignment reference mask. Its hash is stored in the run manifest.
- `face_valid_stride4/8` is not consumed by the point-grid optimizer. It is an
  export/evaluation topology contract. The runner copies both arrays into
  `charts_data.npz`, stores their hashes, and explicitly reports that face
consumption is deferred to explicit-atlas export.

The audited MAtCha implementation was also inspected directly. The depth loss
normalizes over the supplied mask; normal and curvature masks are deterministic
finite-difference erosions of it; and `Matcher3D` excludes both invalid source
pixels and target projections that do not land entirely inside valid support.
Thus the common pixel domain is consumed by all enabled loss terms in this
runner, not merely stored in metadata.

The previous atlas exporter could still delete different vertices/faces in
M1 and M2 by applying arm-specific confidence and depth-range filters after
loading the common face mask. In common-domain mode this has been removed:
the exact frozen pixel mask owns vertex inventory, invalid aligned output now
raises, and the exact frozen face mask owns triangle inventory. Arm-specific
stretch, flip or collapse remains present for the geometry gate to measure
instead of being hidden by deletion.

The exporter also consumes the explicit `charts_data.chart_names` order. The
old implicit lexical-order assumption is retained only as a legacy fallback.

## Real strict-v3 status

The current source tree fully replays its authority hash. The existing strict
DAV2/MoGe initializer files and the all-19 comparison domain are valid mapping
diagnostics, but they are not a runnable primary M1/M2 comparison:

- frozen plan content:
  `ed03adf587000f6d18c901df1ac57f3ea2e4e07d9296b31a4b3e5fd06c3df711`;
- selected operational charts: `0`;
- old all-19 domain content:
  `2793d722a23e933bed520f1d2c886bdfc480f0f452c8c43c43899d5fa9fcbe6f`;
- the old domain predates plan binding and is therefore diagnostic-only.

A real adversarial invocation with a deliberately nonexistent MAtCha path was
blocked by `chart plan has no operational submap for alignment` before any GPU
import and created no output directory. This is the correct result.

## Required next run

1. Obtain a non-empty model-neutral v2 plan from the isolated source run.
2. Freeze its content hash in the run authority/command.
3. Rebuild the comparison domain on its exact ordered names; record and pin
   the resulting domain content hash.
4. Run M1 and M2 with the same two expected hashes and exact domain file.
5. Export both atlases from their explicit chart order and the same common
   face domain, then run the strict geometry gate.

No query image, query pose/GT, held geometry, Gaussian, ALIKE or PnP was added
to this chain.

## Verification

- 29 targeted tests pass across strict runner/domain substitution attacks,
  explicit atlas order/topology, full gate, selector/family, RADIO-UV and
  MoGe initializer contracts.
- Negative tests reject a changed expected plan hash, changed expected domain
  hash, changed initializer bytes and changed explicit chart order.
- `git diff --check` is clean.
