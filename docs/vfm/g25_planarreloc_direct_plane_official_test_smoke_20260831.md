# PlanarReloc-style direct-plane RADIO + 2D–3D PnP official-test smoke

## Outcome

The geometry backbone is not the current blocker.  Exact finite-plane support made from
the full-train 2DGS retains 93.52% macro held-ray depth coverage, whereas the old chart
atlas lost most of that support.  The failed deployable path was the identity bridge
`RADIO child posterior -> voxel child -> finite plane`, not the 2DGS surface itself.

A PlanarReloc-style path now treats each fused plane as a directly retrievable map entity:

1. MoGe3 provides query plane masks only; its depth and metric scale are not used.
2. RADIO query-region descriptors retrieve 2,010 finite map planes directly.
3. Source-view RADIO tokens are restricted by exact plane visibility masks.
4. Frozen 2DGS source-view depth lifts matched source pixels to metric world points.
5. PnP-RANSAC + LM estimates the query pose from query 2D pixels and map 3D points.

The 60 MiB direct field contains 13,043 plane observations from 1,105 mapping views and
does not read pose labels or query GT.  File SHA256:
`63bbf12a6e64e136e0b7ef5df7c052601c758445f656fbb67842ebc57201bc94`.

## Frozen official-test smoke

Before opening test pose errors, 12 images were selected uniformly from each of the three
official test routes (seq13/seq3/seq5), for 36 total images.  Test routes are absent from
the 2DGS train split and from the plane visibility atlas.  Query camera intrinsics came
from camera-only artifacts; pose-bearing query contributor members were opened only after
the PnP outputs were frozen.

| Ranking | seq13 2m/45 | seq3 2m/45 | seq5 2m/45 | aggregate |
|---|---:|---:|---:|---:|
| voxel-child posterior -> plane | 1/12 | 0/12 | 0/12 | 1/36 |
| GT-plane insertion oracle | 11/12 | 11/12 | 12/12 | 34/36 |
| direct plane-ID RADIO | 11/12 | 12/12 | 12/12 | **35/36** |

For direct plane-ID RADIO, aggregate median error is 0.233 m / 0.675 degrees, P90 is
0.506 m / 1.666 degrees, 1m/10-degree recall is 35/36, and 0.5m/5-degree recall is
32/36.  Route report SHA256 values are:

- seq13: `264bf5146701094a8a68010ea493267c9d39d3d7bc26a702af0ca65366c3db33`
- seq3: `d9cf62a37adc84fbe6ea437ec9fac4c5ba886815679f47741beb37ae96743d9c`
- seq5: `252902f8c8b0a37c43434eeb87de7ce01deab003eea93bceb8ba855501538532`

These are semantics-sealed reports: their rows and metrics are unchanged from the original
PnP output, while the retrieval label is corrected from the legacy child-to-plane string
to the direct plane-observation contract and the upstream report byte hash is retained.

The sole gross failure is seq13/frame00255.  It has only 32 PnP inliers from 533 frozen
correspondences (ratio 0.060), versus 0.147 or greater for every successful query in this
smoke.  On the separate 88-query seq10 development control, direct-plane PnP reaches
92.05% at 2m/45 degrees and 89.77% at 1m/10 degrees.  A deterministic development-only
grid selects an inlier-ratio threshold of 0.15: it accepts 80/88 poses with 100% 2m/45
precision.  Applied descriptively to the already-open pilot, it rejects the gross failure
and one otherwise-correct low-confidence pose, leaving 34/36 accepted correct poses.
Because this protocol was created after pilot metrics were observed, it is explicitly
eligible only for the still-unseen official-test complement or a new scene, not as a
preregistered claim on these 36 images.

## Full post-pilot complement

After the confidence file above was sealed, the remaining 494 official-test images were
processed.  The exact pilot names were excluded route by route.  This is a prospective
test of the frozen pose/confidence implementation on additional images, but not a pristine
method-selection test: the representation and algorithm had already been developed using
the 36-image pilot.

| Route | complement N | raw 2m/45 | raw 1m/10 | accepted / N | accepted 2m/45 precision |
|---|---:|---:|---:|---:|---:|
| seq13 | 338 | 86.39% | 84.62% | 300/338 | 95.00% |
| seq3 | 86 | 98.84% | 96.51% | 86/86 | 98.84% |
| seq5 | 70 | 98.57% | 97.14% | 69/70 | 100.00% |
| all | 494 | **90.28%** | **88.46%** | 455/494 | **96.48%** |

Among all usable outputs, the median error is 0.198 m / 0.689 degrees.  The frozen
0.15 inlier-ratio rule rejects 28 bad outputs and 7 good outputs, but accepts 16 bad
outputs.  It therefore helps, but is not an adequate production confidence measure.
The complement report is
`direct_radio_plane_pnp_official_test_unseen_complement494_v1.json`, file SHA256
`6eaad66f5b78bb2ea891cd3483dbfedb886b0a21f19a478ca71d1d8e5d1d06b8`.

The failure distribution is structured.  In seq13 frames 126--150, raw 2m/45 recall
falls to 8/24 while the median fraction of query pixels assigned to MoGe3 planes falls
to 19.6%.  Across seq13, good outputs have medians of 31 matched planes and 245 PnP
inliers; failures have medians of 7 planes and 32.5 inliers.  Full 2DGS rendering remains
dense (about 95% in the hard segment), while about 58% of visible 2DGS mass there belongs
to the finite-plane subset.  Thus the hard segment combines weak query planar support
with repeated-surface aliases; it is not evidence that the full 2DGS map disappeared.

There are also high-confidence aliases that inlier ratio cannot detect.  For example,
seq3/frame00040 has 23/93 inliers across only four matched planes but is wrong by
84.4 m / 159 degrees.  A final system needs an independent global-location prior and/or
pose-conditioned render verification, plus an explicit low-support/degeneracy gate.
These rules must be frozen on development or a new scene; the current complement may
only be used to diagnose them.

## Locked Top5/Top10 multi-basin extension

The plane budget itself creates distinct PnP basins.  On the seq10 development route,
Top3/Top5/Top10 plane budgets obtain 86.36% / 92.05% / 94.32% at 2m/45 degrees.
Top5 union Top10 contains a correct pose for 85/88 queries (96.59%).  Selecting the
candidate with the larger PnP inlier ratio obtains 84/88 (95.45%) without reading a
pose label.  Top3 adds no oracle coverage and is dropped.

This exact rule was then frozen and replayed on all official-test candidates.  The
36-image pilot remains excluded from the table below, and this is labelled a
locked-after-baseline historical validation rather than a pristine blind test.

| Route | complement N | Top5 raw 2m/45 | selected raw 2m/45 | selected raw 1m/10 | accepted precision 2m/45 |
|---|---:|---:|---:|---:|---:|
| seq13 | 338 | 86.39% | **88.17%** | 86.09% | 95.36% |
| seq3 | 86 | 98.84% | 98.84% | 96.51% | 98.84% |
| seq5 | 70 | 98.57% | 98.57% | 95.71% | 100.00% |
| all | 494 | 90.28% | **91.50%** | **89.27%** | **96.72%** |

At the frozen inlier-ratio threshold 0.15, 457/494 candidates are accepted and
selective-system recall is 89.47% at 2m/45 degrees, versus 88.87% for the single Top5
baseline.  Top10 is selected for only 42/494 complement queries.  Thus the change is a
bounded basin supplement rather than a replacement of the Top5 operating point.

The label-free selected-pose inventory is
`direct_plane_pnp_official_all530_top5_top10_inlier_selected_v1.npz`, file SHA256
`8c4af27ec7af0c31c8bc033b48988c556480896122f8391feb1471e9aa8b3b12`, content
SHA256 `147df5f05a698b62db4fe115a5327b6959045154f7b7bfdceddf1282844d1215`.
The post-label complement report is
`direct_plane_pnp_official_complement494_top5_top10_inlier_selected_validation_v1.json`,
file SHA256 `c717e5b43b44218408891e39aa7e85fcb300d64c9233951cf2e5f0aae1e5bdc6`,
content SHA256 `014f37fe999fd5e72640353370d7b443fe36d4bcd36a4ea20742cf94f4842445`.

## Global-context and render-verification ablations

A fixed, no-training source-view planar context field was tested as a coarse global
prior.  It raises seq10 plane R@5 from 63.92% to 67.05%, but geometric-mean fusion lowers
final PnP 2m/45 from 92.05% to 89.77% and 1m/10 from 89.77% to 85.23%.  It is therefore
KILL as a primary fusion rule.  The field is retained only as an ablation; file SHA256
`73c5b54a24a995155685a1dea75ccc099d076aa01c0bfaa06f78ea36c9054fed`,
content SHA256 `42cbc3444ad8c86ae3b9c574914d9a0e050570de38b4a1fc9736aa1351afc97a`.

PnP poses are now separately sealed before any GT member is opened.  A post-pose verifier
renders clean 2DGS dominant depth/normals and compares them to MoGe3 after fitting exactly
one median query-depth scale.  On seq10, scale-free median log-depth error has good/bad
pose AUC about 0.95 and normal consistency AUC about 0.83.  Nevertheless, using minimum
render depth error to choose Top5 versus Top10 on the seq13 complement gives 87.57%, below
the 88.17% inlier-ratio selector: repeated facades can have genuinely similar geometry.
Render verification is therefore KILL as the primary candidate selector and remains a
diagnostic/future appearance-fusion signal.  Only the coverage-corrected v2 report is
valid: file SHA256 `71e4e9611a153f3edcc9089dd8a7c83ed1bafb2a7849653330df832a84acd14a`,
content SHA256 `b6d51fbaef966b183244956d8f99fd4c81ae0c8a4f2d1a9a551366ac34b816c4`.

Additional GT-free PnP diagnostics now report inlier image hull/bbox, 2D covariance,
3D spread, reprojection residual, plane/region diversity, and source-view diversity.
On seq10 all seven 2m/45 failures have inlier ratio below 0.15, while only one correct
pose is below that threshold.  The extra signals explain degeneracy but do not justify
another hard threshold on already-open official-test data.

Two stricter RADIO appearance checks were then tested without changing PnP.  A global
descriptor field stores the normalized mean of every RADIO token for all 1,105 mapping
views, and each candidate is compared only with mapping cameras within 10 m / 45 degrees.
It selects Top10 only once and obtains 92.05% / 89.77% at 2m/45 and 1m/10: effectively
the Top5 baseline.  Replacing the mean descriptor by full-image spatial RADIO matching
(fixed 2x2 token pooling, fixed 128-dimensional Rademacher projection, mutual matches and
homography RANSAC) selects Top10 five times and obtains 94.32% / 89.77%.  It remains below
the frozen PnP-inlier-ratio selector at 95.45% / 89.77%, so both are KILL as primary
selectors.  The post-label reports are, respectively, file/content SHA256
`5b39978eef64d27c430afe5030c6820ad439079fabc691787f2d584ba17681a5` /
`d182a0610e6f08a120cdaa5322e50d32d37288d4958f23a4b962c3882af11b8a`
and `5f9d398d18f64b12e013b2530190af46275b5f4f038dd68b531737ee515c8928` /
`e99dab078f95f641fb506fd2fa3fee0374c114efa30f0e4b4531de24e0d093da`.

A grouped multi-hypothesis diagnostic also freezes PnP seeds from the complete match set
and from its physical-plane, source-view, and query-region groups before opening labels.
Neither raw-inlier, capped-balanced-support, nor supported-entity selection exceeds
90.91% at 2m/45.  Even the post-label oracle over every frozen grouped pose is only
90.91% for this Top10 correspondence inventory.  Of the three queries for which both the
original Top5 and Top10 poses fail, grouping recovers frame00059 to 1.16 m, but the best
generated translations for frame00034 and frame00038 remain 5.65 m and 2.36 m.  Thus the
dominant residual is not merely failure to sample a PnP basin: the frozen correspondence
pool itself lacks a correct metric hypothesis for those cases.  The diagnostic report
file/content SHA256 is `3674d830e3cd490412cffbf9c99581a614b2d61dd97408d8b07cf17d69607914` /
`bb1c43dc893908cf593d99722dfa3789777f403a881238b24ccbc560abd683d1`.

The one-best-3D-hypothesis bottleneck was then relaxed in a controlled H3 diagnostic.
Each query token retains at most three score-ordered 3D hypotheses, while candidate-pose
scoring allows that token to contribute only its single smallest valid reprojection
residual.  This avoids multiplying support by duplicating the same image location.  The
frozen candidate-pool oracle rises to 85/88 (96.59%) at 2m/45; frame00038 now has a
1.56 m candidate and frame00059 a 0.33 m candidate, while frame00034 remains unrecovered
(best 5.16 m).  A development-selected deployment rule compares the original Top5/Top10
candidate with the H3 balanced-support candidate by unique-token inlier ratio.  It reaches
86/88 (97.73%) on seq10, versus 84/88 for Top5/Top10, but selects H3 for 77/88 queries.

The rule was frozen and tested once on the interleaved 88-query seq13 shard0.  The existing
Top5/Top10 ratio rule obtains 76/88 at 2m/45 and 75/88 at 1m/10.  H3 obtains the same
76/88 at 2m/45 but only 72/88 at 1m/10.  It therefore fails the held non-regression gate;
the remaining seq13 shards were not run.  H3 is KILL as a default selector, while the
larger frozen candidate pool remains useful evidence for future candidate-level learning.
The seq10 post-label report file/content SHA256 is
`08fb8ecf952a4e909796e92eb7a706f1d973d8dd6332fb15748e05c9e4b57ac0` /
`fe3f943afbccb2d92aae1d480bfa0554c171e37181905e1e881b8a6aefeb32d0`;
the held shard report is
`c9b1e6f5be311ec0c87bd1289478694eab6e4cbb6146eab92567138de5134f9b` /
`bdf44ff093a0369ab20309ada05f75ef33f812a77197c714ced2dd9b60741d1e`.

## Multi-view consensus and canonical metric-UV ablations

Two follow-up tests directly targeted the remaining within-plane correspondence error.
First, the H3 pool ranked a world hypothesis by the number of independent mapping views
supporting the same physical-plane point within a fixed 0.5 m radius.  This remains fully
pose/label-free, but it lowers the seq10 frozen candidate-pool oracle from 85/88 to 84/88.
Frame00034 remains absent (best 5.48 m), and frame00059 degrades from 0.33 m to 1.27 m.
The assumption that repeated nearby observations necessarily identify the same facade
material point is therefore false.  The branch is KILL and was not opened on held data.
The report file/content SHA256 is
`67bfa53794096a67a97cd970de2ac7e8e30f5f4f9272079a1a643e21d04ac1cc` /
`489b2011a29599dea6f84a438865282e90c0a26ada30713ecb06ded971c1292c`.

Second, a PlanarReloc-style canonical metric UV map was implemented.  Mapping 2DGS token
points are projected into each finite plane's metric frame and grouped only by a fixed
0.5 m cell identity.  A cell is retained only with at least two independent mapping
views; 29,950 such texels cover 92.43% of the original mapping token observations.
Each texel stores up to four deterministic anonymous view prototypes, sharing one exact
view-balanced observed UV/3D coordinate.  The grid centre is deliberately not used as
geometry: an initial implementation exposed up to 0.35 m of avoidable tangential
quantization error and was corrected before the authoritative run.

The canonical map has 95,449 appearance prototypes over 29,950 metric texels.  Query
RADIO tokens are matched to these prototypes, filtered by a metric plane homography,
deduplicated by texel, and passed to grouped PnP.  It obtains only 80.68% at 2m/45 and
70.45% at 1m/10; even the post-label oracle over every frozen pose is only 81.82% and
70.45%.  A GT-pose diagnostic explains the loss: the fraction of unique query tokens
having any <=4 px world hypothesis falls from 45.22% for the source-view H3 pool to
33.98% for canonical UV.  Anonymous prototypes increase match count, but do not preserve
the viewpoint-conditioned material identity carried by the source image.

Thus metric plane UV is a sound geometry/identity carrier, but direct averaging or a
small raw-prototype bank of coarse RADIO tokens is not a sufficient canonical local
descriptor.  Further cell-size, radius, or prototype-count tuning is not justified.
The authoritative UV atlas file/content SHA256 is
`c3abf770b06a7a1a543febded8863569479d7de82e38226b89855a0b2c623097` /
`17c9f2128305b14451139a0fbea4d610441cc8857b0de908ca964463c4befcb3`;
the frozen correspondence inventory is
`c822e34c329f192eec27aaa6795165b5e5bac30ec191129e6d383f9589e599e2` /
`42f34a8a4273dfc7241187f98499acdcc3171b5c790e3a24115c232f99d0a165`;
and the evaluation file/content SHA256 is
`8f5e40b449471aaa484dfe4c4000f34d8c24ffa47f2c30b40a52a6ba1cafec5d` /
`fae4dd7e1b658a13aa7f53148d7d0de4458ddc3542fbce0562654b41917e8fef`.

## PlanarReloc-style high-resolution local matching

The failure of coarse RADIO as a canonical UV descriptor does not invalidate the
plane-map/PnP route.  A separate branch now uses RADIO only to retrieve finite physical
planes and their four best mapping source views.  It extracts OpenCV SIFT on the frozen
1024x576 camera canvas, masks query keypoints by MoGe3 plane regions and mapping
keypoints by rendered finite-plane support, applies mutual Lowe-0.8 matching and a
4-pixel per-plane homography RANSAC, and lifts mapping keypoints with the exact rendered
2DGS finite-plane depth before grouped PnP.  It does not use LoFTR, query depth, a query
scale, or a pose/GT member before all candidate poses have been frozen.

The first implementation exposed a real coordinate-contract bug: Cambridge RGB files
are 1920x1080, while contributor cameras and rendered masks are 1024x576.  Raw SIFT
coordinates were therefore clipped into the low-resolution depth grid.  The fixed
implementation area-resizes every RGB image to its frozen camera canvas before feature
extraction.  On 546 real lifted mapping keypoints, lift-and-reproject error is 2.016 px
median, 4.130 px P90, 6.069 px maximum, and 100% within 8 px.  A later contributor/3D
point cache changes no arithmetic: an independent seq13 frame replay reproduces all 22
candidate poses and inlier counts bit-for-bit.

A support-only selector was frozen on a 15-query seq10 smoke: use SIFT when its best
candidate has at least 16 unique-query-keypoint inliers, otherwise retain the existing
Top5/Top10 RADIO pose.  No threshold was changed after opening the routes below.

| Inventory | N | RADIO 2m/45 | SIFT 2m/45 | frozen hybrid 2m/45 | union oracle 2m/45 |
|---|---:|---:|---:|---:|---:|
| seq10 development | 88 | 84/88 | 83/88 | **87/88** | not used for promotion |
| seq13 interleaved shard0 | 88 | 76/88 | **84/88** | 83/88 | 85/88 |
| seq13 interleaved shard1 | 88 | 77/88 | 84/88 | **85/88** | 86/88 |
| seq3 full route | 98 | **97/98** | 94/98 | 96/98 | 97/98 |

Across the two untouched seq13 shards, the frozen hybrid improves 2m/45 from 153/176
(86.93%) to 168/176 (95.45%), and 1m/10 from 150/176 (85.23%) to 166/176 (94.32%).
The pure SIFT branch is also 168/176 at 2m/45.  This is the strongest evidence so far
that a PlanarReloc-like sequence -- plane retrieval, high-resolution local appearance,
metric plane-supported lifting, then PnP -- is viable on this outdoor scene.

It is not yet a universal default.  On the already strong seq3 route the same frozen
hybrid changes 97/98 to 96/98: frame00038 has 33 internally consistent SIFT inliers but
the entire SIFT candidate pool is on the wrong repeated surface (6.69 m error), while
RADIO is correct.  The union oracle equals RADIO at 97/98, so no local-candidate selector
can improve that route; it can only avoid choosing the alias.  Consequently the branch
is GO as a complementary candidate generator and KILL as an unconditional replacement.
Further tuning of the inlier threshold on seq13/seq3 is explicitly disallowed.

A cautious pose-conditioned selector was therefore evaluated without adding a fitted
parameter.  For the RADIO and SIFT pose separately, it takes the mean of the four best
global RADIO similarities among mapping cameras within the already frozen 10 m / 45
degree pose neighbourhood, and ties to RADIO.  This rule had previously been tested for
Top5/Top10 arbitration and its constants were not changed here.  On seq10 it selects
SIFT for only 5/88 queries and obtains 85/88, versus 84/88 for RADIO.  Replayed once on
held data, it obtains 82/88 on seq13 shard0 and shard1, 84/87 on shard2, and 83/87 on
shard3.  All four shard decisions were frozen before labels were opened; shard2/3 were
additionally bound by
`seq13_shard2_shard3_sift_global_context_preexecution_v1.json` before candidate
generation.  Across complete seq13 it selects SIFT for only 39/350 queries and changes:

| seq13 complete | RADIO | pure SIFT | cautious selector | RADIO union SIFT oracle |
|---|---:|---:|---:|---:|
| 2m/45 | 309/350 (88.29%) | 332/350 (94.86%) | **331/350 (94.57%)** | 343/350 (98.00%) |
| 1m/10 | 302/350 (86.29%) | **324/350 (92.57%)** | 319/350 (91.14%) | not evaluated |

The same cautious selector exactly preserves seq3 at 97/98 while selecting SIFT 7/98
times, and improves seq5 from 81/82 to 82/82 while selecting SIFT 6/82 times.  Thus every
held shard/route tested after the rule was frozen is non-regressive at 2m/45, while the
large difficult-route gain is retained.

A final development-only check replaced the global descriptor with the already frozen
spatial RADIO verifier: 2x2 token pooling to 18x32, a fixed 1280-to-128 Rademacher
projection, mutual matches, and homography RANSAC over the four nearest mapping cameras
inside the same 10 m / 45 degree pose neighbourhood.  On seq10 it selects SIFT only 5/88
times and obtains 86/88 (97.73%) at 2m/45 and 82/88 (93.18%) at 1m/10.  This is one query
better than global context, but below the previously frozen support16 development result
of 87/88 and supplies no clear new operating point.  It is therefore KILL as a selector
and was not opened on held routes.  The selected inventory has file SHA256
`19cad0db643ebc00fa2fb016bb17d57b4cdeda80e892dd5f1fd8cc63bc0ca62d`;
the post-label report has file/content SHA256
`a4fb8997d741d509640a7d77e877b1d528475f4ab962431fd0fd137183ea8aab` /
`6e97f6e695b11c896e95a6d4fc9f1b692ca125ae18cd35166b0e2441cc159370`.
No further hand-written scalar selector is justified on this scene.

This conservative selector is the current bounded GO operating point.  It does not
reach the aggressive support16 hybrid's 95.45% recall on the first two seq13 shards, but
it passes the cross-route non-regression test that support16 fails.  It remains
`production_eligible=false` because
the same official-test pool was already used for earlier method diagnosis; a new scene or
pre-execution held route is still required for a promotion claim.  Its seq13 shard0--3
selected inventories have file SHA256
`d66970a6ca8f06785d534b84a577e5e7f0584df7c0ac06fa0783b9a6844660fc`,
`956e085c15fcb947863561ab5f0444190d6669ad115dbce41b3876ffcff60531`,
`c7307ac9fa073c57704644aed7548235abfc4bf9c447e4cd64672595ddf5ebcc`, and
`1887a13438d9ad75bc8ed6a5772bdb2268fa39058f79d7ea4483ba841947d157`.
The seq3/seq5 inventories are
`205c54dbeb2ce1f07fcfae0c9396274b4fbf8555b1cb7fa18f842af5f49b5c1b` and
`42315ef8a412b762257bf5905e4b1c724f40b9fb09873f764a328d6292c261cc`.

Key artifacts are the seq10 candidate bank file SHA256
`c8064644b78077ffb1e1489d9cbd34a3937d47edb480cdc0ed8a347f4d5b4c17` and hybrid
report file SHA256 `520fb8fe9b8625c6e8d7ee3b6cf4ed7195330132530b5abd2d94705c50e6a626`;
seq13 shard0/shard1 SIFT candidate files
`3bad12484cffa2f2d59015b4eb8610cd0daa3fe94af5bbd2d4c26d2423d9a18c` /
`6570e8d415d86cac7a4e394fdeb9ce98107885487d093c5253ede98255c10ed4`, and hybrid
report files `6de8b3228c2deb6e33f2b9b141a86de4f8da33deb2eac27785ab3df308346c8f` /
`685f99f92e5d34b9c6bd5ca81409a91e55e3f1ba62a81543e5ef612389dba574`.

## Bottleneck diagnosis

Post-label evaluation of the old child-to-plane bridge gives weighted plane R@5 of only
5.70% (seq13), 1.69% (seq3), and 2.73% (seq5), with median correct-plane ranks of
279, 208, and 236.  Direct plane-ID RADIO raises weighted R@5 to 63.54%, 76.41%, and
64.57%, respectively.  The pre-existing official-test retrieval audit independently
reports only 19.84% representable visible-mass ceiling for its sparse canonical field.

Therefore the map correction is architectural: retain exact 2DGS-derived finite support,
make planes first-class retrieval identities, and use source-view observations as the
appearance carrier.  The old voxel/child hierarchy may remain as an auxiliary global
retrieval prior, but must not define the plane identity used by the pose backend.

## Remaining limits

- All 530 official-test images have now been evaluated, but 36 are the method-development
  pilot and 494 are the post-pilot complement; they must not be merged into a pristine
  benchmark claim.
- An exact per-plane observation bank reduces route-balanced 12-query runtime from minutes
  to about 20--24 seconds while reproducing all 36 pilot rows bit-for-bit.  It is still
  2.4 GiB and uses about 2.9--5.2 GiB RSS per process, so compact learned/PCA descriptors
  or a memory-mapped layout remain necessary for online deployment.  Bank file SHA256:
  `8cb54f4bc734d3ba00a17bbb0ce9d0b24ec17e27fcac6487dcde6fef2957c7db`.
- Inlier ratio remains the strongest tested deployment-side selector, but 15 accepted
  complement poses are still wrong at 2m/45.  Scale-free geometry rendering does not
  reliably resolve repeated-facade aliases.  Global-mean and spatial RADIO checks also
  fail to improve the selector on development.  The remaining technical target is now
  within-plane tangential correspondence under repeated facades, followed by a new-scene
  blind validation.  H3 proves that alternate metric basins sometimes exist, but its
  simple inlier selector does not generalize; another hand-designed scalar candidate score
  is not justified by current evidence.
- MoGe3 plane segmentation remains relatively expensive; its output quality is adequate
  on most views, but the seq13 failure interval shows that a plane-only backend needs a
  non-planar point/appearance fallback.
