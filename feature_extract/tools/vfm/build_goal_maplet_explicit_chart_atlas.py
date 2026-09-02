from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import build_strict_explicit_chart_atlas
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def cross_chart_nearest(points,rows):
 """Exact nearest point belonging to another chart.

 A fixed global k-neighbour query is not exact: dense samples from the source
 chart can occupy all k entries even when another chart overlaps nearby.
 """
 out=np.full(len(points),np.inf);nearest=np.full(len(points),-1,np.int64)
 for chart in np.unique(rows):
  source=np.flatnonzero(rows==chart);target=np.flatnonzero(rows!=chart)
  if len(source)==0 or len(target)==0:continue
  distance,index=cKDTree(points[target]).query(points[source],k=1,workers=-1)
  out[source]=distance;nearest[source]=target[index]
 return out,nearest
def summarize_distance(x):
 finite=x[np.isfinite(x)];return {'finite_fraction':float(len(finite)/len(x)),'median_m':float(np.median(finite)),'p90_m':float(np.quantile(finite,.9)),'within_0p1m':float(np.mean(x<=.1)),'within_0p25m':float(np.mean(x<=.25)),'within_0p5m':float(np.mean(x<=.5))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--charts',type=Path,required=True);p.add_argument('--cameras',type=Path,required=True);p.add_argument('--alignment_manifest',type=Path,required=True);p.add_argument('--expected_alignment_manifest_content_sha256',required=True);p.add_argument('--initializer_artifacts',type=Path,required=True);p.add_argument('--initializer_manifest',type=Path,required=True);p.add_argument('--expected_initializer_manifest_content_sha256',required=True);p.add_argument('--comparison_domain_v3',type=Path,required=True);p.add_argument('--expected_comparison_domain_content_sha256',required=True);p.add_argument('--upstream_exact_topology_v2',type=Path,required=True);p.add_argument('--expected_upstream_exact_topology_v2_content_sha256',required=True);p.add_argument('--frozen_submap_plan',type=Path,required=True);p.add_argument('--expected_plan_content_sha256',required=True);p.add_argument('--disjoint_upstream_authority',type=Path,required=True);p.add_argument('--expected_disjoint_authority_content_sha256',required=True);p.add_argument('--bounded_submap',type=Path,required=True);p.add_argument('--expected_bounded_submap_content_sha256',required=True);p.add_argument('--initializer_arm',choices=('DAV2','MoGe3'),required=True);p.add_argument('--geometry_state',choices=('initial','aligned'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,choices=(4,8),default=4);a=p.parse_args()
 if a.output.exists():raise FileExistsError('refusing to overwrite strict explicit chart atlas')
 atlas,audit=build_strict_explicit_chart_atlas(a.charts,a.cameras,alignment_manifest_path=a.alignment_manifest,comparison_domain_path=a.comparison_domain_v3,expected_comparison_domain_content_sha256=a.expected_comparison_domain_content_sha256,upstream_exact_topology_v2_path=a.upstream_exact_topology_v2,expected_upstream_exact_topology_v2_content_sha256=a.expected_upstream_exact_topology_v2_content_sha256,frozen_submap_plan_path=a.frozen_submap_plan,expected_plan_content_sha256=a.expected_plan_content_sha256,authority_path=a.disjoint_upstream_authority,expected_authority_content_sha256=a.expected_disjoint_authority_content_sha256,bounded_submap_path=a.bounded_submap,expected_bounded_submap_content_sha256=a.expected_bounded_submap_content_sha256,expected_alignment_manifest_content_sha256=a.expected_alignment_manifest_content_sha256,initializer_artifacts_path=a.initializer_artifacts,initializer_manifest_path=a.initializer_manifest,expected_initializer_manifest_content_sha256=a.expected_initializer_manifest_content_sha256,initializer_arm=a.initializer_arm,geometry_state=a.geometry_state,stride=a.stride);meta=atlas.save_npz(a.output);before,_=cross_chart_nearest(audit['initial_points'],audit['chart_rows']);after,nearest=cross_chart_nearest(audit['aligned_points'],audit['chart_rows'])
 target=nearest.clip(min=0);source_normal_norm=np.linalg.norm(atlas.normals_world,axis=1);target_normal_norm=source_normal_norm[target];overlap=np.isfinite(after)&(after<=1.)&(nearest>=0)&(source_normal_norm>.5)&(target_normal_norm>.5);delta=atlas.vertices_world-atlas.vertices_world[target];p2plane=np.abs(np.sum(delta*atlas.normals_world[target],axis=1));dot=np.abs(np.sum(atlas.normals_world*atlas.normals_world[target],axis=1));angle=np.degrees(np.arccos(np.clip(dot,-1,1)));surface={'overlap_within_1m_fraction':float(np.mean(np.isfinite(after)&(after<=1.))),'overlap_with_valid_normals_fraction':float(overlap.mean()),'point_to_plane_median_m':float(np.median(p2plane[overlap])) if overlap.any() else None,'point_to_plane_p90_m':float(np.quantile(p2plane[overlap],.9)) if overlap.any() else None,'unsigned_normal_angle_median_deg':float(np.median(angle[overlap])) if overlap.any() else None,'unsigned_normal_angle_p90_deg':float(np.quantile(angle[overlap],.9)) if overlap.any() else None}
 report={'artifact_type':'goal_maplet_strict_explicit_chart_atlas_build_audit_v4','output':str(a.output),'output_file_sha256':sha(a.output),'content_sha256':meta['content_sha256'],'initializer_arm':a.initializer_arm,'geometry_state':a.geometry_state,'source_charts_file_sha256':sha(a.charts),'source_cameras_file_sha256':sha(a.cameras),'alignment_manifest_file_sha256':sha(a.alignment_manifest),'initializer_manifest_file_sha256':sha(a.initializer_manifest),'source_authority_file_sha256':sha(a.disjoint_upstream_authority),'comparison_domain_v3_file_sha256':sha(a.comparison_domain_v3),'upstream_exact_topology_v2_file_sha256':sha(a.upstream_exact_topology_v2),'bounded_submap_file_sha256':sha(a.bounded_submap),'source_reference_edge_safe':meta['source_reference_edge_safe'],'full_submap_gate_eligible':meta['full_submap_gate_eligible'],'chart_count':len(atlas.chart_names),'vertex_count':len(atlas.vertices_world),'face_count':len(atlas.faces),'npz_mib':a.output.stat().st_size/2**20,'initial_cross_chart_nearest':summarize_distance(before),'aligned_cross_chart_nearest':summarize_distance(after),'aligned_surface_consistency':surface,'metadata':meta};report_path=a.output.with_suffix('.json');report_path.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in report.items() if k!='metadata'},indent=2))
if __name__=='__main__':main()
