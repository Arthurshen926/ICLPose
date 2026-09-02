"""Validated compact surface-chart atlas exported from MAtCha chart grids."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .chart_comparison_domain import (
 BASE_ARRAY_NAMES as EXACT_DOMAIN_BASE_ARRAY_NAMES,
 SCHEMA as EXACT_DOMAIN_SCHEMA,
 topology_array_names,
 validate_exact_topology_arrays,
)
from .chart_submap_selection import (
 CARDINALITY_SCHEMA as CARDINALITY_PLAN_SCHEMA,
 ChartSubmapPlan,
 load_model_neutral_alignment_selection,
)
from .lineage import arrays_sha256,canonical_json_sha256,file_sha256

SCHEMA='goal_maplet_explicit_surface_chart_atlas_v1'
BOUNDED_DOMAIN_SCHEMA='goal_maplet_axis_aligned_bounded_submap_v1'
PHYSICAL_DOMAIN_SCHEMA='goal_maplet_chart_comparison_domain_v3'

@dataclass(frozen=True)
class ExplicitChartAtlas:
 chart_names:np.ndarray;chart_vertex_offsets:np.ndarray;vertices_world:np.ndarray
 normals_world:np.ndarray;uv:np.ndarray;confidence:np.ndarray
 chart_face_offsets:np.ndarray;faces:np.ndarray;metadata:dict
 def arrays(self):
  return {name:np.asarray(getattr(self,name)) for name in ('chart_names','chart_vertex_offsets','vertices_world','normals_world','uv','confidence','chart_face_offsets','faces')}
 def validated(self):
  a=self.arrays();c=len(a['chart_names']);v=len(a['vertices_world']);f=len(a['faces'])
  if a['chart_vertex_offsets'].shape!=(c+1,) or a['chart_face_offsets'].shape!=(c+1,):raise ValueError('invalid chart offsets')
  if a['chart_vertex_offsets'][0]!=0 or a['chart_vertex_offsets'][-1]!=v or np.any(np.diff(a['chart_vertex_offsets'])<=0):raise ValueError('empty or invalid chart vertices')
  if a['chart_face_offsets'][0]!=0 or a['chart_face_offsets'][-1]!=f or np.any(np.diff(a['chart_face_offsets'])<0):raise ValueError('invalid chart faces')
  if a['vertices_world'].shape!=(v,3) or a['normals_world'].shape!=(v,3) or a['uv'].shape!=(v,2) or a['confidence'].shape!=(v,) or a['faces'].shape!=(f,3):raise ValueError('invalid chart array shapes')
  if not all(np.isfinite(a[x]).all() for x in ('vertices_world','normals_world','uv','confidence')):raise ValueError('nonfinite chart atlas')
  if np.any((a['faces']<0)|(a['faces']>=v)):raise ValueError('chart face outside vertex inventory')
  for row in range(c):
   vlo,vhi=map(int,a['chart_vertex_offsets'][row:row+2]);flo,fhi=map(int,a['chart_face_offsets'][row:row+2])
   if fhi>flo and np.any((a['faces'][flo:fhi]<vlo)|(a['faces'][flo:fhi]>=vhi)):raise ValueError('face crosses chart boundary')
  if self.metadata.get('artifact_type')!=SCHEMA:raise ValueError('wrong chart atlas schema')
  return self
 def save_npz(self,path):
  arrays=self.validated().arrays();meta=dict(self.metadata);meta['arrays_sha256']=arrays_sha256(arrays);meta['content_sha256']=canonical_json_sha256(meta);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.temporary.npz');np.savez_compressed(tmp,**arrays,metadata_json=np.asarray(json.dumps(meta,sort_keys=True)));tmp.replace(path);return meta
 @classmethod
 def load_npz(cls,path):
  with np.load(path,allow_pickle=False) as data:
   meta=json.loads(str(data['metadata_json'].item()));arrays={name:np.asarray(data[name]) for name in ('chart_names','chart_vertex_offsets','vertices_world','normals_world','uv','confidence','chart_face_offsets','faces')}
  if arrays_sha256(arrays)!=meta.get('arrays_sha256'):raise ValueError('chart atlas arrays differ from lineage')
  return cls(metadata=meta,**arrays).validated()

def _chart_camera_binding(pts,depths,scale,cameras,chart_names=None):
 names=[Path(x).name for x in cameras['filepaths']];index={name:i for i,name in enumerate(names)}
 ordered=(sorted(names) if chart_names is None else np.asarray(chart_names).astype(str).tolist())
 if len(ordered)!=len(pts) or len(set(names))!=len(names) or len(set(ordered))!=len(ordered) or not set(ordered).issubset(names):raise ValueError('chart/camera cardinality differs')
 c2w=np.asarray(cameras['cams2world'],np.float64);errors=[]
 yy=np.arange(4,pts.shape[1],max(1,pts.shape[1]//12));xx=np.arange(4,pts.shape[2],max(1,pts.shape[2]//16))
 for chart,name in enumerate(ordered):
  camera=index[name];C=c2w[camera,:3,3]*scale;R=c2w[camera,:3,:3];z=((pts[chart][yy[:,None],xx]-C)@R)[...,2];d=depths[chart][yy[:,None],xx];valid=np.isfinite(z)&np.isfinite(d)&(d>0)&(np.abs(d)<100)
  if valid.sum()<16:raise ValueError('insufficient chart pixels for camera binding replay')
  errors.append(float(np.median(np.abs(z[valid]-d[valid]))))
 if max(errors)>1e-4:raise ValueError('lexical chart order does not replay camera depth')
 return ordered,np.asarray([index[x] for x in ordered],np.int64),np.asarray(errors)

def build_explicit_chart_atlas(charts_path,cameras_path,*,allowed_routes,stride=8,depth_range_ratio=20.,edge_relative_depth=.05,minimum_edge_m=.5,source_authority=None,comparison_domain=None):
 with np.load(charts_path,allow_pickle=False) as data:
  pts=np.asarray(data['pts'],np.float64);depth=np.asarray(data['depths'],np.float64);prior=np.asarray(data['prior_depths'],np.float64);conf=np.asarray(data['confs'],np.float64);scale=float(data['scale_factor']);source_valid=np.asarray(data['valid'],bool) if 'valid' in data.files else np.ones_like(depth,dtype=bool);source_chart_names=np.asarray(data['chart_names']).astype(str) if 'chart_names' in data.files else None
 if source_valid.shape!=depth.shape:raise ValueError('source valid mask differs from chart depths')
 cameras=json.loads(Path(cameras_path).read_text());names,camera_rows,binding_error=_chart_camera_binding(pts,depth,scale,cameras,source_chart_names);c2w=np.asarray(cameras['cams2world'],np.float64)
 routes=[x.split('__',1)[0] for x in names];allowed=set(allowed_routes);selected=[i for i,x in enumerate(routes) if x in allowed];excluded_seen=sorted(set(routes)-allowed)
 comparison_faces=None;comparison_meta=None
 if comparison_domain is not None:
  with np.load(comparison_domain,allow_pickle=False) as domain:
   domain_arrays={name:np.asarray(domain[name]) for name in ('chart_names','valid','face_valid_stride4','face_valid_stride8')};comparison_meta=json.loads(str(domain['metadata_json'].item()))
  claimed=comparison_meta.pop('content_sha256',None)
  if claimed!=canonical_json_sha256(comparison_meta):raise ValueError('comparison domain content hash differs')
  comparison_meta['content_sha256']=claimed
  if comparison_meta.get('arrays_sha256')!=arrays_sha256(domain_arrays):raise ValueError('comparison domain arrays differ')
  selected_names=[names[i] for i in selected]
  if domain_arrays['chart_names'].astype(str).tolist()!=selected_names:raise ValueError('comparison domain chart inventory differs')
  if domain_arrays['valid'].shape!=source_valid[selected].shape or not np.array_equal(domain_arrays['valid'],source_valid[selected]):raise ValueError('aligned source valid mask differs from frozen comparison domain')
  key=f'face_valid_stride{stride}'
  if key not in domain_arrays:raise ValueError('comparison domain does not define requested atlas stride')
  comparison_faces=np.asarray(domain_arrays[key],bool)
 vertices=[];normals=[];uvs=[];confidence=[];faces=[];voff=[0];foff=[0];kept_names=[];valid_fractions=[];initial_samples=[];aligned_samples=[];sample_chart=[]
 for outrow,chart in enumerate(selected):
  d=depth[chart];p=prior[chart];finite=np.isfinite(d)&np.isfinite(p)&np.isfinite(conf[chart])&np.isfinite(pts[chart]).all(2)&(d>0)&(p>0)
  if comparison_faces is not None:
   # The frozen common pixel/face domain owns topology for paired M1/M2.
   # Arm-specific confidence or depth-range filtering here would silently
   # recreate different vertex/face inventories after a common-domain run.
   if np.any(source_valid[chart]&~finite):raise ValueError('aligned geometry is invalid on the frozen comparison domain')
   q=source_valid[chart].copy()
  else:
   q=source_valid[chart]&finite&(conf[chart]>0);med=float(np.median(d[q]));pmed=float(np.median(p[q]));q&=(d<=med*depth_range_ratio)&(d>=med/depth_range_ratio)&(p<=pmed*depth_range_ratio)&(p>=pmed/depth_range_ratio)
  ys=np.arange(0,d.shape[0],stride);xs=np.arange(0,d.shape[1],stride);grid=pts[chart][ys[:,None],xs]/scale;grid_valid=q[ys[:,None],xs];grid_conf=conf[chart][ys[:,None],xs];metric_depth=d[ys[:,None],xs]/scale
  local=np.full(grid_valid.shape,-1,np.int64);local[grid_valid]=np.arange(grid_valid.sum())+voff[-1];v=grid[grid_valid];vertices.append(v);uv=np.stack(np.meshgrid(xs/(d.shape[1]-1),ys/(d.shape[0]-1)),axis=-1);uvs.append(uv[grid_valid]);confidence.append(grid_conf[grid_valid]);valid_fractions.append(float(q.mean()));kept_names.append(names[chart])
  local_faces=[]
  for y in range(len(ys)-1):
   for x in range(len(xs)-1):
    ids=np.asarray([local[y,x],local[y,x+1],local[y+1,x],local[y+1,x+1]])
    if np.any(ids<0) or (comparison_faces is not None and not comparison_faces[outrow,y,x]):continue
    if comparison_faces is not None:
     # The paired M1/M2 table must use exactly the same triangles.  Any
     # post-alignment stretch/flip is measured as a failure rather than being
     # hidden by arm-specific face deletion.
     local_faces.extend(((ids[0],ids[2],ids[1]),(ids[1],ids[2],ids[3])))
    else:
     xyz=np.asarray([grid[y,x],grid[y,x+1],grid[y+1,x],grid[y+1,x+1]]);threshold=max(minimum_edge_m,edge_relative_depth*float(np.median([metric_depth[y,x],metric_depth[y,x+1],metric_depth[y+1,x],metric_depth[y+1,x+1]])))
     if max(np.linalg.norm(xyz[a]-xyz[b]) for a,b in ((0,1),(0,2),(1,3),(2,3),(0,3),(1,2)))<=threshold:local_faces.extend(((ids[0],ids[2],ids[1]),(ids[1],ids[2],ids[3])))
  local_faces=np.asarray(local_faces,np.int64).reshape(-1,3);faces.append(local_faces);n=np.zeros_like(v)
  if len(local_faces):
   rel=local_faces-voff[-1];fn=np.cross(v[rel[:,1]]-v[rel[:,0]],v[rel[:,2]]-v[rel[:,0]]);length=np.linalg.norm(fn,axis=1);fn/=np.maximum(length[:,None],1e-15)
   for corner in range(3):np.add.at(n,rel[:,corner],fn)
  n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-15);normals.append(n)
  C=c2w[camera_rows[chart],:3,3]*scale;ratio=p[ys[:,None],xs]/d[ys[:,None],xs];initial=(C+(pts[chart][ys[:,None],xs]-C)*ratio[...,None])/scale
  initial_samples.append(initial[grid_valid]);aligned_samples.append(v);sample_chart.append(np.full(len(v),outrow,np.int32));voff.append(voff[-1]+len(v));foff.append(foff[-1]+len(local_faces))
 arrays=dict(chart_names=np.asarray(kept_names),chart_vertex_offsets=np.asarray(voff,np.int64),vertices_world=np.concatenate(vertices),normals_world=np.concatenate(normals),uv=np.concatenate(uvs),confidence=np.concatenate(confidence).astype(np.float32),chart_face_offsets=np.asarray(foff,np.int64),faces=np.concatenate(faces) if faces else np.zeros((0,3),np.int64))
 metadata={'artifact_type':SCHEMA,'representation':'explicit_aligned_surface_charts_with_uv','source_charts_path':str(charts_path),'source_cameras_path':str(cameras_path),'source_authority':source_authority,'comparison_domain':str(comparison_domain) if comparison_domain is not None else None,'comparison_domain_content_sha256':comparison_meta['content_sha256'] if comparison_meta is not None else None,'paired_common_face_inventory':comparison_faces is not None,'source_chart_count':len(names),'chart_count':len(kept_names),'source_routes':sorted(set(routes)),'allowed_routes':sorted(allowed),'excluded_routes_seen_during_alignment':excluded_seen,'optimization_saw_excluded_routes':bool(excluded_seen),'control_only':bool(excluded_seen),'stride':stride,'validity':'exact_frozen_common_pixel_domain_with_finite_output_assertion' if comparison_faces is not None else 'source_valid_mask AND finite_positive_confident_and_per_chart_20x_robust_depth_range','edge_break':'exact_frozen_common_face_inventory_no_arm_specific_face_deletion' if comparison_faces is not None else 'all quad diagonals <= max(0.5m,0.05*median_metric_depth)','camera_binding':'charts_data_explicit_order_depth_replay_or_legacy_lexical_fallback','camera_binding_max_median_error_scaled':float(binding_error.max()),'valid_fraction_mean':float(np.mean(valid_fractions)),'scale_factor':scale}
 atlas=ExplicitChartAtlas(metadata=metadata,**arrays).validated()
 audit_inputs={'initial_points':np.concatenate(initial_samples),'aligned_points':np.concatenate(aligned_samples),'chart_rows':np.concatenate(sample_chart)}
 return atlas,audit_inputs

def _replay_metadata(metadata,label):
 claimed=metadata.get('content_sha256');payload=dict(metadata);payload.pop('content_sha256',None)
 if claimed!=canonical_json_sha256(payload):raise ValueError(f'{label} content hash does not replay')
 return claimed

def _require_sha(value,label):
 text=str(value) if value is not None else ''
 if len(text)!=64 or any(character not in '0123456789abcdef' for character in text):raise ValueError(f'{label} is not a lowercase SHA-256')
 return text

def _bounded_submap_hash(minimum,maximum):
 minimum=np.asarray(minimum,np.float64);maximum=np.asarray(maximum,np.float64)
 if minimum.shape!=(3,) or maximum.shape!=(3,) or not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or np.any(minimum>=maximum):raise ValueError('bounded submap has an invalid AABB')
 return canonical_json_sha256({'artifact_type':BOUNDED_DOMAIN_SCHEMA,'minimum_world':minimum.tolist(),'maximum_world':maximum.tolist()})

def _vertex_normals(vertices,faces):
 vertices=np.asarray(vertices,np.float64);faces=np.asarray(faces,np.int64);normals=np.zeros_like(vertices)
 if len(faces):
  face_normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
  lengths=np.linalg.norm(face_normals,axis=1);face_normals/=np.maximum(lengths[:,None],1e-15)
  for corner in range(3):np.add.at(normals,faces[:,corner],face_normals)
 normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-15)
 return normals

def build_strict_explicit_chart_atlas(
 charts_path,
 cameras_path,
 *,
 alignment_manifest_path,
 comparison_domain_path,
 expected_comparison_domain_content_sha256,
 frozen_submap_plan_path,
 expected_plan_content_sha256,
 authority_path,
 expected_authority_content_sha256,
 bounded_submap_path,
 expected_bounded_submap_content_sha256,
 expected_alignment_manifest_content_sha256,
 initializer_artifacts_path,
 initializer_manifest_path,
 expected_initializer_manifest_content_sha256,
 initializer_arm,
 geometry_state,
 stride=8,
 upstream_exact_topology_v2_path=None,
 expected_upstream_exact_topology_v2_content_sha256=None,
 allow_legacy_v2_diagnostic=False,
):
 """Export one initial/aligned atlas on a sealed physical-safe topology.

 The optimizer consumes the paired v1 pixel domain.  This exporter consumes
 a v3 physical-reference-safe topology, externally replays its v2 exact-
 topology parent and the v1 domain recorded by the alignment manifest, and
 uses only the packed v3 vertices/faces.  No topology is inferred from an
 arm's confidence, depth, or geometry at export time.  Legacy v2 is available
 only through an explicit diagnostic opt-in and can never be gate-eligible.
 """
 charts_path=Path(charts_path);cameras_path=Path(cameras_path);alignment_manifest_path=Path(alignment_manifest_path);initializer_artifacts_path=Path(initializer_artifacts_path);initializer_manifest_path=Path(initializer_manifest_path)
 comparison_domain_path=Path(comparison_domain_path);frozen_submap_plan_path=Path(frozen_submap_plan_path);authority_path=Path(authority_path);bounded_submap_path=Path(bounded_submap_path)
 if initializer_arm not in ('DAV2','MoGe3'):raise ValueError('initializer_arm must be DAV2 or MoGe3')
 if geometry_state not in ('initial','aligned'):raise ValueError('geometry_state must be initial or aligned')
 if stride not in (4,8):raise ValueError('strict atlas stride must be 4 or 8')
 if not isinstance(allow_legacy_v2_diagnostic,bool):raise ValueError('legacy v2 diagnostic opt-in must be boolean')
 for value,label in (
  (expected_comparison_domain_content_sha256,'expected comparison-domain hash'),
  (expected_plan_content_sha256,'expected frozen-plan hash'),
  (expected_authority_content_sha256,'expected disjoint-authority hash'),
  (expected_bounded_submap_content_sha256,'expected bounded-submap hash'),
  (expected_alignment_manifest_content_sha256,'expected alignment-manifest hash'),
  (expected_initializer_manifest_content_sha256,'expected initializer-manifest hash'),
 ):_require_sha(value,label)

 authority=json.loads(authority_path.read_text());authority_hash=_replay_metadata(authority,'disjoint authority')
 if authority_hash!=expected_authority_content_sha256:raise ValueError('disjoint authority differs from experiment pin')
 required_authority_flags=('strict_disjoint_upstream','source_held_image_disjoint','source_held_route_disjoint','physical_source_held_input_roots_disjoint')
 if authority.get('artifact_type')!='goal_maplet_disjoint_chart_upstream_authority_v2' or any(authority.get(key) is not True for key in required_authority_flags):raise ValueError('strict atlas requires physically isolated v2 authority')
 if authority.get('uses_query_or_ground_truth') is not False or authority.get('forbidden_routes_opened') is not False:raise ValueError('disjoint authority opened query/GT or forbidden routes')
 source=authority.get('source')
 if not isinstance(source,dict):raise ValueError('disjoint authority lacks source inventory')
 source_names=[str(name) for name in source.get('ordered_names',[])]
 if not source_names or len(set(source_names))!=len(source_names):raise ValueError('disjoint authority source inventory is empty or ambiguous')
 source_names_hash=canonical_json_sha256(source_names)
 coordinate_cameras_hash=_require_sha(authority.get('posed_colmap_cameras_file_sha256'),'coordinate-camera hash')
 coordinate_images_hash=_require_sha(authority.get('posed_colmap_images_file_sha256'),'coordinate-image hash')

 plan=ChartSubmapPlan.load_npz(frozen_submap_plan_path)
 selection=load_model_neutral_alignment_selection(frozen_submap_plan_path,expected_plan_content_sha256=expected_plan_content_sha256)
 if len(selection.operational_submaps)!=1:raise ValueError('strict atlas requires one operational frozen submap')
 selected_names=list(selection.ordered_names)
 if plan.metadata.get('artifact_type')==CARDINALITY_PLAN_SCHEMA:
  cardinality_config=plan.metadata.get('config')
  if not isinstance(cardinality_config,dict) or cardinality_config.get('minimum_selected_charts_per_submap')!=16 or cardinality_config.get('maximum_selected_charts_per_submap')!=16 or len(selected_names)!=16 or plan.metadata.get('selection_cardinality_frozen_before_held_geometry') is not True or plan.metadata.get('held_geometry_used_for_selection') is not False:raise ValueError('v3 chart plan is not a held-free exact-16 selection authority')
 if plan.chart_names.astype(str).tolist()!=source_names or plan.metadata.get('source_ordered_names_sha256')!=source_names_hash:raise ValueError('frozen plan source inventory differs from authority')
 lineage=plan.metadata.get('lineage')
 if not isinstance(lineage,dict) or lineage.get('disjoint_authority_content_sha256')!=authority_hash or lineage.get('source_tree_sha256')!=source.get('tree_sha256'):raise ValueError('frozen plan lineage differs from authority')

 with np.load(comparison_domain_path,allow_pickle=False) as data:
  base_arrays={name:np.asarray(data[name]) for name in EXACT_DOMAIN_BASE_ARRAY_NAMES}
  topology_arrays={name:np.asarray(data[name]) for name in topology_array_names()}
  domain_metadata=json.loads(str(data['metadata_json'].item()))
 domain_hash=_replay_metadata(domain_metadata,'sealed comparison domain')
 if domain_hash!=expected_comparison_domain_content_sha256:raise ValueError('sealed comparison domain differs from experiment pin')
 domain_schema=domain_metadata.get('artifact_type')
 if domain_schema not in (PHYSICAL_DOMAIN_SCHEMA,EXACT_DOMAIN_SCHEMA):raise ValueError('comparison domain is not a sealed v3/v2 exact topology')
 if domain_schema==EXACT_DOMAIN_SCHEMA and not allow_legacy_v2_diagnostic:raise ValueError('v2 exact topology is legacy diagnostic only; a physical-safe v3 domain is required')
 if domain_schema==PHYSICAL_DOMAIN_SCHEMA and stride!=4:raise ValueError('physical-safe v3 atlas export is formally restricted to primary stride 4')
 all_domain_arrays={**base_arrays,**topology_arrays}
 if domain_metadata.get('arrays_sha256')!=arrays_sha256(all_domain_arrays):raise ValueError('sealed comparison-domain arrays differ')
 validate_exact_topology_arrays(base_arrays,topology_arrays,expected_sha256=domain_metadata.get('exact_topology_arrays_sha256'))
 if domain_metadata.get('exact_pixel_mask_frozen_for_both_arms') is not True or domain_metadata.get('exact_face_indices_frozen_for_both_arms') is not True:raise ValueError('comparison domain lacks a paired exact pixel/face seal')

 upstream_v2_metadata=None
 if domain_schema==PHYSICAL_DOMAIN_SCHEMA:
  if upstream_exact_topology_v2_path is None or expected_upstream_exact_topology_v2_content_sha256 is None:raise ValueError('v3 comparison domain requires an externally pinned upstream v2 artifact')
  upstream_exact_topology_v2_path=Path(upstream_exact_topology_v2_path)
  _require_sha(expected_upstream_exact_topology_v2_content_sha256,'expected upstream v2 comparison-domain hash')
  with np.load(upstream_exact_topology_v2_path,allow_pickle=False) as data:
   upstream_v2_base={name:np.asarray(data[name]) for name in EXACT_DOMAIN_BASE_ARRAY_NAMES}
   upstream_v2_topology={name:np.asarray(data[name]) for name in topology_array_names()}
   upstream_v2_metadata=json.loads(str(data['metadata_json'].item()))
  upstream_v2_hash=_replay_metadata(upstream_v2_metadata,'upstream exact-topology v2 domain')
  if upstream_v2_hash!=expected_upstream_exact_topology_v2_content_sha256 or upstream_v2_metadata.get('artifact_type')!=EXACT_DOMAIN_SCHEMA:raise ValueError('externally pinned upstream exact-topology v2 domain differs')
  upstream_v2_arrays={**upstream_v2_base,**upstream_v2_topology}
  if upstream_v2_metadata.get('arrays_sha256')!=arrays_sha256(upstream_v2_arrays):raise ValueError('upstream exact-topology v2 arrays differ')
  validate_exact_topology_arrays(upstream_v2_base,upstream_v2_topology,expected_sha256=upstream_v2_metadata.get('exact_topology_arrays_sha256'))
  upstream_requirements={
   'upstream_exact_topology_v2_file_sha256':file_sha256(upstream_exact_topology_v2_path),
   'upstream_exact_topology_v2_content_sha256':upstream_v2_hash,
   'upstream_exact_topology_v2_arrays_sha256':upstream_v2_metadata.get('arrays_sha256'),
   'upstream_exact_topology_v2_exact_topology_arrays_sha256':upstream_v2_metadata.get('exact_topology_arrays_sha256'),
   'upstream_optimizer_comparison_domain_v1_file_sha256':upstream_v2_metadata.get('upstream_comparison_domain_file_sha256'),
   'upstream_optimizer_comparison_domain_v1_content_sha256':upstream_v2_metadata.get('upstream_comparison_domain_content_sha256'),
   'upstream_optimizer_comparison_domain_v1_arrays_sha256':upstream_v2_metadata.get('base_domain_arrays_sha256'),
  }
  for key,expected in upstream_requirements.items():
   if domain_metadata.get(key)!=expected:raise ValueError(f'physical-safe v3 comparison domain {key} differs')
  if not np.array_equal(base_arrays['chart_names'],upstream_v2_base['chart_names']) or not np.array_equal(base_arrays['valid'],upstream_v2_base['valid']):raise ValueError('physical-safe v3 changed v2 chart/pixel inventory')
  for stride_value in (4,8):
   final_mask=np.asarray(base_arrays[f'face_valid_stride{stride_value}'],bool);upstream_mask=np.asarray(upstream_v2_base[f'face_valid_stride{stride_value}'],bool)
   if final_mask.shape!=upstream_mask.shape or np.any(final_mask&~upstream_mask):raise ValueError('physical-safe v3 face topology is not a subset of v2')
  required_physical_flags=(
   'source_reference_edge_safe','physical_face_safety_authority',
   'face_valid_v3_subset_of_face_valid_v2',
   'exact_topology_repacked_after_reference_safety',
   'valid_and_chart_names_byte_equal_upstream_v2',
   'source_reference_edge_safety_replayed',
  )
  if domain_metadata.get('full_submap_gate_eligible') is not True or domain_metadata.get('uses_query_or_ground_truth') is not False or any(domain_metadata.get(key) is not True for key in required_physical_flags) or domain_metadata.get('orphan_sampled_vertices_present') is not False:raise ValueError('v3 domain lacks physical-reference safety authority')
  stride_contract_fields=(
   'full_submap_gate_eligible_strides','required_nonempty_face_inventory_strides',
   'full_submap_gate_primary_stride','noneligible_stride_empty_inventory_permitted',
   'face_quad_count_per_chart_by_stride','triangle_count_per_chart_by_stride',
   'packed_vertex_count_per_chart_by_stride','empty_face_chart_names_by_stride',
  )
  if any(key in domain_metadata for key in stride_contract_fields):
   if domain_metadata.get('full_submap_gate_eligible_strides')!=[4] or domain_metadata.get('required_nonempty_face_inventory_strides')!=[4] or domain_metadata.get('full_submap_gate_primary_stride')!=4 or domain_metadata.get('noneligible_stride_empty_inventory_permitted') is not True:raise ValueError('v3 full-submap gate stride authority differs')
   count_fields={
    'face_quad_count_per_chart_by_stride':{},
    'triangle_count_per_chart_by_stride':{},
    'packed_vertex_count_per_chart_by_stride':{},
    'empty_face_chart_names_by_stride':{},
   }
   for stride_value in (4,8):
    key=f'stride{stride_value}';face_count=np.diff(np.asarray(topology_arrays[f'face_offsets_stride{stride_value}'],np.int64));vertex_count=np.diff(np.asarray(topology_arrays[f'sampled_vertex_offsets_stride{stride_value}'],np.int64))
    count_fields['face_quad_count_per_chart_by_stride'][key]=(face_count//2).tolist()
    count_fields['triangle_count_per_chart_by_stride'][key]=face_count.tolist()
    count_fields['packed_vertex_count_per_chart_by_stride'][key]=vertex_count.tolist()
    count_fields['empty_face_chart_names_by_stride'][key]=[selected_names[row] for row,value in enumerate(face_count) if value==0]
   for key,expected in count_fields.items():
    if domain_metadata.get(key)!=expected:raise ValueError(f'v3 {key} does not replay packed topology')
   if any(value==0 for value in count_fields['triangle_count_per_chart_by_stride']['stride4']):raise ValueError('v3 primary stride has an empty chart face inventory')
  face_masks={f'face_valid_stride{value}':np.asarray(base_arrays[f'face_valid_stride{value}'],bool) for value in (4,8)}
  parent_face_masks={f'face_valid_stride{value}':np.asarray(upstream_v2_base[f'face_valid_stride{value}'],bool) for value in (4,8)}
  final_face_hash=arrays_sha256(face_masks);parent_face_hash=arrays_sha256(parent_face_masks)
  if domain_metadata.get('source_reference_edge_safe_face_masks_sha256')!=final_face_hash or domain_metadata.get('final_reference_safe_face_masks_sha256')!=final_face_hash or domain_metadata.get('parent_v2_face_valid_arrays_sha256')!=parent_face_hash:raise ValueError('v3 physical-reference face-mask hash does not replay')
  safety_config=domain_metadata.get('source_reference_edge_safety_config')
  if not isinstance(safety_config,dict) or domain_metadata.get('source_reference_edge_safety_config_sha256')!=canonical_json_sha256(safety_config):raise ValueError('v3 source-reference safety configuration does not replay')
  source_reference_root=Path(str(domain_metadata.get('source_reference_root',''))).resolve()
  if source_reference_root!=Path(str(source.get('root',''))).resolve():raise ValueError('v3 source-reference root differs from disjoint authority')
  source_reference_cameras=source_reference_root/'cameras.json'
  source_reference_cameras_hash=_require_sha(domain_metadata.get('source_reference_cameras_file_sha256'),'v3 source-reference camera hash')
  if not source_reference_cameras.is_file() or file_sha256(source_reference_cameras)!=source_reference_cameras_hash or source_reference_cameras_hash!=source.get('cameras_file_sha256'):raise ValueError('v3 source-reference camera bytes differ from disjoint authority')
  selected_pointmaps=domain_metadata.get('source_reference_selected_pointmap_inventory')
  if not isinstance(selected_pointmaps,dict) or set(selected_pointmaps)!=set(selected_names) or domain_metadata.get('pointmap_inventory')!=selected_pointmaps or domain_metadata.get('source_reference_selected_pointmap_inventory_sha256')!=canonical_json_sha256(selected_pointmaps):raise ValueError('v3 selected source-reference pointmap inventory differs')
  for name in selected_names:
   pointmap_path=source_reference_root/'pointmaps'/Path(name).with_suffix('.json').name
   if not pointmap_path.is_file() or file_sha256(pointmap_path)!=selected_pointmaps[name]:raise ValueError(f'v3 source-reference pointmap bytes differ for {name}')
  if domain_metadata.get('source_reference_full_pointmap_inventory_sha256')!=source.get('pointmap_inventory_sha256'):raise ValueError('v3 full source-reference pointmap inventory differs from authority')
  initializer_domain_metadata=upstream_v2_metadata
  upstream_v1_content=_require_sha(upstream_v2_metadata.get('upstream_comparison_domain_content_sha256'),'upstream v1 comparison-domain hash')
  upstream_v1_file=_require_sha(upstream_v2_metadata.get('upstream_comparison_domain_file_sha256'),'upstream v1 comparison-domain file hash')
 else:
  if upstream_exact_topology_v2_path is not None or expected_upstream_exact_topology_v2_content_sha256 is not None:raise ValueError('legacy v2 diagnostic cannot claim a separate upstream v2 artifact')
  initializer_domain_metadata=domain_metadata
  upstream_v1_content=_require_sha(domain_metadata.get('upstream_comparison_domain_content_sha256'),'upstream v1 comparison-domain hash')
  upstream_v1_file=_require_sha(domain_metadata.get('upstream_comparison_domain_file_sha256'),'upstream v1 comparison-domain file hash')
 if base_arrays['chart_names'].astype(str).tolist()!=selected_names:raise ValueError('sealed comparison domain differs from frozen chart order')
 domain_requirements={
  'disjoint_upstream_authority_file_sha256':file_sha256(authority_path),
  'disjoint_upstream_authority_content_sha256':authority_hash,
  'source_tree_sha256':source.get('tree_sha256'),
  'mapping_source_ordered_names_sha256':source_names_hash,
  'frozen_submap_plan_file_sha256':file_sha256(frozen_submap_plan_path),
  'frozen_submap_plan_content_sha256':expected_plan_content_sha256,
  'selected_chart_names_in_order_sha256':plan.metadata.get('selected_chart_names_in_order_sha256'),
 }
 for key,expected in domain_requirements.items():
  if domain_metadata.get(key)!=expected:raise ValueError(f'sealed comparison domain {key} differs')

 manifest=json.loads(alignment_manifest_path.read_text());manifest_hash=_replay_metadata(manifest,'alignment manifest')
 if manifest_hash!=expected_alignment_manifest_content_sha256:raise ValueError('alignment manifest differs from experiment pin')
 expected_initializer={'DAV2':'dav2','MoGe3':'moge3'}[initializer_arm]
 if manifest.get('artifact_type')!='goal_maplet_masked_chart_alignment_gate_v1' or manifest.get('initializer')!=expected_initializer:raise ValueError('alignment manifest initializer arm differs')
 if manifest.get('uses_query_or_ground_truth') is not False or manifest.get('paired_common_pixel_domain') is not True or manifest.get('common_face_inventory_deferred_to_explicit_atlas_export') is not True or manifest.get('output_restored_to_frozen_plan_order') is not True:raise ValueError('alignment manifest lacks strict paired/export order contract')
 if manifest.get('selection_contract')!=plan.metadata.get('alignment_runner_contract'):raise ValueError('alignment manifest runner contract differs from frozen plan')
 _require_sha(manifest.get('alignment_runner_file_sha256'),'alignment runner file hash');_require_sha(manifest.get('alignment_code_inventory_sha256'),'alignment code inventory hash')
 manifest_requirements={
  'chart_names':selected_names,
  'disjoint_upstream_authority_content_sha256':authority_hash,
  'comparison_domain_content_sha256':upstream_v1_content,
  'comparison_domain_file_sha256':upstream_v1_file,
  'frozen_submap_plan_content_sha256':expected_plan_content_sha256,
  'selected_chart_names_in_order_sha256':plan.metadata.get('selected_chart_names_in_order_sha256'),
  'charts_data_file_sha256':file_sha256(charts_path),
  'subset_cameras_file_sha256':file_sha256(cameras_path),
 }
 for key,expected in manifest_requirements.items():
  if manifest.get(key)!=expected:raise ValueError(f'alignment manifest {key} differs')
 initializer_hashes=manifest.get('initializer_file_sha256')
 if not isinstance(initializer_hashes,list) or len(initializer_hashes)!=len(selected_names):raise ValueError('alignment manifest initializer inventory is not per selected chart')
 initializer_manifest=json.loads(initializer_manifest_path.read_text());initializer_manifest_hash=_replay_metadata(initializer_manifest,'initializer run manifest')
 if initializer_manifest_hash!=expected_initializer_manifest_content_sha256:raise ValueError('initializer run manifest differs from experiment pin')
 initializer_manifest_key='dav2' if initializer_arm=='DAV2' else 'moge'
 expected_run_schema='goal_maplet_dav2_chart_initializer_run_v1' if initializer_arm=='DAV2' else 'goal_maplet_moge3_chart_initializer_run_v2'
 if initializer_manifest.get('artifact_type')!=expected_run_schema or initializer_manifest.get('uses_query_or_ground_truth') is not False:raise ValueError('initializer run manifest has the wrong arm/semantics')
 if initializer_arm=='DAV2' and initializer_manifest.get('disjoint_upstream_authority_content_sha256')!=authority_hash:raise ValueError('DAV2 initializer manifest authority differs')
 domain_manifest_requirements={
  f'{initializer_manifest_key}_initializer_manifest_file_sha256':file_sha256(initializer_manifest_path),
  f'{initializer_manifest_key}_initializer_manifest_content_sha256':initializer_manifest_hash,
 }
 for key,expected in domain_manifest_requirements.items():
  if initializer_domain_metadata.get(key)!=expected:raise ValueError(f'sealed comparison domain lineage {key} differs')
 rows=initializer_manifest.get('rows')
 if not isinstance(rows,list) or initializer_manifest.get('chart_count')!=len(rows):raise ValueError('initializer manifest row inventory differs')
 initializer_rows={str(row.get('name','')):row for row in rows if isinstance(row,dict)}
 if '' in initializer_rows or len(initializer_rows)!=len(rows):raise ValueError('initializer manifest contains empty/duplicate chart names')

 bounds=json.loads(bounded_submap_path.read_text());_replay_metadata(bounds,'bounded submap')
 minimum=np.asarray(bounds.get('minimum_world'),np.float64);maximum=np.asarray(bounds.get('maximum_world'),np.float64);bounded_hash=_bounded_submap_hash(minimum,maximum)
 if bounded_hash!=expected_bounded_submap_content_sha256 or bounds.get('bounded_submap_content_sha256')!=bounded_hash:raise ValueError('bounded submap differs from experiment pin')
 if bounds.get('artifact_type')!=BOUNDED_DOMAIN_SCHEMA or bounds.get('uses_query_or_ground_truth') is not False or bounds.get('held_geometry_consumed') is not False:raise ValueError('bounded submap is not frozen from source-only evidence')
 bounds_requirements={
  'disjoint_upstream_authority_file_sha256':file_sha256(authority_path),
  'disjoint_upstream_authority_content_sha256':authority_hash,
  'mapping_source_ordered_names_sha256':source_names_hash,
  'comparison_domain_file_sha256':file_sha256(comparison_domain_path),
  'comparison_domain_content_sha256':domain_hash,
  'frozen_submap_plan_file_sha256':file_sha256(frozen_submap_plan_path),
  'frozen_submap_plan_content_sha256':expected_plan_content_sha256,
  'coordinate_cameras_file_sha256':coordinate_cameras_hash,
  'coordinate_images_file_sha256':coordinate_images_hash,
 }
 for key,expected in bounds_requirements.items():
  if bounds.get(key)!=expected:raise ValueError(f'bounded submap {key} differs')
 if bounds.get('source_bound_derivation_schema')!='goal_maplet_source_only_submap_bound_derivation_v1' or _require_sha(bounds.get('pre_frozen_builder_config_content_sha256'),'source-bound config hash')!=bounds.get('pre_frozen_builder_config_content_sha256') or not np.isfinite(float(bounds.get('source_bound_margin_m',np.nan))) or float(bounds.get('source_bound_margin_m'))<=0:raise ValueError('bounded submap lacks source-only derivation authority')

 required_chart_arrays=('pts','depths','prior_depths','confs','valid','chart_names','scale_factor','comparison_face_valid_stride4','comparison_face_valid_stride8')
 with np.load(charts_path,allow_pickle=False) as data:
  if any(name not in data.files for name in required_chart_arrays):raise ValueError('charts_data lacks strict paired arrays')
  points=np.asarray(data['pts'],np.float64);depths=np.asarray(data['depths'],np.float64);prior=np.asarray(data['prior_depths'],np.float64);confs=np.asarray(data['confs'],np.float64);source_valid=np.asarray(data['valid'],bool);chart_names=np.asarray(data['chart_names']).astype(str);scale=float(data['scale_factor'])
  source_faces={stride_value:np.asarray(data[f'comparison_face_valid_stride{stride_value}'],bool) for stride_value in (4,8)}
 if chart_names.tolist()!=selected_names:raise ValueError('charts_data differs from frozen chart order')
 if points.ndim!=4 or points.shape[-1]!=3 or depths.shape!=points.shape[:3] or prior.shape!=depths.shape or confs.shape!=depths.shape or source_valid.shape!=depths.shape or not np.isfinite(scale) or scale<=0:raise ValueError('charts_data grid contract differs')
 if source_valid.shape!=base_arrays['valid'].shape or not np.array_equal(source_valid,base_arrays['valid']):raise ValueError('charts_data pixel mask differs from sealed comparison domain')
 for stride_value in (4,8):
  sealed_faces=np.asarray(base_arrays[f'face_valid_stride{stride_value}'],bool)
  if domain_schema==EXACT_DOMAIN_SCHEMA:
   if not np.array_equal(source_faces[stride_value],sealed_faces):raise ValueError('charts_data face mask differs from legacy v2 comparison domain')
  elif sealed_faces.shape!=source_faces[stride_value].shape or np.any(sealed_faces&~source_faces[stride_value]):raise ValueError('physical-safe v3 face mask is not a subset of optimizer v1 faces')
 if manifest.get('common_pixel_mask_sha256')!=arrays_sha256({'valid':source_valid}):raise ValueError('charts_data pixel mask differs from alignment manifest')
 manifest_face_hash=manifest.get('common_face_mask_sha256')
 if not isinstance(manifest_face_hash,dict):raise ValueError('alignment manifest lacks common face hashes')
 for stride_value in (4,8):
  expected_face_hash=arrays_sha256({f'face_valid_stride{stride_value}':source_faces[stride_value]})
  if manifest_face_hash.get(f'stride{stride_value}')!=expected_face_hash:raise ValueError('charts_data face mask differs from alignment manifest')

 cameras=json.loads(cameras_path.read_text());camera_names=[Path(path).name for path in cameras.get('filepaths',[])]
 if camera_names!=selected_names or len(cameras.get('focals',[]))!=len(selected_names) or len(cameras.get('cams2world',[]))!=len(selected_names):raise ValueError('subset cameras differ from frozen chart order')
 c2w=np.asarray(cameras['cams2world'],np.float64)
 if c2w.shape!=(len(selected_names),4,4) or not np.isfinite(c2w).all():raise ValueError('subset camera poses are invalid')
 initial_metric=[];initializer_content_hashes=[]
 for chart,name in enumerate(selected_names):
  initializer_path=initializer_artifacts_path/f'{name}.npz'
  if file_sha256(initializer_path)!=initializer_hashes[chart]:raise ValueError('initializer bytes differ from alignment manifest')
  with np.load(initializer_path,allow_pickle=False) as data:
   initializer_metadata=json.loads(str(data['metadata_json'].item()));initializer_valid=np.asarray(data['valid'],bool)
   if initializer_arm=='DAV2':initializer_points=np.asarray(data['points_world'],np.float64)
   else:initializer_points=np.asarray(data['points_camera'],np.float64)
  initializer_content_hashes.append(_replay_metadata(initializer_metadata,f'{initializer_arm} initializer {name}'))
  initializer_row=initializer_rows.get(name)
  if not isinstance(initializer_row,dict) or initializer_row.get('file_sha256')!=initializer_hashes[chart] or initializer_row.get('content_sha256')!=initializer_content_hashes[-1]:raise ValueError('initializer file/content differs from run manifest row')
  if initializer_valid.shape!=depths.shape[1:] or initializer_points.shape!=points.shape[1:] or np.any(source_valid[chart]&~initializer_valid):raise ValueError('initializer grid/validity differs from alignment common pixels')
  if initializer_arm=='DAV2':
   if initializer_metadata.get('artifact_type')!='goal_maplet_dav2_chart_initializer_v1' or initializer_metadata.get('source_name')!=name or initializer_metadata.get('disjoint_upstream_authority_content_sha256')!=authority_hash or initializer_metadata.get('uses_query_or_ground_truth') is not False:raise ValueError('DAV2 initializer lineage differs')
   world=initializer_points
  else:
   camera_source_path=Path(cameras['filepaths'][chart])
   if initializer_metadata.get('artifact_type')!='goal_maplet_moge3_chart_initializer_v2' or initializer_metadata.get('source_name')!=name or initializer_metadata.get('uses_camera_pose') is not False or initializer_metadata.get('uses_query_or_ground_truth') is not False or initializer_metadata.get('source_image_file_sha256')!=file_sha256(camera_source_path):raise ValueError('MoGe3 initializer lineage differs')
   world=initializer_points@c2w[chart,:3,:3].T+c2w[chart,:3,3]
  if not np.isfinite(world[source_valid[chart]]).all():raise ValueError('initializer is nonfinite on frozen common pixels')
  initial_metric.append(world)
 initial_metric=np.stack(initial_metric)
 centers=c2w[:,:3,3]*scale;rotations=c2w[:,:3,:3]
 replay_depth=np.einsum('nhwc,ncd->nhwd',points-centers[:,None,None,:],rotations)[...,2]
 replay_support=source_valid
 if not np.isfinite(points[replay_support]).all() or not np.isfinite(depths[replay_support]).all() or not np.isfinite(prior[replay_support]).all() or not np.isfinite(confs[replay_support]).all() or np.any(depths[replay_support]<=0) or np.any(prior[replay_support]<=0):raise ValueError('charts_data is nonfinite/nonpositive on frozen pixels')
 max_depth_error=float(np.max(np.abs(replay_depth[replay_support]-depths[replay_support])))
 if max_depth_error>1e-3:raise ValueError('charts_data camera/depth binding does not replay')
 initializer_depth=np.einsum('nhwc,ncd->nhwd',initial_metric-c2w[:,:3,3][:,None,None,:],rotations)[...,2]*scale
 max_prior_error=float(np.max(np.abs(initializer_depth[replay_support]-prior[replay_support])))
 if max_prior_error>1e-3:raise ValueError('raw initializer does not replay frozen prior depth')
 aligned_metric=points/scale

 vertex_offsets=np.asarray(topology_arrays[f'sampled_vertex_offsets_stride{stride}'],np.int64)
 pixel_indices=np.asarray(topology_arrays[f'sampled_vertex_pixel_indices_stride{stride}'],np.int64)
 face_offsets=np.asarray(topology_arrays[f'face_offsets_stride{stride}'],np.int64)
 faces=np.asarray(topology_arrays[f'faces_stride{stride}'],np.int64)
 height,width=depths.shape[1:];vertices=[];initial_vertices=[];aligned_vertices=[];uvs=[];confidence=[];chart_rows=[]
 for chart in range(len(selected_names)):
  lo,hi=map(int,vertex_offsets[chart:chart+2]);pixels=pixel_indices[lo:hi];yy=pixels//width;xx=pixels%width
  if np.any(pixels<0) or np.any(pixels>=height*width) or not source_valid[chart,yy,xx].all():raise ValueError('sealed topology references pixels outside frozen validity')
  initial_chart=initial_metric[chart,yy,xx];aligned_chart=aligned_metric[chart,yy,xx]
  initial_vertices.append(initial_chart);aligned_vertices.append(aligned_chart);vertices.append(initial_chart if geometry_state=='initial' else aligned_chart)
  uvs.append(np.stack((xx/(width-1),yy/(height-1)),axis=1));confidence.append(confs[chart,yy,xx].astype(np.float32));chart_rows.append(np.full(len(pixels),chart,np.int32))
 vertices=np.concatenate(vertices);initial_vertices=np.concatenate(initial_vertices);aligned_vertices=np.concatenate(aligned_vertices);uv=np.concatenate(uvs);confidence=np.concatenate(confidence);chart_rows=np.concatenate(chart_rows)
 if not np.isfinite(vertices).all() or np.any(vertices<minimum-1e-6) or np.any(vertices>maximum+1e-6):raise ValueError('strict atlas geometry is nonfinite or escapes bounded submap')
 atlas_arrays=dict(chart_names=np.asarray(selected_names),chart_vertex_offsets=vertex_offsets,vertices_world=vertices,normals_world=_vertex_normals(vertices,faces),uv=uv,confidence=confidence,chart_face_offsets=face_offsets,faces=faces)
 selected_initializer_inventory=[{'name':name,'file_sha256':initializer_hashes[row],'content_sha256':initializer_content_hashes[row]} for row,name in enumerate(selected_names)]
 metadata={
  'artifact_type':SCHEMA,'representation':'strict_explicit_aligned_surface_charts_with_exact_uv_topology','uses_query_or_ground_truth':False,
  'initializer_arm':initializer_arm,'is_pre_alignment_geometry':geometry_state=='initial','geometry_state':geometry_state,
  'initial_geometry_contract':'raw per-chart initializer bytes consumed by alignment' if geometry_state=='initial' else None,
  'paired_common_face_inventory':True,'exact_topology_consumed':True,'full_submap_gate_eligible':domain_schema==PHYSICAL_DOMAIN_SCHEMA,'source_reference_edge_safe':domain_schema==PHYSICAL_DOMAIN_SCHEMA,'legacy_v2_diagnostic':domain_schema==EXACT_DOMAIN_SCHEMA,'stride':stride,
  'chart_count':len(selected_names),'source_chart_count':len(selected_names),'source_routes':sorted(set(name.split('__',1)[0] for name in selected_names)),
  'optimization_saw_excluded_routes':False,'control_only':False,'comparison_budget':'exact_frozen_source_ordered_pool',
  'held_mapping_images_consumed':False,'outside_frozen_source_mapping_images_consumed':False,
  'disjoint_upstream_authority_content_sha256':authority_hash,'disjoint_upstream_authority_file_sha256':file_sha256(authority_path),
  'mapping_source_ordered_names_sha256':source_names_hash,'source_tree_sha256':source.get('tree_sha256'),
  'coordinate_cameras_file_sha256':coordinate_cameras_hash,'coordinate_images_file_sha256':coordinate_images_hash,
  'comparison_domain_artifact_type':domain_schema,'comparison_domain_content_sha256':domain_hash,'comparison_domain_file_sha256':file_sha256(comparison_domain_path),
  'alignment_upstream_comparison_domain_content_sha256':upstream_v1_content,'alignment_upstream_comparison_domain_file_sha256':upstream_v1_file,
  'exact_topology_arrays_sha256':domain_metadata.get('exact_topology_arrays_sha256'),'selected_chart_names_in_order_sha256':plan.metadata.get('selected_chart_names_in_order_sha256'),
  'frozen_submap_plan_content_sha256':expected_plan_content_sha256,'frozen_submap_plan_file_sha256':file_sha256(frozen_submap_plan_path),
  'bounded_submap_content_sha256':bounded_hash,'bounded_submap_min_world':minimum.tolist(),'bounded_submap_max_world':maximum.tolist(),
  'bounded_submap_artifact_content_sha256':bounds.get('content_sha256'),'bounded_submap_file_sha256':file_sha256(bounded_submap_path),
  'alignment_manifest_path':str(alignment_manifest_path.resolve()),'alignment_manifest_content_sha256':manifest_hash,'alignment_manifest_file_sha256':file_sha256(alignment_manifest_path),
  'initializer_manifest_path':str(initializer_manifest_path.resolve()),'initializer_manifest_content_sha256':initializer_manifest_hash,'initializer_manifest_file_sha256':file_sha256(initializer_manifest_path),
  'initializer_selected_file_inventory':selected_initializer_inventory,'initializer_selected_file_inventory_sha256':canonical_json_sha256(selected_initializer_inventory),
  'initializer_artifact_file_sha256':initializer_hashes,'initializer_artifact_content_sha256':initializer_content_hashes,
  'alignment_runner_contract':manifest.get('selection_contract'),'alignment_runner_file_sha256':manifest.get('alignment_runner_file_sha256'),'alignment_code_inventory_sha256':manifest.get('alignment_code_inventory_sha256'),'source_charts_file_sha256':file_sha256(charts_path),'source_cameras_file_sha256':file_sha256(cameras_path),
  'camera_binding':'explicit_frozen_plan_order_and_exact_depth_replay','camera_binding_max_abs_error_scaled':max_depth_error,'initializer_prior_depth_max_abs_error_scaled':max_prior_error,
  'validity':'v3_exact_reference_safe_packed_vertices_and_faces_without_arm_specific_deletion' if domain_schema==PHYSICAL_DOMAIN_SCHEMA else 'legacy_v2_exact_packed_vertices_and_faces_diagnostic_only','edge_break':'v3_physical_reference_safe_frozen_common_face_inventory' if domain_schema==PHYSICAL_DOMAIN_SCHEMA else 'legacy_v2_exact_frozen_common_face_inventory_not_source_reference_safe',
  'valid_fraction_mean':float(source_valid.reshape(len(source_valid),-1).mean(1).mean()),'scale_factor':scale,
 }
 if domain_schema==PHYSICAL_DOMAIN_SCHEMA:
  metadata.update({
   'upstream_exact_topology_v2_file_sha256':file_sha256(upstream_exact_topology_v2_path),
   'upstream_exact_topology_v2_content_sha256':upstream_v2_metadata.get('content_sha256'),
   'upstream_exact_topology_v2_arrays_sha256':upstream_v2_metadata.get('arrays_sha256'),
   'upstream_exact_topology_v2_exact_topology_arrays_sha256':upstream_v2_metadata.get('exact_topology_arrays_sha256'),
   'source_reference_edge_safe_face_masks_sha256':domain_metadata.get('source_reference_edge_safe_face_masks_sha256'),
   'final_reference_safe_face_masks_sha256':domain_metadata.get('final_reference_safe_face_masks_sha256'),
   'source_reference_root':str(source_reference_root),
   'source_reference_cameras_file_sha256':source_reference_cameras_hash,
   'source_reference_selected_pointmap_inventory_sha256':domain_metadata.get('source_reference_selected_pointmap_inventory_sha256'),
   'source_reference_full_pointmap_inventory_sha256':domain_metadata.get('source_reference_full_pointmap_inventory_sha256'),
   'source_reference_edge_safety_config_sha256':domain_metadata.get('source_reference_edge_safety_config_sha256'),
   'physical_face_safety_authority':True,
   'exact_topology_repacked_after_reference_safety':True,
   'full_submap_gate_eligible_strides':domain_metadata.get('full_submap_gate_eligible_strides',[4]),
   'full_submap_gate_primary_stride':domain_metadata.get('full_submap_gate_primary_stride',4),
  })
 atlas=ExplicitChartAtlas(metadata=metadata,**atlas_arrays).validated()
 return atlas,{'initial_points':initial_vertices,'aligned_points':aligned_vertices,'chart_rows':chart_rows,'domain_metadata':domain_metadata,'alignment_manifest':manifest}

__all__=['ExplicitChartAtlas','build_explicit_chart_atlas','build_strict_explicit_chart_atlas']
