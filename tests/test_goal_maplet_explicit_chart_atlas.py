import json
import numpy as np
import pytest
from feature_extract.vfm.localization_goal_maplet.explicit_chart_atlas import build_explicit_chart_atlas,ExplicitChartAtlas
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256,canonical_json_sha256
from feature_extract.tools.vfm.evaluate_goal_maplet_chart_held_reprojection import _render_triangles
def fixture(tmp_path,wrong_order=False):
 h,w=8,12;names=['seq10__b.png','seq1__a.png'];c2w=np.repeat(np.eye(4)[None],2,0);c2w[0,0,3]=2;c2w[0,2,3]=1
 cameras={'filepaths':[str(tmp_path/x) for x in names],'focals':[10,10],'cams2world':c2w.tolist()};(tmp_path/'cameras.json').write_text(json.dumps(cameras))
 lexical=sorted(names);pts=[];depth=[]
 for name in lexical:
  row=names.index(name);y,x=np.mgrid[:h,:w];d=np.full((h,w),2.);C=c2w[row,:3,3];p=C+np.stack(((x-w/2)/10*d,(y-h/2)/10*d,d),-1);pts.append(p);depth.append(d)
 if wrong_order:pts=pts[::-1]
 np.savez(tmp_path/'charts.npz',pts=np.asarray(pts),depths=np.asarray(depth),prior_depths=np.asarray(depth),confs=np.ones((2,h,w)),scale_factor=np.asarray(1.))
 return tmp_path/'charts.npz',tmp_path/'cameras.json'
def test_chart_camera_lexical_binding_and_roundtrip(tmp_path):
 charts,cameras=fixture(tmp_path);atlas,audit=build_explicit_chart_atlas(charts,cameras,allowed_routes=['seq1'],stride=2,minimum_edge_m=1.);assert atlas.chart_names.tolist()==['seq1__a.png'];assert atlas.metadata['optimization_saw_excluded_routes'];assert len(atlas.faces)>0
 path=tmp_path/'atlas.npz';atlas.save_npz(path);loaded=ExplicitChartAtlas.load_npz(path);assert np.array_equal(loaded.faces,atlas.faces);assert np.allclose(audit['initial_points'],audit['aligned_points'])
def test_wrong_chart_order_fails_depth_replay(tmp_path):
 charts,cameras=fixture(tmp_path,True)
 with pytest.raises(ValueError,match='lexical chart order'):build_explicit_chart_atlas(charts,cameras,allowed_routes=['seq1'],stride=2)
def test_source_valid_mask_excludes_filled_carrier_vertices(tmp_path):
 charts,cameras=fixture(tmp_path)
 with np.load(charts,allow_pickle=False) as data:payload={name:np.asarray(data[name]) for name in data.files}
 payload['valid']=np.ones_like(payload['depths'],dtype=bool);payload['valid'][0,0,0]=False
 np.savez(charts,**payload)
 atlas,_=build_explicit_chart_atlas(charts,cameras,allowed_routes=['seq10'],stride=2,minimum_edge_m=1.)
 assert not np.any(np.all(atlas.uv==np.asarray([0.,0.]),axis=1))
def test_explicit_chart_order_overrides_legacy_lexical_assumption(tmp_path):
 charts,cameras=fixture(tmp_path)
 with np.load(charts,allow_pickle=False) as data:payload={name:np.asarray(data[name]) for name in data.files}
 order=np.asarray([1,0]);payload={name:(value[order] if value.ndim>=1 and value.shape[0]==2 else value) for name,value in payload.items()};payload['chart_names']=np.asarray(['seq1__a.png','seq10__b.png'])
 np.savez(charts,**payload)
 atlas,_=build_explicit_chart_atlas(charts,cameras,allowed_routes=['seq1','seq10'],stride=2,minimum_edge_m=1.)
 assert atlas.chart_names.tolist()==['seq1__a.png','seq10__b.png']
def test_frozen_comparison_domain_prevents_arm_specific_face_deletion(tmp_path):
 charts,cameras=fixture(tmp_path)
 with np.load(charts,allow_pickle=False) as data:base={name:np.asarray(data[name]) for name in data.files}
 names=sorted(['seq10__b.png','seq1__a.png']);valid=np.ones_like(base['depths'],dtype=bool)
 domain_arrays={'chart_names':np.asarray(names),'valid':valid,'face_valid_stride4':np.ones((2,1,2),bool),'face_valid_stride8':np.zeros((2,0,1),bool)}
 meta={'artifact_type':'goal_maplet_chart_comparison_domain_v1','arrays_sha256':arrays_sha256(domain_arrays)};meta['content_sha256']=canonical_json_sha256(meta)
 domain=tmp_path/'domain.npz';np.savez_compressed(domain,**domain_arrays,metadata_json=np.asarray(json.dumps(meta,sort_keys=True)))
 arm1=dict(base);arm1['valid']=valid;arm1['chart_names']=np.asarray(names);arm1['confs']=np.ones_like(base['confs'])
 arm2=dict(arm1);arm2['confs']=np.full_like(base['confs'],-3.);arm2['depths']=base['depths'].copy();arm2['depths'][0,0,0]=30.;arm2['pts']=base['pts'].copy();arm2['pts'][0,0,0,2]=30.
 first=tmp_path/'arm1.npz';second=tmp_path/'arm2.npz';np.savez(first,**arm1);np.savez(second,**arm2)
 atlas1,_=build_explicit_chart_atlas(first,cameras,allowed_routes=['seq1','seq10'],stride=4,comparison_domain=domain)
 atlas2,_=build_explicit_chart_atlas(second,cameras,allowed_routes=['seq1','seq10'],stride=4,comparison_domain=domain)
 assert np.array_equal(atlas1.chart_vertex_offsets,atlas2.chart_vertex_offsets)
 assert np.array_equal(atlas1.chart_face_offsets,atlas2.chart_face_offsets)
 assert np.array_equal(atlas1.faces,atlas2.faces)
 assert np.array_equal(atlas1.uv,atlas2.uv)
def test_triangle_renderer_fills_surface_with_metric_depth():
 vertices=np.asarray([[-.2,-.2,2.],[.2,-.2,2.],[0.,.2,2.]])
 rendered=_render_triangles(vertices,np.asarray([[0,1,2]]),np.eye(4),100.)
 finite=rendered[np.isfinite(rendered)]
 assert len(finite)>10
 assert np.allclose(finite,2.)
