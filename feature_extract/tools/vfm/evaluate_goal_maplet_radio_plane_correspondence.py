"""Pose-free RADIO child posterior to finite physical-plane matching gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions,extract_query_plane_regions
def main():
 p=argparse.ArgumentParser();p.add_argument('--physical_map',type=Path,required=True);p.add_argument('--planar_map',type=Path,required=True);p.add_argument('--incidence',type=Path,required=True);p.add_argument('--moge_dir',type=Path,required=True);p.add_argument('--query_plane_dir',type=Path);p.add_argument('--retrieval_dir',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 physical=GoalMapletPhysicalMap.load_npz(a.physical_map);plane=GeometryNativePlanarMap.load_npz(a.planar_map,primitive_count=len(physical.primitive_ids));primitive_owner=np.full(len(physical.primitive_ids),-1,np.int64)
 for row in range(len(plane.plane_ids)):lo,hi=map(int,plane.member_offsets[row:row+2]);primitive_owner[plane.member_primitive_rows[lo:hi]]=row
 row_by_id=np.full(int(physical.primitive_ids.max())+1,-1,np.int64);row_by_id[physical.primitive_ids]=np.arange(len(physical.primitive_ids))
 with np.load(a.incidence,allow_pickle=False) as d:offset=np.asarray(d['plane_cell_offsets']);child=np.asarray(d['child_rows']);mass=np.asarray(d['surface_mass_m2'])
 child_to_planes={}
 for row in range(len(plane.plane_ids)):
  lo,hi=map(int,offset[row:row+2]);total=max(float(mass[lo:hi].sum()),1e-15)
  for c,m in zip(child[lo:hi].tolist(),mass[lo:hi].tolist()):child_to_planes.setdefault(c,[]).append((row,float(m/total)))
 rows=[]
 for path in sorted(a.moge_dir.glob('*.npz')):
  name=path.name;retrieval=a.retrieval_dir/name
  if not retrieval.exists():continue
  if a.query_plane_dir:q,_=QueryPlaneRegions.load_npz(a.query_plane_dir/name)
  else:
   with np.load(path,allow_pickle=False) as d:q=extract_query_plane_regions(d['points_camera'],d['normal_camera'],d['valid'])
  with np.load(retrieval,allow_pickle=False) as d:token_xy=np.asarray(d['token_xy']);token_child=np.asarray(d['token_child_rows']);token_probability=np.asarray(d['token_child_probabilities'],np.float64)
  with np.load(a.contributors/name,allow_pickle=False) as d:ids=np.asarray(d['topk_ids'][:,:,0],np.int64)
  primitive=np.full(ids.shape,-1,np.int64);valid=(ids>=0)&(ids<row_by_id.size);primitive[valid]=row_by_id[ids[valid]];gtlabel=np.full(ids.shape,-1,np.int64);valid=primitive>=0;gtlabel[valid]=primitive_owner[primitive[valid]]
  records=[]
  for region in range(len(q.normals_camera)):
   mask=q.labels==region;block=mask.reshape(36,4,64,4).mean((1,3));weight=block[token_xy[:,1],token_xy[:,0]];query={}
   for c,pv,wv in zip(token_child.reshape(-1).tolist(),token_probability.reshape(-1).tolist(),np.repeat(weight,token_child.shape[1]).tolist()):
    if c>=0 and wv>0:query[c]=query.get(c,0.)+pv*wv
   total=sum(query.values());score=np.zeros(len(plane.plane_ids),np.float64)
   if total>0:
    for c,value in query.items():
     for candidate,fraction in child_to_planes.get(c,()):score[candidate]+=np.sqrt(value/total*fraction)
   ranking=np.lexsort((np.arange(len(score)),-score));values=gtlabel[mask];values=values[values>=0];gt=-1;purity=0.
   if values.size:
    count=np.bincount(values,minlength=len(score));gt=int(count.argmax());purity=float(count[gt]/values.size)
   rank=int(np.flatnonzero(ranking==gt)[0]+1) if gt>=0 else -1;records.append({'region':region,'pixels':int(mask.sum()),'gt_plane':gt,'gt_purity':purity,'predicted_plane':int(ranking[0]),'predicted_score':float(score[ranking[0]]),'gt_rank':rank,'top10':ranking[:10].tolist(),'top10_scores':score[ranking[:10]].tolist()})
  rows.append({'image':name,'query_plane_count':len(records),'regions':records})
 regions=[z for x in rows for z in x['regions'] if z['gt_plane']>=0 and z['gt_purity']>=.5];weights=np.asarray([x['pixels'] for x in regions],float);ranks=np.asarray([x['gt_rank'] for x in regions]);report={'artifact_type':'goal_maplet_pose_free_radio_to_physical_plane_correspondence_v1','query_count':len(rows),'evaluated_region_count':len(regions),'uses_pose_or_gt_in_scoring':False,'gt_used_only_after_ranking':True,'weighted_recall_at_1':float(np.average(ranks<=1,weights=weights)) if len(regions) else 0.,'weighted_recall_at_5':float(np.average(ranks<=5,weights=weights)) if len(regions) else 0.,'weighted_recall_at_10':float(np.average(ranks<=10,weights=weights)) if len(regions) else 0.,'median_gt_rank':float(np.median(ranks)) if len(regions) else None,'production_eligible':False,'rows':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
