"""Build actual flat/LoD retrieval pairs with route-excluded appearance.

Mapping pose/depth are opened only after pose-free proposals are frozen. Missing,
occluded and boundary-ambiguous support is ignored, never an automatic negative.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from scipy.spatial import cKDTree
from feature_extract.tools.vfm.localization_lod_memory import LocalizationLoD
from feature_extract.tools.vfm.partial_overlap_matcher import make_pair
from feature_extract.tools.vfm.partial_visibility_memory import diverse_tokens
from feature_extract.tools.vfm.build_goal_maplet_native_fine_readout import load_grids
from feature_extract.tools.vfm.build_goal_maplet_structured_memory_features import moge_tokens
from feature_extract.vfm.localization_goal_maplet.structured_local_memory import normalise
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def atlas_uncertainty(arrays):
    covariance=arrays['prototype_world_covariance_m2']
    if covariance.ndim!=3 or covariance.shape[1:]!=(3,3) or not np.isfinite(covariance).all():raise ValueError('invalid native covariance')
    return np.trace(covariance,axis1=-2,axis2=-1)/3


def load_map(base):
    ap=base/'stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz';fp=base/'native_fine_v264/readout/map.npz';gp=base.parent/'stmarys_rendered_ransac_fused_planes_v1.npz'
    with np.load(ap) as f:
        world=f['world_points'];coarse=normalise(f['radio_features']);cov=atlas_uncertainty(f);offset=f['plane_texel_offsets'];am=json.loads(f['metadata_json'].item())
    with np.load(fp) as f:
        assert np.array_equal(world,f['world_points']);fine=normalise(f['fine']);available=f['available'];meta=json.loads(f['metadata_json'].item())
    assert file_sha256(ap)==meta['atlas_sha256'];assert file_sha256(gp)==am['planar_map_file_sha256']
    with np.load(gp) as f:normals=np.repeat(f['normals_world'],np.diff(offset),axis=0)
    return dict(world=world,coarse_map=coarse,fine_map=fine,covariance=cov,map_normals=normals,available=available),meta,{str(p):file_sha256(p) for p in [ap,fp,gp]}


def project(world,pose,K,k1):
    uv=cv2.projectPoints(world,cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2)
    z=(world@pose[:3,:3].T+pose[:3,3])[:,2];inside=np.isfinite(uv).all(1)&(z>0)&(uv>=0).all(1)&(uv<=np.array([255,143])).all(1)
    return uv,z,inside


def supervise(pair,region,world,pose,K,k1,depth):
    uv,z,inside=project(world,pose,K,k1);pix=np.clip(np.rint(np.nan_to_num(uv)).astype(int),[0,0],[255,143]);d=depth[pix[:,1],pix[:,0]];known_depth=np.isfinite(d)&(d>0)
    visible=inside&known_depth&(np.abs(d-z)<=np.maximum(.1,.01*d))
    xy=pair['pixels'];qpix=np.clip(np.rint(xy).astype(int),[0,0],[255,143]);qd=depth[qpix[:,1],qpix[:,0]];qknown=np.isfinite(qd)&(qd>0)
    projection_near=cKDTree(uv[region[inside[region]]]).query(xy)[0] if inside[region].any() else np.full(len(xy),np.inf)
    visible_rows=region[visible[region]];target=np.full(len(xy),-1,np.int8)
    target[qknown&(projection_near>4)]=0
    if len(visible_rows):
        distance,near=cKDTree(uv[visible_rows]).query(xy);consistent=np.abs(z[visible_rows[near]]-qd)<=np.maximum(.1,.01*qd)
        target[qknown&(distance<=2)&consistent]=1
    ids=pair['ids'];error=np.linalg.norm(uv[ids]-xy[:,None],axis=-1)
    positive=visible[ids]&(error<=2)&qknown[:,None]&(np.abs(z[ids]-qd[:,None])<=np.maximum(.1,.01*qd[:,None]))
    # Visible different pixels or genuinely outside-FOV anchors are known
    # non-correspondences. Hidden anchors inside the image remain unknown.
    known=positive|((visible[ids]&(error>4))|(~inside[ids]))
    return target,positive,known


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=False);maps,meta,sources=load_map(a.base);world=maps['world'];tree=cKDTree(world);records=[]
    for route in ['seq1','seq2','seq4','seq6','seq7','seq8','seq11']:
        cp=a.base/'native_hybrid_mapping_v286'/route/'corr.npz';mask=cp.with_name('mask.npz')
        with np.load(cp) as f:names=f['names'].astype(str);Ks=f['camera_matrices'];ks=f['radial_k1'];cm=json.loads(f['metadata_json'].item())
        with np.load(mask) as f:eligible=f['eligible'];mm=json.loads(f['metadata_json'].item())
        assert cm['atlas_member_mask_file_sha256']==file_sha256(mask);assert mm['excluded_mapping_route']==route;assert set(n.split('__')[0] for n in names)=={route}
        with np.load(a.base/'native_region_training_v276/offline_native_contexts.npz') as f:owners=f['source_names'].astype(str)[f['native_source']]
        assert not eligible[np.char.startswith(owners,route+'__')].any()
        lod=LocalizationLoD(world,maps['coarse_map'],eligible);sources[str(cp)]=file_sha256(cp);sources[str(mask)]=file_sha256(mask)
        for i,name in enumerate(names):
            cache=a.base/'adaptive_memory_v234/fine_cache'/name;mg=a.base/'native_region_training_v276/moge'/route/name;grids,ck=load_grids(cache,meta['projection_sha256']);assert ck==meta['checkpoint_sha256'];qp,qn,qv=moge_tokens(mg);sources[str(cache)]=file_sha256(cache);sources[str(mg)]=file_sha256(mg)
            q=normalise(grids[0].reshape(2304,64));tokens=diverse_tokens(q,qv);proposals={key:lod.proposals(q,flat=key=='flat')[0] for key in ['flat','lod']};(o/(name+'.retrieval.json')).write_text(json.dumps(proposals))
            gt=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')/name;sources[str(gt)]=file_sha256(gt)
            with np.load(gt) as f:pose=f['pose_w2c'];depth=f['dominant_depth']
            for rid in dict.fromkeys(proposals['flat']+proposals['lod']):
                region=np.array(tree.query_ball_point(world[rid],6.,return_sorted=True),int);region=region[eligible[region]]
                if len(region)<16 or len(tokens)<16:continue
                pair=make_pair(grids,qp,qn,qv,tokens,region,**maps);target,positive,known=supervise(pair,region,world,pose,Ks[i],float(ks[i]),depth)
                filename=f'{name}.{rid}.npz';np.savez_compressed(o/filename,query=pair['query'].astype(np.float16),map=pair['map'].astype(np.float16),edges=pair['edges'].astype(bool),similarity=pair['similarity'].astype(np.float16),target=target,positive=positive,known=known,ids=pair['ids'],tokens=tokens)
                records.append(dict(path=filename,name=name,route=route,split='train' if route in ['seq1','seq2','seq4','seq6'] else 'validation',region=rid,positive_fraction=float((target==1).mean()),unknown_fraction=float((target<0).mean()),identity_supervised_fraction=float(positive.any(1).mean())))
            print(route,i+1,'pairs',len(records),flush=True)
    sources[str(Path(__file__))]=file_sha256(Path(__file__));(o/'manifest.json').write_text(json.dumps(dict(records=records,sources=sources,query_test_routes_opened=False,training_routes=['seq1','seq2','seq4','seq6'],validation_routes=['seq7','seq8','seq11'],supervision='shared-map rendered-depth proxy; hidden or missing is unknown',geometry_not_rebuilt_per_route=True,coordinate_head_frozen=True),indent=2))
if __name__=='__main__':main()
