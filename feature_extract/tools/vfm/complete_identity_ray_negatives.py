"""Complete mapping-only identity negatives by geometric ray exclusion.

Occlusion uncertainty does not make a distant projected anchor a possible
identity. Near-ray hidden anchors remain unknown. Overlap targets are unchanged.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.prepare_overlap_lod_training import project
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def ray_excluded_known(known, positive, ids, xy, uv, inside, query_depth):
    qknown = np.isfinite(query_depth) & (query_depth > 0)
    error = np.linalg.norm(uv[ids] - xy[:, None], axis=-1)
    # The 4px ambiguity band is exactly the existing visible-negative band.
    excluded = inside[ids] & (error > 4) & qknown[:, None] & ~positive
    return known | excluded


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.load(open(a.input/'manifest.json'));sources={str(a.input/'manifest.json'):file_sha256(a.input/'manifest.json')}
    mp=a.base/'native_fine_v264/readout/map.npz';sources[str(mp)]=file_sha256(mp)
    with np.load(mp) as f:world=f['world_points']
    cameras={};cache={};stats={}
    for r in manifest['records']:
        route=r['route'];assert route in ['seq1','seq2','seq4','seq6','seq7','seq8','seq11']
        if route not in cameras:
            cp=a.base/'native_hybrid_mapping_v286'/route/'corr.npz';sources[str(cp)]=file_sha256(cp)
            with np.load(cp) as f:cameras[route]={n:(K,k) for n,K,k in zip(f['names'].astype(str),f['camera_matrices'],f['radial_k1'])}
        name=r['name'];assert name.startswith(route+'__')
        if name not in cache:
            gp=Path('output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean')/name;sources[str(gp)]=file_sha256(gp)
            with np.load(gp) as f:pose=f['pose_w2c'];depth=f['dominant_depth']
            K,k=cameras[route][name];uv,_,inside=project(world,pose,K,k);cache[name]=(uv,inside,depth)
        uv,inside,depth=cache[name];ip=a.input/r['path'];sources[str(ip)]=file_sha256(ip)
        with np.load(ip) as f:arr={k:f[k] for k in f.files}
        t=arr['tokens'];xy=np.c_[t%64*4+1.5,t//64*4+1.5];pix=np.clip(np.rint(xy).astype(int),[0,0],[255,143]);qd=depth[pix[:,1],pix[:,0]]
        old=arr['known'];arr['known']=ray_excluded_known(old,arr['positive'],arr['ids'],xy,uv,inside,qd)
        st=stats.setdefault(route,dict(candidates=0,unknown_before=0,unknown_after=0,added_negatives=0))
        st['candidates']+=old.size;st['unknown_before']+=int((~old).sum());st['unknown_after']+=int((~arr['known']).sum());st['added_negatives']+=int((arr['known']&~old).sum())
        assert np.all(arr['known'][arr['positive']])
        np.savez_compressed(a.output/r['path'],**arr)
    sources[str(Path(__file__))]=file_sha256(Path(__file__));sources[str(Path(__file__).with_name('prepare_overlap_lod_training.py'))]=file_sha256(Path(__file__).with_name('prepare_overlap_lod_training.py'))
    manifest.update(parent_manifest_sha256=file_sha256(a.input/'manifest.json'),identity_completion='inside image, >4px projected separation, valid query depth; hidden near-ray remains unknown',completion_sources=sources,completion_statistics=stats,query_test_routes_opened=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
