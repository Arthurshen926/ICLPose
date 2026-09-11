"""Read selected metric anchors and query evidence from matched local RGB crops."""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.build_goal_maplet_fine_readout import sample_grid
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def local_pixels(pixels,quadrant):
    origin=np.array([quadrant%2*128,quadrant//2*72])
    return (np.asarray(pixels)+.5-origin)*2-.5


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['plan','cache','source_names','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--candidates',nargs='+',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    names=np.array(json.loads(a.source_names.read_text()))
    with np.load(a.plan/'mapping.npz') as z:m={k:z[k] for k in z.files}
    d=np.zeros((len(m['map_xy']),64),np.float32);valid=m['map_quadrant']>=0
    def read(s):
        with np.load(a.cache/names[s]) as z:
            meta=json.loads(str(z['metadata_json']))
            if meta['query_GT_opened'] is not False or meta['plan_sha256']!=file_sha256(a.plan/'plan.json'):raise ValueError('ROI cache differs')
            return dict(zip(z['quadrants'],z['features'].astype(np.float32)))
    for s in np.unique(m['prototype_source'][valid]):
        grids=read(s)
        for q,g in grids.items():
            rows=np.flatnonzero(valid&(m['prototype_source']==s)&(m['map_quadrant']==q))
            d[rows]=sample_grid(g,local_pixels(m['map_xy'][rows],q))
    ids=np.flatnonzero(valid)
    np.savez_compressed(a.output/'map.npz',unit_anchor_prototype=ids,descriptors=d[ids].astype(np.float16),prototype_world=m['prototype_world'])
    offsets=np.array([(x,y) for y in [-4/3,0,4/3] for x in [-4/3,0,4/3]])
    for candidate in a.candidates:
        with np.load(candidate) as z:c={k:z[k] for k in z.files}
        with np.load(a.plan/(candidate.stem+'.npz')) as z:mask=z['roi_mask'].copy();quadrants=z['query_quadrants']
        mask &= valid[c['prototype_rows']]
        f=np.zeros(len(mask),np.float32);pixels=np.c_[(c['query_token']%64)*4+1.5,(c['query_token']//64)*4+1.5].astype(np.float32)
        for s,q in quadrants:
            rows=np.flatnonzero((c['source_image']==s)&mask);pr=c['prototype_rows'][rows];g=read(s)[q]
            xy=pixels[rows].copy();f[rows]=np.sum(sample_grid(g,local_pixels(xy,q))*d[pr],axis=1)
            probes=xy[:,None]+offsets[None]
            sim=np.sum(sample_grid(g,local_pixels(probes,q))*d[pr,None],axis=-1);sim[:,4]+=1e-7
            pixels[rows]=probes[np.arange(len(rows)),np.argmax(sim,axis=1)]
        np.savez_compressed(a.output/(candidate.stem+'.npz'),cosine=f,roi_mask=mask,refined_pixels=pixels,candidate_sha256=np.asarray(file_sha256(candidate)))
    (a.output/'summary.json').write_text(json.dumps({'scope':__doc__,'map_descriptors':len(ids),'extra_map_descriptor_bytes':len(ids)*64*2,
        'one_query_crop_pixels':768*432,'full_fine_pixels':1536*864,'full_coarse_pixels':1024*576,
        'query_GT_or_poses_opened':False,'new_geometry_or_candidates':False,'plan_sha256':file_sha256(a.plan/'plan.json')},indent=2))


if __name__=='__main__':main()
