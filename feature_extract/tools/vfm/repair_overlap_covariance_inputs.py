"""Migrate a frozen training population to the authoritative covariance field.

All retrieval, pose/depth labels and other features are preserved exactly.
"""
import argparse,json
from pathlib import Path
import numpy as np
from feature_extract.tools.vfm.prepare_overlap_lod_training import load_map
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['base','input','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);maps,_,sources=load_map(a.base);manifest=json.load(open(a.input/'manifest.json'));count=0
    for r in manifest['records']:
        with np.load(a.input/r['path']) as f:arrays={k:f[k] for k in f.files}
        m=arrays['map'].copy();m[...,129]=np.log1p(maps['covariance'][arrays['ids']]).astype(m.dtype)
        assert np.array_equal(m[...,:129],arrays['map'][...,:129]) and np.array_equal(m[...,130:],arrays['map'][...,130:]);arrays['map']=m
        np.savez_compressed(a.output/r['path'],**arrays);count+=1
    manifest['covariance_repair']=dict(authoritative_field='prototype_world_covariance_m2',feature_column=129,parent_manifest_sha256=file_sha256(a.input/'manifest.json'),source_sha256={**sources,str(Path(__file__)):file_sha256(Path(__file__))},retrieval_labels_other_features_unchanged=True)
    for name in ['prepare_overlap_lod_training.py','partial_overlap_matcher.py','localization_lod_memory.py']:
        path=Path(__file__).with_name(name);manifest['sources'][str(path)]=file_sha256(path)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print('repaired',count,'pairs',flush=True)
if __name__=='__main__':main()
