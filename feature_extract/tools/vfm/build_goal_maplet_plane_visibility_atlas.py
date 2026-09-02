from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import build_plane_visibility_atlas
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--observation_dir',type=Path,required=True);p.add_argument('--lineage',type=Path,required=True);p.add_argument('--contributors',type=Path,required=True);p.add_argument('--planar_map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--token_grid',type=int,nargs=2,default=(36,64),metavar=('HEIGHT','WIDTH'));a=p.parse_args()
 with np.load(a.lineage,allow_pickle=False) as data:offset=np.asarray(data['plane_observation_offsets']);rows=np.asarray(data['plane_observation_rows'])
 token_grid=tuple(map(int,a.token_grid));atlas=build_plane_visibility_atlas(a.observation_dir.glob('*.planes.npz'),offset,rows,a.contributors,token_grid=token_grid);a.output.parent.mkdir(parents=True,exist_ok=True)
 meta=atlas.save_npz(a.output,{'planar_map_file_sha256':sha(a.planar_map),'lineage_file_sha256':sha(a.lineage),'observation_manifest_file_sha256':sha(a.observation_dir/'manifest.json'),'uses_query_or_ground_truth':False,'token_grid':list(token_grid),'token_coverage_quantization':'area_fraction_times_16_rounded'})
 print(json.dumps({'output':str(a.output),'file_sha256':sha(a.output),'content_sha256':meta['content_sha256'],'plane_count':int(atlas.plane_offsets.size-1),'observation_count':int(atlas.plane_observation_rows.size)},indent=2))
if __name__=='__main__':main()
