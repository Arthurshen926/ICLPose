"""Fail-closed identity/coordinate contract for indexed verification banks."""
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256


def validate_bank(bank,metadata,world,map_path,map_sha256,atlas_sha256,names,cameras,radial):
    if arrays_sha256(bank)!=metadata.get('arrays_sha256'):raise ValueError('bank array hash mismatch')
    sources={str(Path(p).resolve()):h for p,h in metadata.get('source_sha256',{}).items()}
    if sources.get(str(Path(map_path).resolve()))!=map_sha256:raise ValueError('fine map identity mismatch')
    if atlas_sha256 not in sources.values():raise ValueError('atlas identity mismatch')
    n=len(bank['names']);rows=bank['prototype_rows'];tokens=bank['query_tokens'];offset=bank['offsets']
    if not np.array_equal(bank['names'],names) or len(set(names.astype(str)))!=n:raise ValueError('query order or uniqueness mismatch')
    if not np.array_equal(bank['camera_matrices'],cameras) or not np.array_equal(bank['radial_k1'],radial):raise ValueError('camera mismatch')
    if offset.shape!=(n+1,) or offset.dtype.kind not in 'iu' or offset[0]!=0 or np.any(np.diff(offset)<0) or offset[-1]!=len(rows):raise ValueError('invalid offsets')
    if rows.dtype.kind not in 'iu' or np.any(rows<0) or np.any(rows>=len(world)):raise ValueError('invalid prototype identities')
    if tokens.shape!=rows.shape or tokens.dtype.kind not in 'iu' or np.any(tokens<0) or np.any(tokens>=2304):raise ValueError('invalid token indices')
    if bank['pixels'].shape!=(len(rows),2) or not np.isfinite(bank['pixels']).all() or not np.isfinite(world[rows]).all():raise ValueError('invalid measurement coordinates')


def validate_poses(poses):
    if poses.ndim!=3 or poses.shape[1:]!=(4,4) or not np.isfinite(poses).all():raise ValueError('invalid pose shape or values')
    rot=poses[:,:3,:3]
    if not np.allclose(rot@rot.transpose(0,2,1),np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(rot),1,atol=1e-5) or not np.allclose(poses[:,3],np.array([0,0,0,1])):raise ValueError('invalid rigid transform')
