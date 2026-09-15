"""Explicit per-scene input paths with the original StMarys defaults preserved."""
import json
from pathlib import Path
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def radio_manifests(base):
    p=Path(base)/'scene_inputs.json'
    if not p.exists():
        return [Path('output/vfm_tokens/StMarysChurch/full_1024x576')/(s+'_manifest.json') for s in ['train','test']]
    d=json.loads(p.read_text())
    if d.get('query_pose_values_included') is not False:raise ValueError('Scene input authority differs')
    paths=[Path(v) for v in d['radio_manifests']]
    if len(paths)!=2:raise ValueError('Require full train and test RADIO manifests')
    if any(file_sha256(v)!=d['radio_manifest_sha256'][str(v)] for v in paths):raise ValueError('RADIO inventory changed')
    return paths
