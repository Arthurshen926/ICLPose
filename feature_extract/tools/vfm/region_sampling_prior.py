"""Load frozen, correspondence-bound priors used for sampling only."""
import json
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256


def load_region_sampling_prior(path, correspondence_path, correspondence):
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata_json']))
        arrays = {k:z[k] for k in z.files if k != 'metadata_json'}
    expected = canonical_json_sha256({k:v for k,v in metadata.items() if k != 'content_sha256'})
    if (metadata.get('artifact_type') != 'goal_maplet_region_sampling_prior_v1'
            or metadata.get('query_pose_or_ground_truth_read') is not False
            or metadata.get('content_sha256') != expected
            or metadata.get('arrays_sha256') != arrays_sha256(arrays)
            or metadata.get('correspondence_file_sha256') != file_sha256(correspondence_path)
            or not np.array_equal(arrays.get('names'),correspondence['names'])
            or not np.array_equal(arrays.get('correspondence_offsets'),correspondence['correspondence_offsets'])):
        raise ValueError('context sampling prior lineage differs')
    weights = np.asarray(arrays['sampling_weights'],np.float64)
    if weights.shape != correspondence['query_tokens'].shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('invalid positive sampling weights')
    return weights
