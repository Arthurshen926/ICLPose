"""Threshold-wise frozen candidate coverage; diagnostic only, never a selector."""
import argparse
import json
from pathlib import Path
import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256

THRESHOLDS = ((.1, 1), (.25, 2), (.5, 5), (1, 10), (2, 45))


def pool_decomposition(errors, selected, thresholds=THRESHOLDS):
    """Errors are [query,candidate,(translation,rotation)]; invalids fail.

    Coverage is a separate existential statement at each joint threshold.
    It is not the error vector of any single deployable or GT-selected pose.
    """
    e = np.asarray(errors, dtype=float)
    s = np.asarray(selected)
    if e.ndim != 3 or e.shape[2] != 2 or not e.shape[0] or not e.shape[1]:
        raise ValueError('expected nonempty N x K x 2 errors')
    if s.shape != (len(e),) or not np.issubdtype(s.dtype, np.integer):
        raise ValueError('selected must be one integer candidate index per query')
    if np.any(s < 0) or np.any(s >= e.shape[1]) or np.any(e < 0):
        raise ValueError('invalid selection or negative error')
    result = {}
    for t, r in thresholds:
        hit = np.isfinite(e).all(2) & (e[:, :, 0] <= t) & (e[:, :, 1] <= r)
        available = hit.any(1)
        chosen = hit[np.arange(len(e)), s]
        result[f'{t}m_{r}deg'] = {
            'pool_recall_percent': float(100 * available.mean()),
            'selected_recall_percent': float(100 * chosen.mean()),
            'selection_loss_pp': float(100 * (available & ~chosen).mean()),
            'pool_missing_percent': float(100 * (~available).mean()),
            'conditional_selection_success_percent': float(100 * chosen.sum() / available.sum()) if available.any() else None,
            'selection_failure_query_indices': np.flatnonzero(available & ~chosen).tolist(),
            'pool_missing_query_indices': np.flatnonzero(~available).tolist(),
        }
    return result


def main():
    from feature_extract.tools.vfm.select_goal_maplet_coordinate_pose_geometry_consensus import _load_selected, _load_endpoint_evaluation
    from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import _load_pose_candidate
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--consensus_dirs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    names, errors, selected, sources = [], [], [], {}
    for d in a.consensus_dirs:
        arr, _ = _load_selected(d / 'selected.npz')
        cmd = dict(json.loads((d / 'protocol.json').read_text())['commands'])['selected']
        get = lambda key: Path(cmd[cmd.index('--' + key) + 1])
        poses = [_load_pose_candidate(get(k)) for k in ['primary_pose', 'alternate_pose']]
        for pa, _ in poses:
            if not np.array_equal(pa['names'].astype(str), arr['names'].astype(str)):
                raise ValueError('endpoint query order differs')
        # Bind frozen inference artifacts before opening endpoint evaluation labels.
        for path in [d / 'selected.npz', get('primary_pose'), get('alternate_pose'), d / 'protocol.json']:
            sources[str(path)] = file_sha256(path)
        ev = [_load_endpoint_evaluation(get(k), get(pk), pm['content_sha256'], arr['names'])
              for k, pk, (_, pm) in zip(['primary_evaluation', 'alternate_evaluation'], ['primary_pose', 'alternate_pose'], poses)]
        for i, c in enumerate(arr['selected_branch']):
            if not np.array_equal(arr['pose_w2c'][i], poses[int(c)][0]['pose_w2c'][i], equal_nan=True):
                raise ValueError('selected pose is not a frozen endpoint')
            names.append(str(arr['names'][i]))
            errors.append([[row[i]['translation_error_m'], row[i]['rotation_error_deg']] for row in ev])
            selected.append(int(c))
    if len(names) != len(set(names)):
        raise ValueError('duplicate queries across splits')
    result = pool_decomposition(errors, selected)
    for row in result.values():
        for key in ['selection_failure_query_indices', 'pool_missing_query_indices']:
            row[key.replace('_indices', '_names')] = [names[i] for i in row.pop(key)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(scope='post-label threshold-wise diagnostic; no deployable oracle pose',
        sources=sources, names=names, errors=errors, selected=selected, thresholds=result), indent=2))


if __name__ == '__main__':
    main()
