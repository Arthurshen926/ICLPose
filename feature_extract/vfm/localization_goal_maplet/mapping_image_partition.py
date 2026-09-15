"""Optional train-only image grouping for scenes with a single mapping route."""
import json
from pathlib import Path

import numpy as np
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def image_groups(names, partition_path):
    path = Path(partition_path)
    d = json.loads(path.read_text())
    manifest = Path(d['mapping_manifest'])
    if file_sha256(manifest) != d['mapping_manifest_sha256']:
        raise ValueError('Mapping partition authority changed')
    allowed = {r['image_id'].replace('/', '__') + '.npz'
               for r in json.loads(manifest.read_text())['records']}
    sets = {k: set(d[k]) for k in ['fit', 'validation', 'excluded']}
    if any(len(d[k]) != len(sets[k]) for k in sets):
        raise ValueError('Duplicate partition image')
    if not sets['fit'] or not sets['validation']:
        raise ValueError('Empty mapping fit or validation set')
    if any(sets[x] & sets[y] for x, y in [('fit', 'validation'), ('fit', 'excluded'), ('validation', 'excluded')]):
        raise ValueError('Mapping partition overlap')
    if set.union(*sets.values()) != allowed:
        raise ValueError('Partition must cover exactly the mapping inventory')
    lookup = {n: 'mapping_images_' + group for group, items in sets.items() for n in items}
    if any(str(n) not in lookup for n in names):
        raise ValueError('Observation outside mapping inventory')
    return np.asarray([lookup[str(n)] for n in names])


def partition_metadata(path):
    d = json.loads(Path(path).read_text())
    fit_routes = sorted({n.split('__')[0] for n in d['fit']})
    val_routes = sorted({n.split('__')[0] for n in d['validation']})
    return dict(mapping_image_partition_file_sha256=file_sha256(Path(path)),
                mapping_validation_partition_kind='fixed_train_image_blocks',
                fit_mapping_routes=fit_routes, validation_mapping_route=None,
                validation_mapping_routes=val_routes,
                fit_validation_route_disjoint=not bool(set(fit_routes) & set(val_routes)),
                fit_validation_image_disjoint=True,
                validation_route_absent_from_prototypes=not bool(set(fit_routes) & set(val_routes)),
                validation_images_absent_from_prototypes=True,
                validation_prototype_routes=fit_routes)
