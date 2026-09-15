"""Seal full Cambridge inventories and original-contract, pose-blind cameras.

Only image-name fields of the public split files are used here. Native COLMAP
extrinsics and tracks are skipped by camera_bindings, never decoded. This is
input preparation; it deliberately produces no localization result.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.cambridge_camera_authority import camera_bindings, read_native_work_cameras
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


SCENES = {'GreatCourt': (1532, 760), 'KingsCollege': (1220, 343),
          'OldHospital': (895, 182), 'ShopFacade': (231, 103),
          'StMarysChurch': (1487, 530)}


def names_only(path):
    names = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 8 and '/' in fields[0]:
            names.append(fields[0])
    if not names or len(set(names)) != len(names):
        raise ValueError(f'Empty or duplicate split inventory: {path}')
    return names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=Path('/hy-tmp/Cambridge_stdloc'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for scene, expected in SCENES.items():
        root, out = a.dataset_root / scene, a.output / scene
        out.mkdir(parents=True, exist_ok=True)
        train, test = [names_only(root / f'dataset_{split}.txt') for split in ['train', 'test']]
        if (len(train), len(test)) != expected or set(train) & set(test):
            raise ValueError(f'Official split differs: {scene}')
        source = root / 'sparse/0'
        binding = camera_bindings(source / 'images.bin')
        missing_train = sorted(set(train) - set(binding))
        missing_test = sorted(set(test) - set(binding))
        if missing_test:
            raise ValueError(f'Official test cameras missing: {scene}: {missing_test}')
        image_ids = sorted((set(train) | set(test)) & set(binding))
        names = [n.replace('/', '__') + '.npz' for n in image_ids]
        cams = read_native_work_cameras(scene, names, a.dataset_root)
        native = read_colmap_cameras_binary(source / 'cameras.bin')
        camera_rows = {}
        for image_id in image_ids:
            c = native[binding[image_id]]
            f, cx, cy, k = c.params
            camera_rows[image_id] = dict(model_id=2, width=1024, height=576,
                                        params=[f * 1024 / c.width, cx * 1024 / c.width,
                                                cy * 576 / c.height, k])
        metadata = dict(scene=scene, query_extrinsics_decoded=False, query_pose_values_parsed=False,
                        camera_authority='native SIMPLE_RADIAL; original RADIO-canvas/half-pixel work-grid contract',
                        source_sha256={str(p): file_sha256(p) for p in [source / 'images.bin', source / 'cameras.bin']})
        dest = out / 'native_camera_only.npz'
        if dest.exists():
            raise FileExistsError(dest)
        np.savez_compressed(dest, names=np.array(names),
                            camera_matrices=np.array([cams[n][0] for n in names]),
                            radial_k1=np.array([cams[n][1] for n in names]),
                            metadata_json=np.array(json.dumps(metadata)))
        (out / 'native_camera_manifest.json').write_text(json.dumps(dict(
            format='per_query_colmap_calibration_only_v1', cameras=camera_rows,
            production_contract=dict(contains_camera_pose=False, contains_sfm_points=False, contains_sfm_tracks=False),
            intrinsic_audit=metadata), indent=2))
        report = dict(scene=scene, train_images=len(train), test_images=len(test),
                      mapping_subsampling=False, query_pose_values_parsed=False,
                      mapping_images_with_native_calibration=len(train)-len(missing_train),
                      mapping_missing_native_calibration=missing_train,
                      missing_calibration_policy='report unavailable mapping views; never impute or drop test queries',
                      input_preparation_only=True, localization_complete=False,
                      camera_sha256=file_sha256(dest),
                      train_names=train, test_names=test)
        (out / 'official_inventory.json').write_text(json.dumps(report, indent=2))
        print(scene, len(train), len(test), 'sealed', flush=True)


if __name__ == '__main__':
    main()
