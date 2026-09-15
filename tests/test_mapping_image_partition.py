import json
import pytest
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.mapping_image_partition import image_groups, partition_metadata


def fixture(tmp_path):
    m=tmp_path/'train.json'
    m.write_text(json.dumps({'records':[{'image_id':f'seq2/frame{i}.png'} for i in range(4)]}))
    d=dict(mapping_manifest=str(m),mapping_manifest_sha256=file_sha256(m),
           fit=['seq2__frame0.png.npz','seq2__frame1.png.npz'],
           validation=['seq2__frame3.png.npz'],excluded=['seq2__frame2.png.npz'])
    p=tmp_path/'split.json';p.write_text(json.dumps(d));return p,d


def test_single_route_is_not_claimed_route_disjoint(tmp_path):
    p,d=fixture(tmp_path)
    assert image_groups(d['validation']+d['fit'],p).tolist()==['mapping_images_validation','mapping_images_fit','mapping_images_fit']
    assert partition_metadata(p)['fit_validation_route_disjoint'] is False
    assert partition_metadata(p)['fit_validation_image_disjoint'] is True


def test_reject_test_image_in_partition(tmp_path):
    p,d=fixture(tmp_path);d['validation']=['seq3__frame3.png.npz'];p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='exactly'):image_groups(d['fit'],p)


def test_reject_shared_image(tmp_path):
    p,d=fixture(tmp_path);d['validation']+=d['fit'][:1];p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='overlap'):image_groups(d['fit'],p)


def test_reject_changed_train_authority(tmp_path):
    p,d=fixture(tmp_path)
    from pathlib import Path
    Path(d['mapping_manifest']).write_text('{}')
    with pytest.raises(ValueError,match='authority'):image_groups(d['fit'],p)
