from pathlib import Path

import pytest
import yaml

from feature_extract.tools.vfm.extract_tokens import discover_image_files


CAMBRIDGE_ROOT = Path("/hy-tmp/Cambridge_stdloc")


@pytest.mark.skipif(not CAMBRIDGE_ROOT.exists(), reason="Cambridge_stdloc is not mounted")
@pytest.mark.parametrize(
    ("config_path", "expected_train", "expected_test"),
    [
        ("feature_extract/configs/vfm/data/oldhospital.yaml", 895, 182),
        ("feature_extract/configs/vfm/data/shopfacade.yaml", 231, 103),
        ("feature_extract/configs/vfm/data/kingscollege.yaml", 1220, 343),
        ("feature_extract/configs/vfm/data/greatcourt.yaml", 1532, 760),
        ("feature_extract/configs/vfm/data/stmaryschurch.yaml", 1487, 530),
    ],
)
def test_cambridge_data_configs_resolve_real_split_images(
    config_path,
    expected_train,
    expected_test,
):
    config = yaml.safe_load(Path(config_path).read_text())
    image_root = Path(config["image_root"])

    train = discover_image_files(image_root, Path(config["splits"]["train"]))
    test = discover_image_files(image_root, Path(config["splits"]["test"]))

    assert len(train) == expected_train
    assert len(test) == expected_test
