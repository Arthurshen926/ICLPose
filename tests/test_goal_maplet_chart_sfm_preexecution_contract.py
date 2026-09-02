from pathlib import Path

import pytest

from feature_extract.tools.vfm.build_goal_maplet_chart_sfm_preexecution_contract import (
    _command,
)


def test_preexecution_command_explicitly_enumerates_complete_inventory():
    command = _command(
        Path("/tmp/matcha"),
        Path("/tmp/source"),
        Path("/tmp/output"),
        image_count=18,
    )
    offset = command.index("--image_idx") + 1
    assert command[offset:] == [str(index) for index in range(18)]


@pytest.mark.parametrize("image_count", [0, 1])
def test_preexecution_command_rejects_degenerate_role(image_count):
    with pytest.raises(ValueError, match="at least two images"):
        _command(
            Path("/tmp/matcha"),
            Path("/tmp/source"),
            Path("/tmp/output"),
            image_count=image_count,
        )
