from __future__ import annotations

import json
from pathlib import Path

from feature_extract.tools.vfm.filter_referenced_manifest_by_query_split import main


def test_filter_contract_source_mentions_all_query_splits() -> None:
    # The executable is covered by the artifact integration run; keep a cheap
    # source-level guard against regressing to validation-only filtering.
    source = Path(main.__code__.co_filename).read_text()
    assert '("train", "validation", "test")' in source
    assert "neither_query_nor_reference" in source
