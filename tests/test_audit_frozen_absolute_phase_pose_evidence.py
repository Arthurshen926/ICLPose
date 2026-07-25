from __future__ import annotations

import pytest

from feature_extract.tools.vfm.audit_frozen_absolute_phase_pose_evidence import (
    _validate_pairing,
)


def _metadata(*, control: bool, digest: str = "evidence") -> dict[str, object]:
    return {
        "query_id": "q.png",
        "frozen_query_evidence_sha256": digest,
        "verification_point_selection": {"source": "formal"},
        "strict_frozen_evidence_contract": {
            "fixed_global_topl": True,
            "support_channel_permutation_control_only": control,
        },
    }


def test_pairing_permits_only_the_declared_control_contract_difference() -> None:
    _validate_pairing(
        visual_metadata=[_metadata(control=False)],
        control_metadata=[_metadata(control=True)],
    )


def test_pairing_rejects_different_frozen_query_evidence() -> None:
    with pytest.raises(ValueError, match="differ beyond descriptor control"):
        _validate_pairing(
            visual_metadata=[_metadata(control=False, digest="left")],
            control_metadata=[_metadata(control=True, digest="right")],
        )
