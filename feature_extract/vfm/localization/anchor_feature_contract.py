"""Descriptor contract for sparse observations attached to 2DGS anchors.

ALIKE may select repeatable query locations without contributing an identity
descriptor.  The persistent map and the query must nevertheless agree on
which descriptor is used for identity, so that accidentally concatenating an
ALIKE descriptor cannot silently change the method.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


ALIKE_ANCHOR_FEATURE = "alike_anchor"
ALIKE_RADIO_FINAL_ANCHOR_FEATURE = "alike_anchor_plus_radio_final_context"
RADIO_FINAL_ANCHOR_FEATURE = "radio_final_at_alike_detection"

SUPPORTED_ANCHOR_FEATURES = (
    ALIKE_ANCHOR_FEATURE,
    ALIKE_RADIO_FINAL_ANCHOR_FEATURE,
    RADIO_FINAL_ANCHOR_FEATURE,
)


def anchor_feature_kind(metadata: Mapping[str, object] | None) -> str:
    """Return the explicitly declared sparse-anchor identity feature."""

    values = dict(metadata or {})
    kind = str(values.get("local_feature", "")).strip()
    if not kind:
        # Legacy ALIKE-only banks predate the explicit contract.
        representation = str(values.get("representation", ""))
        if "alike_plus_radio_final" in representation:
            kind = ALIKE_RADIO_FINAL_ANCHOR_FEATURE
        elif "alike" in representation:
            kind = ALIKE_ANCHOR_FEATURE
    if kind not in SUPPORTED_ANCHOR_FEATURES:
        raise ValueError(
            "local descriptor bank does not declare a supported "
            f"anchor identity feature: {kind!r}"
        )
    return kind


def compose_anchor_query_descriptors(
    *,
    alike_descriptors: np.ndarray,
    radio_final_descriptors: np.ndarray | None,
    feature_kind: str,
    expected_dim: int,
) -> np.ndarray:
    """Build query descriptors without allowing an implicit feature fusion."""

    alike = np.asarray(alike_descriptors, dtype=np.float32)
    if alike.ndim != 2:
        raise ValueError("ALIKE descriptors must have shape (N, C)")
    radio = (
        None
        if radio_final_descriptors is None
        else np.asarray(radio_final_descriptors, dtype=np.float32)
    )
    if radio is not None and (
        radio.ndim != 2 or int(radio.shape[0]) != int(alike.shape[0])
    ):
        raise ValueError(
            "RADIO-final descriptors must align with ALIKE detections"
        )
    kind = str(feature_kind)
    if kind == ALIKE_ANCHOR_FEATURE:
        output = alike
    elif kind == ALIKE_RADIO_FINAL_ANCHOR_FEATURE:
        if radio is None:
            raise ValueError("fused anchor identity requires RADIO-final")
        output = np.concatenate([alike, radio], axis=1)
    elif kind == RADIO_FINAL_ANCHOR_FEATURE:
        if radio is None:
            raise ValueError("RADIO-only anchor identity requires RADIO-final")
        output = radio
    else:
        raise ValueError(f"unsupported anchor identity feature: {kind!r}")
    if int(output.shape[1]) != int(expected_dim):
        raise ValueError(
            "query and map anchor descriptor dimensions differ: "
            f"{output.shape[1]} != {int(expected_dim)}"
        )
    output = output / np.maximum(
        np.linalg.norm(output, axis=1, keepdims=True),
        1e-8,
    )
    return output.astype(np.float32)
