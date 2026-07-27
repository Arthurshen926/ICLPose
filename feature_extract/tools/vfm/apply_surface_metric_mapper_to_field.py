"""Apply a trained fine RADIO metric adapter to a surface feature field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField
from feature_extract.vfm.localization.surface_metric_feature_mapper import (
    load_surface_metric_feature_mapper,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_field", required=True)
    parser.add_argument("--metric_mapper_checkpoint", required=True)
    parser.add_argument("--output_field", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_field)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite metric surface field")
    field = SurfaceFeatureField.load_npz(Path(args.input_field))
    mapper = load_surface_metric_feature_mapper(
        Path(args.metric_mapper_checkpoint), device=str(args.device)
    )
    metadata = dict(field.metadata)
    metadata.update(
        {
            "uses_surface_metric_mapper": True,
            "surface_metric_supervision": mapper.metadata.get("supervision"),
        }
    )
    converted = SurfaceFeatureField(
        source_indices=field.source_indices,
        centers=field.centers,
        normals=field.normals,
        tangent1=field.tangent1,
        tangent2=field.tangent2,
        scale1=field.scale1,
        scale2=field.scale2,
        opacity=field.opacity,
        features=mapper.project_points(field.features),
        uncertainty=field.uncertainty,
        confidence=field.confidence,
        support_weight=field.support_weight,
        support_count=field.support_count,
        owner_maplet_ids=field.owner_maplet_ids,
        metadata=metadata,
    )
    converted.save_npz(output)
    print(
        json.dumps(
            {
                "stage": "apply_surface_metric_mapper_to_field",
                "surfel_count": len(converted),
                "output_field": str(output),
                "production_contract": metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
