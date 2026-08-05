"""Freeze an explicit, fail-closed field/query-readout/render contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.canonical_field import CanonicalSurfaceField
from feature_extract.vfm.localization_goal_maplet.feature_contract import FieldFeatureContract
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical_field", required=True)
    parser.add_argument(
        "--query_readout_type",
        required=True,
        choices=("surface_maplet_mapper", "canonical_radio_codec", "raw_radio_final"),
    )
    parser.add_argument("--query_readout", default="")
    parser.add_argument("--render_protocol", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite field feature contract")
    field = CanonicalSurfaceField.load_npz(Path(args.canonical_field))
    readout_path = Path(args.query_readout) if args.query_readout else None
    contract = FieldFeatureContract(
        field.content_sha256,
        str(args.query_readout_type),
        file_sha256(readout_path) if readout_path is not None else "",
        str(args.render_protocol),
        metadata={
            "artifact_type": "goal_maplet_field_feature_contract_v1",
            "canonical_field_path": str(args.canonical_field),
            "query_readout_path": str(args.query_readout),
        },
    )
    contract.validate(field, query_readout_path=readout_path)
    contract.save_json(output)
    print(contract.content_sha256)


if __name__ == "__main__":
    main()
