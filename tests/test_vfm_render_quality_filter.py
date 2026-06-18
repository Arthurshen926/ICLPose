import json
import subprocess
import sys

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.render_quality_filter import (
    RenderQualityRecord,
    filter_candidate_bank_by_render_quality,
)


def _bank():
    return CandidateHypothesisBank.from_candidates(
        protocol_name="stage1_vpr",
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=[
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="bad_render",
                candidate_type="descriptor_retrieval_reference_pose",
                reference_image="bad.png",
                pose_error=PoseCost(translation_m=0.1, rotation_deg=2.0),
                metadata={"retrieval_rank": 1},
            ),
            CandidateHypothesis(
                query_id="q.png",
                candidate_id="good_render",
                candidate_type="descriptor_retrieval_reference_pose",
                reference_image="good.png",
                pose_error=PoseCost(translation_m=0.2, rotation_deg=3.0),
                metadata={"retrieval_rank": 2},
            ),
        ],
    )


def test_render_quality_filter_removes_low_visibility_candidates():
    filtered = filter_candidate_bank_by_render_quality(
        _bank(),
        quality_by_image_id={
            "bad.png": RenderQualityRecord(image_id="bad.png", alpha_coverage=0.1, visible_ratio=0.2),
            "good.png": RenderQualityRecord(image_id="good.png", alpha_coverage=0.9, visible_ratio=0.8),
        },
        min_alpha_coverage=0.5,
        min_visible_ratio=0.5,
    )

    assert [candidate.candidate_id for candidate in filtered.candidates] == ["good_render"]
    assert filtered.candidates[0].metadata["render_quality_filter_passed"] is True
    assert filtered.candidates[0].metadata["render_alpha_coverage"] == 0.9


def test_render_quality_filter_cli_writes_filtered_bank(tmp_path):
    bank_path = tmp_path / "bank.jsonl"
    sidecar_path = tmp_path / "quality.jsonl"
    output = tmp_path / "filtered.jsonl"
    _bank().to_jsonl(bank_path)
    sidecar_path.write_text(
        "\n".join(
            [
                json.dumps({"image_id": "bad.png", "alpha_coverage": 0.1, "visible_ratio": 0.2}),
                json.dumps({"image_id": "good.png", "alpha_coverage": 0.9, "visible_ratio": 0.8}),
            ]
        )
        + "\n"
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "feature_extract.tools.vfm.filter_candidate_bank_render_quality",
            "--candidate_bank",
            str(bank_path),
            "--quality_sidecar",
            str(sidecar_path),
            "--min_alpha_coverage",
            "0.5",
            "--min_visible_ratio",
            "0.5",
            "--output",
            str(output),
        ],
        check=True,
    )

    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert lines[0]["protocol_kind"] == "reference_pose"
    assert lines[1]["candidate_id"] == "good_render"
