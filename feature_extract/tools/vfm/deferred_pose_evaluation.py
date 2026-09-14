"""Seal inference status without loading pose labels or inventing evaluation data."""
import json
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def write_deferred_evaluation(output, frozen_pose, queries):
    output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(artifact_type='pose_inference_without_label_evaluation_v412',
                evaluation_deferred=True,query_pose_or_ground_truth_opened=False,
                frozen_pose_path=str(frozen_pose),frozen_pose_sha256=file_sha256(frozen_pose),
                query_count=int(queries))
    output.write_text(json.dumps(report,indent=2)+'\n')
