from feature_extract.tools.vfm.render_goal_maplet_selected_candidate_env import (
    ENV_ARGUMENTS,
    selected_arguments,
    selected_environment,
)
from feature_extract.tools.vfm.freeze_goal_maplet_g23_configuration import (
    DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS,
)


def test_selected_candidate_environment_preserves_screening_configuration():
    arguments = {key: index for index, key in enumerate(ENV_ARGUMENTS)}
    payload = {
        "artifact_type": "goal_maplet_candidate_oof_development_selection_v1",
        "recommended_tag": "exact64_anchor16x4",
        "recommended": {"inference_arguments": arguments},
    }
    environment = selected_environment(payload)
    assert environment["G23_CANDIDATE_TAG"] == "exact64_anchor16x4"
    assert environment["G23_EXACT_VERIFY_COUNT"] == str(
        arguments["view_geometry_exact_verify_count"]
    )
    assert environment["G23_TRANSLATION_NMS_M"] == str(
        arguments["translation_nms_m"]
    )


def test_selected_candidate_arguments_cover_full_frozen_schema():
    arguments = {
        key: False if key == "view_geometry_visibility_chart" else index
        for index, key in enumerate(DEPLOYMENT_CANDIDATE_ARGUMENT_KEYS)
    }
    payload = {
        "artifact_type": "goal_maplet_candidate_oof_development_selection_v1",
        "recommended": {"inference_arguments": arguments},
    }
    tokens = selected_arguments(payload)
    assert "--view_geometry_exact_pool_semantics" in tokens
    assert "--view_geometry_visibility_chart" not in tokens
    assert "--geometry_support_pairs" in tokens
