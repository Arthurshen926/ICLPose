import numpy as np
import pytest
from feature_extract.tools.vfm.audit_goal_maplet_mapping_joint_geometry import summarize


def test_oracle_decomposition_exact_with_correlated_errors():
    image=np.array([[1.,0.],[0.,2.]])
    uv=np.array([[.3,.2],[1.,-.2]])
    normal=np.array([[.2,0.],[0.,.1]])
    bank=dict(evaluation_rows=np.arange(2),image_error=image,projected_world_error=uv+normal,
              offplane_image_error=normal,residual=image-uv-normal,normal_residual_m=np.array([.1,.2]),
              head_content_sha256=np.asarray('test'))
    report=summarize(bank)
    assert sum(report['squared_residual_decomposition_px2'].values())==pytest.approx(np.mean(np.sum((image-uv-normal)**2,axis=1)))
    assert report['errors']['oracle_image_and_uv']['mean_squared_px']==pytest.approx(.025)
