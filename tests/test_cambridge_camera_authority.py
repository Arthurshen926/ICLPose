import struct
import numpy as np
from feature_extract.tools.vfm.cambridge_camera_authority import camera_bindings
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics


def test_camera_binding_ignores_pose_and_tracks(tmp_path):
    p=tmp_path/'images.bin'
    p.write_bytes(struct.pack('<Q',1)+struct.pack('<i7di',7,*([float('nan')]*7),42)+b'seq13/frame00001.png\0'+struct.pack('<Qddq',1,float('nan'),float('nan'),-1))
    assert camera_bindings(p)=={'seq13/frame00001.png':42}


def test_radial_and_pixel_phase_are_preserved():
    K,k=_scaled_intrinsics(2,np.array([880.,512.,288.,.043]),1024,576)
    np.testing.assert_array_equal(K,np.array([[220.,0.,127.625],[0.,220.,71.625],[0.,0.,1.]]))
    assert k==.043
