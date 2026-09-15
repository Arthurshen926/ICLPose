"""Read only Cambridge image-name/camera bindings and native normalized intrinsics.

The legacy 1024px retriangulated text export is not interchangeable with the
native calibration: its tiny radial numbers must not be used unverified as
OpenCV normalized distortion. This helper replays the existing RADIO-canvas
then work-grid resize contract, without decoding image poses or 3D tracks.
"""
import struct
from pathlib import Path
from feature_extract.vfm.colmap_tracks import read_colmap_cameras_binary
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import _scaled_intrinsics


def camera_bindings(path):
    result={}
    with open(path,'rb') as f:
        total=struct.unpack('<Q',f.read(8))[0]
        for _ in range(total):
            f.seek(4+7*8,1)  # image ID, quaternion and translation are not decoded
            camera_id=struct.unpack('<i',f.read(4))[0];name=bytearray()
            while True:
                b=f.read(1)
                if not b:raise ValueError('truncated image name')
                if b==b'\0':break
                name.extend(b)
            name=name.decode('utf8')
            if name in result:raise ValueError('duplicate image name')
            result[name]=camera_id
            count=struct.unpack('<Q',f.read(8))[0];f.seek(count*24,1)
        if f.tell()!=Path(path).stat().st_size:raise ValueError('invalid track extent or trailing bytes')
    return result


def read_native_work_cameras(scene, names, dataset_root=Path('/hy-tmp/Cambridge_stdloc')):
    model=Path(dataset_root)/scene/'sparse/0';cameras=read_colmap_cameras_binary(model/'cameras.bin');bindings=camera_bindings(model/'images.bin');out={}
    for name in names:
        image=name[:-4].replace('__','/') if name.endswith('.npz') else name;c=cameras[bindings[image]]
        if c.model_id!=2:raise ValueError('expected native SIMPLE_RADIAL authority')
        f,cx,cy,k=c.params
        # Replay existing 1024x576 RADIO canvas camera and half-pixel downsample.
        params=[f*1024/c.width,cx*1024/c.width,cy*576/c.height,k]
        out[name]=_scaled_intrinsics(c.model_id,params,1024,576)
    return out
