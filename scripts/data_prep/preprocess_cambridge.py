"""
Cambridge Landmarks 图像预处理（不含 Mask2Former）
==================================================
1. CLAHE 直方图均衡化
2. 去畸变 (cv2.undistort)
3. 保存到 {source_path}/processed/ 目录

用法:
  python scripts/data_prep/preprocess_cambridge.py \
    --source_path dataset/OldHospital
"""
import argparse
import os
import struct
import collections
import cv2
import numpy as np
from tqdm import tqdm

# ---- COLMAP 读取 (内联) ----
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
}

def read_intrinsics_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("I", f.read(4))[0]
            model_id = struct.unpack("i", f.read(4))[0]
            width = struct.unpack("Q", f.read(8))[0]
            height = struct.unpack("Q", f.read(8))[0]
            num_params = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f"{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = Camera(cam_id, CAMERA_MODELS[model_id].model_name, width, height, np.array(params))
    return cameras

def read_extrinsics_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("I", f.read(4))[0]
            qvec = struct.unpack("4d", f.read(32))
            tvec = struct.unpack("3d", f.read(24))
            cam_id = struct.unpack("I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode()
            num_pts = struct.unpack("Q", f.read(8))[0]
            f.read(num_pts * 24)  # skip 2D points
            images[img_id] = BaseImage(img_id, np.array(qvec), np.array(tvec), cam_id, name, None, None)
    return images

def hist_equalize(image, equalizer):
    b, g, r = cv2.split(image)
    return cv2.merge((equalizer.apply(r), equalizer.apply(g), equalizer.apply(b)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True, help="Cambridge scene root (e.g., dataset/OldHospital)")
    parser.add_argument("--images", default="", help="Image sub-directory under source_path (empty = root)")
    parser.add_argument("--output_folder", default="processed")
    args = parser.parse_args()

    colmap_path = os.path.join(args.source_path, "sparse", "0")
    cam_extrinsics = read_extrinsics_binary(os.path.join(colmap_path, "images.bin"))
    cam_intrinsics = read_intrinsics_binary(os.path.join(colmap_path, "cameras.bin"))

    output_path = os.path.join(args.source_path, args.output_folder)
    os.makedirs(output_path, exist_ok=True)

    hist_equalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print(f"Processing {len(cam_extrinsics)} images...")
    print(f"  Source: {args.source_path}/{args.images}")
    print(f"  Output: {output_path}")

    for key in tqdm(cam_extrinsics, desc="Preprocessing"):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        image_name = extr.name

        image_output_path = os.path.join(output_path, image_name)
        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)

        # 构造相机矩阵和畸变系数
        if intr.model == "SIMPLE_RADIAL":
            camera_matrix = np.array([
                [intr.params[0], 0, intr.params[1]],
                [0, intr.params[0], intr.params[2]],
                [0, 0, 1],
            ], dtype=np.float32)
            distortion_coeffs = np.array([intr.params[3], 0, 0, 0], dtype=np.float32)
        elif intr.model == "PINHOLE":
            camera_matrix = np.array([
                [intr.params[0], 0, intr.params[2]],
                [0, intr.params[1], intr.params[3]],
                [0, 0, 1],
            ], dtype=np.float32)
            distortion_coeffs = np.zeros(4, dtype=np.float32)
        elif intr.model == "SIMPLE_PINHOLE":
            camera_matrix = np.array([
                [intr.params[0], 0, intr.params[1]],
                [0, intr.params[0], intr.params[2]],
                [0, 0, 1],
            ], dtype=np.float32)
            distortion_coeffs = np.zeros(4, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported camera model: {intr.model}")

        # 读取图像
        image_path = os.path.join(args.source_path, args.images, image_name) if args.images else os.path.join(args.source_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  WARNING: Cannot read {image_path}, skipping")
            continue

        # CLAHE
        image = hist_equalize(image, hist_equalizer)

        # 去畸变
        image = cv2.undistort(image, camera_matrix, distortion_coeffs)

        # 保存
        cv2.imwrite(image_output_path, image)

    print(f"Done. Processed images saved to {output_path}")
    print("Note: No masks.pkl generated (Mask2Former not used). Training will proceed without semantic masks.")

if __name__ == "__main__":
    main()
