"""
统一图像预处理脚本 (v2)
=========================
替代 preprocess_cambridge.py，去掉 CLAHE，加入 SegFormer 语义分割 mask。

功能:
  1. 去畸变 (cv2.undistort)  —— 仅此一项图像增强
  2. SegFormer 语义分割 → 生成 masks.pkl (stuff_mask, sky_mask, undistort_mask)
  3. 自动场景分析：统计天空/动态物体比例，输出场景报告
  4. 支持室内/室外/混合场景，无需手动切换

masks.pkl 格式 (与 STDLoc 兼容):
  {image_name: (stuff_mask, sky_mask, undistort_mask)}
  - stuff_mask:    bool Tensor [H, W], True=静态背景, False=动态物体(人/车等)
  - sky_mask:      bool Tensor [H, W], True=非天空, False=天空区域
  - undistort_mask: bool Tensor [H, W], True=有效区域, False=去畸变黑边

用法:
  # 基础用法 (去畸变 + SegFormer mask)
  python scripts/data_prep/preprocess_unified.py \
    --source_path dataset/OldHospital

  # 自定义输出目录名
  python scripts/data_prep/preprocess_unified.py \
    --source_path dataset/OldHospital --output_folder processed_v2

  # 仅去畸变 (不生成 mask, 例如网络不可用、模型未下载)
  python scripts/data_prep/preprocess_unified.py \
    --source_path dataset/OldHospital --no_mask

  # 指定 SegFormer 模型路径 (本地目录)
  python scripts/data_prep/preprocess_unified.py \
    --source_path dataset/OldHospital \
    --segformer_model /path/to/local/segformer-b2-finetuned-ade-512-512

  # 跳过去畸变 (已经做过, 只需生成 mask)
  python scripts/data_prep/preprocess_unified.py \
    --source_path dataset/OldHospital --skip_undistort --mask_only
"""
import argparse
import os
import sys
import struct
import collections
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# =====================================================================
# ADE20K 类别定义 (SegFormer-ADE20K-150 输出 0~149)
# =====================================================================

# 动态物体类别 (会移动的东西) —— 室内室外通用
DYNAMIC_CLASSES = {
    12,   # person
    20,   # car, automobile
    76,   # boat
    80,   # bus, autobus
    83,   # truck
    90,   # airplane
    102,  # van
    103,  # ship
    116,  # minibike, motorbike
    126,  # animal
    127,  # bicycle
}

# 天空类别
SKY_CLASSES = {
    2,    # sky
}

# 所有需要标记的"非静态背景"类别 = 动态 + 天空
MASK_OUT_CLASSES = DYNAMIC_CLASSES | SKY_CLASSES

# ADE20K 类别名称 (用于统计报告)
ADE20K_NAMES = {
    0: "wall", 1: "building", 2: "sky", 3: "floor", 4: "tree",
    5: "ceiling", 6: "road", 7: "bed", 8: "windowpane", 9: "grass",
    10: "cabinet", 11: "sidewalk", 12: "person", 13: "earth", 14: "door",
    15: "table", 16: "mountain", 17: "plant", 18: "curtain", 19: "chair",
    20: "car", 21: "water", 22: "painting", 23: "sofa", 24: "shelf",
    25: "house", 26: "sea", 27: "mirror", 28: "rug", 29: "field",
    30: "armchair", 31: "seat", 32: "fence", 33: "desk", 34: "rock",
    35: "wardrobe", 36: "lamp", 37: "bathtub", 38: "railing", 39: "cushion",
    40: "base", 41: "box", 42: "column", 43: "signboard", 44: "chest",
    45: "counter", 46: "sand", 47: "sink", 48: "skyscraper", 49: "fireplace",
    50: "refrigerator", 51: "grandstand", 52: "path", 53: "stairs",
    54: "runway", 55: "case", 56: "pool", 57: "pillow", 58: "screen",
    59: "stairway", 60: "river", 61: "bridge", 62: "bookcase", 63: "blind",
    64: "coffee table", 65: "toilet", 66: "flower", 67: "book", 68: "hill",
    69: "bench", 70: "countertop", 71: "stove", 72: "palm", 73: "kitchen island",
    74: "computer", 75: "swivel chair", 76: "boat", 77: "bar", 78: "arcade machine",
    79: "hovel", 80: "bus", 81: "towel", 82: "light", 83: "truck", 84: "tower",
    85: "chandelier", 86: "awning", 87: "streetlight", 88: "booth",
    89: "television", 90: "airplane", 91: "dirt track", 92: "apparel",
    93: "pole", 94: "land", 95: "bannister", 96: "escalator", 97: "ottoman",
    98: "bottle", 99: "buffet", 100: "poster", 101: "stage", 102: "van",
    103: "ship", 104: "fountain", 105: "conveyer belt", 106: "canopy",
    107: "washer", 108: "plaything", 109: "swimming pool", 110: "stool",
    111: "barrel", 112: "basket", 113: "waterfall", 114: "tent", 115: "bag",
    116: "minibike", 117: "cradle", 118: "oven", 119: "ball", 120: "food",
    121: "step", 122: "tank", 123: "trade name", 124: "microwave", 125: "pot",
    126: "animal", 127: "bicycle", 128: "lake", 129: "dishwasher",
    130: "screen door", 131: "blanket", 132: "sculpture", 133: "hood",
    134: "sconce", 135: "vase", 136: "traffic light", 137: "tray",
    138: "ashcan", 139: "fan", 140: "pier", 141: "crt screen", 142: "plate",
    143: "monitor", 144: "bulletin board", 145: "shower", 146: "radiator",
    147: "glass", 148: "clock", 149: "flag",
}

# =====================================================================
# COLMAP 读取 (内联，无外部依赖)
# =====================================================================

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
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
            f.read(num_pts * 24)
            images[img_id] = BaseImage(img_id, np.array(qvec), np.array(tvec), cam_id, name, None, None)
    return images


# =====================================================================
# 相机去畸变
# =====================================================================

def get_camera_matrix_and_dist(intr):
    """从 COLMAP Camera 构建 camera_matrix 和 distortion_coeffs."""
    if intr.model == "SIMPLE_RADIAL":
        camera_matrix = np.array([
            [intr.params[0], 0, intr.params[1]],
            [0, intr.params[0], intr.params[2]],
            [0, 0, 1],
        ], dtype=np.float32)
        distortion_coeffs = np.array([intr.params[3], 0, 0, 0], dtype=np.float32)
    elif intr.model == "RADIAL":
        camera_matrix = np.array([
            [intr.params[0], 0, intr.params[1]],
            [0, intr.params[0], intr.params[2]],
            [0, 0, 1],
        ], dtype=np.float32)
        distortion_coeffs = np.array([intr.params[3], intr.params[4], 0, 0], dtype=np.float32)
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
    elif intr.model == "OPENCV":
        camera_matrix = np.array([
            [intr.params[0], 0, intr.params[2]],
            [0, intr.params[1], intr.params[3]],
            [0, 0, 1],
        ], dtype=np.float32)
        distortion_coeffs = np.array(intr.params[4:8], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported camera model: {intr.model}")
    return camera_matrix, distortion_coeffs


def undistort_image(image, camera_matrix, distortion_coeffs):
    """去畸变并返回 (undistorted_image, undistort_mask)."""
    has_distortion = np.any(np.abs(distortion_coeffs) > 1e-8)
    if not has_distortion:
        # 无畸变：全部有效
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        return image, mask

    undistorted = cv2.undistort(image, camera_matrix, distortion_coeffs)
    # undistort_mask: 去畸变后黑色边缘 = 无效
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8) * 255
    return undistorted, mask


# =====================================================================
# SegFormer 语义分割
# =====================================================================

class SegFormerMasker:
    """使用 SegFormer-B2 ADE20K 做语义分割，生成 stuff/sky/dynamic mask."""

    def __init__(self, model_name_or_path="nvidia/segformer-b2-finetuned-ade-512-512",
                 device="cuda"):
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        print(f"[SegFormer] Loading model: {model_name_or_path}")
        self.processor = SegformerImageProcessor.from_pretrained(model_name_or_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name_or_path)
        self.model = self.model.to(device).eval()
        self.device = device
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[SegFormer] Loaded ({n_params:.1f}M params) on {device}")

    @torch.no_grad()
    def predict(self, image_bgr):
        """
        输入: BGR numpy image [H, W, 3] uint8
        输出: segmentation map [H, W] int64, 每个像素的 ADE20K 类别 ID (0~149)
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits  # [1, 150, h, w] (低分辨率)

        # 上采样到原图大小
        h, w = image_bgr.shape[:2]
        logits_upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        seg_map = logits_upsampled.argmax(dim=1).squeeze(0)  # [H, W]
        return seg_map.cpu()

    def get_masks(self, image_bgr):
        """
        输入: BGR numpy image [H, W, 3] uint8
        输出: (stuff_mask, sky_mask, class_stats)
          - stuff_mask:  bool Tensor [H, W], True=静态背景, False=动态物体
          - sky_mask:    bool Tensor [H, W], True=非天空, False=天空区域
          - class_stats: dict {class_id: pixel_count} 用于场景分析
        """
        seg_map = self.predict(image_bgr)  # [H, W] int64

        # stuff_mask: 排除动态物体
        stuff_mask = torch.ones_like(seg_map, dtype=torch.bool)
        for cls_id in DYNAMIC_CLASSES:
            stuff_mask[seg_map == cls_id] = False

        # sky_mask: 排除天空
        sky_mask = torch.ones_like(seg_map, dtype=torch.bool)
        for cls_id in SKY_CLASSES:
            sky_mask[seg_map == cls_id] = False

        # 统计信息
        class_stats = {}
        unique_classes = seg_map.unique().tolist()
        total_pixels = seg_map.numel()
        for cls_id in unique_classes:
            count = (seg_map == cls_id).sum().item()
            class_stats[cls_id] = count

        return stuff_mask, sky_mask, class_stats


# =====================================================================
# 简单天空检测 (备用方案, 无需神经网络)
# =====================================================================

class SimpleSkyDetector:
    """
    基于 HSV 阈值的简单天空检测。
    仅检测天空，不检测动态物体。适用于无 GPU 或模型未下载的情况。
    """

    def get_masks(self, image_bgr):
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # 蓝天: 色相 90-130, 饱和度 30-255, 亮度 100-255
        blue_sky = cv2.inRange(hsv, (90, 30, 100), (130, 255, 255))
        # 灰天/白天: 低饱和度 + 高亮度 + 图像上半部分
        gray_sky = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
        # 只在上半部分搜索灰天 (避免白墙误判)
        upper_mask = np.zeros((h, w), dtype=np.uint8)
        upper_mask[:h // 3, :] = 255
        gray_sky = cv2.bitwise_and(gray_sky, upper_mask)

        sky_binary = cv2.bitwise_or(blue_sky, gray_sky)
        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        sky_binary = cv2.morphologyEx(sky_binary, cv2.MORPH_OPEN, kernel)
        sky_binary = cv2.morphologyEx(sky_binary, cv2.MORPH_CLOSE, kernel)

        sky_mask_np = (sky_binary == 0)  # True = 非天空
        stuff_mask = torch.ones(h, w, dtype=torch.bool)  # 无动态物体检测
        sky_mask = torch.from_numpy(sky_mask_np)

        total = h * w
        sky_count = int((sky_binary > 0).sum())
        # class_stats 需要包含所有像素的类别，这样 analyze_scene 才能正确计算比例
        class_stats = {
            0: total - sky_count,  # 0=wall (代表背景/非天空)
            2: sky_count,          # 2=sky
        }

        return stuff_mask, sky_mask, class_stats


# =====================================================================
# 场景分析报告
# =====================================================================

def analyze_scene(all_stats, num_images):
    """根据所有图像的语义统计，生成场景分析报告."""
    total_pixels = 0
    class_totals = {}
    for stats in all_stats:
        for cls_id, count in stats.items():
            class_totals[cls_id] = class_totals.get(cls_id, 0) + count
            total_pixels += count

    if total_pixels == 0:
        return

    print("\n" + "=" * 60)
    print("  场景语义分析报告")
    print("=" * 60)

    # 天空比例
    sky_pixels = sum(class_totals.get(cls_id, 0) for cls_id in SKY_CLASSES)
    sky_ratio = sky_pixels / total_pixels * 100

    # 动态物体比例
    dynamic_pixels = sum(class_totals.get(cls_id, 0) for cls_id in DYNAMIC_CLASSES)
    dynamic_ratio = dynamic_pixels / total_pixels * 100

    # 需要 mask 的帧数
    frames_with_sky = 0
    frames_with_dynamic = 0
    for stats in all_stats:
        if any(stats.get(cls_id, 0) > 0 for cls_id in SKY_CLASSES):
            frames_with_sky += 1
        if any(stats.get(cls_id, 0) > 0 for cls_id in DYNAMIC_CLASSES):
            frames_with_dynamic += 1

    # 场景类型判断
    is_outdoor = sky_ratio > 1.0  # 超过1%像素是天空
    has_dynamic = dynamic_ratio > 0.1  # 超过0.1%像素是动态物体
    scene_type = []
    if is_outdoor:
        scene_type.append("室外")
    else:
        scene_type.append("室内")
    if has_dynamic:
        scene_type.append("含动态物体")
    scene_type_str = " / ".join(scene_type)

    print(f"\n  场景类型: {scene_type_str}")
    print(f"  总图像数: {num_images}")
    print(f"\n  天空区域: {sky_ratio:.2f}% (占{frames_with_sky}/{num_images}帧)")
    print(f"  动态物体: {dynamic_ratio:.2f}% (占{frames_with_dynamic}/{num_images}帧)")

    # 按比例降序显示前10个类别
    sorted_classes = sorted(class_totals.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'类别':<20} {'比例':>8} {'标记':<10}")
    print(f"  {'-' * 40}")
    for cls_id, count in sorted_classes[:15]:
        ratio = count / total_pixels * 100
        name = ADE20K_NAMES.get(cls_id, f"class_{cls_id}")
        tag = ""
        if cls_id in SKY_CLASSES:
            tag = "[SKY]"
        elif cls_id in DYNAMIC_CLASSES:
            tag = "[DYNAMIC]"
        print(f"  {name:<20} {ratio:>7.2f}% {tag}")

    print("=" * 60)

    # 建议
    print("\n  处理建议:")
    if is_outdoor and has_dynamic:
        print("  → 天空 mask + 动态物体 mask 均已激活 ✓")
    elif is_outdoor:
        print("  → 天空 mask 已激活，无明显动态物体 ✓")
    elif has_dynamic:
        print("  → 动态物体 mask 已激活，无天空 ✓ (室内场景)")
    else:
        print("  → 室内静态场景，mask 影响极小，但仍已生成 ✓")
    print()


# =====================================================================
# 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="统一图像预处理: 去畸变 + SegFormer 语义 mask (无 CLAHE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source_path", required=True,
                        help="场景根目录 (含 sparse/0/)")
    parser.add_argument("--images", default="",
                        help="图像子目录 (默认: 根目录下的图像)")
    parser.add_argument("--output_folder", default="processed",
                        help="输出子目录名 (默认: processed)")
    parser.add_argument("--no_mask", action="store_true",
                        help="不生成语义 mask (仅去畸变)")
    parser.add_argument("--mask_only", action="store_true",
                        help="仅生成 mask (不去畸变/不复制图像)")
    parser.add_argument("--skip_undistort", action="store_true",
                        help="跳过去畸变 (已经做过)")
    parser.add_argument("--segformer_model", default="nvidia/segformer-b2-finetuned-ade-512-512",
                        help="SegFormer 模型名称或本地路径")
    parser.add_argument("--simple_sky", action="store_true",
                        help="使用简单 HSV 天空检测 (无需 GPU/模型)")
    parser.add_argument("--device", default="cuda",
                        help="推理设备 (cuda/cpu)")
    parser.add_argument("--batch_viz", action="store_true",
                        help="保存 mask 可视化图像到 masks_viz/")
    args = parser.parse_args()

    # ---- 读取 COLMAP ----
    colmap_path = os.path.join(args.source_path, "sparse", "0")
    cam_extrinsics = read_extrinsics_binary(os.path.join(colmap_path, "images.bin"))
    cam_intrinsics = read_intrinsics_binary(os.path.join(colmap_path, "cameras.bin"))

    output_path = os.path.join(args.source_path, args.output_folder)
    if not args.mask_only:
        os.makedirs(output_path, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  统一预处理管线 v2 (无 CLAHE)")
    print(f"{'=' * 60}")
    print(f"  场景路径:  {args.source_path}")
    print(f"  图像来源:  {args.images or '(根目录)'}")
    print(f"  输出目录:  {output_path}")
    print(f"  图像数量:  {len(cam_extrinsics)}")
    print(f"  去畸变:    {'跳过' if args.skip_undistort or args.mask_only else '是'}")
    print(f"  语义 Mask: {'否' if args.no_mask else ('简单HSV' if args.simple_sky else 'SegFormer')}")
    print()

    # ---- 初始化分割模型 ----
    masker = None
    if not args.no_mask:
        if args.simple_sky:
            masker = SimpleSkyDetector()
            print("[Mask] 使用简单 HSV 天空检测 (无动态物体检测)")
        else:
            try:
                masker = SegFormerMasker(
                    model_name_or_path=args.segformer_model,
                    device=args.device,
                )
            except Exception as e:
                print(f"\n[ERROR] 无法加载 SegFormer 模型: {e}")
                print("[提示] 请先运行模型下载:")
                print("  python scripts/data_prep/download_segformer.py")
                print("或使用 --simple_sky 备用方案, 或 --no_mask 跳过")
                sys.exit(1)

    masks_dict = {}
    all_stats = []
    viz_dir = None
    if args.batch_viz and masker is not None:
        viz_dir = os.path.join(args.source_path, "masks_viz")
        os.makedirs(viz_dir, exist_ok=True)

    # ---- 逐图处理 ----
    for key in tqdm(cam_extrinsics, desc="Processing"):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        image_name = extr.name

        # 读取图像
        if args.images:
            image_path = os.path.join(args.source_path, args.images, image_name)
        else:
            image_path = os.path.join(args.source_path, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"  WARNING: Cannot read {image_path}, skipping")
            continue

        # ---- 去畸变 ----
        if not args.skip_undistort and not args.mask_only:
            camera_matrix, distortion_coeffs = get_camera_matrix_and_dist(intr)
            image_undist, undistort_mask_np = undistort_image(image, camera_matrix, distortion_coeffs)

            # 保存去畸变图像
            image_output_path = os.path.join(output_path, image_name)
            os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
            cv2.imwrite(image_output_path, image_undist)

            # 用去畸变后的图像做分割
            image_for_seg = image_undist
            undistort_mask = torch.from_numpy(undistort_mask_np > 0)
        elif args.mask_only:
            # mask_only: 直接用输入图像
            image_for_seg = image
            undistort_mask = torch.ones(image.shape[0], image.shape[1], dtype=torch.bool)
        else:
            # skip_undistort but not mask_only: 复制原图
            image_output_path = os.path.join(output_path, image_name)
            os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
            cv2.imwrite(image_output_path, image)
            image_for_seg = image
            undistort_mask = torch.ones(image.shape[0], image.shape[1], dtype=torch.bool)

        # ---- 语义分割 ----
        if masker is not None:
            stuff_mask, sky_mask, class_stats = masker.get_masks(image_for_seg)
            all_stats.append(class_stats)

            # 存入 dict
            masks_dict[image_name] = (stuff_mask, sky_mask, undistort_mask)

            # ---- 可视化 (可选) ----
            if viz_dir is not None:
                vis = image_for_seg.copy()
                # 动态物体 → 红色半透明
                dynamic_mask_np = (~stuff_mask).numpy()
                vis[dynamic_mask_np] = (vis[dynamic_mask_np] * 0.5 + np.array([0, 0, 200]) * 0.5).astype(np.uint8)
                # 天空 → 蓝色半透明
                sky_mask_np = (~sky_mask).numpy()
                vis[sky_mask_np] = (vis[sky_mask_np] * 0.5 + np.array([200, 100, 0]) * 0.5).astype(np.uint8)
                # 去畸变黑边 → 黑色
                distort_mask_np = (~undistort_mask).numpy()
                vis[distort_mask_np] = 0

                safe_name = image_name.replace("/", "_")
                cv2.imwrite(os.path.join(viz_dir, f"mask_{safe_name}"), vis)

    # ---- 保存 masks.pkl ----
    if masker is not None and masks_dict:
        masks_pkl_path = os.path.join(args.source_path, "masks.pkl")
        with open(masks_pkl_path, "wb") as f:
            pickle.dump(masks_dict, f)
        print(f"\n[Mask] 已保存 {len(masks_dict)} 个 mask 到 {masks_pkl_path}")

        # 场景分析
        analyze_scene(all_stats, len(masks_dict))

    if not args.mask_only:
        print(f"[Image] 已保存 {len(cam_extrinsics)} 张去畸变图像到 {output_path}")

    print("完成!")


if __name__ == "__main__":
    main()
