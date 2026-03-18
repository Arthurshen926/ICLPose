import sys, os
sys.path.insert(0, '.')

ok, warn, fail = 0, 0, 0

def check(name, stmt, critical=True):
    global ok, warn, fail
    try:
        exec(stmt, {})
        print(f'  OK  {name}')
        ok += 1
    except Exception as e:
        short = str(e).split('\n')[0][:70]
        if critical:
            print(f'  FAIL {name}: {short}')
            fail += 1
        else:
            print(f'  WARN {name}: {short}')
            warn += 1

print("=== Core ===")
check("torch+GPU",     "import torch; assert torch.cuda.is_available(), 'no GPU'")
check("torchvision",   "import torchvision")
check("numpy",         "import numpy")
check("scipy",         "import scipy")
check("cv2",           "import cv2")
check("PIL",           "import PIL")
check("yaml",          "import yaml")
check("tqdm",          "import tqdm")
check("einops",        "import einops")
check("matplotlib",    "import matplotlib")
check("sklearn",       "import sklearn")
check("tensorboard",   "import tensorboard")

print("\n=== 3D/Geometry ===")
check("gsplat",        "import gsplat")
check("plyfile",       "import plyfile")
check("open3d",        "import open3d")
check("trimesh",       "import trimesh")

print("\n=== DL Ecosystem ===")
check("kornia",        "import kornia")
check("timm",          "import timm")
check("munch",         "import munch")
check("transformers",  "import transformers")
check("wandb",         "import wandb")

print("\n=== CUDA Extensions ===")
check("diff_gauss",    "from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer")
check("simple_knn",    "from simple_knn._C import distCUDA2")
check("tinycudann",    "import tinycudann", critical=False)

print("\n=== ODISE Ecosystem ===")
check("fvcore",        "import fvcore")
check("detectron2",    "import detectron2")
check("mask2former",   "import mask2former", critical=False)
check("odise",         "import odise", critical=False)

print("\n=== Project Modules ===")
check("ICPoseNet",           "from ic_models.ic_pose_net import ICPoseNet")
check("ICPoseNetV3",         "from ic_models.ic_pose_net_v3 import ICPoseNetV3")
check("MultiScaleRenderer",  "from modules.multiscale_renderer import MultiScaleRenderer")
check("GaussianModel",       "from splatloc_modules.gaussian_splatting.scene.gaussian_model import GaussianModel")
check("ViTExtractor",        "from feature_extraction.extractor_dino import ViTExtractor")
check("PoseDatasetV3",       "from data.dataset_v3 import PoseDatasetV3")
check("PoseDatasetV4",       "from data.dataset_v4 import PoseDatasetV4")

print(f'\n=== Result: {ok} OK, {warn} warnings, {fail} failures ===')
if fail == 0:
    print('Environment is ready!')
else:
    print(f'{fail} critical issues need fixing')
