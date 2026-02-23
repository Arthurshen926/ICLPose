"""
损失函数模块初始化文件
"""

from .pose_loss import PoseLoss, PoseLossKendall
from .reprojection_loss import ReprojectionLoss, reprojection_loss, project_3d_to_2d

# C2F (Coarse-to-Fine) 损失函数
from .pose_loss_c2f import (
    PoseLossC2F,
    PoseLossMapFree,
)

# V3: Sequence Loss for iterative refinement
from .sequence_loss import (
    SequenceLoss,
    PoseOnlySequenceLoss,
    rotation_geodesic_loss,
    translation_loss,
    confidence_weighted_flow_loss,
)

__all__ = [
    'PoseLoss',
    'PoseLossKendall',
    'ReprojectionLoss',
    'reprojection_loss',
    'project_3d_to_2d',
    # C2F 损失函数
    'PoseLossC2F',
    'PoseLossMapFree',
    # V3 Sequence Loss
    'SequenceLoss',
    'PoseOnlySequenceLoss',
    'rotation_geodesic_loss',
    'translation_loss',
    'confidence_weighted_flow_loss',
]
