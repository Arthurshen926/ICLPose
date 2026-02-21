"""
Modules package
"""

from .transformer import TransformerLayer, MultiHeadAttention, AttentionLayer, AttentionOutput
from .fusion_module import CrossModalFusionModule, LearnableQueryEmbedding
from .pose_regressor import (
    PoseRegressor, 
    PoseRegressorQuaternion,
    rotation_vector_to_matrix,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d
)

# C2F (Coarse-to-Fine) 模块
from .fusion_module_c2f import (
    CrossModalFusionModuleC2F,
    PositionalEncoding2DIntrinsic,
    PositionalEncoding3DNeRF,
    ICPoseNetC2F,
    PoseRegressionHead,
)

# MaRepo 风格位姿回归器
from .pose_regressor_marepo import (
    PoseRegressorMaRepo,
    PoseRegressorMaRepoIterative,
)

# 重叠检测模块
from .overlap_detection import (
    OverlapEstimator,
    FrustumPosePredictor,
    OverlapDetectionModule,
    OverlapLoss,
)

__all__ = [
    'TransformerLayer',
    'MultiHeadAttention',
    'AttentionLayer',
    'AttentionOutput',
    'CrossModalFusionModule',
    'LearnableQueryEmbedding',
    'PoseRegressor',
    'PoseRegressorQuaternion',
    'rotation_vector_to_matrix',
    'quaternion_to_matrix',
    # C2F 模块
    'CrossModalFusionModuleC2F',
    'PositionalEncoding2DIntrinsic',
    'PositionalEncoding3DNeRF',
    'ICPoseNetC2F',
    'PoseRegressionHead',
    # MaRepo 模块
    'PoseRegressorMaRepo',
    'PoseRegressorMaRepoIterative',
    # 重叠检测模块
    'OverlapEstimator',
    'FrustumPosePredictor',
    'OverlapDetectionModule',
    'OverlapLoss',
]
