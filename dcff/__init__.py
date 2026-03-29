# DCFF: Deferred Cascaded Feature Field
# A hybrid explicit-implicit 3D feature representation for 6-DOF visual localization.
#
# Architecture:
#   Explicit (2DGS surfels + 16d latent z_i) → Fine geometric features (sharp edges)
#   Implicit (Multi-res Hash Grid + MLP)      → Coarse semantic features (smooth gradients)
#   Deferred screen-space rendering decouples 3D complexity from 2D feature decode.

from .hash_grid import SpatialHashGrid
from .deferred_renderer import DeferredCascadedRenderer
from .hybrid_gaussian import HybridGaussianModel
from .losses import DCFFLoss
