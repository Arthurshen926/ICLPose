"""Regional evidence conditioned on an inferred base pose, never a pose label.

Reprojection uses coarse 256 x 144 token centers. A token contributes at most
one vote even if it has several 3D explanations. Consistency with this pose is
conditional evidence: a wrong base pose can also have consistent matches.
"""
import numpy as np
from feature_extract.tools.vfm.native_hybrid_region_value import pooled_additions

FEATURE_NAMES = (
    'base_pose_finite', 'base_supported_image_fraction',
    'base_supported_token_fraction', 'base_clipped_residual',
    'base_supported_quadrants', 'candidate_positive_depth_fraction',
    'candidate_support_2px_fraction', 'candidate_support_4px_fraction',
    'candidate_clipped_residual', 'candidate_novel_support',
    'union_gained_support', 'union_lost_support',
    'candidate_supported_quadrants', 'candidate_support_spread',
)


def token_residuals(world, tokens, pose, K, k1):
    """Return unique tokens, their best valid residual, and positive-depth flags."""
    tokens = np.asarray(tokens, int)
    world = np.asarray(world, float)
    unique, inverse = np.unique(tokens, return_inverse=True)
    if not len(tokens):
        return unique, np.empty(0), np.empty(0, bool)
    cam = world @ pose[:3, :3].T + pose[:3, 3]
    z = cam[:, 2]
    positive = np.isfinite(cam).all(1) & (z > 1e-8)
    with np.errstate(over='ignore', invalid='ignore'):
        xy = cam[:, :2] / np.where(positive, z, 1.)[:, None]
        xy *= 1 + k1 * np.sum(xy * xy, axis=1)[:, None]
        xy = xy * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
        pixel = np.c_[(tokens % 64) * 4 + 1.5, (tokens // 64) * 4 + 1.5]
        residual = np.linalg.norm(xy - pixel, axis=1)
    residual[~positive | ~np.isfinite(residual)] = np.inf
    best = np.full(len(unique), np.inf)
    np.minimum.at(best, inverse, residual)
    has_positive = np.zeros(len(unique), bool)
    np.logical_or.at(has_positive, inverse, positive)
    return unique, best, has_positive


def _quadrants(tokens):
    if not len(tokens):
        return 0.
    return len(np.unique((tokens // 64 >= 18).astype(int) * 2 + (tokens % 64 >= 32))) / 4


def _clipped_residual(residual):
    # Missing candidate evidence must not appear to have zero residual.
    return float(np.mean(np.minimum(residual, 32))) / 32 if len(residual) else 1.


def pose_features(original_tokens, original_prototypes, rt, rp, rs, world, pose, K, k1):
    """Return eight candidate feature rows after the real 1024-row merge cap.

    The action is adding one of ranks 9..16 to the fixed first eight regions.
    Original frontend pairs are always retained and excluded from additions.
    """
    if not np.isfinite(pose).all():
        return np.zeros((8, len(FEATURE_NAMES)))
    bt, bp, _ = pooled_additions(original_tokens, original_prototypes, rt, rp, rs, list(range(8)))
    base_t = np.r_[original_tokens, bt]
    base_p = np.r_[original_prototypes, bp]
    ut, be, _ = token_residuals(world[base_p], base_t, pose, K, k1)
    supported = ut[be <= 4]
    base = [1., len(supported) / 2304, float(np.sum(be <= 4)) / max(len(ut), 1),
            _clipped_residual(be), _quadrants(supported)]
    features = []
    for candidate in range(8, 16):
        ct, cp, _ = pooled_additions(original_tokens, original_prototypes, rt, rp, rs, [candidate])
        nt, np_, _ = pooled_additions(original_tokens, original_prototypes, rt, rp, rs,
                                     list(range(8)) + [candidate])
        t, e, positive = token_residuals(world[cp], ct, pose, K, k1)
        un, ne, _ = token_residuals(world[np.r_[original_prototypes, np_]],
                                   np.r_[original_tokens, nt], pose, K, k1)
        new_support = un[ne <= 4]
        candidate_support = t[e <= 4]
        den = max(len(t), 1)
        spread = (float(np.std(candidate_support % 64) * np.std(candidate_support // 64))
                  / (64 * 36)) if len(candidate_support) else 0.
        features.append(base + [
            float(np.sum(positive)) / den, float(np.sum(e <= 2)) / den,
            float(np.sum(e <= 4)) / den, _clipped_residual(e),
            len(np.setdiff1d(candidate_support, supported)) / 2304,
            len(np.setdiff1d(new_support, supported)) / 2304,
            len(np.setdiff1d(supported, new_support)) / 2304,
            _quadrants(candidate_support), spread,
        ])
    return np.asarray(features)
