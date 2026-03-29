"""
Contrastive losses for GSFFs training.

Implements:
  - L_NCE: Symmetric pixel-level InfoNCE contrastive loss (Eq. 2)
  - L_PRO: Prototypical contrastive loss with Sinkhorn-Knopp assignment (Eq. 3)
  - L_CE: Cross-entropy loss for segmentation alignment (Eq. 5)
"""

import torch
import torch.nn.functional as F


def info_nce_loss(
    feat_3d: torch.Tensor,
    feat_2d: torch.Tensor,
    temperature: float = 0.07,
    max_samples: int = 1024,
) -> torch.Tensor:
    """
    Symmetric pixel-level InfoNCE loss (Eq. 2 in paper).
    
    For each pixel u, the positive pair is (F3D_u, F2D_u).
    Negatives are all other pixels in the batch.
    
    Args:
        feat_3d: [B, D, H, W] rendered 3D feature maps (L2-normalized)
        feat_2d: [B, D, H, W] encoded 2D feature maps (L2-normalized)
        temperature: contrastive temperature τ
        max_samples: max pixels to sample per image (for efficiency)
        
    Returns:
        scalar loss
    """
    B, D, H, W = feat_3d.shape
    
    # Flatten spatial dims: [B, D, HW]
    f3d = feat_3d.reshape(B, D, H * W)
    f2d = feat_2d.reshape(B, D, H * W)
    
    loss = torch.tensor(0.0, device=feat_3d.device)
    
    for b in range(B):
        # [D, HW]
        f3 = f3d[b]
        f2 = f2d[b]
        N = f3.shape[1]
        
        # Subsample if too many pixels
        if N > max_samples:
            idx = torch.randperm(N, device=f3.device)[:max_samples]
            f3 = f3[:, idx]
            f2 = f2[:, idx]
            N = max_samples
        
        # Similarity matrix [N, N]: sim[i,j] = f3_i · f2_j / τ
        sim = (f3.T @ f2) / temperature  # [N, N]
        
        # Symmetric NCE: anchor on 3D side and on 2D side
        labels = torch.arange(N, device=sim.device)
        loss_3d_to_2d = F.cross_entropy(sim, labels)
        loss_2d_to_3d = F.cross_entropy(sim.T, labels)
        
        loss = loss + (loss_3d_to_2d + loss_2d_to_3d) / 2.0
    
    return loss / B


@torch.no_grad()
def sinkhorn_knopp(
    scores: torch.Tensor,
    n_iters: int = 3,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp optimal transport for prototype assignment.
    
    Args:
        scores: [N, K] similarity scores between features and prototypes
        n_iters: number of Sinkhorn iterations
        epsilon: entropy regularization (temperature for softmax)
        
    Returns:
        assignments: [N, K] soft assignment matrix (doubly-stochastic)
    """
    Q = torch.exp(scores / epsilon)
    Q /= Q.sum()
    
    K = Q.shape[1]
    N = Q.shape[0]
    
    for _ in range(n_iters):
        # Row normalization: each sample sums to 1/N
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= N
        # Column normalization: each prototype sums to 1/K
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= K
    
    # Final row normalization for proper assignments
    Q = Q / Q.sum(dim=1, keepdim=True)
    return Q * N  # Scale back


def prototypical_loss(
    feat_3d: torch.Tensor,
    feat_2d: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.07,
    max_samples: int = 2048,
) -> torch.Tensor:
    """
    Prototypical contrastive loss (Eq. 3 in paper).
    
    Encourages both F3D and F2D at each pixel to be close to the
    same cluster prototype, using Sinkhorn-Knopp assignments.
    
    Args:
        feat_3d: [B, D, H, W] rendered 3D feature maps
        feat_2d: [B, D, H, W] encoded 2D feature maps
        prototypes: [K, D] L2-normalized prototype features
        temperature: contrastive temperature
        max_samples: max pixels to sample per image
        
    Returns:
        scalar loss
    """
    B, D, H, W = feat_3d.shape
    K = prototypes.shape[0]
    
    f3d = feat_3d.reshape(B, D, H * W)
    f2d = feat_2d.reshape(B, D, H * W)
    
    loss = torch.tensor(0.0, device=feat_3d.device)
    
    for b in range(B):
        f3 = f3d[b]  # [D, N]
        f2 = f2d[b]  # [D, N]
        N = f3.shape[1]
        
        if N > max_samples:
            idx = torch.randperm(N, device=f3.device)[:max_samples]
            f3 = f3[:, idx]
            f2 = f2[:, idx]
            N = max_samples
        
        # Compute similarities to prototypes
        # [N, D] @ [D, K] = [N, K]
        sim_3d = f3.T @ prototypes.T  # [N, K]
        sim_2d = f2.T @ prototypes.T  # [N, K]
        
        # Get prototype assignments using Sinkhorn-Knopp on average similarity
        avg_sim = (sim_3d + sim_2d) / 2.0
        assignments = sinkhorn_knopp(avg_sim)  # [N, K]
        proto_labels = assignments.argmax(dim=1)  # [N]
        
        # For each pixel, compute joint similarity to assigned prototype
        # sim(F3D_n, p_n) + sim(F2D_n, p_n)
        assigned_protos = prototypes[proto_labels]  # [N, D]
        joint_sim = (
            torch.sum(f3.T * assigned_protos, dim=1) +
            torch.sum(f2.T * assigned_protos, dim=1)
        ) / temperature  # [N]
        
        # Denominator: sum over all prototypes
        all_sim = (sim_3d + sim_2d) / temperature  # [N, K]
        log_denominator = torch.logsumexp(all_sim, dim=1)  # [N]
        
        loss = loss - (joint_sim - log_denominator).mean()
    
    return loss / B


def segmentation_ce_loss(
    seg_2d: torch.Tensor,
    seg_3d: torch.Tensor,
    proto_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for segmentation alignment (Eq. 5 in paper).
    
    Args:
        seg_2d: [B, K, H, W] predicted 2D segmentation logits (from encoder head)
        seg_3d: [B, K, H, W] rendered 3D pseudo-logits 
        proto_labels: [B, H, W] integer labels (prototype assignments per pixel)
        
    Returns:
        scalar loss
    """
    loss_2d = F.cross_entropy(seg_2d, proto_labels, reduction='mean')
    loss_3d = F.cross_entropy(seg_3d, proto_labels, reduction='mean')
    return (loss_2d + loss_3d) / 2.0


def compute_pseudo_logits(
    features: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute soft-assignment pseudo-logits from features and prototypes.
    
    l_ik = exp(g_i · p_k) / sum_k' exp(g_i · p_k')
    
    Args:
        features: [B, D, H, W] feature maps
        prototypes: [K, D] L2-normalized prototypes
        
    Returns:
        logits: [B, K, H, W] pseudo-logits
    """
    B, D, H, W = features.shape
    K = prototypes.shape[0]
    
    # [B, D, HW] -> [B, HW, D] @ [D, K] -> [B, HW, K] -> [B, K, H, W]
    feat_flat = features.reshape(B, D, H * W).permute(0, 2, 1)  # [B, HW, D]
    logits = feat_flat @ prototypes.T  # [B, HW, K]
    logits = logits.permute(0, 2, 1).reshape(B, K, H, W)
    
    return logits


def get_pixel_proto_labels(
    feat_3d: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """
    Get per-pixel prototype labels from rendered 3D features.
    
    Args:
        feat_3d: [B, D, H, W] rendered feature maps
        prototypes: [K, D] prototypes
        
    Returns:
        labels: [B, H, W] integer labels
    """
    logits = compute_pseudo_logits(feat_3d, prototypes)  # [B, K, H, W]
    return logits.argmax(dim=1)  # [B, H, W]
