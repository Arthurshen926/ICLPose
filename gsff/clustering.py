"""
Spectral clustering for GSFFs prototypical regularization.

Implements:
  1. KNN graph of Gaussian centers → sparse adjacency
  2. Graph Laplacian eigenvector computation
  3. K-means on eigenvectors → K spatial clusters
  4. Prototype computation from cluster-average volumetric features

Uses KNN graph instead of Delaunay for tractability with large point clouds.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans


def build_knn_adjacency(xyz: np.ndarray, k: int = 20) -> csr_matrix:
    """
    Build sparse adjacency matrix from KNN graph (fast alternative to Delaunay).
    
    Args:
        xyz: [N, 3] point cloud coordinates
        k: number of nearest neighbors
        
    Returns:
        adjacency: [N, N] sparse CSR adjacency matrix (symmetric, binary)
    """
    N = xyz.shape[0]
    tree = KDTree(xyz)
    distances, indices = tree.query(xyz, k=k + 1)  # +1 for self
    
    rows = np.repeat(np.arange(N), k)
    cols = indices[:, 1:].flatten()  # skip self
    data = np.ones(len(rows), dtype=np.float64)
    
    adj = csr_matrix((data, (rows, cols)), shape=(N, N))
    # Make symmetric
    adj = adj + adj.T
    adj.data[:] = 1.0  # binary
    return adj


def spectral_clustering(
    xyz: np.ndarray,
    n_clusters: int = 34,
    n_eigenvectors: int = 50,
    subsample: int = None,
    use_fast_fallback: bool = True,
) -> np.ndarray:
    """
    Spectral clustering on the KNN graph of Gaussian centers.
    Falls back to K-means on coordinates if spectral is too slow.
    
    Args:
        xyz: [N, 3] Gaussian center positions (numpy)
        n_clusters: K (number of clusters, 34 in paper)
        n_eigenvectors: number of Laplacian eigenvectors to use
        subsample: if set, subsample points first then propagate labels
        use_fast_fallback: use K-means on coordinates (fast, ~2s) instead
                           of spectral clustering (can be very slow)
        
    Returns:
        labels: [N] integer cluster assignments (0..K-1)
    """
    N = xyz.shape[0]
    
    if use_fast_fallback:
        # Fast K-means on spatial coordinates (subsample for speed)
        sub_n = min(subsample or N, 50000)
        if N > sub_n:
            idx = np.random.choice(N, sub_n, replace=False)
            xyz_sub = xyz[idx]
            kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=42)
            labels_sub = kmeans.fit_predict(xyz_sub)
            tree = KDTree(xyz_sub)
            _, nn_idx = tree.query(xyz, k=1)
            labels = labels_sub[nn_idx.flatten()]
        else:
            kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=42)
            labels = kmeans.fit_predict(xyz)
        return labels
    
    if subsample is not None and N > subsample:
        idx = np.random.choice(N, subsample, replace=False)
        xyz_sub = xyz[idx]
        labels_sub = _spectral_cluster_core(xyz_sub, n_clusters, n_eigenvectors)
        
        # Propagate labels to all points via nearest neighbor
        tree = KDTree(xyz_sub)
        _, nn_idx = tree.query(xyz, k=1)
        labels = labels_sub[nn_idx.flatten()]
    else:
        labels = _spectral_cluster_core(xyz, n_clusters, n_eigenvectors)
    
    return labels


def _spectral_cluster_core(
    xyz: np.ndarray,
    n_clusters: int,
    n_eigenvectors: int,
) -> np.ndarray:
    """Core spectral clustering (KNN graph + Laplacian eigenvectors + K-means)."""
    N = xyz.shape[0]
    
    # Build KNN adjacency (much faster than Delaunay for large N)
    adj = build_knn_adjacency(xyz, k=min(20, N - 1))
    
    # Compute graph Laplacian: L = D - A
    degrees = np.array(adj.sum(axis=1)).flatten()
    D = csr_matrix((degrees, (range(N), range(N))), shape=(N, N))
    L = D - adj
    
    # Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
    d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv_sqrt = csr_matrix((d_inv_sqrt, (range(N), range(N))), shape=(N, N))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Compute smallest eigenvectors (skip eigenvalue 0)
    n_eig = min(n_eigenvectors + 1, N - 1)
    eigenvalues, eigenvectors = eigsh(L_norm, k=n_eig, which='SM', tol=1e-4)
    
    # Skip the trivial eigenvector (eigenvalue ≈ 0)
    eigenvectors = eigenvectors[:, 1:]  # [N, n_eig-1]
    
    # Normalize rows
    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    eigenvectors = eigenvectors / norms
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(eigenvectors)
    
    return labels


def compute_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    """
    Compute cluster prototypes by averaging volumetric features.
    
    Args:
        features: [N, D] volumetric features for all Gaussians
        labels: [N] integer cluster labels (0..K-1)
        n_clusters: K
        
    Returns:
        prototypes: [K, D] L2-normalized prototype features
    """
    D = features.shape[1]
    prototypes = torch.zeros(n_clusters, D, device=features.device, dtype=features.dtype)
    counts = torch.zeros(n_clusters, device=features.device)
    
    for k in range(n_clusters):
        mask = labels == k
        if mask.any():
            prototypes[k] = features[mask].mean(dim=0)
            counts[k] = mask.sum()
    
    # L2 normalize
    prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)
    
    return prototypes
