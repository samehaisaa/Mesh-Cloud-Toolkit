"""Surface normals of a point cloud by local PCA."""

import numpy as np
from scipy.spatial import cKDTree


def estimate_normals(points, k: int = 20, tree: cKDTree | None = None) -> np.ndarray:
    """Unit normal at every point, from the PCA of its ``k`` nearest neighbors.

    The normal is the eigenvector of the neighborhood covariance with the
    smallest eigenvalue. Its sign is arbitrary, which is fine for
    point-to-plane ICP since the residual is squared.
    """
    points = np.asarray(points, dtype=float)
    k = min(k, len(points))
    if k < 3:
        raise ValueError("need at least 3 points to estimate normals")
    tree = cKDTree(points) if tree is None else tree
    _, neighbors = tree.query(points, k=k)
    local = points[neighbors]
    local = local - local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", local, local)
    _, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors[:, :, 0]
