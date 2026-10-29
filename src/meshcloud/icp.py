"""Iterative Closest Point registration, point-to-point and point-to-plane."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .normals import estimate_normals
from .transform import apply, make_transform, rotvec_to_matrix

METHODS = ("point-to-point", "point-to-plane")


@dataclass
class ICPResult:
    """Outcome of :func:`icp`.

    Attributes:
        transformation: 4x4 transform that maps the source onto the target.
        rmse: root mean square distance of the accepted correspondences.
        fitness: fraction of source points with an accepted correspondence.
        iterations: number of updates performed.
        converged: whether the RMSE change dropped below the tolerance.
        history: cumulative transform after each update, initial guess first.
        rmse_history: RMSE matching each entry of ``history``.
    """

    transformation: np.ndarray
    rmse: float
    fitness: float
    iterations: int
    converged: bool
    history: list[np.ndarray] = field(default_factory=list)
    rmse_history: list[float] = field(default_factory=list)


def best_fit_transform(source, target, weights=None) -> np.ndarray:
    """Rigid transform minimizing the squared distances between paired points.

    The closed form of Umeyama (1991) without scale. The determinant
    correction keeps the result a rotation even when the best orthogonal fit
    would be a reflection.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    w = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    source_mean = w @ source
    target_mean = w @ target
    covariance = (target - target_mean).T @ ((source - source_mean) * w[:, None])
    u, _, vt = np.linalg.svd(covariance)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    return make_transform(rotation, target_mean - rotation @ source_mean)


def point_to_plane_step(source, target, normals) -> np.ndarray:
    """One linearized point-to-plane update (Chen and Medioni, 1992).

    Minimizes sum(((R p + t - q) . n)^2) with R linearized as I + [w]x, then
    turns w back into an exact rotation. Points are centered first so the
    small-angle step does not depend on where the origin is.
    """
    center = source.mean(axis=0)
    p, q = source - center, target - center
    a = np.hstack([np.cross(p, normals), normals])
    b = -np.einsum("ij,ij->i", p - q, normals)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    rotation = rotvec_to_matrix(x[:3])
    translation = x[3:] + center - rotation @ center
    return make_transform(rotation, translation)


def icp(
    source,
    target,
    *,
    method: str = "point-to-plane",
    init=None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float | None = None,
    overlap: float = 1.0,
    target_normals=None,
    knn: int = 20,
) -> ICPResult:
    """Aligns ``source`` onto ``target`` with ICP.

    Each iteration pairs every moved source point with its nearest target
    point, drops pairs farther than ``max_correspondence_distance``, keeps
    only the ``overlap`` fraction of source points with the closest pairs
    (trimmed ICP, Chetverikov et al. 2002), and solves for the update.

    Args:
        source: (N, 3) points to move.
        target: (M, 3) fixed points.
        method: "point-to-point" (Besl and McKay 1992) or "point-to-plane"
            (Chen and Medioni 1992).
        init: initial 4x4 guess. Defaults to the identity.
        max_iterations: maximum number of updates.
        tolerance: stop when the RMSE changes by less than this fraction of
            the target's bounding box diagonal.
        max_correspondence_distance: reject pairs farther apart than this.
        overlap: fraction of source points expected to overlap the target,
            in (0, 1]. Values below 1 turn on trimming.
        target_normals: (M, 3) normals for point-to-plane. Estimated by PCA
            over ``knn`` neighbors when missing.
        knn: neighborhood size for normal estimation.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got '{method}'")
    if not 0.0 < overlap <= 1.0:
        raise ValueError("overlap must be in (0, 1]")
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    tree = cKDTree(target)
    if method == "point-to-plane" and target_normals is None:
        target_normals = estimate_normals(target, knn, tree)
    diagonal = np.linalg.norm(target.max(axis=0) - target.min(axis=0))
    keep_count = max(int(np.ceil(overlap * len(source))), 1)
    min_pairs = 3 if method == "point-to-point" else 6

    def match(transform):
        moved = apply(transform, source)
        distances, indices = tree.query(moved)
        keep = np.arange(len(source))
        if max_correspondence_distance is not None:
            keep = keep[distances <= max_correspondence_distance]
        if len(keep) > keep_count:
            keep = keep[np.argpartition(distances[keep], keep_count - 1)[:keep_count]]
        rmse = float(np.sqrt(np.mean(distances[keep] ** 2))) if len(keep) else np.inf
        return moved[keep], indices[keep], rmse

    transform = np.eye(4) if init is None else np.array(init, dtype=float)
    moved, indices, rmse = match(transform)
    history, rmse_history = [transform.copy()], [rmse]
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        if len(moved) < min_pairs:
            iterations -= 1
            break
        if method == "point-to-point":
            step = best_fit_transform(moved, target[indices])
        else:
            step = point_to_plane_step(moved, target[indices], target_normals[indices])
        transform = step @ transform
        moved, indices, new_rmse = match(transform)
        history.append(transform.copy())
        rmse_history.append(new_rmse)
        change = abs(rmse - new_rmse)
        rmse = new_rmse
        if change < tolerance * diagonal:
            converged = True
            break

    return ICPResult(
        transformation=transform,
        rmse=rmse,
        fitness=len(moved) / len(source),
        iterations=iterations,
        converged=converged,
        history=history,
        rmse_history=rmse_history,
    )
