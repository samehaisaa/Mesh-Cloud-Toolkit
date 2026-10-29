"""Uniform sampling of points on a triangle mesh surface."""

import numpy as np


def face_areas_and_normals(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    """Area and unit normal of every triangle. Degenerate faces get a zero normal."""
    vertices = np.asarray(vertices, dtype=float)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    cross = np.cross(b - a, c - a)
    length = np.linalg.norm(cross, axis=1)
    normals = np.divide(
        cross, length[:, None], out=np.zeros_like(cross), where=length[:, None] > 0
    )
    return 0.5 * length, normals


def sample_surface(vertices, faces, n: int, rng=None, return_normals: bool = False):
    """Samples ``n`` points uniformly over the area of a triangle mesh.

    A triangle is picked with probability proportional to its area, then a
    point inside it is drawn with the square-root barycentric trick, which is
    uniform over the triangle (Osada et al., Shape Distributions, 2002).

    Args:
        vertices: (N, 3) vertex positions.
        faces: (M, 3) vertex indices of each triangle.
        n: number of points to draw.
        rng: seed or numpy Generator.
        return_normals: also return the normal of the face each point lies on.

    Returns:
        (n, 3) points, and (n, 3) face normals if ``return_normals``.
    """
    rng = np.random.default_rng(rng)
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("the mesh has no faces to sample from")
    areas, normals = face_areas_and_normals(vertices, faces)
    total = areas.sum()
    if total <= 0:
        raise ValueError("the mesh has zero surface area")

    chosen = rng.choice(len(faces), size=n, p=areas / total)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    weights = np.column_stack([1.0 - s, s * (1.0 - r2), s * r2])
    corners = vertices[faces[chosen]]
    points = np.einsum("ij,ijk->ik", weights, corners)
    if return_normals:
        return points, normals[chosen]
    return points
