"""Synthetic test meshes."""

import numpy as np


def bumpy_sphere(n_lat: int = 48, n_lon: int = 96, amplitude: float = 0.25):
    """A smooth, asymmetric closed surface: a unit sphere with low-frequency bumps.

    The bumps break every rotational symmetry, so a registration has a
    single correct answer.

    Returns:
        (vertices, faces) with vertices (N, 3) and triangle faces (M, 3).
    """
    theta = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]  # polar angle, poles excluded
    phi = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    radius = 1.0 + amplitude * (
        0.6 * np.sin(3.0 * t) * np.cos(2.0 * p)
        + 0.4 * np.cos(2.0 * t + 0.7) * np.sin(p + 0.3)
        + 0.3 * np.sin(t) ** 2 * np.cos(3.0 * p - 1.1)
    )
    ring = np.stack(
        [
            radius * np.sin(t) * np.cos(p),
            radius * np.sin(t) * np.sin(p),
            radius * np.cos(t),
        ],
        axis=-1,
    ).reshape(-1, 3)
    # Each pole sits at the mean radius of the ring next to it.
    north = [0.0, 0.0, radius[0].mean()]
    south = [0.0, 0.0, -radius[-1].mean()]
    vertices = np.vstack([ring, north, south])
    north_index, south_index = len(ring), len(ring) + 1

    def index(i, j):
        return i * n_lon + j % n_lon

    faces = []
    for j in range(n_lon):
        faces.append([north_index, index(0, j), index(0, j + 1)])
        faces.append([south_index, index(n_lat - 2, j + 1), index(n_lat - 2, j)])
        for i in range(n_lat - 2):
            a, b = index(i, j), index(i, j + 1)
            c, d = index(i + 1, j), index(i + 1, j + 1)
            faces.append([a, c, b])
            faces.append([b, c, d])
    return vertices, np.asarray(faces, dtype=np.int64)
