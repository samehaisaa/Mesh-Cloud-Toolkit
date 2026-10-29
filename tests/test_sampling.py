import numpy as np
import pytest
from scipy import stats

from meshcloud.sampling import face_areas_and_normals, sample_surface


def test_triangles_are_picked_in_proportion_to_their_area():
    # Two triangles in the plane z = 0, with areas 0.5 and 1.5.
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [5, 0, 0], [2, 1, 0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    n = 40_000
    points = sample_surface(vertices, faces, n, rng=0)
    share = np.mean(points[:, 0] >= 2.0)
    # Binomial standard error is about 0.002, so 0.01 is a 4.6 sigma band.
    assert share == pytest.approx(0.75, abs=0.01)


def test_points_are_uniform_inside_a_square():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    points = sample_surface(vertices, faces, 20_000, rng=1)
    assert np.all((points >= 0) & (points <= 1))
    for axis in (0, 1):
        assert stats.kstest(points[:, axis], "uniform").pvalue > 1e-3
    # The diagonal splits the square in two equal halves.
    assert np.mean(points[:, 0] > points[:, 1]) == pytest.approx(0.5, abs=0.015)


def test_returned_normals_are_the_face_normals():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 3, 1]])
    _, normals = sample_surface(vertices, faces, 100, rng=2, return_normals=True)
    expected = face_areas_and_normals(vertices, faces)[1]
    assert all(
        np.allclose(n, expected[0]) or np.allclose(n, expected[1]) for n in normals
    )


def test_rejects_meshes_without_area():
    vertices = np.zeros((3, 3))
    with pytest.raises(ValueError, match="zero surface area"):
        sample_surface(vertices, np.array([[0, 1, 2]]), 10)
    with pytest.raises(ValueError, match="no faces"):
        sample_surface(vertices, np.empty((0, 3), dtype=int), 10)
