import numpy as np
import pytest

from meshcloud.io import read_mesh, read_points, write_obj, write_ply
from meshcloud.shapes import bumpy_sphere


@pytest.fixture
def mesh():
    return bumpy_sphere(n_lat=8, n_lon=12)


@pytest.mark.parametrize("binary", [True, False])
def test_ply_round_trip(tmp_path, mesh, binary):
    vertices, faces = mesh
    normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    path = tmp_path / "mesh.ply"
    write_ply(path, vertices, normals=normals, faces=faces, binary=binary)
    loaded = read_mesh(path)
    np.testing.assert_array_equal(loaded.vertices, vertices)
    np.testing.assert_array_equal(loaded.normals, normals)
    np.testing.assert_array_equal(loaded.faces, faces)


def test_ply_point_cloud_has_no_faces(tmp_path):
    points = np.random.default_rng(0).normal(size=(50, 3))
    write_ply(tmp_path / "cloud.ply", points)
    loaded = read_mesh(tmp_path / "cloud.ply")
    np.testing.assert_array_equal(loaded.vertices, points)
    assert loaded.faces.shape == (0, 3)
    assert loaded.normals is None


def test_obj_round_trip(tmp_path, mesh):
    vertices, faces = mesh
    write_obj(tmp_path / "mesh.obj", vertices, faces)
    loaded = read_mesh(tmp_path / "mesh.obj")
    np.testing.assert_array_equal(loaded.vertices, vertices)
    np.testing.assert_array_equal(loaded.faces, faces)


def test_obj_quads_slashes_and_negative_indices(tmp_path):
    (tmp_path / "quad.obj").write_text(
        "# a unit square\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\n"
        "f 1/1/1 2/1/1 3/1/1 4/1/1\n"
        "f -4 -3 -2\n"
    )
    loaded = read_mesh(tmp_path / "quad.obj")
    np.testing.assert_array_equal(loaded.faces, [[0, 1, 2], [0, 2, 3], [0, 1, 2]])


def test_binary_ply_with_mixed_polygons_and_extra_properties(tmp_path):
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment written by hand\n"
        "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\n"
        "element face 2\nproperty list uchar int vertex_indices\n"
        "end_header\n"
    )
    vertex = np.zeros(4, dtype=[("p", "<f4", (3,)), ("red", "u1")])
    vertex["p"] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    body = vertex.tobytes()
    body += np.array([4], "u1").tobytes() + np.array([0, 1, 2, 3], "<i4").tobytes()
    body += np.array([3], "u1").tobytes() + np.array([0, 1, 3], "<i4").tobytes()
    (tmp_path / "mixed.ply").write_bytes(header.encode() + body)
    loaded = read_mesh(tmp_path / "mixed.ply")
    np.testing.assert_array_equal(loaded.vertices[2], [1, 1, 0])
    np.testing.assert_array_equal(loaded.faces, [[0, 1, 2], [0, 2, 3], [0, 1, 3]])


def test_read_points_and_unknown_suffix(tmp_path, mesh):
    write_obj(tmp_path / "mesh.obj", *mesh)
    assert read_points(tmp_path / "mesh.obj").shape == mesh[0].shape
    with pytest.raises(ValueError, match="unsupported"):
        read_mesh(tmp_path / "mesh.stl")
