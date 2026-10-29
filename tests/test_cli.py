import subprocess
import sys

import numpy as np

from meshcloud.cli import main
from meshcloud.io import read_mesh, read_points, write_obj, write_ply
from meshcloud.shapes import bumpy_sphere
from meshcloud.transform import apply, make_transform, random_rotation


def test_sample_then_register(tmp_path, capsys):
    vertices, faces = bumpy_sphere()
    write_obj(tmp_path / "shape.obj", vertices, faces)

    assert main(["sample", str(tmp_path / "shape.obj"), "-n", "3000",
                 "-o", str(tmp_path / "target.ply"), "--normals"]) == 0  # fmt: skip
    target = read_mesh(tmp_path / "target.ply")
    assert target.vertices.shape == (3000, 3)
    assert target.normals.shape == (3000, 3)

    moved = make_transform(random_rotation(np.radians(10), rng=0), [0.05, 0.0, 0.0])
    write_ply(tmp_path / "source.ply", apply(moved, target.vertices))
    capsys.readouterr()
    assert main(["register", str(tmp_path / "source.ply"), str(tmp_path / "target.ply"),
                 "-o", str(tmp_path / "aligned.ply")]) == 0  # fmt: skip
    output = capsys.readouterr().out
    assert "converged True" in output
    rmse = float(
        next(line for line in output.splitlines() if line.startswith("rmse")).split()[1]
    )
    assert rmse < 1e-3
    aligned = read_points(tmp_path / "aligned.ply")
    assert np.abs(aligned - target.vertices).max() < 1e-2


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "meshcloud", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "register" in result.stdout
