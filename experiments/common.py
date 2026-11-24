"""Shared setup for the experiments: the test shape, perturbations and metrics."""

import json
from pathlib import Path

import numpy as np

from meshcloud import read_mesh
from meshcloud.shapes import bumpy_sphere
from meshcloud.transform import apply, make_transform, random_rotation, rotation_about

ROOT = Path(__file__).resolve().parent.parent
BUNNY = ROOT / "data" / "bunny" / "bun_zipper.ply"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
METHODS = ("point-to-point", "point-to-plane")
SUCCESS_DEGREES = 5.0


def load_shape():
    """Returns (name, vertices, faces): the bunny if fetched, else the bumpy sphere."""
    if BUNNY.exists():
        mesh = read_mesh(BUNNY)
        return "Stanford Bunny", mesh.vertices, mesh.faces
    print("data/bunny is missing, run experiments/fetch_bunny.py first.")
    print("Falling back to the synthetic bumpy sphere.")
    vertices, faces = bumpy_sphere()
    return "bumpy sphere", vertices, faces


def diagonal(points) -> float:
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def perturbation(center, degrees, rng, shift) -> np.ndarray:
    """Rotation by ``degrees`` about a random axis through ``center``, then a
    shift of length ``shift`` in a random direction."""
    rotation = random_rotation(np.radians(degrees), rng)
    direction = rng.normal(size=3)
    direction *= shift / np.linalg.norm(direction)
    return make_transform(translation=direction) @ rotation_about(rotation, center)


def alignment_error(estimate, truth, points) -> float:
    """RMS distance between the points moved by the estimate and by the truth."""
    gap = apply(estimate, points) - apply(truth, points)
    return float(np.sqrt(np.mean(np.sum(gap**2, axis=1))))


def save_json(name: str, payload) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
