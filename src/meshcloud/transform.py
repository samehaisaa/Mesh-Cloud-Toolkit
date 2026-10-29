"""Rigid transforms in 3D, stored as 4x4 homogeneous matrices."""

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix of a 3-vector, so that skew(a) @ b == cross(a, b)."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def axis_angle_to_matrix(axis, angle: float) -> np.ndarray:
    """Rotation of ``angle`` radians about ``axis`` (Rodrigues' formula)."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("rotation axis must be non-zero")
    k = skew(axis / norm)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def rotvec_to_matrix(rotvec) -> np.ndarray:
    """Rotation matrix of a rotation vector, whose norm is the angle in radians."""
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        k = skew(rotvec)
        return np.eye(3) + k + 0.5 * (k @ k)
    return axis_angle_to_matrix(rotvec, angle)


def random_rotation(angle: float, rng=None) -> np.ndarray:
    """Rotation of exactly ``angle`` radians about a uniformly random axis."""
    rng = np.random.default_rng(rng)
    return axis_angle_to_matrix(rng.normal(size=3), angle)


def make_transform(rotation=None, translation=None) -> np.ndarray:
    """4x4 transform from a 3x3 rotation and a 3-vector translation."""
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = rotation
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def rotation_about(rotation, center) -> np.ndarray:
    """4x4 transform that rotates about ``center`` instead of the origin."""
    center = np.asarray(center, dtype=float)
    return make_transform(rotation, center - rotation @ center)


def invert(transform) -> np.ndarray:
    """Inverse of a rigid transform, without a general matrix inverse."""
    rotation = transform[:3, :3]
    return make_transform(rotation.T, -rotation.T @ transform[:3, 3])


def compose(*transforms) -> np.ndarray:
    """Product of transforms. ``compose(a, b)`` applies ``b`` first, then ``a``."""
    result = np.eye(4)
    for transform in transforms:
        result = result @ transform
    return result


def apply(transform, points) -> np.ndarray:
    """Applies a 4x4 transform to an (N, 3) array of points."""
    points = np.asarray(points, dtype=float)
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_error(estimate, truth) -> float:
    """Angle in degrees of the rotation that takes ``truth`` to ``estimate``.

    Accepts 3x3 rotations or 4x4 transforms.
    """
    relative = np.asarray(estimate)[:3, :3].T @ np.asarray(truth)[:3, :3]
    cos = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def translation_error(estimate, truth) -> float:
    """Distance between the translation parts of two 4x4 transforms."""
    return float(np.linalg.norm(np.asarray(estimate)[:3, 3] - np.asarray(truth)[:3, 3]))
