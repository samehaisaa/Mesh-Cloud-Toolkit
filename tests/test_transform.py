import numpy as np
import pytest

from meshcloud.transform import (
    apply,
    axis_angle_to_matrix,
    compose,
    invert,
    make_transform,
    random_rotation,
    rotation_about,
    rotation_error,
    rotvec_to_matrix,
    translation_error,
)


@pytest.mark.parametrize("angle", [0.0, 0.1, 1.0, np.pi / 2, 3.0])
def test_random_rotation_is_a_rotation_of_the_requested_angle(angle):
    rotation = random_rotation(angle, rng=0)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation_error(rotation, np.eye(3)) == pytest.approx(
        np.degrees(angle), abs=1e-6
    )


def test_axis_angle_matches_the_right_hand_rule():
    rotation = axis_angle_to_matrix([0, 0, 1], np.pi / 2)
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_rotvec_agrees_with_axis_angle_and_handles_tiny_angles():
    rotvec = np.array([0.3, -0.2, 0.5])
    np.testing.assert_allclose(
        rotvec_to_matrix(rotvec),
        axis_angle_to_matrix(rotvec, np.linalg.norm(rotvec)),
        atol=1e-12,
    )
    np.testing.assert_allclose(rotvec_to_matrix(np.zeros(3)), np.eye(3))


def test_invert_and_compose():
    transform = make_transform(random_rotation(0.7, rng=1), [1.0, -2.0, 0.5])
    np.testing.assert_allclose(
        compose(invert(transform), transform), np.eye(4), atol=1e-12
    )
    points = np.random.default_rng(2).normal(size=(10, 3))
    other = make_transform(random_rotation(0.3, rng=3), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        apply(compose(other, transform), points), apply(other, apply(transform, points))
    )


def test_rotation_about_keeps_the_center_fixed():
    center = np.array([1.0, 2.0, 3.0])
    transform = rotation_about(random_rotation(1.2, rng=4), center)
    np.testing.assert_allclose(apply(transform, center[None]), center[None], atol=1e-12)


def test_translation_error():
    a = make_transform(translation=[1.0, 0.0, 0.0])
    b = make_transform(translation=[1.0, 3.0, 4.0])
    assert translation_error(a, b) == pytest.approx(5.0)
