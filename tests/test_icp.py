import numpy as np
import pytest

from meshcloud.icp import best_fit_transform, icp
from meshcloud.normals import estimate_normals
from meshcloud.sampling import sample_surface
from meshcloud.shapes import bumpy_sphere
from meshcloud.transform import (
    apply,
    invert,
    make_transform,
    random_rotation,
    rotation_about,
    rotation_error,
)

VERTICES, FACES = bumpy_sphere()


def perturbation(degrees, seed, shift=0.05):
    """A rotation about the shape's center plus a small random shift."""
    rng = np.random.default_rng(seed)
    rotation = random_rotation(np.radians(degrees), rng)
    shift = shift * rng.normal(size=3) / np.sqrt(3)
    return make_transform(translation=shift) @ rotation_about(rotation, np.zeros(3))


def test_best_fit_transform_recovers_an_exact_transform():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3))
    truth = make_transform(random_rotation(2.0, rng), [3.0, -1.0, 0.5])
    estimate = best_fit_transform(source, apply(truth, source))
    np.testing.assert_allclose(estimate, truth, atol=1e-10)


def test_best_fit_transform_never_returns_a_reflection():
    source = np.random.default_rng(1).normal(size=(30, 3))
    mirrored = source * [1.0, 1.0, -1.0]
    estimate = best_fit_transform(source, mirrored)
    assert np.linalg.det(estimate[:3, :3]) == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["point-to-point", "point-to-plane"])
def test_icp_recovers_a_known_perturbation_on_identical_samples(method):
    target = sample_surface(VERTICES, FACES, 3000, rng=0)
    truth = perturbation(15, seed=1)
    source = apply(invert(truth), target)
    result = icp(source, target, method=method, max_iterations=200, tolerance=1e-9)
    assert result.converged
    assert rotation_error(result.transformation, truth) < 0.01
    assert result.rmse < 1e-4


def test_point_to_plane_is_more_accurate_on_independent_samples():
    # Two different samplings of one surface have no exact point pairs. The
    # point-to-point objective then bottoms out slightly away from the truth
    # (0.5 to 2.6 degrees over 8 seeds), while point-to-plane stays under 0.05.
    target = sample_surface(VERTICES, FACES, 3000, rng=2)
    truth = perturbation(15, seed=3)
    source = apply(invert(truth), sample_surface(VERTICES, FACES, 3000, rng=4))
    errors = {
        method: rotation_error(
            icp(source, target, method=method, max_iterations=200).transformation, truth
        )
        for method in ("point-to-point", "point-to-plane")
    }
    assert errors["point-to-point"] < 3.0
    assert errors["point-to-plane"] < 0.1
    assert errors["point-to-plane"] < errors["point-to-point"]


def test_point_to_plane_converges_in_fewer_iterations_on_a_smooth_surface():
    target = sample_surface(VERTICES, FACES, 3000, rng=5)
    normals = estimate_normals(target)
    counts = {"point-to-point": [], "point-to-plane": []}
    for seed in range(5):
        truth = perturbation(15, seed=10 + seed)
        source = apply(
            invert(truth), sample_surface(VERTICES, FACES, 3000, rng=20 + seed)
        )
        for method in counts:
            result = icp(
                source,
                target,
                method=method,
                max_iterations=200,
                target_normals=normals,
            )
            assert result.converged
            counts[method].append(result.iterations)
    assert np.median(counts["point-to-plane"]) < np.median(counts["point-to-point"])


def test_point_to_point_rmse_never_increases():
    # Besl and McKay's convergence theorem, without trimming or rejection.
    target = sample_surface(VERTICES, FACES, 2000, rng=6)
    source = apply(
        perturbation(30, seed=7), sample_surface(VERTICES, FACES, 2000, rng=8)
    )
    result = icp(source, target, method="point-to-point", max_iterations=100)
    rmse = np.array(result.rmse_history)
    assert np.all(np.diff(rmse) <= 1e-12)
    assert len(result.history) == result.iterations + 1


def test_trimming_aligns_partially_overlapping_clouds():
    points = sample_surface(VERTICES, FACES, 6000, rng=9)
    other = sample_surface(VERTICES, FACES, 6000, rng=10)
    height = points[:, 0]
    target = points[height < np.quantile(height, 0.75)]
    source_points = other[other[:, 0] > np.quantile(other[:, 0], 0.25)]
    overlap = np.mean(source_points[:, 0] < np.quantile(height, 0.75))
    truth = perturbation(10, seed=11)
    source = apply(invert(truth), source_points)
    result = icp(source, target, overlap=overlap, max_iterations=200)
    assert rotation_error(result.transformation, truth) < 1.0
    assert result.fitness == pytest.approx(overlap, abs=1e-3)


def test_rejects_bad_arguments():
    points = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.raises(ValueError, match="method"):
        icp(points, points, method="point-to-line")
    with pytest.raises(ValueError, match="overlap"):
        icp(points, points, overlap=0.0)


def test_stops_when_every_pair_is_rejected():
    points = np.random.default_rng(1).normal(size=(20, 3))
    result = icp(points + 100.0, points, max_correspondence_distance=1.0)
    assert result.iterations == 0
    assert not result.converged
    assert result.fitness == 0.0
