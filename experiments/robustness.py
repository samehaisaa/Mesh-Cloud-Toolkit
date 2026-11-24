"""Robustness of ICP to sensor noise and to partial overlap.

Noise: Gaussian noise of growing strength is added to both clouds, which
start 20 degrees apart. Normals for point-to-plane are estimated from the
noisy target, as they would be in practice.

Overlap: the target keeps one side of the shape and the source the other
side, cut along a random direction, so only part of the source has a true
match. Plain point-to-plane ICP is compared with trimmed point-to-plane ICP,
which is given the true overlap ratio. The clouds start 10 degrees apart.

Writes results/robustness.json and figures/robustness.png.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from common import (
    FIGURES,
    METHODS,
    SUCCESS_DEGREES,
    diagonal,
    load_shape,
    perturbation,
    save_json,
)

from meshcloud import estimate_normals, icp, sample_surface
from meshcloud.transform import apply, invert, rotation_error

NOISE_LEVELS = (0.0, 0.001, 0.0025, 0.005, 0.01)
KEEP_FRACTIONS = (1.0, 0.9, 0.8, 0.7, 0.6)


def summarize(errors) -> dict:
    errors = np.asarray(errors)
    return {
        "success_rate": float(np.mean(errors < SUCCESS_DEGREES)),
        "median_rotation_error_deg": float(np.median(errors)),
    }


def noise_study(vertices, faces, points, trials) -> list[dict]:
    clean_target = sample_surface(vertices, faces, points, rng=0)
    clean_base = sample_surface(vertices, faces, points, rng=1)
    size = diagonal(clean_target)
    rows = []
    for level in NOISE_LEVELS:
        errors = {m: [] for m in METHODS}
        for trial in range(trials):
            rng = np.random.default_rng([int(level * 1e5), trial])
            target = clean_target + rng.normal(
                scale=level * size, size=clean_target.shape
            )
            base = clean_base + rng.normal(scale=level * size, size=clean_base.shape)
            normals = estimate_normals(target, k=20)
            moved = perturbation(base.mean(axis=0), 20, rng, 0.02 * size)
            source = apply(moved, base)
            for method in METHODS:
                result = icp(
                    source, target, method=method, max_iterations=300,
                    target_normals=normals,
                )  # fmt: skip
                errors[method].append(
                    rotation_error(result.transformation, invert(moved))
                )
        row = {"noise": level, **{m: summarize(errors[m]) for m in METHODS}}
        rows.append(row)
        print(
            f"noise {level:.2%} of size  "
            + "  ".join(
                f"{m} {row[m]['median_rotation_error_deg']:.2f} deg" for m in METHODS
            )
        )
    return rows


def overlap_study(vertices, faces, points, trials) -> list[dict]:
    full_target = sample_surface(vertices, faces, 2 * points, rng=2)
    full_base = sample_surface(vertices, faces, 2 * points, rng=3)
    size = diagonal(full_target)
    variants = {"point-to-plane": False, "point-to-plane, trimmed": True}
    rows = []
    for keep in KEEP_FRACTIONS:
        errors = {v: [] for v in variants}
        overlaps = []
        for trial in range(trials):
            rng = np.random.default_rng([int(keep * 100), trial, 7])
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            height_t, height_b = full_target @ direction, full_base @ direction
            cut_t = np.quantile(height_t, keep)
            target = full_target[height_t <= cut_t]
            base = full_base[height_b >= np.quantile(height_b, 1.0 - keep)]
            overlap = float(np.mean(base @ direction <= cut_t))
            overlaps.append(overlap)
            normals = estimate_normals(target, k=20)
            moved = perturbation(base.mean(axis=0), 10, rng, 0.02 * size)
            source = apply(moved, base)
            for variant, trimmed in variants.items():
                result = icp(
                    source, target, method="point-to-plane", max_iterations=300,
                    overlap=overlap if trimmed else 1.0, target_normals=normals,
                )  # fmt: skip
                errors[variant].append(
                    rotation_error(result.transformation, invert(moved))
                )
        row = {
            "keep_fraction": keep,
            "mean_overlap": float(np.mean(overlaps)),
            **{v: summarize(errors[v]) for v in variants},
        }
        rows.append(row)
        print(
            f"overlap {row['mean_overlap']:.0%}  "
            + "  ".join(
                f"{v} {row[v]['median_rotation_error_deg']:.2f} deg "
                f"({row[v]['success_rate']:.0%} ok)"
                for v in variants
            )
        )
    return rows


def plot(results: dict) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=150)
    colors = ("#d1495b", "#00798c")

    noise = results["noise"]
    x = [100 * row["noise"] for row in noise]
    for method, color in zip(METHODS, colors, strict=True):
        y = [row[method]["median_rotation_error_deg"] for row in noise]
        left.plot(x, y, "o-", color=color, label=method, lw=2, ms=4)
    left.set_xlabel("noise std (% of shape size)")
    left.set_ylabel("median rotation error (deg)")
    left.set_title("Noise, 20 degrees apart", fontsize=10)

    overlap = results["overlap"]
    x = [100 * row["mean_overlap"] for row in overlap]
    for variant, color in zip(
        ("point-to-plane", "point-to-plane, trimmed"),
        ("#00798c", "#edae49"),
        strict=True,
    ):
        y = [row[variant]["median_rotation_error_deg"] for row in overlap]
        right.plot(x, y, "o-", color=color, label=variant, lw=2, ms=4)
    right.set_xlabel("source points with a true match (%)")
    right.set_ylabel("median rotation error (deg)")
    right.set_title("Partial overlap, 10 degrees apart", fontsize=10)
    right.invert_xaxis()

    for ax in (left, right):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{results['shape']}, {results['trials']} runs per point", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "robustness.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--points", type=int, default=4000)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    name, vertices, faces = load_shape()
    results = {
        "shape": name,
        "points": args.points,
        "trials": args.trials,
        "success_threshold_deg": SUCCESS_DEGREES,
        "noise": noise_study(vertices, faces, args.points, args.trials),
        "overlap": overlap_study(vertices, faces, args.points, args.trials),
    }
    print("saved", save_json("robustness.json", results))
    plot(results)


if __name__ == "__main__":
    main()
