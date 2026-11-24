"""Convergence basin: how far off can the initial guess be before ICP fails?

Two independent samplings of the shape play source and target. For every
initial rotation angle, the source is rotated about its center by that angle
around a random axis and shifted by 2% of the shape's size, then aligned back
with both ICP variants. A run succeeds when the final rotation error is under
5 degrees, which means ICP found the right basin.

Writes results/basin.json and figures/basin.png.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from common import (
    FIGURES,
    METHODS,
    RESULTS,
    SUCCESS_DEGREES,
    alignment_error,
    diagonal,
    load_shape,
    perturbation,
    save_json,
)

from meshcloud import estimate_normals, icp, sample_surface
from meshcloud.transform import apply, invert, rotation_error


def run(points: int, trials: int, angles, max_iterations: int) -> dict:
    name, vertices, faces = load_shape()
    target = sample_surface(vertices, faces, points, rng=0)
    base = sample_surface(vertices, faces, points, rng=1)
    size = diagonal(target)
    normals = estimate_normals(target, k=20)

    rows = []
    for angle in angles:
        row = {"angle": angle}
        records = {m: [] for m in METHODS}
        for trial in range(trials):
            rng = np.random.default_rng([int(angle), trial])
            moved = perturbation(base.mean(axis=0), angle, rng, 0.02 * size)
            truth = invert(moved)
            source = apply(moved, base)
            for method in METHODS:
                result = icp(
                    source,
                    target,
                    method=method,
                    max_iterations=max_iterations,
                    target_normals=normals,
                )
                records[method].append(
                    (
                        rotation_error(result.transformation, truth),
                        alignment_error(result.transformation, truth, source) / size,
                        result.iterations,
                        result.converged,
                    )
                )
        for method, values in records.items():
            errors, gaps, iterations, converged = (
                np.array(v) for v in zip(*values, strict=True)
            )
            ok = errors < SUCCESS_DEGREES

            def median(x, ok=ok):
                return float(np.median(x[ok])) if ok.any() else None

            row[method] = {
                "success_rate": float(ok.mean()),
                "median_rotation_error_deg": median(errors),
                "median_alignment_error": median(gaps),
                "median_iterations": median(iterations),
                "hit_iteration_cap": float(np.mean(~converged)),
            }
        rows.append(row)
        summary = "  ".join(
            f"{m} {row[m]['success_rate']:.0%} ({row[m]['median_iterations']} it,"
            f" cap {row[m]['hit_iteration_cap']:.0%})"
            for m in METHODS
        )
        print(f"{angle:5.0f} deg  {summary}")
    return {
        "shape": name,
        "points": points,
        "trials": trials,
        "success_threshold_deg": SUCCESS_DEGREES,
        "shift_fraction": 0.02,
        "max_iterations": max_iterations,
        "rows": rows,
    }


def plot(results: dict) -> None:
    FIGURES.mkdir(exist_ok=True)
    angles = [row["angle"] for row in results["rows"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    for method, color in zip(METHODS, ("#d1495b", "#00798c"), strict=True):
        rates = [100 * row[method]["success_rate"] for row in results["rows"]]
        ax.plot(angles, rates, "o-", color=color, label=method, lw=2, ms=4)
    ax.set_xlabel("initial rotation (degrees)")
    ax.set_ylabel(f"runs within {results['success_threshold_deg']:g} degrees (%)")
    ax.set_xticks(angles[::2])
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    title = f"ICP convergence basin, {results['shape']}"
    ax.set_title(f"{title}, {results['trials']} runs per angle", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "basin.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--points", type=int, default=4000)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument(
        "--replot",
        action="store_true",
        help="redraw the figure from results/basin.json",
    )
    args = parser.parse_args()
    if args.replot:
        plot(json.loads((RESULTS / "basin.json").read_text()))
        return
    results = run(args.points, args.trials, range(0, 181, 15), args.max_iterations)
    print("saved", save_json("basin.json", results))
    plot(results)


if __name__ == "__main__":
    main()
