"""Shaded 3D Stanford Bunny animation comparing point-to-point and point-to-plane ICP.

Writes figures/registration.webp.
"""

import argparse
import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from common import FIGURES, METHODS, diagonal, load_shape, perturbation  # noqa: E402
from PIL import Image  # noqa: E402
from style import (  # noqa: E402
    INK,
    P2L_COLOR,
    P2P_COLOR,
    SOURCE_RGB,
    TARGET_RGB,
    apply_paper_style,
    camera_basis,
    orient_faces_outward,
    panel_label,
    render_meshes,
)

from meshcloud import estimate_normals, icp, sample_surface  # noqa: E402
from meshcloud.transform import apply, invert, rotation_error  # noqa: E402


def render_frame(
    vertices,
    faces,
    source_vertices,
    histories,
    errors,
    finished,
    frame,
    bounds,
    elev,
    azim,
    max_iter,
) -> Image.Image:
    apply_paper_style()
    fig = plt.figure(figsize=(10.2, 3.55), dpi=115)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.14)
    ax_p2p = fig.add_subplot(gs[0, 0])
    ax_p2l = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[0, 2])

    for ax, method, lbl, title_base in (
        (ax_p2p, "point-to-point", "(a)", "Point-to-point"),
        (ax_p2l, "point-to-plane", "(b)", "Point-to-plane"),
    ):
        hist = histories[method]
        step = min(frame, len(hist) - 1)
        moved_verts = apply(hist[step], source_vertices)
        render_meshes(
            ax,
            [
                (vertices, faces, TARGET_RGB, 1.0),
                (moved_verts, faces, SOURCE_RGB, 1.0),
            ],
            elev=elev,
            azim=azim,
            bounds=bounds,
            two_sided=True,
        )
        err = errors[method][step]
        if step == len(hist) - 1 and finished[method]:
            subtitle = f"converged, iter {step} ({err:.2f}°)"
        else:
            subtitle = f"iter {step} ({err:.1f}°)"
        ax.set_title(f"{title_base}\n{subtitle}", fontsize=9.2, pad=3, color=INK)
        panel_label(ax, lbl, x=-0.04, y=1.05)

    # Panel (c): live log-scale rotation error vs iteration
    for method, color in (("point-to-point", P2P_COLOR), ("point-to-plane", P2L_COLOR)):
        curve = errors[method]
        step = min(frame, len(curve) - 1)
        xs = np.arange(step + 1)
        ax_err.plot(xs, curve[: step + 1], color=color, lw=1.8, label=method)
        ax_err.scatter(
            [step],
            [curve[step]],
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
    ax_err.set_yscale("log")
    ax_err.set_xlim(-1, max_iter + 1)
    ax_err.set_ylim(0.015, 60.0)
    ax_err.set_yticks([0.03, 0.1, 0.3, 1, 3, 10, 40])
    ax_err.set_yticklabels(["0.03", "0.1", "0.3", "1", "3", "10", "40"])
    ax_err.set_xlabel("Iteration")
    ax_err.set_ylabel("Rotation error (deg)")
    ax_err.set_title("Convergence\n(40° initial rotation)", fontsize=9.2, pad=2)
    ax_err.grid(True, which="major", zorder=0)
    ax_err.legend(frameon=False, loc="upper right", fontsize=8.0)
    panel_label(ax_err, "(c)", x=-0.16, y=1.04)

    handles = [
        patches.Patch(color=TARGET_RGB, label="target"),
        patches.Patch(color=SOURCE_RGB, label="source"),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        frameon=False,
        fontsize=8.2,
        bbox_to_anchor=(0.02, 0.01),
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.17)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return Image.open(buf).convert("RGB")


def frame_durations(step_list: list[int], split_iter: int) -> list[int]:
    """Quick start into slow-motion convergence, then steady tail."""
    durations = []
    for idx, step in enumerate(step_list):
        if idx == 0:
            durations.append(450)
        elif idx == len(step_list) - 1:
            durations.append(2400)
        elif step < split_iter:
            durations.append(380)
        elif step == split_iter:
            durations.append(950)
        elif step <= split_iter + 4:
            durations.append(max(150, 240 - 25 * (step - split_iter)))
        else:
            durations.append(140)
    return durations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--angle", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=50)
    args = parser.parse_args()

    name, vertices, faces = load_shape()
    faces = orient_faces_outward(vertices, faces)

    target = sample_surface(vertices, faces, 4000, rng=0)
    base = sample_surface(vertices, faces, 4000, rng=1)
    size = diagonal(target)
    normals = estimate_normals(target, k=20)
    rng = np.random.default_rng(args.seed)
    moved = perturbation(base.mean(axis=0), args.angle, rng, 0.02 * size)
    truth = invert(moved)
    source = apply(moved, base)
    source_vertices = apply(moved, vertices)

    histories, errors, finished = {}, {}, {}
    for method in METHODS:
        result = icp(
            source, target, method=method, max_iterations=300, target_normals=normals
        )
        histories[method] = result.history
        finished[method] = result.converged
        errors[method] = [rotation_error(t, truth) for t in result.history]
        print(
            f"{method}: {result.iterations} iterations, converged {result.converged}, "
            f"rotation error {errors[method][-1]:.3f} deg"
        )

    elev, azim = 14.0, -65.0
    basis = camera_basis(elev, azim)
    all_v = np.vstack([vertices, source_vertices])
    proj = all_v[:, [0, 2, 1]] @ basis.T
    xmin, ymin = proj[:, :2].min(axis=0)
    xmax, ymax = proj[:, :2].max(axis=0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) * 0.56
    bounds = (cx - half, cx + half, cy - half, cy + half)

    max_iter = max(len(h) for h in histories.values()) - 1
    # Dense early steps so point-to-plane's rapid drop is shown step by step,
    # then every second step for point-to-point's slow tail.
    step_list = list(range(0, min(13, max_iter + 1)))
    if max_iter >= 13:
        step_list.extend(range(14, max_iter, 2))
        if step_list[-1] != max_iter:
            step_list.append(max_iter)
    step_list = step_list[: args.max_frames]

    frames = [
        render_frame(
            vertices,
            faces,
            source_vertices,
            histories,
            errors,
            finished,
            f,
            bounds,
            elev,
            azim,
            max_iter,
        )
        for f in step_list
    ]

    count = len(frames)
    p2l_done = len(histories["point-to-plane"]) - 1
    durations = frame_durations(step_list, p2l_done)
    resized = [
        f.resize((900, int(f.height * 900 / f.width)), Image.Resampling.LANCZOS)
        for f in frames
    ]
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "registration.webp"
    resized[0].save(
        path,
        save_all=True,
        append_images=resized[1:],
        duration=durations,
        loop=0,
        quality=82,
        method=6,
    )
    print(f"{name}: {count} frames -> {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
