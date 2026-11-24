"""Generate the schematic, benchmark and partial overlap figures.

Writes:
- figures/objectives.png       (point-to-point vs point-to-plane geometry)
- figures/benchmark.png        (4-panel summary from results/*.json)
- figures/partial_overlap.png  (shaded Stanford Bunny at 33% overlap: plain vs trimmed)
"""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from common import (  # noqa: E402
    FIGURES,
    RESULTS,
    diagonal,
    load_shape,
    perturbation,
)
from scipy.spatial import cKDTree  # noqa: E402
from style import (  # noqa: E402
    INK,
    INLIER_RGB,
    MUTED,
    OUTLIER_RGB,
    P2L_COLOR,
    P2P_COLOR,
    SOURCE_RGB,
    TARGET_RGB,
    TRIM_COLOR,
    apply_paper_style,
    camera_basis,
    orient_faces_outward,
    panel_label,
    render_meshes,
)

from meshcloud import estimate_normals, icp, sample_surface  # noqa: E402
from meshcloud.transform import apply, invert, rotation_error  # noqa: E402


def make_objectives_figure() -> None:
    """Schematic comparing point-to-point and point-to-plane residuals."""
    apply_paper_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.4, 3.3), dpi=240)

    def curve(x):
        return 0.34 * np.sin(0.95 * x + 0.15) + 0.08 * x

    def curve_deriv(x):
        return 0.34 * 0.95 * np.cos(0.95 * x + 0.15) + 0.08

    xs = np.linspace(-0.2, 3.4, 300)
    ys = curve(xs)

    # Sample target points q_i on S and slightly rotated/shifted source points p_i.
    qx = np.array([0.35, 1.05, 1.78, 2.48, 3.05])
    qy = curve(qx)
    theta = np.radians(10.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    px_base = np.array([0.15, 0.85, 1.56, 2.26, 2.84])
    p_pts = np.column_stack([px_base, curve(px_base)])
    p_center = p_pts.mean(axis=0)
    p_moved = (p_pts - p_center) @ rot.T + p_center + np.array([-0.14, 0.34])

    s_curve = np.column_stack([xs, ys])
    s_moved = (s_curve - p_center) @ rot.T + p_center + np.array([-0.14, 0.34])

    for ax, title, lbl in (
        (ax_l, "Point-to-point (Besl & McKay 1992)", "(a)"),
        (ax_r, "Point-to-plane (Chen & Medioni 1992)", "(b)"),
    ):
        ax.plot(xs, ys, color=P2L_COLOR, lw=1.6, alpha=0.85, zorder=2)
        ax.plot(
            s_moved[:, 0],
            s_moved[:, 1],
            color=P2P_COLOR,
            lw=1.3,
            ls="--",
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            qx,
            qy,
            s=36,
            color=P2L_COLOR,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
            label=r"target $q_i$",
        )
        ax.scatter(
            p_moved[:, 0],
            p_moved[:, 1],
            s=36,
            color=P2P_COLOR,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
            label=r"source $p_i$",
        )
        ax.set_title(title, fontsize=9.8, pad=8, color=INK)
        panel_label(ax, lbl, x=0.0, y=1.03)
        ax.set_xlim(-0.35, 3.55)
        ax.set_ylim(-0.32, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")

    # Left panel: Euclidean arrows p_i -> q_i (includes tangential drag)
    for p, q in zip(p_moved, np.column_stack([qx, qy]), strict=True):
        ax_l.annotate(
            "",
            xy=q,
            xytext=p,
            arrowprops={
                "arrowstyle": "-|>,head_length=0.45,head_width=0.22",
                "color": P2P_COLOR,
                "lw": 1.15,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=4,
        )
    ax_l.text(
        1.65,
        -0.24,
        r"$\min_{R,t}\ \sum_i \|R p_i + t - q_i\|_2^2$",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    ax_l.text(
        0.52,
        0.97,
        r"source $\mathcal{P}$",
        color=P2P_COLOR,
        fontsize=8.5,
    )
    ax_l.text(
        2.55,
        0.03,
        r"target $\mathcal{Q}$",
        color=P2L_COLOR,
        fontsize=8.5,
    )
    ax_l.text(
        1.78,
        0.54,
        "tangential drag",
        color=P2P_COLOR,
        fontsize=8,
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.85},
    )

    # Right panel: tangent planes T_{q_i}, normals n_i, and orthogonal projections
    for idx, (p, qx_i, qy_i) in enumerate(zip(p_moved, qx, qy, strict=True)):
        q = np.array([qx_i, qy_i])
        slope = curve_deriv(qx_i)
        t_vec = np.array([1.0, slope])
        t_vec /= np.linalg.norm(t_vec)
        n_vec = np.array([-t_vec[1], t_vec[0]])

        # Tangent plane segment
        seg = np.vstack([q - 0.34 * t_vec, q + 0.34 * t_vec])
        ax_r.plot(seg[:, 0], seg[:, 1], color=MUTED, lw=1.0, ls=":", zorder=3)

        # Normal vector arrow (short)
        n_tip = q + 0.24 * n_vec
        ax_r.annotate(
            "",
            xy=n_tip,
            xytext=q,
            arrowprops={
                "arrowstyle": "-|>,head_length=0.35,head_width=0.18",
                "color": MUTED,
                "lw": 0.9,
                "shrinkA": 3,
                "shrinkB": 0,
            },
            zorder=3,
        )

        # Orthogonal foot of p onto tangent plane through q
        signed_dist = float(np.dot(p - q, n_vec))
        foot = p - signed_dist * n_vec
        ax_r.plot(
            [p[0], foot[0]],
            [p[1], foot[1]],
            color=P2L_COLOR,
            lw=1.35,
            zorder=4,
        )
        # Right-angle marker at foot
        sq = 0.055
        s_dir = np.sign(signed_dist) * n_vec
        t_dir = t_vec if np.dot(q - foot, t_vec) >= 0 else -t_vec
        corner = np.vstack(
            [foot + sq * s_dir, foot + sq * (s_dir + t_dir), foot + sq * t_dir]
        )
        ax_r.plot(corner[:, 0], corner[:, 1], color=P2L_COLOR, lw=0.75, zorder=4)

        if idx == 2:
            ax_r.text(
                n_tip[0] + 0.06,
                n_tip[1] + 0.02,
                r"$n_i$",
                color=MUTED,
                fontsize=8.5,
            )
            ax_r.text(
                q[0] + 0.32,
                q[1] - 0.14,
                r"$T_{q_i}$",
                color=MUTED,
                fontsize=8.5,
            )

    ax_r.text(
        1.65,
        -0.24,
        r"$\min_{R,t}\ \sum_i \left((R p_i + t - q_i)\cdot n_i\right)^2$",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    ax_r.text(
        0.58,
        0.97,
        "slides along tangent",
        color=P2L_COLOR,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.85},
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.06)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "objectives.png", dpi=240)
    plt.close(fig)


def make_benchmark_figure() -> None:
    """Four-panel summary from results/basin.json and results/robustness.json."""
    apply_paper_style()
    basin = json.loads((RESULTS / "basin.json").read_text())
    robust = json.loads((RESULTS / "robustness.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), dpi=240)
    (ax_basin, ax_iter), (ax_noise, ax_over) = axes

    # (a) Convergence basin
    angles = [r["angle"] for r in basin["rows"]]
    ax_basin.axvspan(0, 60, color="#eef4f6", alpha=0.85, zorder=0)
    ax_basin.text(
        30,
        14,
        "100% recovery\n(both variants)",
        ha="center",
        va="center",
        fontsize=7.8,
        color=MUTED,
    )
    for method, color, marker in (
        ("point-to-point", P2P_COLOR, "o"),
        ("point-to-plane", P2L_COLOR, "s"),
    ):
        rates = [100 * r[method]["success_rate"] for r in basin["rows"]]
        ax_basin.plot(
            angles,
            rates,
            f"{marker}-",
            color=color,
            label=method,
            lw=1.8,
            ms=4.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    ax_basin.annotate(
        "75°: 95% vs 50%",
        xy=(75, 50),
        xytext=(102, 68),
        fontsize=8,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8},
    )
    ax_basin.set_xlabel("Initial rotation (deg)")
    ax_basin.set_ylabel("Runs within 5° (%)")
    ax_basin.set_xticks(range(0, 181, 30))
    ax_basin.set_ylim(-4, 106)
    ax_basin.grid(True, zorder=0)
    ax_basin.legend(frameon=False, loc="upper right")
    ax_basin.set_title("Convergence basin", pad=6)
    panel_label(ax_basin, "(a)")

    # (b) Median iterations to converge (successful runs)
    for method, color, marker in (
        ("point-to-point", P2P_COLOR, "o"),
        ("point-to-plane", P2L_COLOR, "s"),
    ):
        pts = [
            (r["angle"], r[method]["median_iterations"])
            for r in basin["rows"]
            if r[method]["median_iterations"] is not None and r["angle"] <= 135
        ]
        ax_iter.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            f"{marker}-",
            color=color,
            label=method,
            lw=1.8,
            ms=4.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    ax_iter.set_yscale("log")
    ax_iter.set_xlabel("Initial rotation (deg)")
    ax_iter.set_ylabel("Median iterations (log scale)")
    ax_iter.set_xticks(range(0, 150, 30))
    ax_iter.set_yticks([3, 10, 30, 100, 200])
    ax_iter.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_iter.grid(True, which="major", zorder=0)
    ax_iter.legend(frameon=False, loc="upper left")
    ax_iter.set_title("Convergence speed", pad=6)
    panel_label(ax_iter, "(b)")

    # (c) Sensor noise
    noise = robust["noise"]
    nx = [100 * r["noise"] for r in noise]
    for method, color, marker in (
        ("point-to-point", P2P_COLOR, "o"),
        ("point-to-plane", P2L_COLOR, "s"),
    ):
        ny = [r[method]["median_rotation_error_deg"] for r in noise]
        ax_noise.plot(
            nx,
            ny,
            f"{marker}-",
            color=color,
            label=method,
            lw=1.8,
            ms=4.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    ax_noise.set_xlabel("Gaussian noise std (% of shape diagonal)")
    ax_noise.set_ylabel("Median rotation error (deg)")
    ax_noise.set_ylim(-0.04, 0.98)
    ax_noise.grid(True, zorder=0)
    ax_noise.legend(frameon=False, loc="upper left")
    ax_noise.set_title("Sensor noise (20° start)", pad=6)
    panel_label(ax_noise, "(c)")

    # (d) Partial overlap
    overlap = robust["overlap"]
    ox = [100 * r["mean_overlap"] for r in overlap]
    y_plain = [r["point-to-plane"]["median_rotation_error_deg"] for r in overlap]
    y_trim = [
        r["point-to-plane, trimmed"]["median_rotation_error_deg"] for r in overlap
    ]
    ax_over.fill_between(ox, y_trim, y_plain, color=P2L_COLOR, alpha=0.08, zorder=1)
    ax_over.plot(
        ox,
        y_plain,
        "s-",
        color=P2L_COLOR,
        label="point-to-plane",
        lw=1.8,
        ms=4.2,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    ax_over.plot(
        ox,
        y_trim,
        "D-",
        color=TRIM_COLOR,
        label="trimmed point-to-plane",
        lw=1.8,
        ms=4.2,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )
    ax_over.annotate(
        "0.06° at 33%",
        xy=(ox[-1], y_trim[-1]),
        xytext=(58, 4.2),
        fontsize=8,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": TRIM_COLOR, "lw": 0.9},
    )
    ax_over.set_xlabel("True overlap (% of source points)")
    ax_over.set_ylabel("Median rotation error (deg)")
    ax_over.invert_xaxis()
    ax_over.set_ylim(-1.0, 23.0)
    ax_over.grid(True, zorder=0)
    ax_over.legend(frameon=False, loc="upper left")
    ax_over.set_title("Partial overlap (10° start)", pad=6)
    panel_label(ax_over, "(d)")

    fig.tight_layout(h_pad=1.8, w_pad=2.0)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "benchmark.png", dpi=240)
    plt.close(fig)


def make_partial_overlap_figure() -> None:
    """3-panel shaded Stanford Bunny showing plain vs trimmed ICP at ~33% overlap."""
    apply_paper_style()
    _, vertices, faces = load_shape()
    faces = orient_faces_outward(vertices, faces)

    # Split mesh horizontally along x so both halves are clearly visible side by side.
    direction = np.array([1.0, 0.0, 0.0])
    keep = 0.60
    v_proj = vertices @ direction
    cut_t = np.quantile(v_proj, keep)
    cut_s = np.quantile(v_proj, 1.0 - keep)

    face_proj = v_proj[faces].mean(axis=1)
    target_faces = faces[face_proj <= cut_t]
    source_faces = faces[face_proj >= cut_s]

    target_pts = sample_surface(vertices, target_faces, 4000, rng=2)
    base_pts = sample_surface(vertices, source_faces, 4000, rng=3)
    overlap = float(np.mean(base_pts @ direction <= cut_t))

    size = diagonal(vertices)
    rng = np.random.default_rng(11)
    moved = perturbation(base_pts.mean(axis=0), 15.0, rng, 0.03 * size)
    truth = invert(moved)
    source_pts = apply(moved, base_pts)
    source_verts = apply(moved, vertices)

    normals = estimate_normals(target_pts, k=20)
    res_plain = icp(
        source_pts,
        target_pts,
        method="point-to-plane",
        max_iterations=200,
        target_normals=normals,
    )
    res_trim = icp(
        source_pts,
        target_pts,
        method="point-to-plane",
        max_iterations=200,
        overlap=overlap,
        target_normals=normals,
    )

    err_plain = rotation_error(res_plain.transformation, truth)
    err_trim = rotation_error(res_trim.transformation, truth)

    # Determine which source faces are inliers vs trimmed at convergence
    aligned_trim_verts = apply(res_trim.transformation, source_verts)
    tri_centers = aligned_trim_verts[source_faces].mean(axis=1)
    tree = cKDTree(target_pts)
    dists, _ = tree.query(tri_centers)
    thresh = np.quantile(dists, overlap)
    trim_colors = np.where(
        (dists <= thresh)[:, None],
        INLIER_RGB[None, :],
        OUTLIER_RGB[None, :],
    )

    aligned_plain_verts = apply(res_plain.transformation, source_verts)

    # Compute shared tight orthographic bounds across all three panels
    elev, azim = 14.0, -72.0
    basis = camera_basis(elev, azim)
    all_v = np.vstack([vertices, source_verts, aligned_plain_verts, aligned_trim_verts])
    proj = all_v[:, [0, 2, 1]] @ basis.T
    xmin, ymin = proj[:, :2].min(axis=0)
    xmax, ymax = proj[:, :2].max(axis=0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) * 0.54
    bounds = (cx - half, cx + half, cy - half, cy + half)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.65), dpi=240)
    panels = [
        (
            axes[0],
            "(a)",
            f"Input ({overlap:.0%} overlap, 15° off)",
            [
                (vertices, target_faces, TARGET_RGB, 1.0),
                (source_verts, source_faces, SOURCE_RGB, 1.0),
            ],
        ),
        (
            axes[1],
            "(b)",
            f"Point-to-plane ({err_plain:.1f}° error)",
            [
                (vertices, target_faces, TARGET_RGB, 1.0),
                (aligned_plain_verts, source_faces, SOURCE_RGB, 1.0),
            ],
        ),
        (
            axes[2],
            "(c)",
            f"Trimmed ICP ({err_trim:.2f}° error)",
            [
                (vertices, target_faces, TARGET_RGB, 1.0),
                (aligned_trim_verts, source_faces, trim_colors, 1.0),
            ],
        ),
    ]

    for ax, lbl, title, mesh_list in panels:
        render_meshes(
            ax, mesh_list, elev=elev, azim=azim, bounds=bounds, two_sided=True
        )
        ax.set_title(title, fontsize=9.5, pad=6, color=INK)
        panel_label(ax, lbl, x=-0.02, y=1.02)

    # Compact legend at bottom
    handles = [
        patches.Patch(color=TARGET_RGB, label="target half"),
        patches.Patch(color=SOURCE_RGB, label="source half"),
        patches.Patch(color=INLIER_RGB, label="trimmed inliers"),
        patches.Patch(color=OUTLIER_RGB, label="trimmed outliers"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.3,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.11, wspace=0.04)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "partial_overlap.png", dpi=240)
    plt.close(fig)
    print(
        f"partial_overlap: overlap={overlap:.1%}, "
        f"plain={err_plain:.2f} deg, trimmed={err_trim:.2f} deg"
    )


def main() -> None:
    make_objectives_figure()
    make_benchmark_figure()
    make_partial_overlap_figure()
    print("wrote objectives.png, benchmark.png, partial_overlap.png")


if __name__ == "__main__":
    main()
