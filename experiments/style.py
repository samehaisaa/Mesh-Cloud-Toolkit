"""Shared plotting style and shaded 3D mesh renderer."""

import matplotlib as mpl
import numpy as np
from matplotlib.collections import PolyCollection

INK = "#1a1d21"
MUTED = "#5a626a"
RULE = "#d4d9df"
P2P_COLOR = "#b8474b"
P2L_COLOR = "#1b6b7a"
TRIM_COLOR = "#c98a2e"
TARGET_RGB = np.array([0.33, 0.46, 0.59])  # matte slate porcelain
SOURCE_RGB = np.array([0.82, 0.43, 0.28])  # warm terracotta clay
INLIER_RGB = np.array([0.14, 0.50, 0.56])  # deep teal inliers
OUTLIER_RGB = np.array([0.78, 0.79, 0.81])  # muted grey trimmed points


def apply_paper_style() -> None:
    """Configure Matplotlib for clean white-background serif figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.75,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "grid.linestyle": "--",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
        }
    )


def panel_label(ax, label: str, x: float = -0.14, y: float = 1.07) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
    )


def camera_basis(elev_deg: float = 12.0, azim_deg: float = -60.0) -> np.ndarray:
    """3x3 matrix mapping bunny-upright (x, z, y) into camera (right, up, depth)."""
    el = np.radians(elev_deg)
    az = np.radians(azim_deg)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    return np.vstack([right, up, view])


def orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return faces wound consistently outward from the mesh centroid."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0 - vertices.mean(axis=0)
    if np.median(np.einsum("ij,ij->i", normals, centroids)) < 0:
        return faces[:, [0, 2, 1]]
    return faces


def _shade_faces(
    normals_cam: np.ndarray,
    base_rgb: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Matte 3-point studio lighting with subtle Fresnel rim on camera-space normals."""
    key = np.array([-0.42, 0.62, 0.66])
    key /= np.linalg.norm(key)
    fill = np.array([0.70, 0.15, 0.55])
    fill /= np.linalg.norm(fill)
    half_vec = key + np.array([0.0, 0.0, 1.0])
    half_vec /= np.linalg.norm(half_vec)

    ndotl1 = np.clip(normals_cam @ key, 0.0, 1.0)
    ndotl2 = np.clip(normals_cam @ fill, 0.0, 1.0)
    ndotv = np.clip(normals_cam[:, 2], 0.0, 1.0)
    spec = np.clip(normals_cam @ half_vec, 0.0, 1.0) ** 22
    rim = (1.0 - ndotv) ** 2.2

    light = 0.26 + 0.64 * ndotl1 + 0.20 * ndotl2
    base = np.broadcast_to(base_rgb, (len(normals_cam), 3))
    rgb = base * light[:, None] + (0.09 * spec + 0.10 * rim)[:, None]
    rgb = np.clip(rgb, 0.0, 1.0)
    rgba = np.empty((len(normals_cam), 4), dtype=float)
    rgba[:, :3] = rgb
    rgba[:, 3] = alpha
    return rgba


def render_meshes(
    ax,
    meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    *,
    elev: float = 12.0,
    azim: float = -60.0,
    bounds: tuple[float, float, float, float] | None = None,
    two_sided: bool = False,
) -> tuple[float, float, float, float]:
    """Render one or more shaded 3D triangle meshes sorted together by depth.

    Each entry in ``meshes`` is ``(vertices, faces, rgb, alpha)`` where ``rgb``
    is either a length-3 color or an ``(F, 3)`` per-face color array.
    """
    basis = camera_basis(elev, azim)
    polys_all = []
    depths_all = []
    colors_all = []

    for vertices, faces, rgb, alpha in meshes:
        if len(faces) == 0:
            continue
        upright = vertices[:, [0, 2, 1]]
        cam = upright @ basis.T
        tri = cam[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        tri = tri[valid]
        normals = normals[valid] / norms[valid, None]

        if two_sided:
            back = normals[:, 2] < 0
            normals = np.where(back[:, None], -normals, normals)
            if rgb.ndim == 2:
                face_rgb = rgb[valid].copy()
            else:
                face_rgb = np.broadcast_to(rgb, (len(normals), 3)).copy()
            # Darken interior cut faces slightly so open shells read solidly.
            face_rgb[back] *= 0.62
        else:
            front = normals[:, 2] > -0.04
            tri = tri[front]
            normals = normals[front]
            if len(tri) == 0:
                continue
            face_rgb = rgb[valid][front] if rgb.ndim == 2 else rgb

        polys_all.append(tri[:, :, :2])
        depths_all.append(tri[:, :, 2].mean(axis=1))
        colors_all.append(_shade_faces(normals, face_rgb, alpha=alpha))

    polys = np.concatenate(polys_all, axis=0)
    depths = np.concatenate(depths_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)

    order = np.argsort(depths)
    polys = polys[order]
    colors = colors[order]

    collection = PolyCollection(
        polys,
        facecolors=colors,
        edgecolors=colors,
        linewidths=0.18,
        antialiaseds=True,
    )
    ax.add_collection(collection)

    if bounds is None:
        xmin, ymin = polys.reshape(-1, 2).min(axis=0)
        xmax, ymax = polys.reshape(-1, 2).max(axis=0)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        half = max(xmax - xmin, ymax - ymin) * 0.56
        bounds = (cx - half, cx + half, cy - half, cy + half)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")
    return bounds
