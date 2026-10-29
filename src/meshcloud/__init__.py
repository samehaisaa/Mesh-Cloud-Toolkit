"""Point cloud registration from scratch, on NumPy and SciPy."""

from .icp import ICPResult, best_fit_transform, icp
from .io import Mesh, read_mesh, read_points, write_obj, write_ply
from .normals import estimate_normals
from .sampling import sample_surface

__all__ = [
    "ICPResult",
    "Mesh",
    "best_fit_transform",
    "estimate_normals",
    "icp",
    "read_mesh",
    "read_points",
    "sample_surface",
    "write_obj",
    "write_ply",
]
__version__ = "0.1.0"
