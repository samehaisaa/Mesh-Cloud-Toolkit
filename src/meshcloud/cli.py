"""Command line interface: ``meshcloud sample`` and ``meshcloud register``."""

import argparse
import sys

import numpy as np

from .icp import METHODS, icp
from .io import read_mesh, read_points, write_ply
from .sampling import sample_surface
from .transform import apply


def _sample(args) -> int:
    mesh = read_mesh(args.mesh)
    sampled = sample_surface(
        mesh.vertices,
        mesh.faces,
        args.points,
        rng=args.seed,
        return_normals=args.normals,
    )
    points, normals = sampled if args.normals else (sampled, None)
    write_ply(args.output, points, normals=normals)
    print(f"sampled {len(points)} points from {len(mesh.faces)} faces -> {args.output}")
    return 0


def _register(args) -> int:
    source = read_points(args.source)
    target = read_points(args.target)
    result = icp(
        source,
        target,
        method=args.method,
        max_iterations=args.max_iterations,
        max_correspondence_distance=args.max_distance,
        overlap=args.overlap,
        knn=args.knn,
    )
    print("transformation (source -> target):")
    print(np.array2string(result.transformation, precision=6, suppress_small=True))
    print(f"rmse {result.rmse:.6g}")
    print(f"fitness {result.fitness:.3f}")
    print(f"iterations {result.iterations}")
    print(f"converged {result.converged}")
    if args.output:
        write_ply(args.output, apply(result.transformation, source))
        print(f"aligned source -> {args.output}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="meshcloud", description="Mesh sampling and ICP point cloud registration."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="sample a point cloud from a mesh")
    sample.add_argument("mesh", help="input .obj or .ply mesh")
    sample.add_argument("-n", "--points", type=int, default=20000)
    sample.add_argument("-o", "--output", required=True, help="output .ply cloud")
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--normals", action="store_true", help="store face normals")
    sample.set_defaults(run=_sample)

    register = commands.add_parser(
        "register", help="align a source cloud onto a target"
    )
    register.add_argument("source", help="cloud to move, .ply or .obj")
    register.add_argument("target", help="fixed cloud, .ply or .obj")
    register.add_argument("--method", choices=METHODS, default="point-to-plane")
    register.add_argument("-o", "--output", help="write the aligned source to .ply")
    register.add_argument("--max-iterations", type=int, default=50)
    register.add_argument(
        "--max-distance", type=float, help="reject pairs farther apart than this"
    )
    register.add_argument(
        "--overlap",
        type=float,
        default=1.0,
        help="expected overlap ratio, trims below 1",
    )
    register.add_argument("--knn", type=int, default=20, help="neighbors for normals")
    register.set_defaults(run=_register)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
