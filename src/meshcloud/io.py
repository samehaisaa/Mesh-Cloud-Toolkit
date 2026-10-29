"""Minimal readers and writers for OBJ and PLY meshes and point clouds.

Only geometry is handled: vertex positions, optional vertex normals and
triangle faces. Polygons with more than three vertices are split into a fan
of triangles. Colors, texture coordinates and materials are ignored.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip


@dataclass
class Mesh:
    """Vertices (N, 3), triangle faces (M, 3) and optional vertex normals."""

    vertices: np.ndarray
    faces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    normals: np.ndarray | None = None


def read_mesh(path) -> Mesh:
    """Reads an OBJ or PLY file. Point clouds come back with no faces."""
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        return _read_obj(path)
    if suffix == ".ply":
        return _read_ply(path)
    raise ValueError(f"unsupported file type '{suffix}', expected .obj or .ply")


def read_points(path) -> np.ndarray:
    """Reads the vertex positions of an OBJ or PLY file as an (N, 3) array."""
    return read_mesh(path).vertices


def write_ply(path, points, normals=None, faces=None, binary=True) -> None:
    """Writes points, optional normals and optional triangles to a PLY file."""
    points = np.asarray(points, dtype="<f8")
    columns = [points]
    header = [
        "ply",
        f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
        f"element vertex {len(points)}",
        "property double x",
        "property double y",
        "property double z",
    ]
    if normals is not None:
        columns.append(np.asarray(normals, dtype="<f8"))
        header += ["property double nx", "property double ny", "property double nz"]
    faces = np.empty((0, 3), dtype=np.int64) if faces is None else np.asarray(faces)
    if len(faces):
        header += [
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
        ]
    header.append("end_header")
    vertex_data = np.hstack(columns)

    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if binary:
            f.write(vertex_data.astype("<f8").tobytes())
            if len(faces):
                records = np.empty(
                    len(faces), dtype=[("n", "u1"), ("idx", "<i4", (3,))]
                )
                records["n"] = 3
                records["idx"] = faces
                f.write(records.tobytes())
        else:
            np.savetxt(f, vertex_data, fmt="%.17g")
            if len(faces):
                np.savetxt(f, np.hstack([np.full((len(faces), 1), 3), faces]), fmt="%d")


def write_obj(path, vertices, faces=None) -> None:
    """Writes vertices and optional triangles to an OBJ file."""
    with open(path, "w", encoding="ascii") as f:
        np.savetxt(f, np.asarray(vertices, dtype=float), fmt="v %.17g %.17g %.17g")
        if faces is not None and len(faces):
            np.savetxt(f, np.asarray(faces) + 1, fmt="f %d %d %d")


def _fan(polygon: list[int]) -> list[list[int]]:
    return [
        [polygon[0], polygon[i], polygon[i + 1]] for i in range(1, len(polygon) - 1)
    ]


def _read_obj(path) -> Mesh:
    vertices, faces = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            words = line.split()
            if not words:
                continue
            if words[0] == "v":
                vertices.append([float(x) for x in words[1:4]])
            elif words[0] == "f":
                polygon = []
                for word in words[1:]:
                    index = int(word.split("/")[0])
                    # OBJ indices are 1-based, negative ones count from the end.
                    polygon.append(index - 1 if index > 0 else len(vertices) + index)
                faces.extend(_fan(polygon))
    return Mesh(
        vertices=np.asarray(vertices, dtype=float).reshape(-1, 3),
        faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3),
    )


def _parse_ply_header(f):
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt, elements = None, []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PLY header has no end_header")
        words = line.decode("ascii", "replace").split()
        if not words:
            continue
        if words[0] == "format":
            fmt = words[1]
        elif words[0] == "element":
            elements.append((words[1], int(words[2]), []))
        elif words[0] == "property":
            if words[1] == "list":
                prop = (words[4], ("list", _PLY_TYPES[words[2]], _PLY_TYPES[words[3]]))
            else:
                prop = (words[2], _PLY_TYPES[words[1]])
            elements[-1][2].append(prop)
        elif words[0] == "end_header":
            return fmt, elements


def _read_ply_ascii(f, elements):
    tokens = np.array(f.read().split(), dtype=np.float64)
    pos, data = 0, {}
    for name, count, props in elements:
        has_list = any(isinstance(kind, tuple) for _, kind in props)
        if not has_list:
            block = tokens[pos : pos + count * len(props)].reshape(count, len(props))
            pos += count * len(props)
            data[name] = {p: block[:, i] for i, (p, _) in enumerate(props)}
            continue
        columns = {p: [] for p, _ in props}
        for _ in range(count):
            for p, kind in props:
                if isinstance(kind, tuple):
                    n = int(tokens[pos])
                    columns[p].append(tokens[pos + 1 : pos + 1 + n].astype(np.int64))
                    pos += 1 + n
                else:
                    columns[p].append(tokens[pos])
                    pos += 1
        data[name] = columns
    return data


def _read_ply_binary(f, elements, endian):
    buffer, pos, data = f.read(), 0, {}
    for name, count, props in elements:
        list_props = [p for p, kind in props if isinstance(kind, tuple)]
        if count == 0:
            data[name] = {p: [] for p, _ in props}
            continue
        if not list_props:
            dtype = np.dtype([(p, endian + kind) for p, kind in props])
            block = np.frombuffer(buffer, dtype=dtype, count=count, offset=pos)
            pos += count * dtype.itemsize
            data[name] = {p: block[p].astype(np.float64) for p, _ in props}
            continue
        # Fast path: every list in this element has the same length as the first.
        fields, first = [], pos
        for p, kind in props:
            if isinstance(kind, tuple):
                length = int(np.frombuffer(buffer, endian + kind[1], 1, first)[0])
                fields += [
                    (p + "_n", endian + kind[1]),
                    (p, endian + kind[2], (length,)),
                ]
                first += (
                    np.dtype(kind[1]).itemsize + length * np.dtype(kind[2]).itemsize
                )
            else:
                fields.append((p, endian + kind))
                first += np.dtype(kind).itemsize
        dtype = np.dtype(fields)
        if pos + count * dtype.itemsize <= len(buffer):
            block = np.frombuffer(buffer, dtype=dtype, count=count, offset=pos)
            uniform = all(
                np.all(block[p + "_n"] == block.dtype[p].shape[0]) for p in list_props
            )
            if uniform:
                pos += count * dtype.itemsize
                data[name] = {p: block[p] for p, _ in props}
                continue
        columns = {p: [] for p, _ in props}
        for _ in range(count):
            for p, kind in props:
                if isinstance(kind, tuple):
                    n = int(np.frombuffer(buffer, endian + kind[1], 1, pos)[0])
                    pos += np.dtype(kind[1]).itemsize
                    items = np.frombuffer(buffer, endian + kind[2], n, pos)
                    columns[p].append(items.astype(np.int64))
                    pos += n * np.dtype(kind[2]).itemsize
                else:
                    columns[p].append(np.frombuffer(buffer, endian + kind, 1, pos)[0])
                    pos += np.dtype(kind).itemsize
        data[name] = columns
    return data


def _read_ply(path) -> Mesh:
    with open(path, "rb") as f:
        fmt, elements = _parse_ply_header(f)
        if fmt == "ascii":
            data = _read_ply_ascii(f, elements)
        elif fmt == "binary_little_endian":
            data = _read_ply_binary(f, elements, "<")
        elif fmt == "binary_big_endian":
            data = _read_ply_binary(f, elements, ">")
        else:
            raise ValueError(f"unknown PLY format '{fmt}'")

    vertex = data.get("vertex")
    if vertex is None:
        raise ValueError("PLY file has no vertex element")
    vertices = np.column_stack([np.asarray(vertex[c], dtype=float) for c in "xyz"])
    normals = None
    if all(c in vertex for c in ("nx", "ny", "nz")):
        normals = np.column_stack(
            [np.asarray(vertex[c], dtype=float) for c in ("nx", "ny", "nz")]
        )

    faces = np.empty((0, 3), dtype=np.int64)
    face = data.get("face", {})
    indices = face.get("vertex_indices", face.get("vertex_index"))
    if indices is not None and len(indices):
        if isinstance(indices, np.ndarray) and indices.shape[1] == 3:
            faces = indices.astype(np.int64)
        else:
            faces = np.asarray(
                [tri for polygon in indices for tri in _fan(list(polygon))],
                dtype=np.int64,
            ).reshape(-1, 3)
    return Mesh(vertices=vertices, faces=faces, normals=normals)
