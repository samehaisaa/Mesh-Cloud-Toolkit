"""Downloads the Stanford Bunny mesh to data/bunny/bun_zipper.ply.

Tries the Stanford 3D Scanning Repository first, then the copy that Open3D
redistributes on GitHub when Stanford's server cannot be reached.
"""

import io
import tarfile
import urllib.request
from pathlib import Path

STANFORD = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
MEMBER = "bunny/reconstruction/bun_zipper.ply"
MIRROR = (
    "https://github.com/isl-org/open3d_downloads/releases/download/"
    "20220201-data/BunnyMesh.ply"
)
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "bunny" / "bun_zipper.ply"


def _download(url: str, timeout: float = 30.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def main() -> None:
    if OUTPUT.exists():
        print(f"already there: {OUTPUT}")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        archive = tarfile.open(fileobj=io.BytesIO(_download(STANFORD)), mode="r:gz")
        data = archive.extractfile(MEMBER).read()
        source = STANFORD
    except OSError as error:
        print(f"Stanford server unreachable ({error}), using the Open3D mirror")
        data = _download(MIRROR)
        source = MIRROR
    if not data.startswith(b"ply"):
        raise RuntimeError(f"{source} did not return a PLY file")
    OUTPUT.write_bytes(data)
    (OUTPUT.parent / "SOURCE.txt").write_text(source + "\n")
    print(f"saved {len(data)} bytes from {source} -> {OUTPUT}")


if __name__ == "__main__":
    main()
