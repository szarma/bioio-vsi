"""SHA-256 over all planes in (t, c, z) order."""

from __future__ import annotations

import hashlib

from bioio_vsi._stack import VsiImage


def all_planes_sha256(img: VsiImage) -> str:
    h = hashlib.sha256()
    t, c, z = img.shape_tcz
    for ti in range(t):
        for ci in range(c):
            for zi in range(z):
                h.update(img.read_plane(t=ti, c=ci, z=zi).tobytes())
    return h.hexdigest()
