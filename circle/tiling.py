"""Voronoi tiling clipped to a disc.

The tiling is a trivalent network:

* interior Voronoi vertices that lie inside the disc are junctions;
* every crossing of a Voronoi ridge with the circle is a junction joining one
  cell-cell contact and two outer films;
* every outer film is a single edge between consecutive boundary junctions of
  one cell, sampled along the circular arc.  The arc samples are film nodes,
  never junctions.

Pure geometry: no physics, no periodic box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Voronoi


@dataclass
class DiscTiling:
    """Junction positions, cell loops and outer-film polylines of a clipped tiling."""
    pos: np.ndarray                 # (Nv, 2) junction positions, absolute coordinates
    cells: list                     # per cell: counter-clockwise list of junction ids
    arcs: dict                      # (a, b) -> (n, 2) polyline on the wall, a -> b counter-clockwise
    radius: float
    center: np.ndarray
    n_interior: int                 # junctions that are Voronoi vertices inside the disc
    n_crossing: int                 # junctions on the wall
    seeds: np.ndarray = field(default=None, repr=False)

    @property
    def n_cells(self):
        return len(self.cells)

    @property
    def n_outer_films(self):
        return len(self.arcs)


class DegenerateTiling(ValueError):
    """The seed set does not give a valid trivalent clipped network."""


def arc_polyline(center, radius, start, stop, spacing):
    """Points on the circle from angle ``start`` counter-clockwise to ``stop``.

    Interior points follow the film rule of the planar model
    (``floor(length/spacing)`` of them, evenly spaced); the two ends are the
    boundary junctions themselves.
    """
    sweep = (stop - start) % (2.0 * np.pi)
    count = int(np.floor(radius * sweep / spacing))
    angles = start + sweep * np.arange(count + 2) / (count + 1)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])


def _circle_roots(p, q, radius):
    """Parameters t in (0, 1) where the segment p + t (q - p) meets |x| = radius."""
    d = q - p
    a = d @ d
    if a == 0.0:
        return []
    b = 2.0 * (p @ d)
    c = p @ p - radius * radius
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return []
    root = np.sqrt(disc)
    return [t for t in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if 0.0 < t < 1.0]


def _clip(seeds, radius):
    """Clip the Voronoi regions of ``seeds`` (centred on the origin) to the disc."""
    n = len(seeds)
    ring = 16.0 * radius * np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 32, endpoint=False)),
                                            np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False))])
    vor = Voronoi(np.vstack([seeds, ring]))
    verts = vor.vertices
    inside = np.einsum("ij,ij->i", verts, verts) < radius * radius

    ids = {}                 # junction key -> id
    points = []

    def junction(key, point):
        if key not in ids:
            ids[key] = len(points)
            points.append(point)
        return ids[key]

    def crossing(a, b, t):
        """Junction where ridge (a, b) meets the wall; ``t`` runs from a to b."""
        lo, hi = (a, b) if a < b else (b, a)
        roots = _circle_roots(verts[lo], verts[hi], radius)
        canonical = t if a < b else 1.0 - t
        rank = int(np.argmin([abs(r - canonical) for r in roots]))
        point = verts[lo] + roots[rank] * (verts[hi] - verts[lo])
        point = point * (radius / np.hypot(*point))       # exactly on the wall
        return junction(("x", lo, hi, rank), point)

    cells, arcs = [], {}
    for seed in range(n):
        region = vor.regions[vor.point_region[seed]]
        if not region or -1 in region:
            raise DegenerateTiling("unbounded Voronoi region")
        corners = verts[region] - seeds[seed]
        order = np.argsort(np.arctan2(corners[:, 1], corners[:, 0]))
        region = [region[k] for k in order]

        loop, exits, entries = [], [], []
        for k, a in enumerate(region):
            b = region[(k + 1) % len(region)]
            if inside[a]:
                loop.append(junction(("v", a), verts[a]))
            roots = _circle_roots(verts[a], verts[b], radius)
            if inside[a] and not inside[b]:
                if len(roots) != 1:
                    raise DegenerateTiling("ridge grazes the wall")
                loop.append(crossing(a, b, roots[0])); exits.append(len(loop) - 1)
            elif inside[b] and not inside[a]:
                if len(roots) != 1:
                    raise DegenerateTiling("ridge grazes the wall")
                loop.append(crossing(a, b, roots[0])); entries.append(len(loop) - 1)
            elif not inside[a] and len(roots) == 2:
                loop.append(crossing(a, b, roots[0])); entries.append(len(loop) - 1)
                loop.append(crossing(a, b, roots[1])); exits.append(len(loop) - 1)
            elif roots:
                raise DegenerateTiling("ridge grazes the wall")
        if len(loop) < 3 or len(set(loop)) != len(loop):
            raise DegenerateTiling("cell with fewer than three junctions")
        if len(exits) != len(entries):
            raise DegenerateTiling("unmatched wall crossings")
        for k in exits:                       # wall arc: exit crossing -> next junction
            nxt = (k + 1) % len(loop)
            if nxt not in entries:
                raise DegenerateTiling("wall arc does not end at a crossing")
            arcs[(loop[k], loop[nxt])] = None
        cells.append(loop)
    return np.array(points), cells, arcs, sum(key[0] == "v" for key in ids)


def _check_network(pos, cells, arcs, min_length):
    sides = {}
    for cell, loop in enumerate(cells):
        for k, a in enumerate(loop):
            b = loop[(k + 1) % len(loop)]
            sides.setdefault((min(a, b), max(a, b)), []).append(cell)
    degree = np.zeros(len(pos), dtype=int)
    for (a, b), owners in sides.items():
        outer = (a, b) in arcs or (b, a) in arcs
        if len(owners) != (1 if outer else 2) or len(set(owners)) != len(owners):
            raise DegenerateTiling("edge with the wrong number of cells")
        if np.hypot(*(pos[a] - pos[b])) < min_length:
            raise DegenerateTiling("edge shorter than the minimum length")
        degree[a] += 1
        degree[b] += 1
    if np.any(degree != 3):
        raise DegenerateTiling("junction that is not trivalent")


def clipped_voronoi(n_cells, rng, radius=None, center=(0.0, 0.0), spacing=0.15,
                    min_length=1e-6, max_tries=200):
    """Poisson-Voronoi tiling of ``n_cells`` cells clipped to a disc.

    ``radius`` defaults to ``sqrt(n_cells/pi)`` so that the mean area per cell
    is one, the convention of the periodic box.  Degenerate seed sets (a
    two-junction cell, a ridge grazing the wall, a repeated vertex pair) are
    redrawn; ``DegenerateTiling`` is raised after ``max_tries`` attempts.
    """
    if n_cells < 3:
        raise ValueError("a clipped tiling needs at least three cells")
    radius = float(np.sqrt(n_cells / np.pi) if radius is None else radius)
    center = np.asarray(center, dtype=float)
    error = None
    for _ in range(max_tries):
        r = radius * np.sqrt(rng.random(n_cells))
        theta = 2.0 * np.pi * rng.random(n_cells)
        seeds = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        try:
            pos, cells, arcs, n_interior = _clip(seeds, radius)
            _check_network(pos, cells, arcs, min_length)
        except DegenerateTiling as exc:
            error = exc
            continue
        angle = np.arctan2(pos[:, 1], pos[:, 0])
        arcs = {(a, b): arc_polyline(center, radius, angle[a], angle[b], spacing)
                for a, b in arcs}
        pos = pos + center
        for (a, b), line in arcs.items():      # ends are the junctions, to the last bit
            line[0], line[-1] = pos[a], pos[b]
        return DiscTiling(pos=pos, cells=cells, arcs=arcs, radius=radius, center=center,
                          n_interior=n_interior, n_crossing=len(pos) - n_interior,
                          seeds=seeds + center)
    raise DegenerateTiling(f"no valid tiling in {max_tries} tries: {error}")
