"""Periodic confluent vertex-network topology.

This module builds and maintains the combinatorial structure of a 2D confluent
tissue on a periodic (torus) square box:

    * vertices  -- 3-valent triple junctions (physical vertices)
    * edges     -- cell-cell contacts, each shared by exactly two cells
    * cells     -- ordered CCW loops of vertex ids

The construction mirrors the paper's initialisation (Poisson-Voronoi tiling in a
periodic box, Kim et al., Nature Physics 17, 859 (2021), Methods) but keeps a
single straight segment per contact.  As shown in the accompanying notes, the
straight-edge tension + pressure forces are *exactly* the vertex-model forces of
the reference MATLAB code (`ts_iteration.m` / `ts_edgeNormalForce.m`) in the
confluent regime.

Everything here is pure geometry/bookkeeping -- no physics.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi


# --------------------------------------------------------------------------- #
# minimum-image helpers (periodic box of side L, origin at 0)
# --------------------------------------------------------------------------- #
def wrap(pos: np.ndarray, L: float) -> np.ndarray:
    """Wrap coordinates into [0, L)."""
    return np.mod(pos, L)


def min_image(dr: np.ndarray, L: float) -> np.ndarray:
    """Minimum-image displacement(s) for a periodic box of side L."""
    return dr - L * np.round(dr / L)


class Tissue:
    """Combinatorial + geometric state of a periodic confluent vertex network.

    Attributes
    ----------
    L : float
        Periodic box side length.
    pos : (Nv, 2) float array
        Vertex positions, always wrapped into [0, L).
    cells : list[list[int]]
        For each cell, an ordered (CCW) list of vertex indices.
    """

    def __init__(self, L: float, pos: np.ndarray, cells: list[list[int]]):
        self.L = float(L)
        self.pos = wrap(np.asarray(pos, float).copy(), self.L)
        self.cells = [list(c) for c in cells]
        self._rebuild_edges()

    # ------------------------------------------------------------------ #
    # topology bookkeeping
    # ------------------------------------------------------------------ #
    def _rebuild_edges(self):
        """Recompute the edge<->cell maps from the cell vertex loops."""
        edge_cells: dict[tuple[int, int], list[int]] = {}
        for ci, loop in enumerate(self.cells):
            n = len(loop)
            for k in range(n):
                a, b = loop[k], loop[(k + 1) % n]
                key = (a, b) if a < b else (b, a)
                edge_cells.setdefault(key, []).append(ci)
        self.edge_cells = edge_cells
        # vertex -> incident cells
        v_cells: dict[int, set[int]] = {}
        for ci, loop in enumerate(self.cells):
            for v in loop:
                v_cells.setdefault(v, set()).add(ci)
        self.vertex_cells = v_cells

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_vertices(self) -> int:
        return len(self.pos)

    def edges(self) -> list[tuple[int, int]]:
        """List of interior edges (shared by exactly two cells)."""
        return [e for e, cs in self.edge_cells.items() if len(cs) == 2]

    # ------------------------------------------------------------------ #
    # geometry (all periodic, minimum-image)
    # ------------------------------------------------------------------ #
    def cell_polygon(self, ci: int) -> np.ndarray:
        """Unwrapped CCW polygon coordinates for cell ci (minimum-image chained)."""
        loop = self.cells[ci]
        p = self.pos[loop]
        out = np.empty_like(p)
        out[0] = p[0]
        for k in range(1, len(loop)):
            out[k] = out[k - 1] + min_image(p[k] - out[k - 1], self.L)
        return out

    def cell_area_perimeter(self, ci: int) -> tuple[float, float]:
        poly = self.cell_polygon(ci)
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        d = np.roll(poly, -1, axis=0) - poly
        perim = np.sum(np.sqrt(np.sum(d * d, axis=1)))
        return abs(area), perim

    def all_areas_perimeters(self) -> tuple[np.ndarray, np.ndarray]:
        A = np.empty(self.n_cells)
        P = np.empty(self.n_cells)
        for ci in range(self.n_cells):
            A[ci], P[ci] = self.cell_area_perimeter(ci)
        return A, P

    def shape_factors(self) -> np.ndarray:
        A, P = self.all_areas_perimeters()
        return P / np.sqrt(A)

    def neighbor_number(self) -> np.ndarray:
        """Number of edges (contacts) per cell = z per cell."""
        z = np.zeros(self.n_cells)
        for e, cs in self.edge_cells.items():
            if len(cs) == 2:
                z[cs[0]] += 1
                z[cs[1]] += 1
        return z


# --------------------------------------------------------------------------- #
# construction from a periodic Poisson-Voronoi tessellation
# --------------------------------------------------------------------------- #
def build_periodic_voronoi(n_side: int, rng: np.random.Generator) -> Tissue:
    """Build a confluent periodic tissue of ``n_side**2`` cells.

    Mirrors ``ts_randomVoronoi.m``: random seed points in a square box are tiled
    into the 3x3 periodic replicas, a Voronoi diagram is computed, and the
    central cells define the tissue.  The resulting network is guaranteed
    3-valent (generic seeds) and periodic.
    """
    L = float(n_side)
    n = n_side * n_side
    seeds = rng.random((n, 2)) * L

    # 3x3 tiling
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], float) * L
    tiled = (seeds[None, :, :] + shifts[:, None, None, :]).reshape(-1, 2)
    # keep track of which tiled point maps to which base cell
    base_id = np.tile(np.arange(n), len(shifts))

    vor = Voronoi(tiled)

    # For each base seed (the 9 central copies -> we pick the copy at shift 0),
    # collect its Voronoi region.  Region vertices are shared across cells; we
    # deduplicate them on a grid to recover the triple-junction network.
    center_block = np.where((shifts == 0).all(axis=1))[0][0]  # index of (0,0) shift
    center_start = center_block * n

    # map raw voronoi vertex coords -> canonical wrapped vertex id
    def key_of(pt):
        # round to a fine grid to merge duplicates, then wrap
        w = np.mod(pt, L)
        return (round(w[0] * 1e6), round(w[1] * 1e6))

    vid_of_key: dict[tuple[int, int], int] = {}
    vpos: list[np.ndarray] = []

    def get_vid(pt):
        k = key_of(pt)
        if k not in vid_of_key:
            vid_of_key[k] = len(vpos)
            vpos.append(np.mod(pt, L))
        return vid_of_key[k]

    cells: list[list[int]] = []
    for c in range(n):
        pidx = center_start + c
        region = vor.regions[vor.point_region[pidx]]
        if len(region) == 0 or -1 in region:
            raise RuntimeError("open Voronoi region; retry with different seed")
        poly = np.array([vor.vertices[r] for r in region])
        # order CCW about the seed
        centroid = poly.mean(axis=0)
        ang = np.arctan2(poly[:, 1] - centroid[1], poly[:, 0] - centroid[0])
        order = np.argsort(ang)
        poly = poly[order]
        loop = [get_vid(p) for p in poly]
        # drop accidental consecutive duplicates
        dedup = [loop[0]]
        for v in loop[1:]:
            if v != dedup[-1]:
                dedup.append(v)
        if dedup[0] == dedup[-1]:
            dedup.pop()
        cells.append(dedup)

    return Tissue(L, np.array(vpos), cells)
