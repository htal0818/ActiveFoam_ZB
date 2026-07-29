"""Correctness checks for the vertex-model reproduction.

Run with:  python -m pytest tests/  (or just execute the file).
These assert the structural invariants that must hold for a confluent periodic
tissue and that the vectorised forces match a direct reference implementation.
"""
import os, sys
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.model import ActiveFoam
from activefoam.topology import build_periodic_voronoi, min_image


def _invariants(af):
    T = af.tissue
    val = Counter()
    for loop in T.cells:
        for v in loop:
            val[v] += 1
    assert set(val.values()) == {3}, "all vertices must be trivalent"
    share = Counter(len(cs) for cs in T.edge_cells.values())
    assert set(share.keys()) == {2}, "every edge must be shared by exactly 2 cells"
    A, _ = T.all_areas_perimeters()
    assert abs(A.sum() - T.L ** 2) < 1e-6, "confluent: total area == box area"
    assert np.all(af.cell_areas() > 0), "all cells CCW / positive area"
    assert abs(T.neighbor_number().mean() - 6.0) < 1e-9, "Euler: <z>=6"


def test_construction_invariants():
    af = ActiveFoam(n_side=6, seed=0)
    assert af.tissue.n_cells == 36
    assert af.tissue.n_vertices == 72          # 2N triple junctions
    _invariants(af)


def test_forces_match_reference():
    """Vectorised pressure+tension force == explicit per-cell/per-edge reference."""
    af = ActiveFoam(n_side=6, dT=0.0, seed=2)
    for _ in range(50):
        af.step()
    F = af.compute_forces()

    T = af.tissue
    Fref = np.zeros((T.n_vertices, 2))
    # pressure
    for ci, loop in enumerate(T.cells):
        poly = T.cell_polygon(ci)
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        p = af.P0 * (af.A0 / abs(area) - 1.0) * np.sign(area)
        n = len(loop)
        for k in range(n):
            gx = 0.5 * (y[(k + 1) % n] - y[(k - 1) % n])
            gy = 0.5 * (x[(k - 1) % n] - x[(k + 1) % n])
            Fref[loop[k]] += p * np.array([gx, gy])
    # tension
    for i, (a, b) in enumerate(af.E):
        t = max(af.Tvals[i], 0.0)
        dr = min_image(T.pos[b] - T.pos[a], T.L)
        Ln = np.hypot(*dr)
        f = t * dr / Ln
        Fref[a] += f
        Fref[b] -= f
    assert np.allclose(F, Fref, atol=1e-9), "vectorised forces disagree with reference"


def test_t1_conserves_topology():
    af = ActiveFoam(n_side=6, w=1.0, dT=0.8, seed=5)
    for it in range(3000):
        af.step()
        if it % 3 == 0:
            af.do_t1_transitions()
    _invariants(af)


if __name__ == "__main__":
    test_construction_invariants()
    test_forces_match_reference()
    test_t1_conserves_topology()
    print("all invariants + force checks passed")
