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


def test_foam_transitions_stable():
    """Foam model: build -> spaces (T2-reverse) -> relax with T1+T2, invariants hold."""
    import numpy as np
    from activefoam.topology import build_periodic_voronoi
    from activefoam.foam_full import FoamTissue, SPACE

    T = build_periodic_voronoi(6, np.random.default_rng(3))
    ft = FoamTissue(T, w=0.3, rho=0.9, seed=1)
    for _ in range(120):
        ft.step(mu=0.0)
    assert abs(ft.f_area.sum() - ft.bs ** 2) < 1e-3          # confluent area
    nv0 = len(ft.vpos)
    for v in range(nv0):
        ft.t2_reverse(v)                                     # create spaces
    ft._update_faces()
    assert len(ft.vpos) == 3 * nv0                           # each vertex -> triangle
    assert int(np.sum(np.any(ft.e_f == SPACE, axis=1))) == 3 * nv0
    for it in range(200):
        ft.step(mu=0.4 if it < 100 else 0.0)
        if it % 20 == 0:
            ft.do_transitions()
    # invariants: trivalent vertices, valid cells
    assert np.all(np.sum(ft.v_e != -1, axis=1) == 3)
    assert all(len(ft.f_v[c]) >= 3 and ft.f_area[c] > 1e-4 for c in range(ft.nFa))


def test_t2_annihilation_reverses_creation():
    import numpy as np
    from activefoam.topology import build_periodic_voronoi
    from activefoam.foam_full import FoamTissue, SPACE
    T = build_periodic_voronoi(6, np.random.default_rng(3))
    ft = FoamTissue(T, w=1.5, rho=1.0, seed=1)
    for _ in range(120):
        ft.step(mu=0.0)
    nv0 = len(ft.vpos)
    for v in range(nv0):
        ft.t2_reverse(v)
    ft._update_faces()
    for _ in range(150):
        ft.step(mu=0.0)                                      # high W closes spaces
    n = ft.t2_annihilate_spaces(area_thr=1e-2)               # generous -> remove all
    assert len(ft.vpos) == nv0                               # back to confluent
    assert int(np.sum(np.any(ft.e_f == SPACE, axis=1))) == 0
    assert abs(ft.f_area.sum() - ft.bs ** 2) < 1e-2


def test_t4_adjacent_resolves_fold():
    """T4-adjacent uncrosses two curved edges that share a vertex."""
    import numpy as np
    from activefoam.foam_full import FoamTissue, edge_len
    ft = FoamTissue.__new__(FoamTissue)
    ft.bs = 100.0; ft.sstn = 0.0; ft.edpc = 0.15
    # e1 bows out and back, e2 runs straight; both start at vertex 0.
    # They cross away from the shared vertex at (1.5, 1.5).
    ft.e_mid = [np.array([[0., 0.], [3., 1.], [0., 2.]]),
                np.array([[0., 0.], [2., 2.]])]
    ft.e_v = np.array([[0, 1], [0, 2]]); ft.e_rfn = np.zeros(2, np.int64)
    n, pt, s1, s2 = ft._edge_cross_adjacent(0, 1, 0)
    assert n == 1 and abs(pt[0] - 1.5) < 1e-9 and abs(pt[1] - 1.5) < 1e-9
    ft._trim_edge_at(0, 0, pt, s1)
    ft._trim_edge_at(1, 0, pt, s2)
    n2, _, _, _ = ft._edge_cross_adjacent(0, 1, 0)
    assert n2 == 0                                          # fold resolved
    assert edge_len(ft.e_mid[0]).sum() > 0
    assert np.allclose(ft.e_mid[0][0], pt) and np.allclose(ft.e_mid[1][0], pt)


def test_t4_reduces_folds_in_run():
    """Enabling T4-adjacent leaves fewer residual adjacent folds after a run."""
    import numpy as np
    from activefoam.topology import build_periodic_voronoi
    from activefoam.foam_full import FoamTissue
    import activefoam.foam_full as FF

    def folds(ft):
        c = 0
        for v in range(len(ft.vpos)):
            eids = [int(e) for e in ft.v_e[v] if e >= 0]
            ring = eids + [eids[0]] if len(eids) >= 2 else []
            for i in range(len(ring) - 1):
                if ring[i] != ring[i + 1]:
                    c += ft._edge_cross_adjacent(ring[i], ring[i + 1], v)[0]
        return c

    def run(use_t4, seed=1, mu=1.5):
        T = build_periodic_voronoi(6, np.random.default_rng(100 + seed))
        ft = FoamTissue(T, w=0.0, rho=0.8, seed=seed)
        for _ in range(120):
            ft.step(mu=mu)
        for v in range(len(ft.vpos)):
            ft.t2_reverse(v)
        ft._update_faces()
        orig = FF.FoamTissue.t4_adjacent
        if not use_t4:
            FF.FoamTissue.t4_adjacent = lambda self, v: False
        try:
            for it in range(160):
                ft.step(mu=mu)
                if it % 10 == 0:
                    ft.do_transitions(mu=mu)
                if len(ft.vpos) > 320:
                    break
        finally:
            FF.FoamTissue.t4_adjacent = orig
        return folds(ft)

    assert run(True, 2) <= run(False, 2)                    # T4 does not worsen tangling


def test_t4_transition_cuts_lens():
    """Non-adjacent T4: two space-bordering edges crossing at two points are
    cut into first/second pieces + a bridge (2 new vertices, 3 new edges)."""
    import numpy as np
    from activefoam.topology import build_periodic_voronoi
    from activefoam.foam_full import (FoamTissue, crd_local,
                                       edge_mid_vrtx_average, edge_len, SPACE)
    T = build_periodic_voronoi(6, np.random.default_rng(5))
    ft = FoamTissue(T, w=0.3, rho=1.0, seed=1)
    for _ in range(120):
        ft.step(mu=0.4)
    for v in range(len(ft.vpos)):
        ft.t2_reverse(v)
    ft._update_faces()
    for it in range(150):
        ft.step(mu=0.4)
        if it % 15 == 0:
            ft.do_transitions(mu=0.4)

    def mid_of(e):
        return crd_local(ft.e_mid[e], ft.bs, ft.sstn).mean(axis=0)

    # non-adjacent, different-cell space-bordering edges (independent of the
    # bbox prefilter, which only fires once e1 has bulged toward e2)
    def eligible(e1):
        adj = {int(x) for v in ft.e_v[e1] for x in ft.v_e[v] if x >= 0}
        cell1 = int(ft.e_f[e1, 1])
        return [e for e in range(len(ft.e_v))
                if e != e1 and e not in adj
                and int(ft.e_f[e, 0]) == SPACE
                and int(ft.e_f[e, 1]) != cell1]

    committed = False
    for e1 in range(len(ft.e_v)):
        if int(ft.e_f[e1, 0]) != SPACE:
            continue
        cands = eligible(e1)
        if not cands:
            continue
        m1 = mid_of(e1)

        def dist(e):
            dm = crd_local(np.vstack([m1, mid_of(e)]), ft.bs, ft.sstn)
            return np.hypot(*(dm[1] - dm[0]))
        e2 = min(cands, key=dist)
        P1 = crd_local(ft.e_mid[e1], ft.bs, ft.sstn)
        s, eend = P1[0], P1[-1]
        m2 = crd_local(ft.e_mid[e2], ft.bs, ft.sstn).mean(axis=0)
        for scale in (2.0, 2.6, 3.2, 1.6):
            arc = np.array([s, m2 + scale * (m2 - m1), eend])
            mid, rfn = edge_mid_vrtx_average(crd_local(arc, ft.bs, ft.sstn), ft.edpc)
            old = ft.e_mid[e1]
            ft.e_mid[e1] = mid; ft.e_rfn[e1] = rfn
            if ft._edge_cross_check(e1, e2)[0] == 2:
                break
            ft.e_mid[e1] = old
        else:
            continue
        f1, f2 = int(ft.e_f[e1, 1]), int(ft.e_f[e2, 1])
        nE, nV = len(ft.e_v), len(ft.vpos)
        if ft.t4_transition(e1):
            assert len(ft.e_v) == nE + 3 and len(ft.vpos) == nV + 2
            assert np.sum(ft.v_e[-1] >= 0) == 3 and np.sum(ft.v_e[-2] >= 0) == 3
            assert all(len(ft.f_v[f]) >= 3 and not ft._cell_self_intersects(f)
                       for f in (f1, f2))
            assert all(edge_len(ft.e_mid[e]).sum() > 1e-9
                       for e in range(len(ft.e_v)))
            committed = True
            break
    assert committed, "no non-adjacent T4 cut could be exercised"


if __name__ == "__main__":
    test_construction_invariants()
    test_forces_match_reference()
    test_t1_conserves_topology()
    test_foam_transitions_stable()
    test_t2_annihilation_reverses_creation()
    test_t4_adjacent_resolves_fold()
    test_t4_reduces_folds_in_run()
    test_t4_transition_cuts_lens()
    print("all checks passed (confluent + foam transitions + full T4)")
