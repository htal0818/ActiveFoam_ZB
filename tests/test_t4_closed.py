"""Deterministic contact-formation fixtures for the MATLAB closed-edge cases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from activefoam.foam_full import FoamTissue, SPACE, crd_local, edge_len


def fixture(closed=(True, True), shift=(0., 0.), shear=0.):
    ft = FoamTissue.__new__(FoamTissue)
    ft.bs = 20.; ft.sstn = shear; ft.edpc = .05
    ft.lsc = ft.psi = ft.rho = ft.A0 = 1.
    ft.w = ft.gam = .3; ft.beta = 0.; ft.dt = .001
    ft.P0 = 10.; ft.tauT = 10.; ft.rng = np.random.default_rng(0)
    ft.nFa = 2
    ft.vpos = np.empty((0, 2)); ft.v_e = np.empty((0, 3), int)
    ft.v_f = np.empty((0, 3), int)
    ft.e_v = np.empty((0, 2), int); ft.e_f = np.empty((0, 2), int)
    ft.e_t = np.empty(0); ft.e_rfn = np.empty(0, int); ft.e_mid = []
    ft.f_v = []; ft.f_e = [[], []]
    ft.f_area = np.zeros(2); ft.f_center = np.zeros((2, 2))
    # Two overlapping circles, sampled off the crossing locations.
    for cell, cx in enumerate((5., 6.4)):
        start = np.pi+.17 if cell == 0 else .17
        theta = np.linspace(start, 2*np.pi+start, 97)
        poly = np.column_stack([cx+np.cos(theta), 5.+np.sin(theta)]) + shift
        poly[-1] = poly[0]
        if closed[cell]:
            v = ft._add_vertex(poly[0]); ft.f_v.append([v])
            e = ft._add_edge(v, v, SPACE, cell, poly, len(poly)-2, 1.+.2*cell)
            ft.v_e[v] = [-1, -1, e]; ft.v_f[v] = [-1, -1, cell]
        else:
            # Three open edges forming the same circle. The contact-facing
            # arc is contained in one edge for either cell.
            cuts = (0, 24, 72, 96)
            vs = [ft._add_vertex(poly[i]) for i in cuts[:-1]]
            ft.f_v.append(vs)
            for j in range(3):
                a, b = vs[j], vs[(j+1)%3]
                m = poly[cuts[j]:cuts[j+1]+1]
                if a > b: m = m[::-1]
                e = ft._add_edge(*sorted([a,b]), SPACE, cell, m, len(m)-2, 1.+.2*cell)
                for v in (a,b):
                    row = [x for x in ft.v_e[v] if x >= 0] + [e]
                    ft.v_e[v] = [-1]*(3-len(row)) + row
                    ft.v_f[v] = [-1, -1, cell]
    ft._wrap_vertices()
    for c in range(2): ft._recompute_cell(c)
    return ft


def check(ft):
    assert np.all(np.isfinite(ft.f_area)) and np.all(ft.f_area > 0)
    for e, (a,b) in enumerate(ft.e_v):
        assert e in ft.v_e[a] and e in ft.v_e[b]
        for point, v in ((ft.e_mid[e][0],a), (ft.e_mid[e][-1],b)):
            loc = crd_local(np.array([ft.vpos[v], point]), ft.bs, ft.sstn)
            np.testing.assert_allclose(loc[0],loc[1],atol=1e-10)
        assert edge_len(ft.e_mid[e]).min() > 1e-10
    for f, seq in enumerate(ft.f_e):
        ids = [e if e>=0 else -e-1 for e in seq]
        assert len(ids) == len(set(ids))
        assert all(f in ft.e_f[e] for e in ids)


@pytest.mark.parametrize('closed', [(True,True),(True,False),(False,True),(False,False)])
@pytest.mark.parametrize('shift,shear', [((0.,0.),0.), ((-5.5,-5.),.3)])
def test_closed_contact_formation(closed, shift, shear):
    ft = fixture(closed, shift, shear)
    # Apply a lattice translation to one boundary to exercise periodic search.
    if shear:
        for e in np.flatnonzero(ft.e_f[:,1] == 1):
            ft.e_mid[e] += [ft.bs*shear, ft.bs]
    nv, ne = len(ft.vpos),len(ft.e_v)
    success = any(ft.t4_transition(e) for e in range(ne))
    assert success
    assert len(ft.vpos) == nv + 2-sum(closed)
    assert len(ft.e_v) == ne + 3-sum(closed)
    bridges = np.flatnonzero(np.all(ft.e_f >= 0,axis=1))
    assert len(bridges) == 1
    np.testing.assert_allclose(ft.e_t[bridges[0]],1.8)
    assert all(len(ft.f_v[c]) == (2 if closed[c] else 5) for c in range(2))
    check(ft)
    for _ in range(5): ft.step()
    check(ft)


def test_vertical_segment_cut():
    mid = np.array([[0.,0.],[0.,1.],[0.,2.],[1.,2.]])
    assert FoamTissue._cut_index(np.array([0.,1.5]),mid) == 1


def test_closed_face_orientation():
    ft = fixture()
    ft.e_mid[0] = ft.e_mid[0][::-1]
    ft._recompute_cell(0)
    assert ft.f_area[0] > 3.
    assert ft.f_e[0] == [-1]

@pytest.mark.parametrize('closed', [(True,True),(True,False),(False,True)])
def test_formed_contact_can_separate_again(closed):
    ft = fixture(closed)
    assert any(ft.t4_transition(e) for e in range(len(ft.e_v)))
    e = int(np.flatnonzero(np.all(ft.e_f>=0,axis=1))[0])
    assert ft.t1_foam(e)
    assert ft.t3_reverse(e)
    assert not np.any(np.all(ft.e_f>=0,axis=1))
    check(ft)
    for c in range(2):
        if closed[c]: assert len(ft.f_v[c]) == 1


def test_closed_t4_rejection_restores_state():
    ft=fixture()
    before=ft._snapshot()
    ft._cell_self_intersects=lambda f: True
    assert not ft.t4_transition(0)
    after=ft._snapshot()
    for a,b in zip(before,after):
        if isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        else:np.testing.assert_array_equal(a,b)
