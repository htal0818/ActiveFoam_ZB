"""T1 geometry must preserve the reference length and match cell incidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from types import SimpleNamespace
from activefoam.foam_full import FoamTissue,crd_local,edge_mid_vrtx_average
from activefoam.foam_measure import quality, validate
from activefoam.topology import build_periodic_voronoi


def test_t1_rotation_preserves_length_and_endpoints():
    f=FoamTissue(build_periodic_voronoi(6,np.random.default_rng(7)),w=1.,seed=3)
    accepted=False
    for e,(a,b) in enumerate(f.e_v):
        if any(len(f.f_v[c])<=3 for c in f.e_f[e]):continue
        snap=f._snapshot()
        p=crd_local(f.vpos[[a,b]],f.bs);mid=p.mean(0);dr=p[1]-p[0]
        dr=dr/np.linalg.norm(dr)*.02
        f.vpos[a]=mid-.5*dr;f.vpos[b]=mid+.5*dr;f._wrap_vertices()
        for ee in set(f.v_e[a])|set(f.v_e[b]):
            if ee<0:continue
            pts=crd_local(f.vpos[f.e_v[ee]],f.bs)
            f.e_mid[ee],f.e_rfn[ee]=edge_mid_vrtx_average(pts,f.edpc)
        f._update_faces()
        if not f.t1_foam(e):f._restore(snap);continue
        new=crd_local(f.vpos[[a,b]],f.bs);newdr=new[1]-new[0]
        np.testing.assert_allclose(np.linalg.norm(newdr),.02,atol=1e-12)
        np.testing.assert_allclose(np.dot(newdr,dr),0.,atol=1e-12)
        for ee,ends in enumerate(f.e_v):
            for point,v in zip(f.e_mid[ee][[0,-1]],ends):
                pair=crd_local(np.array([point,f.vpos[v]]),f.bs)
                np.testing.assert_allclose(pair[0],pair[1],atol=1e-12)
        assert np.all(f.f_area>0)
        accepted=True;break
    assert accepted,'no valid short-edge exchange was accepted'


def test_neighbor_count_deduplicates_disjoint_contacts():
    f=FoamTissue.__new__(FoamTissue);f.nFa=3
    f.e_f=np.array([[-1,0],[0,1],[0,1],[1,2],[-1,2]])
    np.testing.assert_array_equal(f.neighbor_number(),[1.,2.,1.])


def test_space_seeding_keeps_curvature_and_inherits_tension_fluctuations():
    f = FoamTissue(build_periodic_voronoi(6, np.random.default_rng(7)), w=1.)
    e = int(np.argmax(f.edge_lengths()))
    vertex = int(f.e_v[e, 0])
    incident = f.v_e[vertex].copy()
    first_new = len(f.e_v)
    fluctuations = dict(zip(incident, [.2, -.1, .4]))
    for film, fluctuation in fluctuations.items():
        f.e_t[film] += fluctuation
    mid = f.e_mid[e].copy()
    chord = mid[-1] - mid[0]
    normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
    mid[1:-1] += .08 * normal
    f.e_mid[e] = mid

    f.t2_reverse(vertex)

    curved = f.e_mid[e]
    assert np.max((curved[1:-1] - curved[0]) @ normal) > .04
    for new in range(first_new, first_new + 3):
        bordering = [film for junction in f.e_v[new] for film in f.v_e[junction]
                     if film in fluctuations]
        assert len(bordering) == 2
        np.testing.assert_allclose(f.e_t[new] - f.fixed_tension(new),
                                   np.mean([fluctuations[film] for film in bordering]))


def test_reference_geometry_policy_flags_crossings_but_rejects_invalid_area():
    pytest.importorskip("numba")
    points = np.array([[0., 0.], [.4, .4], [0., .4], [.4, 0.],
                       [.5, 0.], [.5, .5], [0., .5]])
    foam = SimpleNamespace(
        f_area=np.array([.1]), nFa=1, bs=2., sstn=0., vpos=points,
        f_v=[list(range(len(points)))], e_mid=[points],
        _cell_self_intersects=lambda _: True, _cell_polyline=lambda _: points,
    )
    with pytest.raises(ValueError, match="resolved boundary crossing"):
        validate(foam)

    foam.geometry_policy = "reference"
    assert validate(foam) == pytest.approx(.025)
    assert quality(foam)["resolved_crossing_checks"] == 2
    foam.f_area[0] = 0.
    with pytest.raises(ValueError, match="nonpositive cell area"):
        validate(foam)
