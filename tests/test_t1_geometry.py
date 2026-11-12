"""T1 geometry must preserve the reference length and match cell incidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from activefoam.foam_full import FoamTissue,crd_local,edge_mid_vrtx_average
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
