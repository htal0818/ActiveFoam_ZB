import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from activefoam.foam_full import FoamTissue
from activefoam.topology import build_periodic_voronoi

@pytest.mark.parametrize('shear',[0., .3])
def test_accelerated_trajectory_matches_python(shear):
    pytest.importorskip('numba')
    ref=FoamTissue(build_periodic_voronoi(4,np.random.default_rng(2)),w=.4,rho=.9,seed=1)
    for v in range(len(ref.vpos)): ref.t2_reverse(v)
    if shear: ref.apply_affine_shear(shear)
    fast=copy.deepcopy(ref);fast.accelerated=True
    for i in range(15):
        ref.step(.5);fast.step(.5)
        np.testing.assert_allclose(fast.vpos,ref.vpos,atol=2e-11,rtol=2e-11)
        np.testing.assert_allclose(fast.f_area,ref.f_area,atol=2e-11,rtol=2e-11)
        np.testing.assert_allclose(fast.e_t,ref.e_t,atol=1e-14)
        np.testing.assert_array_equal(fast.e_rfn,ref.e_rfn)
        for a,b in zip(fast.e_mid,ref.e_mid):
            np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-11)
    np.testing.assert_allclose(fast.shape_factors(), ref.shape_factors(), atol=2e-11)
    np.testing.assert_allclose(fast.shear_stress(), ref.shear_stress(), atol=2e-11)

def test_accelerated_transition_search_matches_python():
    pytest.importorskip('numba')
    import copy
    ref=FoamTissue(build_periodic_voronoi(4,np.random.default_rng(5)),w=.1,rho=.9,seed=3)
    for v in range(len(ref.vpos)):ref.t2_reverse(v)
    for _ in range(40):ref.step(.5)
    fast=copy.deepcopy(ref);fast.accelerated=True
    assert fast.do_transitions() == ref.do_transitions()
    np.testing.assert_array_equal(fast.e_v,ref.e_v)
    np.testing.assert_array_equal(fast.e_f,ref.e_f)
    np.testing.assert_allclose(fast.vpos,ref.vpos,atol=1e-11)
    np.testing.assert_allclose(fast.f_area,ref.f_area,atol=1e-11)


def test_sheared_domain_wrap_preserves_lattice_equivalence():
    from activefoam.foam_full import crd_domain,crd_local
    p=np.array([[2.,-.1],[4.,12.1],[-3.,-10.]])
    q=crd_domain(p,4.,1.5)
    assert np.all((q[:,1]>=0)&(q[:,1]<4))
    assert np.all((q[:,0]-1.5*q[:,1]>=0)&(q[:,0]-1.5*q[:,1]<4))
    for a,b in zip(p,q):
        local=crd_local(np.array([a,b]),4.,1.5)
        np.testing.assert_allclose(local[0],local[1],atol=1e-12)


def test_polygon_centroid_independent_of_boundary_sampling():
    pytest.importorskip('numba')
    from numba.typed import List
    from activefoam._kernels import polygon_centroids

    loop = List([np.array([0], dtype=np.int64)])
    coarse = np.array([[0., 0.], [4., 0.], [2., 2.], [0., 2.], [0., 0.]])
    resampled = np.insert(coarse, 1, [[1., 0.], [2., 0.]], axis=0)
    for points in (coarse, resampled):
        centers = polygon_centroids(List([points]), loop, 10., 0.)
        np.testing.assert_allclose(centers[0], [14/9, 8/9], atol=1e-12)
