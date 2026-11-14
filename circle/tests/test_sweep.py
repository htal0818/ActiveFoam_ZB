import numpy as np
import pytest
from circle.sweep import geometry_parameters,unwrap_centroids,motion_statistics,aggregate,tasks


def test_curvature_sweep_preserves_density_and_has_matched_controls():
    a,b=geometry_parameters(36,1.3),geometry_parameters(144,1.3)
    assert b['radius']==pytest.approx(2*a['radius'])
    assert b['curvature']==pytest.approx(a['curvature']/2)
    assert a['nominal_density']==pytest.approx(b['nominal_density'])
    assert a['nominal_density']==pytest.approx(1.3)
    all_tasks=tasks((36,100),(1,))
    for t in all_tasks:
        assert any(u.domain!=t.domain and (u.n,u.w,u.rho,u.mu,u.mode,u.seed)==
                   (t.n,t.w,t.rho,t.mu,t.mode,t.seed) for u in all_tasks)


def test_periodic_unwrapping_recovers_boundary_crossing_msd():
    x=np.arange(0.,6.,.25)
    track=np.zeros((len(x),1,2));track[:,0,0]=(x+3.8)%4
    unwrapped=unwrap_centroids(track,4.)
    np.testing.assert_allclose(unwrapped[:,0,0],x+3.8)
    result=motion_statistics(unwrapped,.25,1.)
    np.testing.assert_allclose(result['msd'],result['lag']**2)


def test_failed_seed_is_not_zero_and_one_seed_has_no_sem():
    records=[dict(accepted=True,value=2.),dict(accepted=False,value=0.)]
    assert aggregate(records,'value')==dict(n=1,attempted=2,mean=2.,sem=None)
    records.append(dict(accepted=True,value=4.))
    out=aggregate(records,'value')
    assert out['mean']==3 and out['sem']==pytest.approx(1.)
