"""Measurement checks independent of any proposed collective response."""
import numpy as np
import pytest
from circle.collective import variance_matched_mu,trajectory
from circle import force_probe as fp, protocol
from circle.tests.test_force_probe import representation


def test_fixed_variance_matches_discrete_ou_not_mu_or_continuum_approximation():
    # The intentionally coarse dt makes an accidental continuum formula detectable.
    dt=.2;reference=.7;ref_tau=10
    expected=reference**2/(ref_tau*(2-dt/ref_tau))
    for tau in [1.,2.,10.,20.]:
        mu=variance_matched_mu(tau,dt,reference,ref_tau)
        assert mu**2/(tau*(2-dt/tau))==pytest.approx(expected,rel=1e-13)
    with pytest.raises(ValueError):variance_matched_mu(.05,.2)


def test_returned_probe_state_never_aliases_input(monkeypatch):
    f=protocol.periodic_foam(4,1.,1.3,seed=1)
    before=representation(vars(f))
    edge=next(e for e in range(len(f.e_f)) if min(f.e_f[e])>=0 and fp.separation(f,f.e_v[e])[0]>.1)
    def settled(work,e,vertices,target,config,short):
        return dict(separation=target,film_length=target,force=1.,converged=True,duration=1.)
    monkeypatch.setattr(fp,'hold',settled)
    result,work=fp.probe(f,edge,fp.ProbeConfig(increments=2,apply_t1=False),return_state=True)
    assert result['complete']
    work.vpos[:]=42;work.rng.random(5)
    assert representation(vars(f))==before


def test_trajectory_unwraps_periodic_motion_and_removes_translation(monkeypatch):
    # Synthetic coherent translation crossing the box boundary must not look fluid.
    from circle import collective
    class Moving:
        bs=4.;dt=.1;nFa=2;gam=1.
        e_f=np.array([[0,1]]);e_t=np.array([1.]);f_area=np.ones(2)
        def __init__(self):self.centers=np.array([[3.95,.5],[1.,2.]])
        def advance_scheduled(self,k,mu):self.centers=(self.centers+[.1,0])%self.bs
        def shape_factors(self):return np.full(2,4.)
        def volume_fraction(self):return .125
    monkeypatch.setattr(collective,'cell_centroids',lambda f:f.centers.copy())
    monkeypatch.setattr(fp,'validate',lambda f:{})
    result=trajectory(Moving(),1.,sample=.1)
    assert not result['changes']
    assert max(row['msd'] for row in result['records'])<1e-25
    assert all(row['contact_survival']==1 for row in result['records'])


def test_zero_duration_reversal_does_not_reset_physical_dwell():
    from circle.scripts.audit_collective_division import collapse_same_step
    from circle.scripts.division_contact_analysis import persistent_changes
    p=(0,1)
    events=[dict(time=.1,lost=[p],gained=[]),
            dict(time=.2,lost=[],gained=[p]),dict(time=.2,lost=[p],gained=[])]
    net=collapse_same_step([p],events,.001)
    changes=persistent_changes([p],net,0.,.35,.2)
    assert len(changes)==1 and changes[0]['time']==pytest.approx(.3)
