import copy
import numpy as np
import pytest
from circle import division,state
from circle.disc import DiscFoam
from circle.biology import BiologyFoam
from circle.force_probe import isolated_copy


def tissue(n=8):return BiologyFoam.from_disc(DiscFoam.random(n,seed=4,rho=1.2,w=1.,edpc=.05))


@pytest.mark.parametrize('n',[8,16])
def test_random_wave_conserves_area_pressure_and_simulation_rng(n):
    f=tissue(n);old_area=f.f_area.copy();old_preferred=f.preferred.copy()
    rng=copy.deepcopy(f.rng.bit_generator.state);clock=f.bio_time
    assert f.divide_synchronously(np.random.default_rng(1)),f.division_log[-1]
    assert f.nFa==2*n and f.rng.bit_generator.state==rng and f.bio_time==clock
    np.testing.assert_allclose(f.f_area[:n]+f.f_area[n:],old_area,rtol=1e-10)
    np.testing.assert_allclose(f.preferred[:n]+f.preferred[n:],old_preferred,rtol=1e-13)
    assert np.all(f.birth_time==clock)
    for event in f.division_log[-1]['events']:
        np.testing.assert_allclose(event['pressure_daughters'],event['pressure_before'],rtol=1e-10)
        assert event['new_tension']==2-f.gam and 0<=event['angle']<np.pi
        assert event['angle_adjustment']==0 # redraws, not deterministic angular nudges
    assert not state.unclassified(f)


def test_random_wave_reproducible_and_planned_before_any_mutation(monkeypatch):
    a=tissue();b=isolated_copy(a);original=division.intersections;seen=[]
    def observe(f,*args):
        seen.append(f.nFa)
        return original(f,*args)
    monkeypatch.setattr(division,'intersections',observe)
    assert a.divide_synchronously(np.random.default_rng(9))
    # At least all eight planning calls precede the first split; intersection
    # checks during commitment then see increasing cell numbers.
    first_growth=next(i for i,n in enumerate(seen) if n>8)
    assert first_growth>=9
    assert b.divide_synchronously(np.random.default_rng(9))
    for key in state.physical_fields(a):assert state._same(getattr(a,key),getattr(b,key)),key
    with pytest.raises(ValueError,match='separate'):a.divide_synchronously(a.rng)


def test_invalid_random_proposal_is_redrawn_without_partial_division(monkeypatch):
    f=tissue();snap=state.snapshot(f);original=division.intersections;calls=[]
    def first_invalid(foam,cell,*args):
        calls.append(cell)
        if len(calls)==1:raise division.DivisionRejected('controlled invalid proposal')
        assert foam.nFa==8
        return original(foam,cell,*args)
    monkeypatch.setattr(division,'intersections',first_invalid)
    plans,draws=division.random_plans(f,np.random.default_rng(8))
    assert draws[0]>=2 and len(plans)==8 and not state.differences(f,snap)


@pytest.mark.parametrize('kind',[ValueError,RuntimeError])
def test_entire_random_wave_rolls_back_after_late_failure(monkeypatch,kind):
    f=tissue();snap=state.snapshot(f);original=division.divide_cell
    def fail(foam,cell,plan):
        event=original(foam,cell,plan)
        if cell==1:foam.rng.random();raise kind('after second split')
        return event
    monkeypatch.setattr(division,'divide_cell',fail)
    if kind is ValueError:assert not f.divide_synchronously(np.random.default_rng(9))
    else:
        with pytest.raises(RuntimeError):f.divide_synchronously(np.random.default_rng(9))
    assert not state.differences(f,snap)


def test_stress_uses_inherited_osmotic_content_after_division():
    f=tissue();f.e_t[:]=0.;before=f.stress_tensor().copy()
    assert f.divide_synchronously(np.random.default_rng(1))
    f.e_t[:]=0. # isolate pressure from the newly inserted tensile interfaces
    np.testing.assert_allclose(f.stress_tensor(),before,atol=1e-11)
    expected=-f.rho*f.P0*np.sum(f.preferred-f.f_area)/f.disc_area
    np.testing.assert_allclose(f.stress_tensor(),np.eye(2)*expected,atol=1e-11)


def test_measurement_wrapper_leaves_dynamics_and_ou_draws_unchanged():
    from circle.cleavage import MeasuredFoam
    base=DiscFoam.random(8,seed=3,rho=1.2,edpc=.05)
    f=BiologyFoam.from_disc(base);g=MeasuredFoam.from_disc(base)
    for k in range(1,31):f.advance_scheduled(k,.25);g.advance_scheduled(k,.25)
    for key in state.physical_fields(f):assert state._same(getattr(f,key),getattr(g,key)),key
    assert f.rng.bit_generator.state==g.rng.bit_generator.state
    assert not state.unclassified(g)


def test_junction_force_measurement_matches_original_integrator():
    from circle.cleavage import junction_forces
    f=tissue();force,_=junction_forces(f);before=f.vpos.copy()
    f.dt=1e-7;f.remesh=False;f.step(0.)
    actual=(f.vpos-before)/(f.psi*f.dt)-f.wall.junction_reaction
    np.testing.assert_allclose(force,actual,rtol=1e-7,atol=1e-7)


def test_no_area_or_pressure_jump_is_reported_as_old_uniform_area_stress():
    from circle.cleavage import anisotropy
    theta=np.linspace(0,2*np.pi,256,endpoint=False)
    circle=np.column_stack((np.cos(theta),np.sin(theta)))
    assert anisotropy(circle)<1e-12
    ellipse=circle*np.array([2.,1.])
    assert anisotropy(ellipse)==pytest.approx(.6,abs=1e-10)
