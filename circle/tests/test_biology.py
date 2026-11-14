import copy
import numpy as np
import pytest
from circle.disc import DiscFoam,circle_polyline
from circle.biology import BiologyFoam
from circle import division,state,protocol


def test_disabled_biology_matches_original_exactly():
    original=DiscFoam.random(12,seed=2,rho=1.2,w=1.)
    extended=BiologyFoam.from_disc(original)
    for k in range(1,101):original.advance_scheduled(k,.25);extended.advance_scheduled(k,.25)
    for key in ('vpos','e_f','e_t','f_area'):np.testing.assert_array_equal(getattr(original,key),getattr(extended,key))
    assert original.rng.bit_generator.state==extended.rng.bit_generator.state
    assert not state.unclassified(extended)


def test_delayed_strain_has_no_early_response(monkeypatch):
    f=BiologyFoam.from_disc(DiscFoam.random(3,seed=1),gain=2,delay=.3,memory_spacing=.05)
    length=[1.]
    monkeypatch.setattr(BiologyFoam,'contact_lengths',lambda self:{(10,11):length[0]})
    f._feedback_signal()
    outputs=[]
    for i in range(1,15):
        f.bio_time=i*.05;length[0]=1.1 if i>=4 else 1.
        outputs.append((f.bio_time,f._feedback_signal()[(10,11)]))
    assert all(abs(value)<1e-12 for t,value in outputs if t<.5-1e-10)
    assert outputs[9][1]==pytest.approx(np.tanh(1.))
    assert max(abs(v) for _,v in outputs)<=1


def test_lost_and_reformed_contact_gets_zero_prehistory(monkeypatch):
    f=BiologyFoam.from_disc(DiscFoam.random(3,seed=1),gain=1,delay=1)
    lengths={(10,11):1.};monkeypatch.setattr(BiologyFoam,'contact_lengths',lambda self:lengths.copy())
    f._feedback_signal();lengths.clear();f.bio_time=.05;assert f._feedback_signal()=={}
    lengths[10,11]=2.;f.bio_time=.1
    assert f._feedback_signal()=={(10,11):0.}
    assert f.feedback_reference[0]==2


@pytest.mark.parametrize('n',[3,12,36])
def test_synchronous_round_conserves_area_pressure_and_wall(n):
    f=BiologyFoam.from_disc(DiscFoam.random(n,seed=1,rho=1.2),gain=.5,delay=2)
    before_area=f.f_area.copy();before_preferred=f.preferred.copy();radius=f.radius;center=f.center.copy()
    generator=copy.deepcopy(f.rng.bit_generator.state);vertices=len(f.vpos);edges=len(f.e_v)
    assert f.divide_synchronously(),f.division_log[-1]
    assert f.nFa==2*n and len(f.vpos)==vertices+2*n and len(f.e_v)==edges+3*n
    np.testing.assert_allclose(f.f_area[:n]+f.f_area[n:],before_area,rtol=1e-10)
    np.testing.assert_allclose(f.preferred[:n]+f.preferred[n:],before_preferred,rtol=1e-13)
    np.testing.assert_allclose(f.preferred[:n]/f.f_area[:n],before_preferred/before_area,rtol=1e-10)
    np.testing.assert_allclose(f.preferred[n:]/f.f_area[n:],before_preferred/before_area,rtol=1e-10)
    assert np.all(np.sum(f.v_e>=0,axis=1)==3)
    assert len(set(f.cell_uid))==2*n and set(f.cell_uid).isdisjoint(range(n))
    assert np.all(f.birth_time==f.bio_time) and np.all(f.parent_uid[:n]==np.arange(n))
    assert np.all(f.parent_uid[n:]==np.arange(n))
    assert f.radius==radius;np.testing.assert_array_equal(f.center,center)
    assert f.rng.bit_generator.state==generator and not state.unclassified(f)
    protocol.validate(f)
    for k in range(1,21):f.advance_scheduled(k,.05)
    protocol.validate(f)


def test_single_closed_cell_can_divide():
    center=np.array([12.,12.]);ring=circle_polyline(center,1.,.075)
    f=BiologyFoam.from_disc(DiscFoam.from_closed_cells(3.,[ring],center=center,rho=1.2))
    assert f.divide_synchronously(),f.division_log[-1]
    assert f.nFa==2;protocol.validate(f)


@pytest.mark.parametrize('error',[ValueError,RuntimeError])
def test_whole_round_rollback_after_second_mother_fails(monkeypatch,error):
    f=BiologyFoam.from_disc(DiscFoam.random(12,seed=1),gain=1,delay=2)
    snap=state.snapshot(f);actual=division.divide_cell
    def partial(foam,cell,plan):
        result=actual(foam,cell,plan)
        if cell==1:
            foam.rng.random();foam.feedback_history[:]=9
            raise error('controlled late failure')
        return result
    monkeypatch.setattr(division,'divide_cell',partial)
    if error is ValueError:assert not f.divide_synchronously()
    else:
        with pytest.raises(RuntimeError):f.divide_synchronously()
    assert state.differences(f,snap)==[]
    assert len(f.division_log)==1 and not f.division_log[-1]['accepted']
    protocol.validate(f)


def test_two_rounds_double_only_current_mothers():
    f=BiologyFoam.from_disc(DiscFoam.random(3,seed=4))
    total=f.preferred.sum()
    assert f.divide_synchronously()
    f.bio_time=10.
    assert f.divide_synchronously()
    assert f.nFa==12 and f.preferred.sum()==pytest.approx(total)
    assert np.all(f.birth_time==10.)
    assert [r['mothers'] for r in f.division_log]==[3,6]


def test_growth_changes_preferred_area_without_changing_units_or_wall():
    f=BiologyFoam.from_disc(DiscFoam.random(3,seed=1),growth_rate=np.log(2))
    before=f.preferred.copy();constants=(f.psi,f.lsc,f.A0,f.radius)
    f.step(0.)
    np.testing.assert_allclose(f.preferred,before*2**f.dt)
    assert constants==(f.psi,f.lsc,f.A0,f.radius)
    e=np.flatnonzero(np.any(f.e_f<0,axis=1))[0];c=f.e_f[e,1]
    assert f.normal_force(e,2.)==pytest.approx(f.P0*(f.preferred[c]/f.f_area[c]-1)/f.lsc)


def test_existing_transition_guard_restores_biology_fields():
    f=BiologyFoam.from_disc(DiscFoam.random(3,seed=1),gain=1,delay=1)
    snap=state.snapshot(f)
    def invalid():f.preferred*=2;f.feedback_history[:]=3;f.nFa+=1;return False
    assert not f._guard('fixture',0,f.center,invalid,lambda result,snapshot:result,False)
    assert state.differences(f,snap)==[]
