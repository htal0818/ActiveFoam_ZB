import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from circle.disc import DiscFoam
from circle.ellipse import EllipseFoam
from circle.ellipse_kernels import project_ellipse,ellipse_normal
from circle.experiments import Recorder,observe_transitions,Experiment,activity,clusters,motion_analysis
from circle.measure import neighbor_pairs
from circle import state,protocol


def test_ellipse_projection_is_nearest_and_normal():
    rng=np.random.default_rng(7)
    for angle in rng.uniform(.01,np.pi/2-.01,12):
        a,b=3.,1.5;x,y=1.4*a*np.cos(angle),1.4*b*np.sin(angle)
        p=np.array(project_ellipse(x,y,a,b))
        optimum=minimize_scalar(lambda t:(a*np.cos(t)-x)**2+(b*np.sin(t)-y)**2,
                                bounds=(0,np.pi/2),method='bounded',options={'xatol':1e-14})
        assert np.dot(p-[x,y],p-[x,y])==pytest.approx(optimum.fun,abs=1e-10)
        assert (p[0]/a)**2+(p[1]/b)**2==pytest.approx(1.,abs=2e-14)
        n=ellipse_normal(*p,a,b);d=p-[x,y]
        assert abs(d[0]*n[1]-d[1]*n[0])<1e-12


def test_circle_limit_and_observation_leave_dynamics_unchanged():
    plain=DiscFoam.random(12,seed=3,w=1.,rho=1.2)
    ellipse=EllipseFoam.random(12,seed=3,w=1.,rho=1.2,aspect=1.)
    recorder=Recorder(ellipse);observe_transitions(ellipse,recorder);recorder.begin(ellipse)
    for k in range(1,21):
        plain.advance_scheduled(k,.5);ellipse.advance_scheduled(k,.5)
        recorder.contact_step(k*plain.dt,ellipse)
    np.testing.assert_allclose(ellipse.vpos,plain.vpos,rtol=0,atol=1e-13)
    np.testing.assert_array_equal(ellipse.e_t,plain.e_t)
    assert ellipse.rng.bit_generator.state==plain.rng.bit_generator.state
    assert not state.unclassified(ellipse)


def test_ellipse_confinement_area_and_complete_guard_rollback():
    foam=EllipseFoam.random(12,seed=4,rho=1.3,aspect=2.)
    assert foam.disc_area==pytest.approx(12.)
    for _ in range(20):foam.step(.5)
    assert foam.max_radius()<=foam.radius*(1+1e-12)
    snap=state.snapshot(foam)
    def invalid():foam.vpos[0]=foam.center+[2*foam.axes[0],0];return True
    result=foam._guard('fixture',0,foam.center,invalid,lambda result,snap:result,False)
    assert result is False and not state.differences(foam,snap)


def test_pulse_returns_to_baseline_and_matches_film_count():
    foam=DiscFoam.random(12,seed=1)
    task=Experiment('pulse',1,pattern='sector',mu=.25)
    assert activity(task,foam,0)==.25
    assert activity(task,foam,25)==.25
    amplitude=activity(task,foam,15)
    assert len(amplitude)==len(foam.e_f)
    assert set(amplitude)=={.25,1.}


def test_cage_relative_motion_removes_uniform_translation():
    samples=[dict(time=float(t),centers=np.array([[t,0.],[t+1,0.],[t+2,0.]])) for t in range(9)]
    result=motion_analysis(samples,{(0,1),(1,2)},1.)
    for row in result['lags']:
        assert row['msd']==pytest.approx(row['lag']**2)
        assert row['cage_msd']==pytest.approx(0.)


def test_contact_survival_distinguishes_reformation_and_censors_end():
    class Foam:dt=.1
    r=Recorder(Foam());r.initial_pairs={(0,1)};r.previous_pairs={(0,1)};r.surviving={(0,1)};r.birth={(0,1):(0.,True)}
    class Graph:
        e_f=np.array([[0,2]])
    r.contact_step(1.,Graph())
    Graph.e_f=np.array([[0,1],[0,2]])
    r.contact_step(2.,Graph());r.close(3.)
    assert not r.surviving and (0,1) in r.previous_pairs
    assert r.episodes[0]['left_censored'] and not r.episodes[0]['right_censored']
    assert sum(e['right_censored'] for e in r.episodes)==2


def test_cascade_criterion_uses_space_and_time():
    events=[dict(time=0.,location=[0,0]),dict(time=.1,location=[.1,0]),dict(time=.2,location=[20,0])]
    assert sorted(clusters(events,.5,1.))==[1,2]


def test_nested_transition_rolled_back_by_parent_is_not_recorded():
    foam=DiscFoam.random(12,seed=2)
    rec=Recorder(foam);observe_transitions(foam,rec);rec.begin(foam)
    snap=state.snapshot(foam)
    def parent_operation():
        foam._guard('inner',0,foam.center,lambda:True,lambda result,snap:result,False)
        foam.vpos[0]+=0.01
        return False
    foam._guard('outer',0,foam.center,parent_operation,lambda result,snap:result,False)
    assert not rec.events and not state.differences(foam,snap)


def test_elliptical_kernel_preserves_force_law_away_from_wall():
    from activefoam._kernels import boundary_normal_signs
    from circle.kernels import advance_disc
    from circle.ellipse_kernels import advance_ellipse
    f=DiscFoam.random(12,seed=5)
    mids,loops=f._fast_geometry_inputs()
    args=(f.vpos,f.e_v,f.e_f,f.e_t,mids,f.f_area,f.psi,f.dt,f.P0,f.lsc,f.edpc,
          boundary_normal_signs(f.e_f,loops),*f.center,100.,1e-9,True,False)
    circle=advance_disc(*args);ellipse=advance_ellipse(*args,2.)
    np.testing.assert_array_equal(circle[0],ellipse[0])
    for a,b in zip(circle[1],ellipse[1]):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(circle[2],ellipse[2])


def test_real_neighbor_change_is_recorded_once_without_changing_dynamics():
    plain=DiscFoam.random(36,seed=2,w=1.)
    observed=DiscFoam.random(36,seed=2,w=1.)
    rec=Recorder(observed);observe_transitions(observed,rec);rec.begin(observed)
    for edge in np.flatnonzero(np.all(plain.e_f>=0,axis=1)):
        before=neighbor_pairs(plain)
        result=plain.t1_foam(int(edge))
        assert observed.t1_foam(int(edge))==result
        if result and neighbor_pairs(plain)!=before:break
    else:pytest.fail('fixture produced no real neighbor-changing T1')
    assert len(rec.events)==1 and len(rec.pair_changes)==1
    assert set(map(tuple,rec.events[0]['lost']))==before-neighbor_pairs(plain)
    assert set(map(tuple,rec.events[0]['gained']))==neighbor_pairs(plain)-before
    np.testing.assert_array_equal(plain.vpos,observed.vpos)
    np.testing.assert_array_equal(plain.e_t,observed.e_t)
    assert plain.rng.bit_generator.state==observed.rng.bit_generator.state


def test_ellipse_compiled_and_python_paths_agree_at_wall():
    from activefoam._kernels import boundary_normal_signs
    from circle.ellipse_kernels import advance_ellipse
    f=EllipseFoam.random(12,seed=5,aspect=1.5,rho=1.4)
    mids,loops=f._fast_geometry_inputs()
    args=(f.vpos,f.e_v,f.e_f,f.e_t,mids,f.f_area,f.psi,f.dt,f.P0,f.lsc,f.edpc,
          boundary_normal_signs(f.e_f,loops),*f.center,f.radius,f.wall_tol,True,False,1.5)
    compiled=advance_ellipse(*args);python=advance_ellipse.py_func(*args)
    for i,(a,b) in enumerate(zip(compiled,python)):
        if i==1:
            for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=0,atol=2e-14)
        else:np.testing.assert_allclose(a,b,rtol=0,atol=2e-12)
