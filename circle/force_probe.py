"""External force/length probes of the unchanged overdamped foam dynamics.

No energy is assumed. Work is the signed integral of equilibrated external
compressive force over shortening. Probe execution never mutates its input.
"""
from dataclasses import dataclass, asdict
import copy
import numpy as np
from numba.typed import List
from activefoam import foam_measure
from . import protocol, state
from .disc import DiscFoam

T1_LENGTH = .01 * 2 * np.sqrt(np.pi)


def isolated_copy(foam):
    """Copy actual instance state, including extensions, RNG and diagnostics.

    Identity-keyed caches are invalid on the copy, not on the original. The
    original objects and cached keys never need reconstruction or restoration.
    """
    def clone(value):
        if isinstance(value, np.ndarray): return value.copy()
        if isinstance(value, List): return List([clone(v) for v in value])
        if isinstance(value, list): return [clone(v) for v in value]
        if isinstance(value, tuple): return tuple(clone(v) for v in value)
        if isinstance(value, dict): return {k:clone(v) for k,v in value.items()}
        return copy.deepcopy(value)
    result=object.__new__(type(foam))
    result.__dict__={key:clone(value) for key,value in vars(foam).items()}
    for key in state.KEYED_CACHES:
        result.__dict__.pop(key,None)
    return result


def pairs(foam):
    faces=foam.e_f[np.all(foam.e_f>=0,axis=1)]
    return set(map(tuple,np.sort(faces,axis=1).tolist()))


def separation(foam, vertices):
    positions=foam._local(foam.vpos[list(vertices)])
    delta=positions[1]-positions[0]
    length=float(np.linalg.norm(delta))
    if length<=1e-12: raise ProbeStop('coincident probe junctions')
    return length,delta/length


class ProbeStop(ValueError): pass


def move_junctions(foam, vertices, shifts):
    """External split displacement; shared film endpoints follow the junctions.

    A force requiring penetration of the unilateral wall is preempted, not
    silently clipped into a different, non-central external force.
    """
    moved=foam.vpos[list(vertices)]+shifts
    if isinstance(foam,DiscFoam) and np.any(np.linalg.norm(moved-foam.center,axis=1)>
                                          foam.radius*(1+foam.REJECT_TOL)):
        raise ProbeStop('external constraint would cross the rigid wall')
    foam.vpos[list(vertices)]=moved
    foam._wrap_vertices()
    for edge in np.unique(foam.v_e[list(vertices)]):
        if edge<0: continue
        points=np.array(foam.e_mid[edge])
        points[0]=foam.vpos[foam.e_v[edge,0]]
        points[-1]=foam.vpos[foam.e_v[edge,1]]
        foam.e_mid[edge]=foam._local(points)
    foam._update_faces()


def clamp_step(foam, vertices, target):
    foam.step(0.)                 # OU drift, force law, remeshing all unchanged
    trial,direction=separation(foam,vertices)
    correction=trial-target
    force=correction/(2*foam.psi*foam.dt)
    move_junctions(foam,vertices,np.array([direction,-direction])*correction/2)
    return float(force)


def force_step(foam, vertices, force):
    foam.step(0.)
    _,direction=separation(foam,vertices)
    move_junctions(foam,vertices,np.array([direction,-direction])*foam.psi*foam.dt*force)


def validate(foam):
    if isinstance(foam,DiscFoam): return protocol.validate(foam)
    foam_measure.validate(foam)
    return foam_measure.quality(foam)


@dataclass(frozen=True)
class ProbeConfig:
    increments:int=24
    min_hold:float=15.
    max_hold:float=300.
    window:float=5.
    stable_windows:int=3
    force_atol:float=2e-4
    force_rtol:float=1e-3
    check_interval:float=.25
    release:float=5.
    target:float=T1_LENGTH*.98
    apply_t1:bool=True
    stop_on_other_short:bool=True


def prerequisites(foam, edge, config):
    if getattr(foam,'reference_pressure_lag',False) or foam.pressure_normals!='boundary':
        raise ValueError('probes require boundary normals and current-step pressure')
    if foam.sstn!=0: raise ValueError('probe protocol requires zero applied shear')
    if getattr(foam,'growth_rate',0) or getattr(foam,'feedback_gain',0):
        raise ValueError('growth and feedback must be disabled for the passive probe')
    if not (0<=edge<len(foam.e_f)) or np.any(foam.e_f[edge]<0):
        raise ValueError('select a cell-cell contact')
    if config.increments<2 or config.window<=0 or config.max_hold<config.min_hold or config.stable_windows<2:
        raise ValueError('invalid hold/step configuration')
    if not 0<config.target<separation(foam,foam.e_v[edge])[0]:
        raise ValueError('target must be positive and shorter than initial separation')


def hold(foam, edge, vertices, target, config, initial_short):
    count=max(1,round(config.window/foam.dt));check=max(1,round(config.check_interval/foam.dt))
    history=[];forces=[];quality=None
    for iteration in range(1,round(config.max_hold/foam.dt)+1):
        forces.append(clamp_step(foam,vertices,target))
        if iteration%check==0:
            try: quality=validate(foam)
            except ValueError as exc: raise ProbeStop(f'geometry: {exc}') from exc
            if config.stop_on_other_short:
                now=set(np.flatnonzero(foam.edge_lengths()<T1_LENGTH))-initial_short-{edge}
                if now: raise ProbeStop(f'other film reached T1 length: {sorted(now)}')
        if iteration%count==0:
            block=np.array(forces[-count:]);history.append(dict(time=iteration*foam.dt,
                mean=float(block.mean()),sd=float(block.std())))
            recent=history[-config.stable_windows:]
            if len(recent)==config.stable_windows and iteration*foam.dt>=config.min_hold:
                means=np.array([h['mean'] for h in recent])
                tolerance=config.force_atol+config.force_rtol*abs(means.mean())
                # Remeshing can produce a stationary force ripple even at zero
                # tension noise. Test drift of consecutive block means, retain
                # the instantaneous SD, and independently gate dt/hold refinement.
                if np.ptp(means)<=tolerance:
                    return dict(separation=target,film_length=float(foam.edge_lengths()[edge]),
                        force=float(means.mean()),force_sd=float(max(h['sd'] for h in recent)),
                        duration=iteration*foam.dt,converged=True,history=history,quality=quality)
    return dict(separation=target,film_length=float(foam.edge_lengths()[edge]),
        force=float(np.mean(forces[-count:])),force_sd=float(np.std(forces[-count:])),
        duration=config.max_hold,converged=False,history=history,quality=quality)


def curve_metrics(curve):
    if not curve: return {}
    s=curve[0]['separation']-np.array([row['separation'] for row in curve])
    force=np.array([row['force'] for row in curve]);excess=force-force[0]
    cumulative=np.r_[0,np.cumsum((force[:-1]+force[1:])/2*np.diff(s))]
    baseline_work=force[0]*s[-1]
    k=min(4,len(s))
    return dict(baseline_force=float(force[0]),shortening=float(s[-1]),
        stiffness=float(np.polyfit(s[:k],excess[:k],1)[0]) if k>1 else None,
        peak_force=float(force.max()),peak_excess_force=float(excess.max()),
        work=float(cumulative[-1]),excess_work=float(cumulative[-1]-baseline_work),
        peak_cumulative_work=float(cumulative.max()),cumulative_work=cumulative.tolist(),
        work_per_shortening=float(cumulative[-1]/s[-1]) if s[-1]>0 else None,
        excess_work_per_shortening=float((cumulative[-1]-baseline_work)/s[-1]) if s[-1]>0 else None)


def probe(foam, edge, config=ProbeConfig(), progress=None):
    """Quasi-static compression followed by the original T1 and free release.

    Preemption/nonconvergence retains the partial curve, marked incomplete; it
    is never silently interpreted as a zero-work or successful exchange.
    """
    work=isolated_copy(foam)
    prerequisites(work,edge,config)
    vertices=tuple(map(int,work.e_v[edge]));old_pair=tuple(sorted(map(int,work.e_f[edge])))
    opposite=tuple(work._third_face(v,set(old_pair)) for v in vertices)
    q0,_=separation(work,vertices)
    initial_short=set(np.flatnonzero(work.edge_lengths()<T1_LENGTH))
    result=dict(edge=int(edge),vertices=vertices,old_pair=old_pair,opposite=opposite,
        config=asdict(config),dt=work.dt,psi=work.psi,n=work.nFa,
        domain='disc' if isinstance(work,DiscFoam) else 'periodic',
        shape_mean=float(work.shape_factors().mean()),packing=float(work.volume_fraction()),
        initial_short_edges=sorted(initial_short),curve=[],complete=False,status='preempted')
    try:
        for target in np.linspace(q0,config.target,config.increments+1):
            row=hold(work,edge,vertices,float(target),config,initial_short)
            result['curve'].append(row)
            if progress is not None: progress(len(result['curve'])-1,row)
            if not row['converged']: raise ProbeStop('holding force did not converge')
        result['complete']=True
        if not config.apply_t1: result['status']='curve_only'
        elif work.edge_lengths()[edge]>T1_LENGTH:
            raise ProbeStop('junction target reached but curved film remains above T1 length')
        else:
            accepted=work.t1_foam(edge)
            if not accepted: result['status']='T1 declined'
            else:
                result['immediate_pairs']=sorted(pairs(work))
                for iteration in range(1,round(config.release/work.dt)+1):
                    work.advance_scheduled(iteration,0.)
                    if iteration%max(1,round(config.check_interval/work.dt))==0: validate(work)
                validate(work);final=pairs(work)
                if old_pair in final: result['status']='reversal'
                elif min(opposite)>=0 and tuple(sorted(opposite)) in final: result['status']='exchange'
                else: result['status']='space'
                result['final_pairs']=sorted(final)
    except ProbeStop as exc:
        result['status']='preempted';result['reason']=str(exc);result['complete']=False
    except ValueError as exc:
        result['status']='preempted';result['reason']=f'geometry or T1/release: {exc}';result['complete']=False
    result.update(curve_metrics(result['curve']))
    return result


def pulse(foam, edge, force, duration=50., sample_interval=.25):
    """Constant compressive force and zero-force sham, connectivity frozen."""
    runs=[]
    for amplitude in (0.,float(force)):
        work=isolated_copy(foam);vertices=tuple(map(int,work.e_v[edge]));q0,_=separation(work,vertices)
        records=[]
        for iteration in range(1,round(duration/work.dt)+1):
            force_step(work,vertices,amplitude)
            if iteration%max(1,round(sample_interval/work.dt))==0:
                validate(work);q,_=separation(work,vertices)
                records.append(dict(time=iteration*work.dt,separation=q,shortening=q0-q))
        runs.append(records)
    return dict(force=force,duration=duration,sham=runs[0],forced=runs[1],
        sham_subtracted_shortening=runs[1][-1]['shortening']-runs[0][-1]['shortening'])
