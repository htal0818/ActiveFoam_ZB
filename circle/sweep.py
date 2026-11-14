"""Resumable exploratory disk-curvature sweeps with periodic size controls.

R=sqrt(N/pi) holds nominal density fixed as wall curvature changes. Curvature
and finite size remain coupled; the tissue surface itself has zero curvature.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import time
import numpy as np


@dataclass(frozen=True)
class Task:
    domain: str
    mode: str
    n: int
    w: float
    rho: float
    mu: float
    seed: int
    dt: float = .005
    spacing: float = .15
    warmup: float = 10.
    duration: float = 50.
    sample: float = .5
    budget: float = 300.

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:20]


def geometry_parameters(n, rho):
    if n < 3 or rho <= 0: raise ValueError('positive density and at least three cells required')
    radius=math.sqrt(n/math.pi)
    return dict(radius=radius, curvature=1/radius,
                dimensionless_curvature=math.sqrt(rho)/radius,
                nominal_density=n*rho/(math.pi*radius**2))


def tasks(sizes=(36,100,196), seeds=(1,2,3), duration=50.):
    result=[]
    for domain in ('disc','periodic'):
        for n in sizes:
            if math.isqrt(n)**2 != n: raise ValueError('matched periodic controls need square cell counts')
            for seed in seeds:
                for w in (.5,1.,1.5):
                    for rho in (1.,1.2,1.4):
                        result.append(Task(domain,'structure',n,w,rho,0.,seed,duration=duration))
                for mu in (0.,.5,1.,1.5):
                    result.append(Task(domain,'dynamics',n,1.,1.3,mu,seed,duration=duration))
    return result


def serializable(value):
    if isinstance(value,np.ndarray): return serializable(value.tolist())
    if isinstance(value,np.generic): return serializable(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,dict): return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [serializable(v) for v in value]
    return value


def source_digest(root):
    paths=sorted((root/'activefoam').glob('*.py'))+[root/'circle'/f'{name}.py' for name in
        ('disc','kernels','protocol','measure','state','tiling','sweep')]+[root/'requirements-lock.txt']
    digest=hashlib.sha256()
    for p in paths:
        digest.update(str(p.relative_to(root)).encode());digest.update(p.read_bytes())
    return digest.hexdigest()


def unwrap_centroids(track,box):
    """Unwrap successive samples; valid when cell motion per sample is < box/2."""
    increments=np.diff(track,axis=0)
    increments-=box*np.rint(increments/box)
    return np.concatenate([track[:1],track[:1]+np.cumsum(increments,axis=0)])


def motion_statistics(track,interval,area,mask=None):
    """Time-origin averaged MSD, lags <= T/4 to retain >=75% of origins."""
    track=np.asarray(track)
    if mask is not None: track=track[:,mask]
    maximum=max(1,(len(track)-1)//4)
    lags=np.unique(np.rint(np.geomspace(1,maximum,min(24,maximum))).astype(int))
    if not track.shape[1]: return dict(lag=lags*interval,msd=np.full(len(lags),np.nan))
    return dict(lag=lags*interval,msd=np.array([
        np.mean(np.sum((track[k:]-track[:-k])**2,axis=2))/area for k in lags]))


def run_task(task):
    from activefoam.budget import call_with_budget,BudgetExceeded
    record=dict(task=asdict(task),key=task.key,accepted=False,stage='preparation')
    start=time.monotonic()
    try:
        call_with_budget(task.budget,_simulate,task,record)
        record['accepted']=True
    except (ValueError,FloatingPointError,BudgetExceeded) as exc:
        record['error']=str(exc)
    record['seconds']=round(time.monotonic()-start,3)
    return serializable(record)


def _simulate(task,out):
    from activefoam._kernels import polygon_centroids
    from activefoam import foam_measure
    from . import protocol
    from .measure import zone_statistics,neighbor_pairs,wall_load,wall_layer,cell_centroids
    disc=task.domain=='disc'
    if task.domain not in ('disc','periodic'): raise ValueError('unknown domain')
    kw=dict(w=task.w,rho=task.rho,seed=task.seed,dt=task.dt,edpc=task.spacing)
    foam=(protocol.equilibrated(task.n,**kw) if disc else
          protocol.periodic_equilibrated(math.isqrt(task.n),**kw))
    out['geometry_parameters']=geometry_parameters(task.n,task.rho)
    if not disc:
        out['geometry_parameters']['radius']=None
        out['geometry_parameters']['box_side']=math.sqrt(task.n)
        out['geometry_parameters']['curvature']=None
        out['geometry_parameters']['dimensionless_curvature']=None
    quality=dict(checks=0,local_microfold_checks=0,max_local_microfolds=0)
    def check():
        if disc:
            r=protocol.validate(foam)
            quality['checks']+=1
            quality['local_microfold_checks']+=int(r['local_film_microfolds']>0)
            quality['max_local_microfolds']=max(quality['max_local_microfolds'],r['local_film_microfolds'])
            return r
        foam_measure.validate(foam)
        return dict(policy='periodic per-cell crossing policy',**foam_measure.quality(foam))
    out['quality']=quality
    out['initial_quality']=check()
    if disc:
        metrics=zone_statistics(foam)
        load=wall_load(foam.wall)['total'] if foam.wall else 0.
        metrics.update(wall_load=load,mean_wall_traction=load/(2*np.pi*foam.radius),
                       wall_fraction=metrics['n_touching']/task.n)
    else:
        metrics=dict(packing=foam.volume_fraction(),z=float(foam.neighbor_number().mean()),
                     shape=float(foam.shape_factors().mean()),area=float(foam.f_area.mean()))
    metrics.update(z_all=float(foam.neighbor_number().mean()),shape_all=float(foam.shape_factors().mean()))
    out['structure']=metrics
    if task.mode=='structure':
        # Keep geometry to regenerate snapshots without rerunning the simulations.
        if disc and task.seed==1:
            out['snapshot']=dict(polygons=[foam._cell_polyline(i) for i in range(task.n)],
                                 center=foam.center,radius=foam.radius)
        out['stage']='complete'
        return
    out['stage']='warmup'
    index=max(0,foam.log.iteration) if disc else round(4*foam.tauT/foam.dt)
    def evolve_steps(count,observer=None,event_observer=None):
        nonlocal index
        for k in range(1,count+1):
            index+=1
            foam.advance_scheduled(index,task.mu,contact_observer=event_observer)
            if k%100==0: check()
            if observer is not None: observer(k)
        check()
    evolve_steps(round(task.warmup/foam.dt))
    out['stage']='measurement'
    def centers():
        if disc: return cell_centroids(foam)
        mids,loops=foam._fast_geometry_inputs()
        return polygon_centroids(mids,loops,foam.bs,foam.sstn)
    masks={'all':np.ones(task.n,dtype=bool)}
    if disc:
        masks['wall']=wall_layer(foam);masks['bulk']=~masks['wall']
    frames=[centers()];pairs=neighbor_pairs(foam);changes=np.zeros(task.n)
    loads=[];shapes=[np.asarray(foam.shape_factors()).copy()];events=Counter()
    contact_steps=[]
    stride=round(task.sample/foam.dt)
    steps=round(task.duration/foam.dt)
    if stride<1 or steps%stride or not np.isclose(stride*foam.dt,task.sample):
        raise ValueError('duration and sample interval must be exact multiples of dt')
    def sample(k):
        nonlocal pairs
        if k%stride: return
        frames.append(centers())
        current=neighbor_pairs(foam)
        step_changes=np.zeros(task.n)
        for a,b in pairs^current: step_changes[a]+=1;step_changes[b]+=1
        changes[:] += step_changes
        contact_steps.append(step_changes)
        pairs=current
        shapes.append(np.asarray(foam.shape_factors()).copy())
        if disc: loads.append(wall_load(foam.wall)['total'])
    def event(iteration,kind,current):events[kind]+=1
    evolve_steps(steps,sample,event)
    track=np.asarray(frames)
    if not disc: track=unwrap_centroids(track,foam.bs)
    relative=track-track.mean(axis=1,keepdims=True)
    shape_track=np.asarray(shapes)
    step_changes=np.asarray(contact_steps)
    out['dynamics']=dict(duration=steps*foam.dt,sample_interval=stride*foam.dt,
        tauT=foam.tauT,shape_mean=float(np.mean(shape_track[1:])),events=dict(events),
        shape_time=np.arange(len(shape_track))*task.sample,
        shape_series={name:shape_track[:,mask].mean(axis=1) if mask.any() else np.full(len(shape_track),np.nan)
                      for name,mask in masks.items()},
        shape_change={name:float((shape_track[-1,mask]-shape_track[0,mask]).mean()) if mask.any() else None
                      for name,mask in masks.items()},
        contact_rate_series={name:step_changes[:,mask].mean(axis=1)/task.sample if mask.any() else np.full(len(step_changes),np.nan)
                             for name,mask in masks.items()},
        cell_shape_initial=shape_track[0],cell_shape_final=shape_track[-1],cell_contact_changes=changes,
        cell_initial_zone_wall=masks.get('wall',np.zeros(task.n,dtype=bool)),
        t1_rate=events['T1']/(task.n*steps*foam.dt),
        motion={name:motion_statistics(track,task.sample,foam.A0,mask) for name,mask in masks.items()},
        com_subtracted=motion_statistics(relative,task.sample,foam.A0),
        contact_change_rate={name:float(changes[mask].mean()/(steps*foam.dt)) if mask.any() else None
                             for name,mask in masks.items()},
        zone_counts={name:int(mask.sum()) for name,mask in masks.items()},
        wall_load_mean=float(np.mean(loads)) if loads else None)
    out['final_quality']=check()
    out['stage']='complete'


def aggregate(records,key):
    """Mean/SEM over independent accepted seeds, not cells or time samples."""
    values=[]
    for record in records:
        if not record['accepted']: continue
        value=record
        for part in key.split('.'): value=value.get(part) if isinstance(value,dict) else None
        if value is not None and np.isfinite(value):values.append(float(value))
    return dict(n=len(values),attempted=len(records),mean=float(np.mean(values)) if values else None,
                sem=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None)
