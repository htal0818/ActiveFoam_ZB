"""Pilots for delayed contact feedback and synchronous divisions, with controls."""
from dataclasses import dataclass,asdict
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.signal import detrend
from .biology import BiologyFoam
from .experiments import Experiment,prepare,source_hash
from .measure import cell_centroids,wall_load
from .sweep import serializable
from . import protocol


@dataclass(frozen=True)
class BioTask:
    family:str
    seed:int
    gain:float=0.
    delay:float=0.
    n:int=16
    rho:float=1.2
    dt:float=.005
    spacing:float=.1
    history_spacing:float=.05
    duration:float=150.
    sample:float=.25
    mu:float=0.
    growth:bool=False
    divide:bool=False
    division_times:tuple=(20.,)
    budget:float=600.

    @property
    def key(self):return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:20]


def uid_pairs(f):
    return {tuple(sorted(int(f.cell_uid[c]) for c in pair)) for pair in f.e_f if np.all(pair>=0)}


def watch(f,events):
    base=type(f);depth=[0]
    class Observed(base):
        def _guard(self,kind,ident,site,operation,accepted_of,rejected_value):
            outer=depth[0]==0
            if outer:before=uid_pairs(self);location=np.array(site).copy();start=len(self.log.accepted)
            depth[0]+=1
            try:result=super()._guard(kind,ident,site,operation,accepted_of,rejected_value)
            finally:depth[0]-=1
            if outer and len(self.log.accepted)>start and self.log.accepted[-1]==(self.log.iteration,kind,ident):
                after=uid_pairs(self);lost=before-after;gained=after-before
                if lost or gained:events.append(dict(time=self.bio_time,kind=kind,location=location,lost=sorted(lost),gained=sorted(gained)))
            return result
    f.__class__=Observed


def geometry(f,label):
    return dict(time=f.bio_time,label=label,polygons=[f._cell_polyline(c) for c in range(f.nFa)],
                uids=f.cell_uid.copy(),parents=f.parent_uid.copy(),center=f.center,radius=f.radius,
                shape=f.shape_factors(),preferred=f.preferred.copy())


def sample(f,events,reference):
    contacts=np.all(f.e_f>=0,axis=1);tensions=f.e_t[contacts]
    pairs=uid_pairs(f)
    return dict(time=f.bio_time,n=f.nFa,uids=f.cell_uid.copy(),parents=f.parent_uid.copy(),
        centers=cell_centroids(f),shape=f.shape_factors(),area=f.f_area.copy(),preferred=f.preferred.copy(),
        neighbors=f.neighbor_number(),packing=f.volume_fraction(),nominal_density=f.nominal_density,
        tension_mean=float(tensions.mean()) if len(tensions) else None,
        tension_rms=float(np.sqrt(np.mean((tensions-(2-f.gam))**2))) if len(tensions) else None,
        tension_contacts=tensions,wall_load=wall_load(f.wall)['total'] if f.wall is not None else 0.,
        transactions=len(events),generation_neighbor_overlap=len(pairs&reference)/len(reference) if reference else None)


def spectral_summary(samples):
    """Late local fluctuation diagnostics; no automatic oscillation classification."""
    # A spectrum across division would mix different cells and count changes.
    counts=np.array([s['n'] for s in samples]);start=int(np.flatnonzero(counts==counts[-1])[0])
    kept=samples[start:];q=np.array([s['shape'] for s in kept]);times=np.array([s['time'] for s in kept])
    if len(times)<20:return {}
    late=q[len(q)//2:];dt=float(np.median(np.diff(times)))
    local=detrend(late,axis=0);window=np.hanning(len(late))[:,None]
    power=np.mean(abs(np.fft.rfft(local*window,axis=0))**2,axis=1);freq=np.fft.rfftfreq(len(late),dt)
    power[0]=0.;peak=int(np.argmax(power));total=power.sum()
    blocks=[float(np.sqrt(np.mean(np.var(block,axis=0)))) for block in np.array_split(q,4)]
    return dict(generation_start=times[0],local_shape_rms_blocks=blocks,
        late_local_detrended_rms=float(np.sqrt(np.mean(local**2))),
        peak_frequency=float(freq[peak]),peak_period=float(1/freq[peak]) if freq[peak]>0 else None,
        spectral_concentration=float(power[max(1,peak-1):peak+2].sum()/total) if total>0 else 0.,
        frequency=freq,power=power,
        interpretation='Detrended local shape fluctuations in the last half of the final generation. Peak detection alone is not evidence for a sustained oscillation.')


def run(task):
    from activefoam.budget import call_with_budget,BudgetExceeded
    out=dict(task=asdict(task),key=task.key,accepted=False,stage='preparation');start=time.monotonic()
    try:call_with_budget(task.budget,simulate,task,out)
    except (ValueError,FloatingPointError,BudgetExceeded) as exc:out['error']=str(exc)
    out['seconds']=time.monotonic()-start
    return serializable(out)


def simulate(task,out):
    prepared=prepare(Experiment('biology',task.seed,n=task.n,rho=task.rho,dt=task.dt,spacing=task.spacing))
    f=BiologyFoam.from_disc(prepared,gain=task.gain,delay=task.delay,memory_spacing=task.history_spacing)
    events=[];watch(f,events);reference=uid_pairs(f)
    # Shared deterministic perturbation, then zero stochastic drive in feedback runs.
    if task.family.startswith('feedback'):
        for e,faces in enumerate(f.e_f):
            if np.all(faces>=0):f.e_t[e]+=.1*np.cos(1.7*sum((faces+1)**2))
    samples=[sample(f,events,reference)];frames=[geometry(f,'initial')]
    steps=round(task.duration/task.dt);stride=round(task.sample/task.dt)
    schedule={round(t/task.dt) for t in task.division_times} if task.divide else set()
    out['stage']='evolution';exposure=0.;previous_pairs=reference.copy();pair_changes=[]
    try:
        for k in range(1,steps+1):
            out['elapsed']=k*task.dt
            f.growth_rate=np.log(2)/task.division_times[0] if task.growth and k<=round(task.division_times[0]/task.dt) else 0.
            before=f.nFa;f.advance_scheduled(max(0,f.log.iteration)+1,task.mu);exposure+=before*task.dt
            after=uid_pairs(f)
            if after!=previous_pairs:pair_changes.append(dict(time=f.bio_time,lost=sorted(previous_pairs-after),gained=sorted(after-previous_pairs)))
            previous_pairs=after
            if k in schedule:
                frames.append(geometry(f,'before_division'))
                if not f.divide_synchronously():raise ValueError('synchronous round rejected: '+f.division_log[-1]['reason'])
                frames.append(geometry(f,'after_division'));reference=uid_pairs(f);previous_pairs=reference.copy()
            if k%max(1,round(.1/task.dt))==0:protocol.validate(f)
            if k%stride==0:samples.append(sample(f,events,reference))
        out['final_quality']=protocol.validate(f);frames.append(geometry(f,'final'))
        out['summary']=dict(event_count=len(events),event_rate=len(events)/exposure,cell_time=exposure,
            shape_final=float(np.mean(samples[-1]['shape'])),packing_final=f.volume_fraction(),
            nominal_density_final=f.nominal_density,n_final=f.nFa,spectrum=spectral_summary(samples))
        out['accepted']=True;out['stage']='complete'
    finally:
        out['samples']=samples;out['events']=events;out['pair_changes']=pair_changes
        out['frames']=frames;out['division_log']=f.division_log
        if not out['accepted']:out['failure_geometry']=geometry(f,'failed')


def tasks():
    output=[]
    for seed in (1,2,3):
        output.append(BioTask('feedback',seed))
        for gain in (.5,1.,2.,4.):
            for delay in (0.,2.,10.):output.append(BioTask('feedback',seed,gain=gain,delay=delay))
        for growth in (False,True):
            for divide in (False,True):output.append(BioTask('division',seed,rho=.6,duration=60.,mu=.05,growth=growth,divide=divide))
    return output
