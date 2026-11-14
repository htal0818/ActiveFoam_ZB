"""Event-resolved circular/elliptical foam experiments. No planar force-law edits."""
from dataclasses import dataclass,asdict
from collections import deque
from pathlib import Path
import hashlib,json,time
import numpy as np
from . import protocol
from .disc import DiscFoam
from .ellipse import EllipseFoam
from .measure import cell_centroids,neighbor_pairs,wall_load
from .sweep import serializable


@dataclass(frozen=True)
class Experiment:
    family:str
    seed:int
    n:int=36
    w:float=1.
    rho:float=1.2
    mu:float=.5
    tau:float=10.
    dt:float=.005
    spacing:float=.15
    warmup:float=50.
    duration:float=100.
    sample:float=.5
    aspect:float=1.
    pattern:str='uniform'
    peak:float=1.
    pulse_start:float=10.
    pulse_end:float=20.
    check_every:int=100
    budget:float=600.

    @property
    def key(self):return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:20]


def source_hash(root):
    paths=sorted((root/'activefoam').glob('*.py'))+sorted((root/'circle').glob('*.py'))
    h=hashlib.sha256()
    for p in paths:h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def prepare(task):
    cls=EllipseFoam if task.aspect!=1. else DiscFoam
    extra={'aspect':task.aspect} if task.aspect!=1. else {}
    common=dict(seed=task.seed,dt=task.dt,edpc=task.spacing,tauT=10.,**extra)
    relaxed=cls.random(task.n,**(common|protocol.RELAX))
    protocol.relax_confluent(relaxed,round(7.5/task.dt))
    foam=cls.random(task.n,w=task.w,rho=task.rho,**common)
    protocol._adopt(foam,relaxed);protocol._seed_spaces(foam)
    protocol.evolve(foam,10.,.5,check_every=round(.5/task.dt))
    foam.e_t=np.array([foam.fixed_tension(e) for e in range(len(foam.e_v))])
    protocol.evolve(foam,30.,0.,check_every=round(.5/task.dt))
    foam.tauT=task.tau
    return foam


def boundary_geometry(foam,positions):
    """Distance/curvature from a 1024-point wall sample; projection itself is exact."""
    aspect=getattr(foam,'aspect',1.);a=foam.radius*np.sqrt(aspect);b=foam.radius/np.sqrt(aspect)
    if aspect==1.:
        distance=foam.radius-np.linalg.norm(positions-foam.center,axis=1)
        return distance,np.full(len(positions),1/foam.radius)
    theta=np.linspace(0,2*np.pi,1024,endpoint=False)
    wall=foam.center+np.column_stack([a*np.cos(theta),b*np.sin(theta)])
    distance2=np.sum((positions[:,None,:]-wall[None,:,:])**2,axis=2)
    nearest=distance2.argmin(axis=1)
    curvature=a*b/(a*a*np.sin(theta[nearest])**2+b*b*np.cos(theta[nearest])**2)**1.5
    return np.sqrt(distance2[np.arange(len(positions)),nearest]),curvature


def touching_and_free(foam):
    touching=np.zeros(foam.nFa,dtype=bool);free=touching.copy()
    aspect=getattr(foam,'aspect',1.);axes=foam.radius*np.array([np.sqrt(aspect),1/np.sqrt(aspect)])
    for e,faces in enumerate(foam.e_f):
        if np.any(faces<0):
            for f in faces[faces>=0]:
                free[f]=True
                level=np.linalg.norm((np.asarray(foam.e_mid[e])-foam.center)/axes,axis=1)
                if np.max(level)>=1-1e-6:touching[f]=True
    return touching,free


class Recorder:
    """External diagnostic state; never consumes the simulation random stream."""
    def __init__(self,foam):
        self.enabled=False;self.depth=0;self.events=[];self.samples=[];self.pair_changes=[]
        self.episodes=[];self.birth={};self.initial_pairs=set();self.previous_pairs=set();self.surviving=set()
        self.origin=0;self.dt=foam.dt

    def begin(self,foam):
        self.origin=max(0,foam.log.iteration);self.enabled=True
        self.initial_pairs=neighbor_pairs(foam);self.previous_pairs=self.initial_pairs.copy()
        self.surviving=self.initial_pairs.copy();self.birth={p:(0.,True) for p in self.initial_pairs}
        self.sample(0.,foam)

    def contact_step(self,t,foam):
        current=neighbor_pairs(foam)
        lost=self.previous_pairs-current;gained=current-self.previous_pairs
        if lost or gained:self.pair_changes.append(dict(time=t,lost=sorted(lost),gained=sorted(gained)))
        for pair in lost:
            birth,left=self.birth.pop(pair)
            self.episodes.append(dict(pair=pair,birth=birth,end=t,duration=t-birth,left_censored=left,right_censored=False))
        for pair in gained:self.birth[pair]=(t,False)
        self.surviving-=lost;self.previous_pairs=current

    def sample(self,t,foam):
        centers=cell_centroids(foam);touching,free=touching_and_free(foam)
        distance,curvature=boundary_geometry(foam,centers)
        lengths=foam.edge_lengths();tensions=np.zeros(foam.nFa);weight=np.zeros(foam.nFa)
        mean_length=np.zeros(foam.nFa);count=np.zeros(foam.nFa)
        for e,faces in enumerate(foam.e_f):
            for c in faces[faces>=0]:
                tensions[c]+=lengths[e]*foam.e_t[e];weight[c]+=lengths[e]
                mean_length[c]+=lengths[e];count[c]+=1
        nematic=[]
        for c in range(foam.nFa):
            p=foam._cell_polyline(c)-centers[c];q=np.roll(p,-1,axis=0)
            cross=p[:,0]*q[:,1]-p[:,1]*q[:,0];area=.5*cross.sum()
            xx=np.sum(cross*(p[:,0]**2+p[:,0]*q[:,0]+q[:,0]**2))/(12*area)
            yy=np.sum(cross*(p[:,1]**2+p[:,1]*q[:,1]+q[:,1]**2))/(12*area)
            xy=np.sum(cross*(2*p[:,0]*p[:,1]+p[:,0]*q[:,1]+q[:,0]*p[:,1]+2*q[:,0]*q[:,1]))/(24*area)
            nematic.append([(xx-yy)/(xx+yy),2*xy/(xx+yy)])
        denominator=max(1,len(self.initial_pairs))
        self.samples.append(dict(time=t,centers=centers,shape=foam.shape_factors(),neighbors=foam.neighbor_number(),
            area=foam.f_area.copy(),packing=foam.volume_fraction(),wall_load=wall_load(foam.wall)['total'] if foam.wall is not None else None,
            touching=touching,free_surface=free,wall_distance=distance,wall_curvature=curvature,
            mean_tension=tensions/weight,mean_film_length=mean_length/count,nematic=np.array(nematic),
            neighbor_overlap=len(self.previous_pairs & self.initial_pairs)/denominator,
            continuous_survival=len(self.surviving)/denominator))

    def close(self,t):
        for pair,(birth,left) in self.birth.items():
            self.episodes.append(dict(pair=pair,birth=birth,end=t,duration=t-birth,left_censored=left,right_censored=True))
        self.birth={}


def observe_transitions(foam,recorder):
    """Record outer committed transactions, avoiding events rolled back by a parent."""
    base=type(foam)
    class Observed(base):
        def _guard(self,kind,ident,site,operation,accepted_of,rejected_value):
            outer=recorder.enabled and recorder.depth==0 and kind!='T2r'
            if outer:
                pairs=neighbor_pairs(self);location=np.asarray(site).copy();start=len(self.log.accepted)
            recorder.depth+=1
            try:result=super()._guard(kind,ident,site,operation,accepted_of,rejected_value)
            finally:recorder.depth-=1
            if outer and len(self.log.accepted)>start and self.log.accepted[-1]==(self.log.iteration,kind,ident):
                after=neighbor_pairs(self);lost=pairs-after;gained=after-pairs
                recorder.events.append(dict(time=(self.log.iteration-recorder.origin)*self.dt,kind=kind,
                    location=location,lost=sorted(lost),gained=sorted(gained),
                    cells=sorted({c for pair in lost|gained for c in pair})))
                if lost or gained:recorder.contact_step((self.log.iteration-recorder.origin)*self.dt,self)
            return result
    foam.__class__=Observed
    return foam


def activity(task,foam,t):
    if task.pattern=='uniform' or not task.pulse_start<=t<task.pulse_end:return task.mu
    if task.pattern=='pulse':return task.peak
    centers=np.array([np.mean(mid,axis=0) for mid in foam.e_mid])-foam.center
    if task.pattern=='wall':mask=np.linalg.norm(centers,axis=1)>foam.radius-1.
    elif task.pattern=='sector':mask=np.abs(np.arctan2(centers[:,1],centers[:,0]))<np.pi/4
    else:raise ValueError('unknown activity pattern')
    return np.where(mask,task.peak,task.mu)


def motion_analysis(samples,initial_pairs,area):
    x=np.array([s['centers'] for s in samples]);n=x.shape[1];dt=samples[1]['time']-samples[0]['time']
    adj=[[] for _ in range(n)]
    for a,b in initial_pairs:adj[a].append(b);adj[b].append(a)
    lags=np.unique(np.rint(np.geomspace(1,max(1,(len(x)-1)//4),20)).astype(int))
    rows=[]
    for lag in lags:
        d=x[lag:]-x[:-lag];r2=np.sum(d*d,axis=2)/area
        cage=d.copy()
        for i,neighbors in enumerate(adj):
            if neighbors:cage[:,i]-=d[:,neighbors].mean(axis=1)
        q=(r2<.3**2).mean(axis=1);mean=r2.mean()
        rows.append(dict(lag=lag*dt,msd=mean,cage_msd=np.mean(np.sum(cage*cage,axis=2))/area,
            overlap=q.mean(),chi4=n*np.var(q),non_gaussian=np.mean(r2*r2)/(2*mean**2)-1 if mean>0 else None))
    final=np.sqrt(np.sum((x[lags[-1]:]-x[:-lags[-1]])**2,axis=2)/area).ravel()
    hist,edges=np.histogram(final,bins=30,density=True)
    return dict(lags=rows,displacement_histogram=dict(density=hist,edges=edges,lag=lags[-1]*dt),
                definition='2D alpha2; chi4 over finite-window time origins, not independent replicates; cage neighbors fixed at measurement start.')


def clusters(events,dt,distance,positions=None):
    if not events:return []
    t=np.array([e['time'] for e in events]);p=np.array([e['location'] for e in events]) if positions is None else positions
    parent=list(range(len(t)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for j in range(len(t)):
        for i in range(j-1,-1,-1):
            if t[j]-t[i]>dt:break
            if np.linalg.norm(p[j]-p[i])<=distance:parent[root(j)]=root(i)
    _,counts=np.unique([root(i) for i in range(len(t))],return_counts=True)
    return counts.tolist()


def summarize(recorder,foam,task):
    samples=recorder.samples;t=np.array([s['time'] for s in samples])
    changing=[e for e in recorder.events if e['lost'] or e['gained']]
    changing.sort(key=lambda e:e['time'])
    rng=np.random.default_rng(80000+task.seed);cascade=[]
    for window in (.1*task.tau,.5*task.tau):
        for radius in (np.sqrt(task.rho),2*np.sqrt(task.rho)):
            sizes=clusters(changing,window,radius)
            null=[]
            if changing:
                pos=np.array([e['location'] for e in changing])
                for _ in range(16):null.append(max(clusters(changing,window,radius,pos[rng.permutation(len(pos))])))
            cascade.append(dict(time_window=window,distance=radius,cluster_sizes=sizes,
                                largest=max(sizes) if sizes else 0,shuffled_largest=null))
    shape=np.array([s['shape'] for s in samples]);tension=np.array([s['mean_tension'] for s in samples])
    lagcorr=[]
    for lag in (1,2,5,10):
        if lag>=len(t):continue
        a=(tension[:-lag]-tension[:-lag].mean(axis=0)).ravel()
        b=(shape[lag:]-shape[:-lag]).ravel();b-=b.mean()
        denom=np.linalg.norm(a)*np.linalg.norm(b)
        lagcorr.append(dict(lag=lag*task.sample,tension_to_shape_change=float(a@b/denom) if denom>0 else None))
    profiles=[]
    x=np.array([s['centers'] for s in samples]);speed=np.linalg.norm(np.diff(x,axis=0),axis=2)/task.sample
    dist=np.array([s['wall_distance'] for s in samples[:-1]])
    for lo,hi in ((0.,.5),(.5,1.),(1.,2.),(2.,float('inf'))):
        mask=(dist>=lo)&(dist<hi)
        profiles.append(dict(distance_from_wall=[lo,hi],samples=int(mask.sum()),
            mean_speed=float(speed[mask].mean()) if mask.any() else None,
            mean_shape=float(shape[:-1][mask].mean()) if mask.any() else None))
    blocks=[]
    for ids in np.array_split(np.arange(1,len(samples)),4):
        if len(ids):blocks.append(dict(start=t[ids[0]],end=t[ids[-1]],shape=float(shape[ids].mean()),
            packing=float(np.mean([samples[i]['packing'] for i in ids]))))
    return dict(motion=motion_analysis(samples,recorder.initial_pairs,task.rho),cascades=cascade,
        tension_shape_lags=lagcorr,wall_profiles=profiles,stationarity_blocks=blocks,
        event_count=len(recorder.events),contact_changing_transactions=len(changing),
        event_rate=len(changing)/(task.n*task.duration),
        waiting_times=np.diff([e['time'] for e in changing]),
        shape_mean=float(shape[1:].mean()),packing_mean=float(np.mean([s['packing'] for s in samples[1:]])),
        final_neighbor_overlap=samples[-1]['neighbor_overlap'],final_continuous_survival=samples[-1]['continuous_survival'])


def run_experiment(task):
    from activefoam.budget import call_with_budget,BudgetExceeded
    out=dict(task=asdict(task),key=task.key,accepted=False,stage='preparation');start=time.monotonic()
    try:call_with_budget(task.budget,_run,task,out)
    except (ValueError,FloatingPointError,BudgetExceeded) as exc:out['error']=str(exc)
    out['seconds']=time.monotonic()-start
    return serializable(out)


def _run(task,out):
    foam=prepare(task);rec=Recorder(foam);observe_transitions(foam,rec)
    recent=deque(maxlen=20);out['preparation_quality']=protocol.validate(foam)
    def advance(duration,stage):
        out['stage']=stage;steps=round(duration/task.dt);stride=round(task.sample/task.dt)
        if steps%stride:raise ValueError('sampling and duration mismatch')
        for k in range(1,steps+1):
            t=k*task.dt;out['elapsed']=t
            idx=max(0,foam.log.iteration)+1;before=len(foam.log.accepted)
            foam.advance_scheduled(idx,activity(task,foam,t-task.dt) if stage=='measurement' else task.mu)
            recent.append(dict(time=t,kinds=[e[1] for e in foam.log.accepted[before:]]))
            if stage=='measurement':rec.contact_step(t,foam)
            if k%task.check_every==0:protocol.validate(foam)
            if stage=='measurement' and k%stride==0:rec.sample(t,foam)
        protocol.validate(foam)
    try:
        advance(task.warmup,'warmup');rec.begin(foam)
        advance(task.duration,'measurement');rec.close(task.duration)
        out['summary']=summarize(rec,foam,task);out['final_quality']=protocol.validate(foam)
        out['final_geometry']=dict(polygons=[foam._cell_polyline(c) for c in range(foam.nFa)],
            center=foam.center,radius=foam.radius,aspect=task.aspect)
        out['accepted']=True;out['stage']='complete'
    finally:
        out['recent_steps']=list(recent)
        out['events']=rec.events;out['pair_changes']=rec.pair_changes;out['contact_episodes']=rec.episodes
        out['samples']=rec.samples
        if not out['accepted']:
            out['failure_geometry']=dict(vertices=foam.vpos,films=list(foam.e_mid),faces=foam.e_f,
                                         tension=foam.e_t,area=foam.f_area,center=foam.center,radius=foam.radius)
