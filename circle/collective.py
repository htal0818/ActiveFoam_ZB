"""Paired experiments without altering forces, transitions or wall constraints."""
import numpy as np
import json
from . import force_probe as fp, protocol
from .disc import DiscFoam
from .measure import cell_centroids, touching_wall
from .scripts.division_contact_analysis import persistent_changes
from .scripts.division_dynamics import nonaffine_msd
from .sweep import serializable


def save_result(path, result):
    """Publish complete JSON only after serialization and writing finish."""
    temporary=path.with_suffix('.writing')
    temporary.write_text(json.dumps(serializable(result))+'\n')
    temporary.replace(path)


def contact_pairs(f):
    faces=f.e_f[np.all(f.e_f>=0,axis=1)]
    if hasattr(f,'cell_uid'):faces=f.cell_uid[faces]
    return set(map(tuple,np.sort(faces,axis=1).tolist()))


def variance_matched_mu(tau,dt,reference_mu=3.,reference_tau=10.):
    """Same stationary variance under the actual Euler OU update, not RMS=mu."""
    if not 0<dt<2*min(tau,reference_tau):raise ValueError('unstable OU time step')
    return reference_mu*np.sqrt(tau*(2-dt/tau)/(reference_tau*(2-dt/reference_tau)))


def trajectory(f,duration,mu=0.,start=0,sample=.1,dwell=.2,site=None,observer=None):
    """Observe every step; filter persistent step-end contact changes.

    Net changes within a timestep are observable; a loss and recovery inside
    one step is deliberately unresolved. No change to transition scheduling.
    Periodic centroid trajectories are unwrapped before displacement statistics.
    """
    initial=contact_pairs(f);previous=set(initial);events=[];records=[];checks=[]
    ref=cell_centroids(f);last=ref.copy();unwrapped=ref.copy();mean_area=float(f.f_area.mean())
    ids=np.array(getattr(f,'cell_uid',np.arange(f.nFa))).copy()
    if site is None:site=ref.mean(axis=0)
    delta=ref-site
    if not isinstance(f,DiscFoam):delta-=np.rint(delta/f.bs)*f.bs
    distance=np.linalg.norm(delta,axis=1)
    def capture(t):
        nonlocal last,unwrapped
        now=cell_centroids(f);step=now-last
        if not isinstance(f,DiscFoam):step-=np.rint(step/f.bs)*f.bs
        unwrapped+=step;last=now.copy()
        displacement=unwrapped-ref;displacement-=displacement.mean(axis=0)
        contact_tensions=f.e_t[np.all(f.e_f>=0,axis=1)]
        records.append(dict(time=t,shape=float(f.shape_factors().mean()),
            neighbors=2*len(previous)/f.nFa,packing=float(f.volume_fraction()),
            msd=float(np.mean(np.sum(displacement**2,axis=1)))/mean_area,
            residual_msd=(nonaffine_msd(ref,unwrapped)/mean_area if isinstance(f,DiscFoam) else None),
            cell_displacement=displacement.copy(),centers=now,mean_area=float(f.f_area.mean()),
            tension_variance=float(np.mean((contact_tensions-(2-f.gam))**2)) if len(contact_tensions) else None))
        if observer is not None:records[-1].update(observer(f))
    capture(0.)
    every=max(1,round(sample/f.dt));check=max(1,round(.5/f.dt))
    for k in range(1,round(duration/f.dt)+1):
        f.advance_scheduled(start+k,mu)
        current=contact_pairs(f)
        if current!=previous:
            events.append(dict(time=k*f.dt,lost=sorted(previous-current),gained=sorted(current-previous)))
            previous=current
        if k%check==0:checks.append(dict(time=k*f.dt,**fp.validate(f)))
        if k%every==0:capture(k*f.dt)
    fp.validate(f)
    changes=persistent_changes(initial,events,0.,duration,dwell)
    lost=set();stable=set(initial);cursor=0
    for r in records:
        while cursor<len(changes) and changes[cursor]['time']<=r['time']+1e-9:
            c=changes[cursor];p=tuple(c['pair'])
            if c['present']:stable.add(p)
            else:stable.remove(p);lost.add(p)
            cursor+=1
        r.update(persistent_changes=cursor,contact_survival=len(initial-lost)/len(initial) if initial else None,
                 contact_retention=len(initial&stable)/len(initial) if initial else None)
    return dict(n=f.nFa,duration=duration,mu=mu,dt=f.dt,dwell=dwell,records=records,
                initial_pairs=sorted(initial),final_pairs=sorted(previous),events=events,changes=changes,
                checks=checks,cell_ids=ids,initial_centers=ref,site=np.asarray(site),distance=distance,
                wall_touching=int(touching_wall(f).sum()) if isinstance(f,DiscFoam) else None,
                counting='Persistent changes in step-end contact graph; division excluded; pair loss and gain counted separately.')


def prepare(domain,n,seed,w=1.,rho=1.3,quench=3.):
    fn=protocol.equilibrated if domain=='disc' else protocol.periodic_equilibrated
    arg=n if domain=='disc' else int(np.sqrt(n))
    if domain!='disc' and arg*arg!=n:raise ValueError('periodic preparation currently requires square N')
    f=fn(arg,w,rho,seed=seed,quench_tauT=quench,edpc=.1)
    fp.validate(f)
    if isinstance(f,DiscFoam) and not touching_wall(f).any():
        raise ValueError('comparison requires tissue pressing on wall')
    return f
