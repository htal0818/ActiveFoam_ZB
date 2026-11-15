"""Measurements and prescribed synchronous cleavage; no cell-cycle scheduler."""
import numpy as np
from activefoam.foam_full import boundary_normal_signs
from .biology import BiologyFoam
from .disc import DiscFoam,circle_polyline
from . import protocol,state
from .measure import wall_load,touching_wall
from .force_probe import isolated_copy
from .collective import trajectory

T1_LENGTH=.01*2*np.sqrt(np.pi)


class MeasuredFoam(BiologyFoam):
    """Observe junction motion before topology updates can rename vertices."""
    STATE_DIAGNOSTICS=BiologyFoam.STATE_DIAGNOSTICS+('junction_motion',)

    def step(self,mu=0.):
        before=self.vpos.copy()
        super().step(mu)
        displacement=self.vpos-before
        design=np.column_stack((before-before.mean(axis=0),np.ones(len(before))))
        residual=None
        if len(before)>=3 and np.linalg.matrix_rank(design)==3:
            fit=np.linalg.lstsq(design,displacement,rcond=None)[0]
            residual=float(np.sqrt(np.mean(np.sum((displacement-design@fit)**2,axis=1)))/self.dt)
        self.junction_motion=dict(nonaffine_speed=residual,
            rms_speed=float(np.sqrt(np.mean(np.sum(displacement**2,axis=1)))/self.dt))


def junction_forces(f):
    """Current unconstrained junction forces, using precisely the segment-end rule."""
    out=np.zeros_like(f.vpos);signs=boundary_normal_signs(f.e_f,f._fast_geometry_inputs()[1])
    pressure=f.P0*(f.preferred/f.f_area-1)/f.lsc
    for e,line in enumerate(f.e_mid):
        a,b=f.e_f[e];pn=(pressure[b] if b>=0 else 0)-(pressure[a] if a>=0 else 0)
        for endpoint,index,direction in [(0,0,1),(1,len(line)-2,-1)]:
            segment=line[index+1]-line[index];length=np.linalg.norm(segment)
            if length<1e-14:continue
            tangent=segment/length;normal=signs[e]*np.array([-tangent[1],tangent[0]])
            out[f.e_v[e,endpoint]]+=direction*max(0.,f.e_t[e])*tangent+pn*length/2*normal
    allowed=out.copy();radial=f.vpos-f.center;r=np.linalg.norm(radial,axis=1)
    for v in np.flatnonzero(r>=f.radius-f.wall_tol):
        n=radial[v]/r[v];allowed[v]-=max(0.,out[v]@n)*n
    return out,allowed


def anisotropy(p):
    """Area-moment anisotropy (lambda_max-lambda_min)/(lambda_max+lambda_min)."""
    q=np.roll(p,-1,axis=0);cross=p[:,0]*q[:,1]-p[:,1]*q[:,0];a=.5*cross.sum()
    center=np.sum((p+q)*cross[:,None],axis=0)/(6*a)
    p=p-center;q=np.roll(p,-1,axis=0);cross=p[:,0]*q[:,1]-p[:,1]*q[:,0]
    xx=np.sum(cross*(p[:,0]**2+p[:,0]*q[:,0]+q[:,0]**2))/(12*a)
    yy=np.sum(cross*(p[:,1]**2+p[:,1]*q[:,1]+q[:,1]**2))/(12*a)
    xy=np.sum(cross*(2*p[:,0]*p[:,1]+p[:,0]*q[:,1]+q[:,0]*p[:,1]+2*q[:,0]*q[:,1]))/(24*a)
    values=np.linalg.eigvalsh([[xx,xy],[xy,yy]])
    return float((values[1]-values[0])/max(values.sum(),1e-30))


def measure(f,include_geometry=False,after_division=False):
    pressure=f.P0*(f.preferred/f.f_area-1)/f.lsc
    forces,allowed=junction_forces(f);magnitudes=np.linalg.norm(forces,axis=1)
    residual=np.linalg.norm(allowed,axis=1);polygons=[f._cell_polyline(c) for c in range(f.nFa)]
    q=np.array([anisotropy(p) for p in polygons]);shape=f.shape_factors();length=np.sqrt(f.f_area.mean())
    stress=f.stress_tensor();deviator=stress-np.trace(stress)/2*np.eye(2)
    neighbors=f.neighbor_number();touch=touching_wall(f)
    row=dict(absolute_time=f.bio_time,n=f.nFa,total_area=float(f.f_area.sum()),
        preferred_total=float(f.preferred.sum()),packing=float(f.volume_fraction()),
        pressure_mean=float(pressure.mean()),pressure_variance=float(pressure.var()),pressure=pressure.copy(),
        shape=float(shape.mean()),shape_values=shape,anisotropy_mean=float(q.mean()),anisotropy_values=q,
        cell_areas=f.f_area.copy(),neighbors_values=neighbors,wall_fraction=float(touch.mean()),
        wall_reaction=None if after_division or f.wall is None else wall_load(f.wall)['total'],
        wall_work=0.,junction_force_rms=float(np.sqrt(np.mean(magnitudes**2))),
        allowed_force_rms=float(np.sqrt(np.mean(residual**2))),junction_force_values=magnitudes,
        allowed_force_values=residual,stress=stress,shear_stress=float(stress[0,1]),
        deviatoric_stress=float(np.sqrt(np.sum(deviator**2))),
        mean_cell_length=float(length),spacing_over_cell_length=float(f.edpc/length),
        t1_threshold_over_cell_length=float(T1_LENGTH/length),
        raw_T1_count=sum(kind=='T1' for _,kind,_ in f.log.accepted),
        vertex_nonaffine_speed=None if after_division else getattr(f,'junction_motion',{}).get('nonaffine_speed'))
    if include_geometry:row.update(polygons=polygons,center=f.center.copy(),radius=f.radius)
    return row


def prepare(founders=1,seed=1,dt=.001,spacing=.0125,density=1.3):
    radius=np.sqrt(founders/(np.pi*density));center=np.full(2,4*radius)
    if founders==1:
        ring=circle_polyline(center,radius,spacing)
        base=DiscFoam.from_closed_cells(radius,[ring],center=center,rho=1.,w=1.,dt=dt,edpc=spacing,seed=seed)
        protocol.evolve(base,2.,0.,start=0)
    else:
        base=protocol.equilibrated(founders,w=1.,rho=1.,seed=seed,radius=radius,dt=dt,edpc=spacing,
            anneal_mu=0.,anneal_tauT=.5,quench_tauT=1.,relax_steps=round(1.5/dt))
    f=MeasuredFoam.from_disc(base,gain=0.,growth_rate=0.)
    protocol.validate(f)
    return f


def random_wave(f,rng,max_wave_attempts=8):
    """All attempts occur at the same clock time; never postpone one mother.

    Invalid proposals are redrawn during planning. If final joint validation
    declines a wave, its entire physical state is restored before redrawing a
    complete plan. Orientation draws are consumed; simulation RNG is unchanged.
    """
    before=measure(f,include_geometry=True)
    for attempt in range(1,max_wave_attempts+1):
        if f.divide_synchronously(rng):break
    else:raise ValueError(f'whole synchronous wave declined {max_wave_attempts} times: {f.division_log[-1]}')
    f.junction_motion={};f.wall=None
    after=measure(f,include_geometry=True,after_division=True)
    entry=f.division_log[-1]
    return dict(before=before,after=after,attempts=attempt,division=entry,
        area_error=after['total_area']-before['total_area'],
        preferred_error=after['preferred_total']-before['preferred_total'],
        max_pressure_error=max(float(np.max(abs(e['pressure_daughters']-e['pressure_before']))) for e in entry['events']))


def stage(f,duration,mu=0.,geometry=False):
    return trajectory(f,duration,mu,start=round(f.bio_time/f.dt),sample=.1,
                      observer=lambda f:measure(f,include_geometry=geometry))


def shear_relaxation(f,strain=.005,duration=8.):
    """Internal small-strain pulse and sham with a fixed wall, on isolated copies.

    An external agent instantaneously deforms the tissue with det(F)=1. No wall
    moves. Reject if any point would cross the wall; this is not wall-imposed
    shear. Subsequent dynamics and the transition schedule are unchanged.
    """
    runs=[]
    for amplitude in (0.,strain):
        work=isolated_copy(f);old_area=work.f_area.copy()
        work.vpos[:,0]+=amplitude*(work.vpos[:,1]-work.center[1])
        for e,line in enumerate(work.e_mid):
            points=np.array(line);points[:,0]+=amplitude*(points[:,1]-work.center[1]);work.e_mid[e]=points
        for key in state.KEYED_CACHES+state.STATE_CACHES:work.__dict__.pop(key,None)
        work._update_faces();protocol.validate(work)
        np.testing.assert_allclose(work.f_area,old_area,rtol=1e-9)
        runs.append(stage(work,duration,0.))
    time=np.array([r['time'] for r in runs[0]['records']])
    response=np.array([a['shear_stress']-b['shear_stress'] for a,b in zip(runs[1]['records'],runs[0]['records'])])
    fit=dict(status='unresolved')
    if abs(response[0])>1e-8:
        from scipy.optimize import curve_fit
        try:
            fn=lambda t,a,tau,plateau:a*np.exp(-t/tau)+plateau
            params,_=curve_fit(fn,time,response,p0=(response[0],1.,0.),bounds=([-np.inf,.01,-np.inf],[np.inf,10*duration,np.inf]))
            residual=np.sum((response-fn(time,*params))**2);total=np.sum((response-response.mean())**2)
            r2=1-residual/max(total,1e-30);tau=params[1]
            fit=dict(status='resolved' if r2>=.95 and 3*tau<=duration else 'unresolved',
                     tau=float(tau),r2=float(r2),amplitude=float(params[0]),plateau=float(params[2]))
        except (RuntimeError,ValueError):pass
    return dict(strain=strain,duration=duration,time=time,response=response,fit=fit,
        sham=runs[0],perturbed=runs[1],boundary='fixed; internal displacement pulse, no wall work')
