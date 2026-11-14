"""Measurement protocols using only the curved-edge FoamTissue model.

Time is in tau_R; mu is the MATLAB noise coefficient (not stationary OU SD).
No confluent/deformable surrogate or fit-derived measurements are used.
"""
from copy import deepcopy
from functools import lru_cache

import numpy as np
from .foam_full import FoamTissue, edge_len
from .topology import build_periodic_voronoi


def validate(ft):
    if not np.all(np.isfinite(ft.f_area)) or np.any(ft.f_area <= 1e-8):
        raise ValueError('nonfinite or nonpositive cell area')
    phi = ft.f_area.sum()/ft.bs**2
    if phi > 1.01:
        raise ValueError(f'overlapping geometry: volume fraction {phi:.6g}')
    # The MATLAB reference protocol records crossings without stopping.
    if getattr(ft, 'accelerated', False):
        from ._kernels import List, cell_crossing_report, finite_films
        mids, loops = ft._fast_geometry_inputs()
        if not finite_films(mids):
            raise ValueError('nonfinite boundary')
        key = tuple(tuple(vertices) for vertices in ft.f_v)
        if key != getattr(ft, '_validation_vertex_key', None):
            ft._validation_vertex_key = key
            ft._validation_vertices = List([np.asarray(vertices, dtype=np.int64)
                                            for vertices in ft.f_v])
        folds, max_depth = cell_crossing_report(
            mids, loops, ft.vpos, ft._validation_vertices, ft.bs, ft.sstn)
    else:
        from ._kernels import crossing_depth
        if any(not np.all(np.isfinite(m)) for m in ft.e_mid):
            raise ValueError('nonfinite boundary')
        folds, max_depth = 0, 0.
        for c in range(ft.nFa):
            if ft._cell_self_intersects(c):
                count, depth = crossing_depth(ft._cell_polyline(c), ft.vpos[ft.f_v[c]],
                                              ft.bs, ft.sstn)
                folds += count
                max_depth = max(max_depth, depth)
    ft._quality_checks = getattr(ft,'_quality_checks',0)+1
    resolved = max_depth > .01*2*np.sqrt(np.pi)
    ft._microfold_checks = getattr(ft,'_microfold_checks',0)+(folds>0 and not resolved)
    ft._resolved_crossing_checks = getattr(ft,'_resolved_crossing_checks',0)+resolved
    ft._max_crossing_depth = max(getattr(ft,'_max_crossing_depth',0.),max_depth)
    if resolved and getattr(ft, 'geometry_policy', 'strict') == 'strict':
        raise ValueError(f'resolved boundary crossing: junction distance {max_depth:.6g}')
    return phi


def quality(ft):
    return dict(quality_checks=getattr(ft,'_quality_checks',0),
                microfold_checks=getattr(ft,'_microfold_checks',0),
                resolved_crossing_checks=getattr(ft,'_resolved_crossing_checks',0),
                max_crossing_depth=getattr(ft,'_max_crossing_depth',0.))


def evolve(ft, duration, mu, start=0, sample_every=0, observer=None, post_shear=False,
           contact_observer=None):
    count = int(round(duration/ft.dt))
    k = 0
    try:
        for k in range(1, count+1):
            ft.advance_scheduled(start+k, mu, post_shear=post_shear,
                                 contact_observer=contact_observer)
            if k % 100 == 0:
                validate(ft)
            if observer is not None and k % sample_every == 0:
                observer(k*ft.dt, ft)
        validate(ft)
    except ValueError as exc:
        raise ValueError(f'{exc} at elapsed time {k*ft.dt:g} tau_R, strain {ft.sstn:g}') from exc
    return start+count


def initial_voronoi(n_side, seed, accelerated, **kwargs):
    tissue = build_periodic_voronoi(n_side, np.random.default_rng(100+seed))
    return FoamTissue(tissue, seed=seed, accelerated=accelerated,
                      reference_pressure_lag=True, **kwargs)


@lru_cache(maxsize=8)
def relaxed_tiling(n_side, seed, accelerated, pressure_normals="reference"):
    """Confluent preparation in ts_initialConf, shared across parameter points."""
    ft = initial_voronoi(n_side, seed, accelerated, edpc=.2, pressure_normals=pressure_normals)
    previous_lengths = ft.edge_lengths()
    short_scale = 10
    for iteration in range(1, 1501):
        ft.step(0.)
        if iteration % 5:
            continue
        lengths = ft.edge_lengths()
        short = np.flatnonzero(lengths < .001 * 2 * np.sqrt(np.pi) * short_scale)
        if not short.size:
            short_scale = min(short_scale + 1, 90)
        for e in short:
            a, b = ft.e_f[e]
            if (min(len(ft.f_v[a]), len(ft.f_v[b])) > 3
                    and lengths[e] < previous_lengths[e]):
                ft.t1_foam(int(e))
        previous_lengths = lengths
    return ft._snapshot(typed_films=False)


def initialize(w, rho, seed, n_side=6, accelerated=True, prerelax_steps=0,
               anneal_mu=.5, anneal_tauT=1., geometry_policy='strict', pressure_normals='reference'):
    if geometry_policy not in {'strict', 'reference'}:
        raise ValueError(f'unknown geometry policy: {geometry_policy}')
    ft = initial_voronoi(n_side, seed, accelerated, w=w, rho=rho, pressure_normals=pressure_normals)
    ft.geometry_policy = geometry_policy
    ft._restore(deepcopy(relaxed_tiling(n_side, seed, accelerated, pressure_normals)))
    ft.e_t = np.array([ft.fixed_tension(e) for e in range(len(ft.e_v))])
    for _ in range(prerelax_steps):
        ft.step(0.)
    for v in range(len(ft.vpos)):ft.t2_reverse(v)
    evolve(ft, anneal_tauT*ft.tauT, anneal_mu)
    return ft


def equilibrated_foam(w,rho,seed,n_side=6,prerelax_steps=0,geometry_policy='strict',
                      pressure_normals='reference'):
    # ts_simVolumeFraction seeds spaces immediately after constructing the tiling.
    ft=initialize(w,rho,seed,n_side,prerelax_steps=prerelax_steps,geometry_policy=geometry_policy,
                  pressure_normals=pressure_normals)
    ft.e_t=np.array([ft.fixed_tension(e) for e in range(len(ft.e_v))])
    evolve(ft, 3*ft.tauT, 0., start=round(ft.tauT/ft.dt))
    return ft


def equilibrium(w,rho,seed,n_side=6,prerelax_steps=0,geometry_policy='strict'):
    ft = equilibrated_foam(w, rho, seed, n_side, prerelax_steps, geometry_policy)
    return dict(phi=validate(ft),z=float(ft.neighbor_number().mean()),
                area=ft.f_area.copy(),shape=ft.shape_factors(),edge_faces=ft.e_f.copy(),**quality(ft))


def dynamics(w,mu,seed,n_side=6,burn_tauT=0.,duration_tauT=120.,sample_every=10,
             record_msd=True,geometry_policy='strict',msd_center='matlab',rho=1.,
             measurements=None,pressure_normals='reference'):
    from ._kernels import polygon_centroids

    if msd_center not in {'matlab', 'area'}:
        raise ValueError(f'unknown MSD center definition: {msd_center}')

    def centers(foam):
        if msd_center == 'matlab':
            return foam.f_center.copy()
        return polygon_centroids(*foam._fast_geometry_inputs(), foam.bs, foam.sstn)

    ft=initialize(w, rho, seed, n_side, anneal_mu=mu, geometry_policy=geometry_policy,
                  pressure_normals=pressure_normals)
    start = evolve(ft, burn_tauT*ft.tauT, mu) if burn_tauT else 0
    # ts_faceArea reports the mean of the polygon's sampled boundary vertices.
    # Keep the area centroid available for checking estimator sensitivity.
    prev = unwrapped = None
    trajectory, shapes, phis = [], [], []
    if record_msd:
        prev = centers(ft)
        unwrapped = prev.copy()
        trajectory.append(unwrapped.copy())
    if measurements is not None:
        measurements.begin(ft, start)
    def observe(t,f):
        nonlocal prev,unwrapped
        shapes.append(float(np.mean(f.shape_factors())))
        phis.append(float(f.f_area.sum()/f.bs**2))
        if measurements is not None:
            measurements.sample(t, f)
        if record_msd:
            cur=centers(f)
            d=cur-prev;d-=f.bs*np.rint(d/f.bs)
            unwrapped+=d;prev=cur
            trajectory.append(unwrapped.copy())
    evolve(ft, duration_tauT*ft.tauT, mu, start=start, sample_every=sample_every,
           observer=observe, contact_observer=None if measurements is None else measurements.contacts)
    result = dict(shape_mean=np.mean(shapes), shape_time_sd=np.std(shapes),
                  phi_mean=np.mean(phis), sample_dt=sample_every*ft.dt, **quality(ft))
    if measurements is not None:
        result.update(measurements.finish(ft))
    if not record_msd:
        return result
    traj=np.asarray(trajectory)
    # Paper Eq. 3: cell trajectories, averaged over cells and time origins.
    # Keep lab-frame and COM-subtracted values separately instead of hiding drift.
    lags=np.unique(np.round(np.geomspace(1,min(len(traj)-1,int(100*ft.tauT/(sample_every*ft.dt))),60)).astype(int))
    msd=[];msd_com=[]
    centered=traj-traj.mean(axis=1,keepdims=True)
    for lag in lags:
        msd.append(np.mean(np.sum((traj[lag:]-traj[:-lag])**2,axis=2))/ft.A0)
        msd_com.append(np.mean(np.sum((centered[lag:]-centered[:-lag])**2,axis=2))/ft.A0)
    return dict(result, tlag=lags*sample_every*ft.dt/ft.tauT, msd=np.array(msd),
                msd_com=np.array(msd_com), trajectory=traj, msd_center=np.array(msd_center))


def stress_relaxation(mu,seed,n_side=6,w=.5,pre_tauT=4.,relax_tauR=580.,active_tauR=1000.,strain=1.5,
                      geometry_policy='strict'):
    ft=initialize(w, 1., seed, n_side, anneal_mu=mu, anneal_tauT=pre_tauT,
                  geometry_policy=geometry_policy)
    # In the reference, the step follows an additional 20 tau_R at zero strain.
    pre_steps = evolve(ft, min(20., active_tauR), mu)
    sigma0=np.sqrt(ft.rho)
    sigma_a = 0.
    if mu > 0.:
        # The unperturbed baseline uses a separate branch of the trajectory.
        pre_shear_geometry = ft._snapshot()
        active_stress = []
        evolve(ft, active_tauR, mu, start=pre_steps, sample_every=10,
               observer=lambda t,f:active_stress.append(f.shear_stress()))
        sigma_a = np.std(active_stress)/sigma0
        ft._restore(pre_shear_geometry)
        ft._prev_lens = None
    ft.apply_affine_shear(strain)
    times=[0.];stress=[ft.shear_stress()/sigma0]
    def observe(t,f):times.append(t);stress.append(f.shear_stress()/sigma0)
    evolve(ft, relax_tauR, mu, start=pre_steps, sample_every=10, observer=observe,
           post_shear=True)
    return dict(time=np.array(times),stress=np.array(stress),sigma_A=sigma_a,**quality(ft))
