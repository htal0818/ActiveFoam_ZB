"""Checkpointed full-foam runs for Figs 2b, 2c, 3a, 4a and 4g.

Example: .venv/bin/python scripts/reproduce_requested.py --workers 4
Each completed seed is saved independently. --resume reuses only records with
matching configuration AND source hash. Failed jobs retain explicit diagnostics.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse,fcntl,hashlib,json,sys,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from activefoam.foam_measure import equilibrium,dynamics,stress_relaxation
from activefoam.paper_measure import ShearRunError


def source_hash():
    h=hashlib.sha256()
    for p in sorted((ROOT/'activefoam').glob('*.py')):
        h.update(p.name.encode());h.update(p.read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def _write_record(filename, result, job, elapsed):
    result['metadata'] = np.array(json.dumps(job))
    result['elapsed_seconds'] = elapsed
    temporary = filename.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **result)
    temporary.replace(filename)


def task(job,out,handlers=None):
    stamp=time.time();kind=job['kind'];args=job['parameters']
    filename=Path(out)/f"{job['id']}.npz"
    try:
        if handlers is None:
            handlers = {'equilibrium':equilibrium,'dynamics':dynamics,'stress':stress_relaxation,
                        'phase_stress':stress_relaxation}
        result=handlers[kind](**args)
        _write_record(filename, result, job, time.time()-stamp)
        return job['id'],'ok',time.time()-stamp
    except Exception as exc:
        error=dict(job=job,traceback=traceback.format_exc(),elapsed_seconds=time.time()-stamp)
        if isinstance(exc, ShearRunError):
            _write_record(filename.with_suffix('.partial.npz'), exc.partial, job, time.time()-stamp)
        filename.with_suffix('.error.json').write_text(json.dumps(error,indent=2))
        return job['id'],'FAILED',time.time()-stamp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--n-side',type=int,default=6)
    p.add_argument('--figures', nargs='+', choices=['2b','2c','3a','4a','4g'],
                   default=['2b','2c','3a','4a','4g'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--coarse-grid', action='store_true',
                   help='diagnostic parameter spacing at full simulation durations')
    p.add_argument('--map-grid', type=Path,
                   help='JSON with the explicit w and rho axes for 2b/2c')
    p.add_argument('--phase-stress', action='store_true',
                   help='also measure stress relaxation at the 4g phase-marker points')
    p.add_argument('--map-prerelax-steps', type=int, default=0,
                   help='extra confluent initialization steps for a map sensitivity check')
    p.add_argument('--geometry-policy', choices=('strict', 'reference'), default='strict',
                   help='reference records boundary crossings without rejecting the run')
    p.add_argument('--msd-center', choices=('matlab', 'area'), default='matlab',
                   help='boundary-vertex mean from ts_faceArea or polygon area centroid')
    p.add_argument('--out',type=Path,default=ROOT/'data'/'full_foam')
    p.add_argument('--pilot',action='store_true',help='short pipeline validation; never publication data')
    a=p.parse_args()
    if a.map_grid and (a.coarse_grid or a.pilot):
        p.error('--map-grid cannot be combined with --coarse-grid or --pilot')
    if a.phase_stress and ('4g' not in a.figures or a.pilot):
        p.error('--phase-stress requires figure 4g and full simulation durations')
    if min(a.seeds, a.workers, a.n_side) < 1 or a.map_prerelax_steps < 0:
        p.error('--seeds, --workers, and --n-side must be positive and map prerelax nonnegative')
    a.out.mkdir(parents=True,exist_ok=True)
    # Keep the lock open until the manifest and worker outputs are finalized.
    output_lock = (a.out / '.run.lock').open('a')
    try:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error(f'another runner is writing to {a.out}')
    version=source_hash();jobs=[];job_ids=set()
    def add(kind,params):
        params['n_side']=a.n_side
        params['geometry_policy']=a.geometry_policy
        spec=dict(kind=kind,parameters=params,source_hash=version,pilot=a.pilot)
        spec['id']=kind+'_'+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
        if spec['id'] not in job_ids:
            job_ids.add(spec['id'])
            jobs.append(spec)
    seeds=range(1,a.seeds+1)
    map_grid_source = None
    if set(a.figures)&{'2b','2c'}:
        # Publisher Source Data Fig. 2, worksheet 1, rows 6–661.
        ws = np.round(np.arange(41) * .02, 4)
        rhos = np.round((.90 + np.arange(16) * .02) ** 2, 4)
        map_grid_source = 'Kim et al. Source Data Fig. 2, 16 × 41 grid'
        if a.coarse_grid:
            ws = ws[::5]
            rhos = rhos[[0, 2, 5, 7, 9, 11, 13, 15]]
            map_grid_source += ' (subset)'
        if a.pilot:
            ws, rhos = ws[[0, 16, 40]], rhos[[0, 5, 15]]
            map_grid_source += ' (pilot subset)'
        if a.map_grid:
            try:
                grid = json.loads(a.map_grid.read_text())
                ws, rhos = (np.asarray(grid[key], dtype=float) for key in ('w', 'rho'))
                for name, values in [('w', ws), ('rho', rhos)]:
                    if (values.ndim != 1 or len(values) < 2
                            or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0)
                            or np.any(values < 0) or (name == 'rho' and values[0] == 0)):
                        raise ValueError(f'{name} must be an increasing finite axis')
            except (OSError, KeyError, TypeError, ValueError) as exc:
                p.error(f'invalid --map-grid: {exc}')
            map_grid_source = grid.get('source', str(a.map_grid))
        for seed in seeds:
            for rho in rhos:
                for w in ws:
                    add('equilibrium', dict(w=float(w), rho=float(rho), seed=seed,
                                            prerelax_steps=a.map_prerelax_steps))
    activities=[.25,.5,.75,1.,1.25,1.5]
    if set(a.figures)&{'3a','4g'}:
        ws=np.round(np.arange(0,1.201,.1),3) if '4g' in a.figures else [1.]
        if a.coarse_grid and '4g' in a.figures:ws=np.round(np.arange(0,1.201,.2),3)
        if a.pilot:ws=[0.,.6,1.,1.2] if '4g' in a.figures else [1.]
        for seed in seeds:
            for mu in activities:
                for w in ws:
                    add('dynamics',dict(w=float(w),mu=mu,seed=seed,
                        burn_tauT=0.,
                        record_msd=bool(w==1. and '3a' in a.figures),
                        msd_center=a.msd_center,
                        duration_tauT=2. if a.pilot else (25. if a.coarse_grid and w!=1. else 120.)))
                    if a.phase_stress and mu >= .5:
                        # Fig. 4d/e phase grid, using tau_d = 100 tau_T.
                        add('phase_stress', dict(w=float(w),mu=mu,seed=seed,
                            pre_tauT=4.,relax_tauR=1000.,active_tauR=1000.))
    if '4a' in a.figures:
        for seed in seeds:
            for mu in [0.,.5,1.,1.5]:
                add('stress',dict(mu=mu,seed=seed,pre_tauT=1. if a.pilot else 4.,
                    relax_tauR=10. if a.pilot else 580.,active_tauR=10. if a.pilot else 1000.))
    # Make the requested long MSD and stress curves available before the
    # independent shape/phase sweep consumes the remaining worker time.
    def priority(job):
        if job['kind'] == 'dynamics' and job['parameters']['record_msd']:
            return 0
        return {'stress': 1, 'equilibrium': 2, 'dynamics': 3, 'phase_stress': 4}[job['kind']]
    jobs.sort(key=priority)
    manifest=dict(source_hash=version,model='FoamTissue',pilot=a.pilot,n_side=a.n_side,
                  seeds=a.seeds,dt=.005,edpc=.15,figures=a.figures,
                  geometry_policy=a.geometry_policy,msd_center=a.msd_center,
                  map_grid_source=map_grid_source,jobs=jobs)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    pending=[j for j in jobs if not(a.resume and (a.out/f"{j['id']}.npz").exists())]
    print(f'{len(jobs)} jobs; {len(pending)} pending; {a.workers} workers; source={version}',flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(task,j,str(a.out)) for j in pending]
        for i,f in enumerate(as_completed(futures),1):
            name,status,elapsed=f.result()
            print(f'[{i}/{len(pending)}] {name} {status} {elapsed:.1f}s',flush=True)
    failures=[j['id'] for j in jobs if not (a.out/f"{j['id']}.npz").exists()]
    (a.out/'status.json').write_text(json.dumps(dict(total=len(jobs),completed=len(jobs)-len(failures),failed=failures),indent=2))
    if failures:sys.exit(1)
if __name__=='__main__':main()
