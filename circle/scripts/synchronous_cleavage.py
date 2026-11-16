"""Fixed-time synchronous cleavage, independent random orientations, no growth."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from circle import cleavage as c,protocol
from circle.collective import save_result
from circle.force_probe import isolated_copy
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'circle/data/synchronous_cleavage_20261115'


def run(task):
    seed,interval,mu,dt,spacing,founders,rounds,family=task
    path=DATA/family/f'n{founders}_seed{seed}_gap{interval:g}_mu{mu:g}_dt{dt:g}_h{spacing:g}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return
    p=dict(seed=seed,interval=interval,mu=mu,dt=dt,spacing=spacing,founders=founders,
        rounds=rounds,initial_nominal_density=1.3,first_division=2.,final_observation=8.,
        W=1.,P0=10.,tauT=10.,growth=0.,feedback=0.,orientation='uniform proposals, geometric rejection',
        schedule='strictly synchronous; no timing noise')
    result=dict(parameters=p,status='running',waves=[],stages=[])
    try:
        f=c.prepare(founders,seed,dt,spacing)
        orientations=np.random.default_rng(10000+seed)
        initial=c.measure(f,include_geometry=True)
        protocol.evolve(f,2.,mu,start=0)
        result['initial']=initial
        for wave in range(rounds):
            # A matched branch skips this and every later division.
            skip=isolated_copy(f) if wave==rounds-1 else None
            record=c.random_wave(f,orientations);result['waves'].append(record)
            if wave==0:
                np.testing.assert_allclose(record['before']['preferred_total'],founders,rtol=1e-13)
            assert f.nFa==founders*2**(wave+1)
            duration=interval if wave<rounds-1 else 8.
            observed=c.stage(f,duration,mu,geometry=(seed==1 and interval==4 and mu==0))
            result['stages'].append(observed)
            save_result(path.with_suffix('.partial.json'),result)
            print(path.name,'wave',wave+1,'N',f.nFa,'changes',len(observed['changes']),
                  'area',round(f.f_area.sum(),4),flush=True)
        result['skip_final']=c.stage(skip,8.,mu)
        result['final']=c.measure(f,include_geometry=True)
        result['division_log']=f.division_log
        # Relaxation calibration is performed only for the passive reference.
        if founders==1 and seed==1 and interval==4 and mu==0 and family=='main':
            result['shear_probes']=[]
            for strain in [.005,.01]:
                try:result['shear_probes'].append(c.shear_relaxation(f,strain,duration=8.))
                except ValueError as exc:result['shear_probes'].append(dict(strain=strain,status='preempted',reason=str(exc)))
        result['status']='complete'
    except Exception as exc:result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    save_result(path,result);path.with_suffix('.partial.json').unlink(missing_ok=True)
    print(path.name,result['status'],result.get('error',''),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',choices=['pilot','main','gates','refine'],default='pilot')
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.batch=='pilot':tasks=[(1,4.,0.,.001,.0125,1,5,'main')]
    elif args.batch=='main':tasks=[(s,g,m,.001,.0125,1,5,'main') for s in [1,2] for g in [2.,4.,8.] for m in [0.,.25]]
    elif args.batch=='gates':tasks=[(1,4.,0.,.001,.025,n,1,'gates') for n in [8,16]]
    else:tasks=[(1,4.,0.,dt,h,1,5,'refinement') for dt,h in [(.0005,.0125),(.001,.00625),(.0005,.00625)]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run,tasks))
if __name__=='__main__':main()
