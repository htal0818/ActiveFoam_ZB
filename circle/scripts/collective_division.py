"""Matched five-round 8→256 experiments: synchronous division, no growth."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from circle import protocol,force_probe as fp,collective as c
from circle.biology import BiologyFoam
from circle.biology_experiments import geometry
from circle.sweep import serializable
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'circle/data/collective_dynamics_20261114/division'

def job(interval):
    path=OUT/f'interval{interval:g}.json'
    if path.exists():return
    p=dict(interval=interval,first_division=2.,rounds=5,founders=8,seed=1,
           W=1.,mu=0.,growth=0.,feedback=0.,dt=.001,spacing=.025,
           wall_radius=float(.7*np.sqrt(8)),post_final_duration=4.)
    result=dict(parameters=p,status='running',stages=[],frames=[])
    def checkpoint():
        tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(serializable(result))+'\n')
    try:
        base=protocol.equilibrated(8,w=1.,rho=1.,seed=1,radius=p['wall_radius'],dt=.001,
                                  edpc=.025,anneal_mu=0.,anneal_tauT=.5,
                                  quench_tauT=1.,relax_steps=1500)
        f=BiologyFoam.from_disc(base,gain=0.,growth_rate=0.)
        protocol.evolve(f,2.,0.,start=0)
        result['frames'].append(geometry(f,'Before first division'))
        for r in range(5):
            before=f.nFa;area=f.f_area.sum();initial_time=f.bio_time
            if r==4:skip=fp.isolated_copy(f)
            if not f.divide_synchronously():raise ValueError(f'division rejected {f.division_log[-1]}')
            assert f.nFa==2*before
            np.testing.assert_allclose(f.preferred.sum(),8.,rtol=1e-12)
            np.testing.assert_allclose(f.f_area.sum(),area,rtol=1e-10)
            protocol.validate(f)
            result['frames'].append(geometry(f,f'{before} → {f.nFa}'))
            duration=interval if r<4 else 4.
            run=c.trajectory(f,duration,start=round(initial_time/f.dt),sample=.1)
            result['stages'].append(dict(round=r+1,division_time=initial_time,run=run))
            print(interval,'round',r+1,'N',f.nFa,'changes',len(run['changes']),flush=True)
            checkpoint()
        result['frames'].append(geometry(f,'Four relaxation times after final division'))
        result['skip_final']=c.trajectory(skip,4.,start=round(skip.bio_time/skip.dt),sample=.1)
        result.update(status='complete',preferred_total=float(f.preferred.sum()),
                      final_preferred=f.preferred,division_log=f.division_log)
    except Exception as exc:result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    c.save_result(path,result)
    path.with_suffix('.partial.json').unlink(missing_ok=True)
    print('division',interval,result['status'],result.get('error',''),flush=True)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(job,[2.,4.,8.]))
