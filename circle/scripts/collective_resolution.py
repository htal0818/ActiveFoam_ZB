"""Selected activity failures: coupled dt/spacing refinement, unchanged equations."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from circle import collective as c,protocol
from circle.sweep import serializable
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'circle/data/collective_dynamics_20261114/resolution'

def job(task):
    domain,reference_mu=task;path=OUT/f'{domain}_mu{reference_mu:g}.json'
    if path.exists():return
    p=dict(domain=domain,n=36,seed=1,w=1.5,rho=1.3,tau=10.,reference_mu=reference_mu,
           preparation_dt=.005,preparation_spacing=.1,dt=.0025,spacing=.05,burn=50.,duration=100.,
           note='Same initial prepared state; refined noise increments do not give a pathwise paired realization.')
    phase='preparation'
    try:
        f=c.prepare(domain,36,1,w=1.5);f.dt=.0025;f.edpc=.05
        target=reference_mu**2/(20-.005);mu=float(np.sqrt(target*(20-f.dt)))
        p.update(mu=mu,target_raw_contact_variance=target)
        phase='burn'
        iteration=protocol.evolve(f,50.,mu,start=0)
        phase='observation'
        result=dict(parameters=p,status='complete',run=c.trajectory(f,100.,mu,start=iteration,sample=.25))
    except Exception as exc:result=dict(parameters=p,status='failed',phase=phase,error=f'{type(exc).__name__}: {exc}')
    c.save_result(path,result)
    print(path.name,result['status'],result.get('phase',''),result.get('error',''),flush=True)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(job,[(d,m) for m in [.5,3.] for d in ['disc','periodic']]))
