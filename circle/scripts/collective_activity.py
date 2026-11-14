"""Equal-variance OU persistence: same initial state, two domains, two seeds."""
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from circle import collective as c,force_probe as fp,protocol
from circle.sweep import serializable
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'circle/data/collective_dynamics_20261114/activity'

def job(task):
    domain,seed,tau,reference_mu=task;out=OUT/str(reference_mu);out.mkdir(parents=True,exist_ok=True);path=out/f'{domain}_seed{seed}_tau{tau:g}.json'
    if path.exists():return
    params=dict(domain=domain,seed=seed,tau=tau,n=36,w=1.5,rho=1.3,
                dt=.005,spacing=.1,burn=5*tau,duration=100.,mu_reference=reference_mu)
    try:
        f=c.prepare(domain,36,seed,w=1.5);f.tauT=tau
        mu=c.variance_matched_mu(tau,f.dt,reference_mu=reference_mu)
        # Burn for five persistence times; reset observation after equilibration.
        iteration=protocol.evolve(f,5*tau,mu,start=0)
        run=c.trajectory(f,100.,mu,start=iteration,sample=.25)
        result=dict(status='complete',parameters={**params,'mu':mu,
           'target_raw_contact_variance':reference_mu**2/(10*(2-.005/10))},run=run)
    except Exception as exc:
        result=dict(status='failed',parameters=params,error=f'{type(exc).__name__}: {exc}')
    c.save_result(path,result)
    print(path.name,result['status'],result.get('error',''),flush=True)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--mu',type=float,nargs='+',default=[.25,.5,3.]);args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=1) as pool:
        list(pool.map(job,[(d,s,t,m) for m in args.mu for d in ['disc','periodic'] for s in [1,2] for t in ([10.] if m==0 else [2.,10.,20.])]))
