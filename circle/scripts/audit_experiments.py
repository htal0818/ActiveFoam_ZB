"""Check selected baseline trajectories at every timestep, without changing dynamics."""
import gzip,json,sys
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle.experiments import Experiment,run_experiment,source_hash


def plan():
    return [Experiment('validation_cadence',1,n=n,w=w,mu=mu,check_every=1)
            for n,w,mu in ((36,.5,.5),(36,1.,.25),(36,1.,.5),(36,1.5,.5),(100,1.,.25),(100,1.,.5))]


def main():
    out=ROOT/'circle/data/experiments_cadence_audit_20261114';out.mkdir(exist_ok=True);runs=out/'runs';runs.mkdir(exist_ok=True)
    tasks=plan();manifest=dict(tasks=[t.key for t in tasks],source_hash=source_hash(ROOT),
        definition='Same preparation, seed and dynamics as corresponding adhesion_activity runs, with geometry checks every timestep after preparation. A finer observation cadence can detect transient crossings missed at 0.5 tauR checks.')
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('audit manifest mismatch')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run_experiment,t):t for t in tasks if not (runs/f'{t.key}.json.gz').exists()}
        for future in as_completed(futures):
            t=futures[future];r=future.result()
            with gzip.open(runs/f'{t.key}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,separators=(',',':'))
            print(t.n,t.w,t.mu,r['accepted'],r['stage'],r.get('elapsed'),r.get('error'),flush=True)


if __name__=='__main__':main()
