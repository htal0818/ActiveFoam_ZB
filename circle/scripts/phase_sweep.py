"""Active density slices for contact-number and shape-index response maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from circle.sweep import Task,run_task,source_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'circle/data/phase_sweep_20261114')
    p.add_argument('--sizes',type=int,nargs='+',default=[36,100,196])
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--densities',type=float,nargs='+',default=[1.,1.2,1.3,1.4])
    p.add_argument('--duration',type=float,default=50.)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.workers<1 or a.duration<=0 or min(a.densities)<=0:p.error('positive workers, duration and densities required')
    todo=[Task('disc','dynamics',n,1.,rho,mu,seed,duration=a.duration)
          for n in a.sizes for rho in a.densities for mu in (0.,.5,1.,1.5) for seed in a.seeds]
    if len({t.key for t in todo})!=len(todo):p.error('duplicate grid entries')
    manifest=dict(source_digest=source_digest(ROOT),tasks=[t.key for t in todo],
                  sizes=a.sizes,seeds=a.seeds,densities=a.densities,activities=[0.,.5,1.,1.5],
                  duration=a.duration,w=1.,base_commit=subprocess.check_output(
                      ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  working_tree_dirty=bool(subprocess.check_output(
                      ['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
                  measurement='Time-mean contact number, shape index and packing during the active measurement window; fixed wall/interior zones after warmup.',
                  convention='mu is the input corresponding to Delta T/T0 under unit white-noise covariance in code time units.',
                  geometry='Flat surface; wall curvature 1/R, R=sqrt(N/pi); radius and cell count coupled.')
    a.out.mkdir(parents=True,exist_ok=True);runs=a.out/'runs';runs.mkdir(exist_ok=True)
    path=a.out/'manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('source_digest','tasks'):
            if old[key]!=manifest[key]:raise ValueError(f'{key} differs; choose a fresh output directory')
    else:path.write_text(json.dumps(manifest,indent=2)+'\n')
    pending=[t for t in todo if not (runs/f'{t.key}.json').exists()]
    print(f'{len(todo)} tasks; {len(pending)} remaining',flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_task,t):t for t in pending}
        for i,future in enumerate(as_completed(futures),1):
            t=futures[future];record=future.result()
            tmp=runs/f'{t.key}.tmp';tmp.write_text(json.dumps(record,allow_nan=False)+'\n')
            tmp.replace(runs/f'{t.key}.json')
            print(f'{i}/{len(pending)} N={t.n} rho={t.rho} mu={t.mu} seed={t.seed}: '+
                  ('accepted' if record['accepted'] else record.get('error','failed')),flush=True)
    records=[json.loads((runs/f'{t.key}.json').read_text()) for t in todo]
    fields=['domain','mode','n','w','rho','mu','seed','accepted','stage','seconds','error']
    with (a.out/'outcomes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in records:writer.writerow({k:({**r['task'],**r}).get(k,'') for k in fields})
    print(f"Finished: {sum(r['accepted'] for r in records)}/{len(records)} accepted",flush=True)


if __name__=='__main__':main()
