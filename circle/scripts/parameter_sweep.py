"""Run/resume structure, activity and wall-curvature sweeps, with size controls."""
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
from circle.sweep import tasks,run_task,source_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'circle/data/curvature_sweep')
    p.add_argument('--sizes',type=int,nargs='+',default=[36,100,196])
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--duration',type=float,default=50.)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--mode',choices=['all','structure','dynamics'],default='all')
    a=p.parse_args()
    if a.workers<1 or a.duration<=0:p.error('workers and duration must be positive')
    todo=[t for t in tasks(a.sizes,a.seeds,a.duration) if a.mode=='all' or t.mode==a.mode]
    manifest=dict(source_digest=source_digest(ROOT),tasks=[t.key for t in todo],
                  sizes=a.sizes,seeds=a.seeds,duration=a.duration,mode=a.mode,
                  base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
                  provenance='source_digest identifies simulation files; base_commit may predate uncommitted sweep code',
                  curvature='wall curvature only; R=sqrt(N/pi), size and curvature coupled',
                  policy='disk: resolved crossings rejected, local microfolds reported; periodic: original per-cell checks')
    a.out.mkdir(parents=True,exist_ok=True);runs=a.out/'runs';runs.mkdir(exist_ok=True)
    mpath=a.out/'manifest.json'
    if mpath.exists():
        old=json.loads(mpath.read_text())
        for key in ('source_digest','tasks'):
            if old[key]!=manifest[key]:raise ValueError(f'{key} differs; choose a new output directory')
    else:mpath.write_text(json.dumps(manifest,indent=2)+'\n')
    pending=[t for t in todo if not (runs/f'{t.key}.json').exists()]
    print(f'{len(todo)} tasks; {len(pending)} remaining',flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_task,t):t for t in pending}
        for i,future in enumerate(as_completed(futures),1):
            t=futures[future];record=future.result()
            tmp=runs/f'{t.key}.tmp';tmp.write_text(json.dumps(record,allow_nan=False)+'\n')
            tmp.replace(runs/f'{t.key}.json')
            print(f'{i}/{len(pending)} {t.domain} {t.mode} N={t.n} W={t.w} rho={t.rho} mu={t.mu} seed={t.seed}: '+
                  ('accepted' if record['accepted'] else record.get('error','failed')),
                  flush=True)
    records=[json.loads((runs/f'{t.key}.json').read_text()) for t in todo]
    fields=['domain','mode','n','w','rho','mu','seed','accepted','stage','seconds','error']
    with (a.out/'outcomes.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in records:writer.writerow({key:({**r['task'],**r}).get(key,'') for key in fields})
    print(f"Finished: {sum(r['accepted'] for r in records)}/{len(records)} accepted",flush=True)


if __name__=='__main__':main()
