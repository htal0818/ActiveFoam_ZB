"""Run delayed-feedback and synchronous-division pilots, preserving failed records."""
import os,sys,json,gzip,csv,subprocess,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle.biology_experiments import tasks,run
from circle.experiments import source_hash


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'circle/data/biology_20261114');p.add_argument('--workers',type=int,default=4);p.add_argument('--family',nargs='+');a=p.parse_args()
    todo=[t for t in tasks() if not a.family or t.family in a.family];a.out.mkdir(parents=True,exist_ok=True);runs=a.out/'runs';runs.mkdir(exist_ok=True)
    manifest=dict(source_hash=source_hash(ROOT),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tasks=[t.key for t in todo],division='Every current mother divides at the same scheduled time; the entire round is atomic. No random draws or integration between splits.',
        feedback='tauT*dT=-(T-Tfixed)+gain*tanh(delayed_contact_strain/0.1)+mu*zeta. Feedback runs have mu=0 after a common deterministic tension kick.',
        growth='With growth enabled, preferred area doubles exponentially over 20 tauR, then stops. Cleavage partitions K in proportion to equal daughter areas.',
        validation='Every 0.1 tauR during evolution; standard preparation retains 0.5 tauR checks and its existing microfold tolerance.')
    path=a.out/'manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('source_hash','tasks'):
            if old[key]!=manifest[key]:raise ValueError('manifest mismatch; choose new output')
    else:path.write_text(json.dumps(manifest,indent=2)+'\n')
    pending=[t for t in todo if not (runs/f'{t.key}.json.gz').exists()]
    print(len(todo),'tasks;',len(pending),'remaining',flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run,t):t for t in pending}
        for i,future in enumerate(as_completed(futures),1):
            t=futures[future];r=future.result()
            with gzip.open(runs/f'{t.key}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,separators=(',',':'))
            print(i,'/',len(pending),t.family,'seed',t.seed,'gain',t.gain,'delay',t.delay,'growth',t.growth,'divide',t.divide,':',r['accepted'],r.get('error',''),flush=True)
    records=[]
    for t in todo:
        with gzip.open(runs/f'{t.key}.json.gz','rt') as f:records.append(json.load(f))
    fields=['family','seed','gain','delay','growth','divide','accepted','stage','elapsed','seconds','error']
    with (a.out/'outcomes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in records:w.writerow({k:(r['task']|r).get(k,'') for k in fields})
    print('Completed',sum(r['accepted'] for r in records),'/',len(records),flush=True)


if __name__=='__main__':main()
