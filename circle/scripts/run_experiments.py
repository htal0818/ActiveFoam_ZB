"""Run the proposed experiments with complete compressed event/trajectory records."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv,gzip,json,os
from pathlib import Path
import subprocess,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle.experiments import Experiment,run_experiment,source_hash


def plan():
    tasks=[]
    for seed in (1,2):
        for dt in (.005,.0025):
            for spacing in (.15,.075):
                tasks.append(Experiment('refinement',seed,rho=1.3,mu=1.5,dt=dt,spacing=spacing,
                                        warmup=10,duration=50,check_every=1))
    for n in (36,100):
        for w in (.5,1.,1.5):
            for mu in (.25,.5,1.):
                for seed in (1,2,3):tasks.append(Experiment('adhesion_activity',seed,n=n,w=w,mu=mu))
    for w in (.25,.35):
        for rho in (.9,1.):
            for seed in (1,2,3):tasks.append(Experiment('opening_spaces',seed,w=w,rho=rho))
    for tau in (2.,5.,10.,20.):
        for protocol in ('fixed_amplitude','fixed_variance'):
            mu=.5 if protocol=='fixed_amplitude' else .5*((2*tau-.005)/(20-.005))**.5
            for seed in (1,2,3):tasks.append(Experiment('persistence_'+protocol,seed,tau=tau,mu=mu,
                                                     warmup=5*tau,duration=10*tau))
    for w in (1.,1.5):
        for seed in (1,2,3):tasks.append(Experiment('long',seed,w=w,warmup=200,duration=500,budget=900))
    for n in (36,100):
        for pattern in ('uniform','pulse','wall','sector'):
            for seed in (1,2,3):tasks.append(Experiment('pulse',seed,n=n,mu=.25,pattern=pattern,duration=70))
    for aspect in (1.,1.5,2.):
        for seed in (1,2,3):tasks.append(Experiment('geometry',seed,aspect=aspect))
    return tasks


def load_records(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    records=[]
    for key in manifest['tasks']:
        with gzip.open(directory/'runs'/f'{key}.json.gz','rt') as f:records.append(json.load(f))
    return manifest,records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'circle/data/experiments_20261114')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--family',nargs='+')
    a=p.parse_args();todo=[t for t in plan() if a.family is None or t.family in a.family]
    if not todo:raise ValueError('no experiments selected')
    a.out.mkdir(parents=True,exist_ok=True);runs=a.out/'runs';runs.mkdir(exist_ok=True)
    manifest=dict(tasks=[t.key for t in todo],source_hash=source_hash(ROOT),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preparation='All persistence variants share tauT=10 preparation; change tauT only before warmup. Relax for 7.5 tauR at each dt. Preparation geometry checks every 0.5 tauR; unresolved transient crossings can occur between checks.',
        event_definition='Committed outer transition transactions and their immediate contact changes, plus net contact changes after every scheduled timestep.',
        refinement='Same seed labels, not pathwise-coupled Brownian increments across mesh/timestep changes.',
        noise='mu matches Delta T/T0 under unit white-noise covariance in code time units.',
        budget='Wall-clock timeouts are failures, never zero-valued observations.')
    path=a.out/'manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('tasks','source_hash'):
            if old[key]!=manifest[key]:raise ValueError(f'{key} differs; use a fresh directory')
    else:path.write_text(json.dumps(manifest,indent=2)+'\n')
    pending=[t for t in todo if not (runs/f'{t.key}.json.gz').exists()]
    print(f'{len(todo)} experiments; {len(pending)} remaining',flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_experiment,t):t for t in pending}
        for i,future in enumerate(as_completed(futures),1):
            t=futures[future];r=future.result();temporary=runs/f'{t.key}.tmp'
            with gzip.open(temporary,'wt') as f:json.dump(r,f,allow_nan=False,separators=(',',':'))
            temporary.replace(runs/f'{t.key}.json.gz')
            print(f'{i}/{len(pending)} {t.family} N={t.n} W={t.w} rho={t.rho} mu={t.mu:.4g} tau={t.tau} aspect={t.aspect} {t.pattern} seed={t.seed}: '+
                  ('accepted' if r['accepted'] else f"{r['stage']}: {r.get('error')}"),flush=True)
    _,records=load_records(a.out)
    fields=['family','n','w','rho','mu','tau','aspect','pattern','seed','dt','spacing','accepted','stage','elapsed','seconds','error']
    with (a.out/'outcomes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in records:writer.writerow({k:(r['task']|r).get(k,'') for k in fields})
    print(f"Completed: {sum(r['accepted'] for r in records)}/{len(records)} accepted",flush=True)


if __name__=='__main__':main()
