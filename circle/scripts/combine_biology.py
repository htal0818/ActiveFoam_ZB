"""Combine the validated moderate feedback setting with synchronous growth/division."""
import sys,json,gzip
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle.biology_experiments import BioTask,run
from circle.experiments import source_hash


def main():
    tasks=[BioTask('division_feedback',seed,rho=.6,gain=.5,delay=2.,growth=True,divide=True,duration=60.,mu=.05) for seed in (1,2,3)]
    out=ROOT/'circle/data/biology_combined_20261114';out.mkdir(exist_ok=True);runs=out/'runs';runs.mkdir(exist_ok=True)
    manifest=dict(tasks=[t.key for t in tasks],source_hash=source_hash(ROOT),note='Same growth/division protocol and seeds as the main pilot, adding g=0.5 and delay=2. No deterministic kick; mu=0.05 as in matched division controls. Selected after the component pilots.')
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('manifest mismatch')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(run,t):t for t in tasks if not (runs/f'{t.key}.json.gz').exists()}
        for future in as_completed(pending):
            r=future.result();t=pending[future]
            with gzip.open(runs/f'{t.key}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,separators=(',',':'))
            print(t.seed,r['accepted'],r.get('error'),flush=True)


if __name__=='__main__':main()
