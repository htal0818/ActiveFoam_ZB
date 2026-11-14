"""Longer feedback runs under joint numerical refinement and one every-step audit."""
import os,sys,json,gzip,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle import biology_experiments as experiments
from circle.biology_experiments import BioTask
from circle.biology import BiologyFoam
from circle import protocol
from circle.experiments import source_hash


class AuditedBiologyFoam(BiologyFoam):
    def advance_scheduled(self,*args,**kwargs):
        result=super().advance_scheduled(*args,**kwargs)
        protocol.validate(self)
        return result


def execute(task):
    original=experiments.BiologyFoam
    if task.family=='feedback_every_step':experiments.BiologyFoam=AuditedBiologyFoam
    try:return experiments.run(task)
    finally:experiments.BiologyFoam=original


def main():
    tasks=[]
    for gain,delay in ((0.,0.),(.5,2.),(.5,10.)):
        for dt,h,history in ((.005,.1,.05),(.0025,.05,.025)):
            tasks.append(BioTask('feedback_refinement',1,gain=gain,delay=delay,dt=dt,spacing=h,
                                 history_spacing=history,duration=300.,budget=900))
    tasks.append(BioTask('feedback_every_step',1,gain=.5,delay=2.,duration=150.,budget=900))
    out=ROOT/'circle/data/biology_refinement_20261114';out.mkdir(exist_ok=True);runs=out/'runs';runs.mkdir(exist_ok=True)
    manifest=dict(tasks=[t.key for t in tasks],source_hash=source_hash(ROOT),script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Selected from the seed-1 pilot. Full preparation is repeated at each dt/spacing, so trajectories need not match pointwise. Refine dt, film spacing and history sampling jointly. The every-step run adds validation only and should reproduce its matching pilot.')
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('manifest mismatch')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(execute,t):t for t in tasks if not (runs/f'{t.key}.json.gz').exists()}
        for future in as_completed(pending):
            t=pending[future];r=future.result()
            with gzip.open(runs/f'{t.key}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,separators=(',',':'))
            print(t.family,t.gain,t.delay,t.dt,r['accepted'],r.get('error'),flush=True)


if __name__=='__main__':main()
