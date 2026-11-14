"""Reproducible acceptance checks for the external mechanical apparatus."""
import json
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from circle import protocol, probe, force_probe as fp
from circle.sweep import serializable
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'circle/data/collective_dynamics_20261114/gates'


def job(task):
    domain,variant=task
    path=OUT/f'{domain}_{variant}.json'
    if path.exists():return domain,variant
    f=(protocol.equilibrated(16,1.,1.3,seed=1) if domain=='disc' else
       protocol.periodic_equilibrated(4,1.,1.3,seed=1))
    e=max((e for e in probe.contacts(f) if probe.zone(f,e)!='wall'),
          key=lambda e:f.edge_lengths()[e])
    cfg=fp.ProbeConfig(increments=24,apply_t1=False)
    if variant=='finer':cfg=replace(cfg,increments=48)
    if variant=='longer':cfg=replace(cfg,window=10,min_hold=30,max_hold=600)
    if variant=='dt_half':f.dt/=2
    if variant=='pulse':result=fp.pulse(f,e,.1,duration=100)
    else:result=fp.probe(f,e,cfg)
    path.write_text(json.dumps(serializable(result))+'\n')
    print(domain,variant,result.get('status','pulse'),result.get('work'),flush=True)
    return domain,variant


def summarize():
    result={}
    for domain in ['disc','periodic']:
        runs={v:json.loads((OUT/f'{domain}_{v}.json').read_text())
              for v in ['base','finer','longer','dt_half','pulse']}
        base=runs['base']; row={'complete':all(runs[v]['complete'] for v in ['base','finer','longer','dt_half'])}
        row['work']={v:runs[v]['work'] for v in ['base','finer','longer','dt_half']}
        row['work_relative_errors']={v:abs(runs[v]['work']/base['work']-1) for v in ['finer','longer','dt_half']}
        pulse=runs['pulse']
        delta=np.array([a['separation']-b['separation'] for a,b in zip(pulse['sham'],pulse['forced'])])
        # Require the last two 20-time-unit block means to settle independently.
        late=delta[-160:]; observed=float(late.mean()); drift=abs(late[:80].mean()-late[80:].mean())
        curve=runs['finer']['curve'];s=curve[0]['separation']-np.array([r['separation'] for r in curve])
        force=np.array([r['force'] for r in curve])-curve[0]['force']
        expected=float(np.interp(.1,force,s))
        row.update(pulse_shortening=observed,curve_shortening=expected,pulse_drift=float(drift),
                   pulse_relative_error=abs(observed/expected-1),pulse_duration=100)
        row['pass']=bool(row['complete'] and max(row['work_relative_errors'].values())<.05
                         and abs(observed-expected)<=max(.0001,.05*abs(expected))
                         and drift<=max(.0001,.05*abs(observed)))
        result[domain]=row
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(job,[(d,v) for d in ['disc','periodic'] for v in ['base','finer','longer','dt_half','pulse']]))
    summarize()
