"""Local T1 release, facilitation, boundary and preparation pilots with controls."""
import json
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from circle import collective as c,force_probe as fp,probe,protocol
from circle.disc import DiscFoam
from circle.measure import touching_wall
from circle.sweep import serializable
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'circle/data/collective_dynamics_20261114/mechanics'


def site_of(f,e):return f._local(f.vpos[f.e_v[e]]).mean(axis=0)

def facilitation(before,after,sham,site,excluded):
    common=fp.pairs(before)&fp.pairs(after)&fp.pairs(sham)
    candidates=[]
    for pair in common:
        if pair in excluded:continue
        edge=probe.film_between(before,pair)
        if edge is None:continue
        if fp.separation(before,before.e_v[edge])[0]<.2:continue
        d=site_of(before,edge)-site
        if not isinstance(before,DiscFoam):d-=np.rint(d/before.bs)*before.bs
        candidates.append((float(np.linalg.norm(d)),pair))
    candidates.sort()
    if not candidates:return []
    selected=[('near',candidates[0]),('far',candidates[-1])]
    results=[]
    for zone,(distance,pair) in selected:
        row=dict(zone=zone,distance=distance,pair=pair,curves={})
        for label,f in [('before',before),('after',after),('sham',sham)]:
            e=probe.film_between(f,pair);q=fp.separation(f,f.e_v[e])[0]
            cfg=fp.ProbeConfig(increments=4,target=.9*q,apply_t1=False)
            row['curves'][label]=fp.probe(f,e,cfg)
        results.append(row)
    return results


def job(task):
    domain,zone,seed,quench=task;path=OUT/f'{domain}_{zone}_seed{seed}_quench{quench:g}.json'
    if path.exists():return
    p=dict(domain=domain,zone=zone,seed=seed,quench_tauT=quench,n=36,w=1.,rho=1.3,
           spacing=.1,dt=.005,release=10.,edge_selection='longest two eligible, first complete path')
    result=dict(parameters=p,status='running',attempts=[])
    def checkpoint():path.with_suffix('.partial.json').write_text(json.dumps(serializable(result))+'\n')
    try:
        f=c.prepare(domain,36,seed,quench=quench)
        result['wall_touching_initial']=int(touching_wall(f).sum()) if domain=='disc' else None
        eligible=[e for e in probe.contacts(f) if probe.zone(f,e)==zone]
        eligible.sort(key=lambda e:f.edge_lengths()[e],reverse=True)
        for e in eligible[:2]:
            cfg=fp.ProbeConfig(apply_t1=False)
            curve,loaded=fp.probe(f,e,cfg,return_state=True)
            result['attempts'].append(curve);checkpoint()
            if curve['complete'] and loaded.edge_lengths()[e]<=fp.T1_LENGTH:break
        else:raise fp.ProbeStop('no complete candidate reaching the film T1 threshold')
        site=site_of(f,e);elapsed=sum(r['duration'] for r in curve['curve'])
        sham=fp.isolated_copy(f)
        # Same elapsed zero-noise time and frozen topology as loading.
        for k in range(1,round(elapsed/f.dt)+1):
            sham.step(0.)
            if k%max(1,round(1/f.dt))==0:fp.validate(sham)
        fp.validate(sham)
        after=fp.isolated_copy(loaded);old=fp.pairs(after)
        if not after.t1_foam(e):raise fp.ProbeStop('original T1 operation declined')
        fp.validate(after);direct=old^fp.pairs(after)
        result.update(selected_edge=e,site=site,loading_duration=elapsed,direct_pairs=sorted(direct),
                      curve=curve,branches={},facilitation=[])
        # Immediate local response of common contacts before any free release.
        result['facilitation']=facilitation(loaded,after,sham,site,direct)
        checkpoint()
        for label,branch in [('forced',after),('loaded_no_T1',loaded),('sham',sham)]:
            result['branches'][label]=c.trajectory(branch,10.,site=site,start=round(elapsed/f.dt))
            checkpoint()
        final=set(map(tuple,result['branches']['forced']['final_pairs']))
        original=tuple(curve['old_pair']);opposite=tuple(sorted(curve['opposite']))
        result['outcome']='reversal' if original in final else 'exchange' if opposite in final else 'space'
        result['status']='complete'
    except Exception as exc:result.update(status='preempted' if isinstance(exc,fp.ProbeStop) else 'failed',
                                           error=f'{type(exc).__name__}: {exc}')
    c.save_result(path,result);path.with_suffix('.partial.json').unlink(missing_ok=True)
    print(path.name,result['status'],result.get('outcome',result.get('error','')),flush=True)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    gates=json.loads((OUT.parent/'gates/summary.json').read_text())
    if not all(row['pass'] for row in gates.values()):raise RuntimeError('mechanical gates failed')
    tasks=[(d,z,s,q) for s in [1,2] for q in [1.,5.]
           for d,z in [('disc','interior'),('disc','wall'),('periodic','periodic')]]
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(job,tasks))
