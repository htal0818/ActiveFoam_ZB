"""Verify synchronous lineage/conservation and deterministic feedback controls."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle.scripts.plot_biology import load
from circle.experiments import source_hash


def main():
    folder=ROOT/'circle/data/biology_20261114';records=load(folder)
    refinement=load(ROOT/'circle/data/biology_refinement_20261114')
    assert source_hash(ROOT)==json.loads((folder/'manifest.json').read_text())['source_hash']
    combined=load(ROOT/'circle/data/biology_combined_20261114')
    divisions=0
    for r in records+combined:
        if not r['accepted']:continue
        t=r['task'];frames=r['frames']
        for s in r['samples']:
            assert len(set(s['uids']))==s['n']
            assert len(s['preferred'])==s['n'] and np.all(np.array(s['area'])>0)
            assert np.isclose(np.sum(s['preferred'])/t['n'],s['nominal_density'],rtol=1e-12)
        if t['divide']:
            log=r['division_log'];assert len(log)==1 and log[0]['accepted'];entry=log[0]
            assert entry['mothers']==t['n'] and entry['daughters']==2*t['n']
            assert all(e['time']==entry['time'] for e in entry['events'])
            assert np.isclose(entry['time'],t['division_times'][0],atol=1e-9)
            before=next(f for f in frames if f['label']=='before_division');after=next(f for f in frames if f['label']=='after_division')
            assert before['time']==after['time'] and before['radius']==after['radius']
            assert np.isclose(sum(before['preferred']),sum(after['preferred']),rtol=1e-12)
            assert set(before['uids']).isdisjoint(after['uids'])
            for polygon in after['polygons']:
                assert np.linalg.norm(np.array(polygon)-after['center'],axis=1).max()<=after['radius']*(1+1e-9)
            divisions+=1
    prefixes=0;audit=False
    for r in refinement:
        if not r['accepted'] or r['task']['dt']!=.005:continue
        t=r['task'];pilot=next(x for x in records if x['task']['family']=='feedback' and x['task']['seed']==t['seed'] and x['task']['gain']==t['gain'] and x['task']['delay']==t['delay'])
        assert r['samples'][:len(pilot['samples'])]==pilot['samples']
        events=[e for e in r['events'] if e['time']<=pilot['samples'][-1]['time']+1e-10]
        assert events==pilot['events']
        prefixes+=1
        if t['family']=='feedback_every_step':
            for key in ('samples','events','frames','summary'):assert r[key]==pilot[key]
            audit=True
    report=dict(main_completed=sum(r['accepted'] for r in records),main_attempted=len(records),
        combined_completed=sum(r['accepted'] for r in combined),combined_attempted=len(combined),
        refinement_completed=sum(r['accepted'] for r in refinement),refinement_attempted=len(refinement),
        synchronous_rounds_verified=divisions,matching_feedback_prefixes=prefixes,every_step_audit_exact_match=audit,
        source_hash=source_hash(ROOT))
    (folder/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
