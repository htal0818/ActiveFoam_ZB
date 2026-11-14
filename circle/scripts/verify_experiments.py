"""Check experiment provenance, complete trajectories and matched controls."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle.scripts.run_experiments import load_records
from circle.experiments import source_hash


def main():
    directory=ROOT/'circle/data/experiments_20261114';manifest,records=load_records(directory)
    assert manifest['source_hash']==source_hash(ROOT),'core source changed during/after the batch'
    accepted=[r for r in records if r['accepted']]
    for r in accepted:
        t=r['task'];samples=r['samples']
        assert len(samples)==round(t['duration']/t['sample'])+1
        np.testing.assert_allclose([s['time'] for s in samples],np.arange(len(samples))*t['sample'],rtol=0,atol=1e-10)
        for s in samples:
            for key in ('centers','shape','neighbors','area','wall_distance','wall_curvature','nematic'):
                assert np.all(np.isfinite(np.asarray(s[key],float))),(r['key'],key)
            assert s['neighbor_overlap']>=s['continuous_survival']
        g=r['final_geometry'];axes=g['radius']*np.array([np.sqrt(g['aspect']),1/np.sqrt(g['aspect'])])
        for p in g['polygons']:assert np.linalg.norm((np.array(p)-g['center'])/axes,axis=1).max()<=1+2e-9
        assert r['final_quality']['resolved_film_crossings']==0 and r['final_quality']['nested']==0
        for event in r['events']:assert 0<=event['time']<=t['duration']+1e-9
    # Identical tau=10 runs are intentional controls, not additional replicas.
    duplicates=0
    for seed in (1,2,3):
        pair=[r for r in records if r['task']['family'].startswith('persistence_') and r['task']['tau']==10 and r['task']['seed']==seed]
        assert len(pair)==2
        for key in ('accepted','samples','events','pair_changes','contact_episodes','summary','final_geometry'):
            assert pair[0].get(key)==pair[1].get(key),(seed,key)
        duplicates+=1
    # Paired pulses and shams must agree before the intervention, when both exist.
    paired=0
    shams={(r['task']['n'],r['task']['seed']):r for r in records if r['task']['family']=='pulse' and r['task']['pattern']=='uniform'}
    for r in records:
        t=r['task']
        if t['family']!='pulse' or t['pattern']=='uniform':continue
        sham=shams[t['n'],t['seed']]
        before=[s for s in r.get('samples',[]) if s['time']<=t['pulse_start']]
        control=[s for s in sham.get('samples',[]) if s['time']<=t['pulse_start']]
        if before and control:
            count=min(len(before),len(control));assert before[:count]==control[:count]
            paired+=1
    result=dict(core_source_hash=manifest['source_hash'],runs=len(records),completed=len(accepted),
        duplicate_persistence_controls_verified=duplicates,pre_pulse_sham_matches=paired,
        checks=['recorded sample times and finite geometry','final polygon confinement','no resolved final crossings/nesting',
                'contact survival bounded by neighbor overlap','event time ranges','duplicate persistence controls','pre-pulse matched shams'])
    (directory/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
