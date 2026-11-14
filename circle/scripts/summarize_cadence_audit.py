"""Compare every-step monitoring with the matching sampled-check trajectories."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle.scripts.run_experiments import load_records


def main():
    directory=ROOT/'circle/data/experiments_cadence_audit_20261114'
    manifest,audit=load_records(directory);base_manifest,base=load_records(ROOT/'circle/data/experiments_20261114')
    assert manifest['source_hash']==base_manifest['source_hash']
    rows=[]
    for r in audit:
        t=r['task'];reference=next(x for x in base if x['task']['family']=='adhesion_activity' and all(x['task'][k]==t[k] for k in ('n','w','mu','rho','seed')))
        samples=r.get('samples',[]);control=reference.get('samples',[]);common=min(len(samples),len(control))
        assert samples[:common]==control[:common],f"monitoring changed sampled dynamics: {r['key']}"
        if r['accepted'] and reference['accepted']:
            for key in ('events','pair_changes','contact_episodes','summary','final_geometry'):
                assert r[key]==reference[key],(r['key'],key)
        rows.append(dict(n=t['n'],w=t['w'],mu=t['mu'],seed=t['seed'],sampled_checks_passed=reference['accepted'],
            every_step_checks_passed=r['accepted'],stage=r['stage'],elapsed=r.get('elapsed'),
            common_samples_identical=common,error=r.get('error','')))
    with (directory/'outcomes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    result=dict(completed=sum(r['accepted'] for r in audit),attempted=len(audit),core_source_hash=manifest['source_hash'],
        matching_prefixes_verified=True,
        interpretation='This audit strengthens the selected controls only. It does not validate all seeds, all other protocols, or between-check preparation geometry. The existing local microfold tolerance remains in effect.',rows=rows)
    (directory/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
