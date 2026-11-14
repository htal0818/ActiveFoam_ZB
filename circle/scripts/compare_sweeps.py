"""Compare fresh sweep records exactly, excluding measured wall-clock runtime."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def compare(reference,rerun,reference_figures=None,rerun_figures=None):
    old=json.loads((reference/'manifest.json').read_text())
    new=json.loads((rerun/'manifest.json').read_text())
    if old['tasks']!=new['tasks']:
        raise ValueError('Task lists differ; this is not a rerun of the same sweep.')
    result=dict(reference=str(reference),rerun=str(rerun),
                same_simulation_source=old['source_digest']==new['source_digest'],
                task_count=len(new['tasks']),excluded_fields=['seconds'],runs=[])
    for key in new['tasks']:
        a=json.loads((reference/'runs'/f'{key}.json').read_text())
        b=json.loads((rerun/'runs'/f'{key}.json').read_text())
        row=dict(key=key,task=b['task'],accepted=b['accepted'],
                 reference_seconds=a.pop('seconds'),rerun_seconds=b.pop('seconds'))
        row.update(exact_match=digest(a)==digest(b),reference_digest=digest(a),rerun_digest=digest(b))
        result['runs'].append(row)
    result['matching_runs']=sum(r['exact_match'] for r in result['runs'])
    result['accepted_runs']=sum(r['accepted'] for r in result['runs'])
    result['rejected_runs']=result['task_count']-result['accepted_runs']
    if (reference_figures is None)!=(rerun_figures is None):
        raise ValueError('Supply both figure directories, or neither.')
    if reference_figures is not None:
        names={p.name for p in reference_figures.glob('*.png')}
        other={p.name for p in rerun_figures.glob('*.png')}
        if not names or names!=other:raise ValueError('PNG figure sets differ or are empty.')
        result['png_exact_matches']={name:hashlib.sha256((reference_figures/name).read_bytes()).hexdigest()==
                                    hashlib.sha256((rerun_figures/name).read_bytes()).hexdigest()
                                    for name in sorted(names)}
    result['passed']=(result['same_simulation_source'] and
                      result['matching_runs']==result['task_count'] and
                      all(result.get('png_exact_matches',{}).values()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    p.add_argument('rerun',type=Path)
    p.add_argument('--reference-figures',type=Path)
    p.add_argument('--rerun-figures',type=Path)
    a=p.parse_args()
    result=compare(a.reference,a.rerun,a.reference_figures,a.rerun_figures)
    (a.rerun/'reproducibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"Exact scientific-record matches: {result['matching_runs']}/{result['task_count']}; "
          f"{result['accepted_runs']} accepted, {result['rejected_runs']} rejected.")
    if 'png_exact_matches' in result:
        print(f"Identical PNGs: {sum(result['png_exact_matches'].values())}/{len(result['png_exact_matches'])}")
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
