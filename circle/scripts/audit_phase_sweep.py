"""Check new active observables and reproduce the shared rho=1.3 trajectories."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from circle.sweep import source_digest

ADDED=('neighbor_mean','neighbor_series','cell_neighbors_initial','cell_neighbors_final',
       'packing_mean','packing_series','area_mean')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=ROOT/'circle/data/curvature_sweep_rerun_20261114')
    p.add_argument('--active',type=Path,default=ROOT/'circle/data/phase_sweep_20261114')
    a=p.parse_args()
    manifest=json.loads((a.active/'manifest.json').read_text())
    assert manifest['source_digest']==source_digest(ROOT),'Simulation source changed during or after the active sweep.'
    accepted=0;shared=[]
    for key in manifest['tasks']:
        r=json.loads((a.active/'runs'/f'{key}.json').read_text())
        if r['accepted']:
            accepted+=1;d=r['dynamics'];n=r['task']['n']
            for name,mean in d['neighbor_mean'].items():
                if d['zone_counts'][name]:
                    values=np.array(d['neighbor_series'][name])
                    assert np.isfinite(values).all() and ((values>=0)&(values<=n-1)).all()
                    assert np.isclose(mean,values[1:].mean())
            for name in ('cell_neighbors_initial','cell_neighbors_final'):
                values=np.array(d[name]);assert np.all(values==np.round(values))
                assert int(values.sum())%2==0
            assert np.isclose(np.mean(d['cell_neighbors_final']),d['neighbor_series']['all'][-1])
            assert np.isclose(d['packing_mean'],np.mean(d['packing_series'][1:]))
            radius=r['geometry_parameters']['radius']
            assert np.isclose(d['packing_mean'],d['area_mean']['all']*n/(np.pi*radius**2))
            assert np.isclose(d['shape_mean'],np.mean(d['shape_series']['all'][1:]))
        oldpath=a.reference/'runs'/f'{key}.json'
        if oldpath.exists():
            old=json.loads(oldpath.read_text());old.pop('seconds');r.pop('seconds')
            if 'dynamics' in r:
                for field in ADDED:r['dynamics'].pop(field,None)
            assert old==r,f'Existing observables changed for {key}'
            shared.append(key)
    result=dict(total=len(manifest['tasks']),accepted=accepted,shared_records_exactly_reproduced=len(shared),
                shared_keys=shared,added_measurements=list(ADDED),
                checks='Contact means, degree parity, mean shape, packing/area identity, source hash and unchanged prior observables passed.')
    (a.active/'measurement_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"Validated {accepted} accepted runs; {len(shared)} shared trajectories exactly reproduce prior observables.")


if __name__=='__main__':main()
