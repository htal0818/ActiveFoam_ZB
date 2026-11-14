"""Compare the unchanged interval-four simulation with the historical event log.

The old observer records several operations at the same physical timestamp.
Collapse those to their net step-end change before applying finite-time dwell.
This audit distinguishes a measurement convention from changed dynamics.
"""
import gzip,json
from pathlib import Path
import numpy as np
from circle.scripts.division_contact_analysis import persistent_changes
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'circle/data/collective_dynamics_20261114'


def collapse_same_step(initial,events,dt):
    current=set(map(tuple,initial));previous=current.copy();out=[];time=None
    for event in events:
        t=round(event['time']/dt)
        if t!=time and time is not None:
            if current!=previous:out.append(dict(time=time*dt,lost=sorted(previous-current),gained=sorted(current-previous)))
            previous=current.copy()
        current-=set(map(tuple,event['lost']));current|=set(map(tuple,event['gained']));time=t
    if current!=previous:out.append(dict(time=time*dt,lost=sorted(previous-current),gained=sorted(current-previous)))
    return out


def main():
    old=json.load(gzip.open(ROOT/'circle/data/division_8_to_256_20261114/simulation.json.gz','rt'))
    new=json.loads((DATA/'division/interval4.json').read_text())
    audit=dict(geometry=[],counting=[])
    for frame in new['frames']:
        candidates=[f for f in old['frames'] if len(f['polygons'])==len(frame['polygons']) and abs(f['time']-frame['time'])<1e-6]
        if not candidates:continue
        diff=min(max((float(np.max(abs(np.array(a)-b))) if np.shape(a)==np.shape(b) else np.inf)
             for a,b in zip(frame['polygons'],x['polygons'])) for x in candidates)
        audit['geometry'].append(dict(n=len(frame['polygons']),time=frame['time'],max_difference=diff))
        assert diff==0,'Numerical replay differs from historical simulation'
    for oldstage,newstage in zip(old['stages'],new['stages']):
        initial=oldstage['records'][0]['pairs'];events=collapse_same_step(initial,oldstage['events'],.001)
        counts=[]
        for dwell in [.1,.2,.5]:
            a=persistent_changes(initial,events,oldstage['time'],oldstage['end'],dwell)
            r=newstage['run'];b=persistent_changes(r['initial_pairs'],r['events'],0.,r['duration'],dwell)
            assert len(a)==len(b)
            assert len(events)==len(r['events'])
            counts.append(dict(dwell=dwell,historical_grouped=len(a),new_step_end=len(b)))
        raw=persistent_changes(initial,oldstage['events'],oldstage['time'],oldstage['end'],.2)
        audit['counting'].append(dict(n=oldstage['n'],old_operation_counter=len(raw),checks=counts))
    (DATA/'division/replay_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
