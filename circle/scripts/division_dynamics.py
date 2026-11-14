"""Contact rearrangements after division, excluding the division operation itself."""
import copy
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from circle import protocol, state
from circle.benchmarks import free_area
from circle.biology import BiologyFoam
from circle.biology_experiments import uid_pairs, watch
from circle.disc import DiscFoam, circle_polyline
from circle.experiments import source_hash
from circle.measure import cell_centroids
from circle.sweep import serializable


def clone(foam):
    result = object.__new__(BiologyFoam)
    for key, value in vars(foam).items():
        setattr(result, key, state._copy(value))
    result.rng = copy.deepcopy(foam.rng)
    result.log = copy.deepcopy(foam.log)
    return result


def nonaffine_msd(reference, current):
    """Remove translation, proper rotation, and uniform contraction/expansion."""
    x = reference - reference.mean(axis=0)
    y = current - current.mean(axis=0)
    if np.sum(x*x) < 1e-20:
        return 0.
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.diag([1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    rotated = x @ rotation
    scale = np.sum(rotated*y) / np.sum(x*x)
    return float(np.mean(np.sum((y-scale*rotated)**2, axis=1)))


def baseline(foam, events):
    return dict(time=foam.bio_time, pairs=uid_pairs(foam), centers=cell_centroids(foam),
                event_start=len(events), mean_area=float(foam.f_area.mean()))


def sample(foam, events, ref):
    changes = events[ref['event_start']:]
    lost = {tuple(pair) for event in changes for pair in event['lost']}
    pairs = uid_pairs(foam)
    count = len(ref['pairs'])
    return dict(time=foam.bio_time, lag=foam.bio_time-ref['time'], n=foam.nFa,
                pairs=sorted(pairs), contacts=len(pairs), mean_neighbors=2*len(pairs)/foam.nFa,
                survival=len(ref['pairs']-lost)/count if count else None,
                retention=len(ref['pairs'] & pairs)/count if count else None,
                changes=len(changes), lost=sum(len(e['lost']) for e in changes),
                gained=sum(len(e['gained']) for e in changes),
                shape=float(foam.shape_factors().mean()), area=float(foam.f_area.sum()),
                residual_msd_over_area=nonaffine_msd(ref['centers'],cell_centroids(foam))/ref['mean_area'])


def evolve(foam, end, events, ref, records, checks):
    start = round(foam.bio_time/foam.dt)
    for iteration in range(start+1, round(end/foam.dt)+1):
        foam.advance_scheduled(iteration, mu=0.)
        if iteration % 50 == 0:
            checks.append(dict(time=foam.bio_time, **protocol.validate(foam)))
        if iteration % 100 == 0:
            records.append(sample(foam, events, ref))


def run():
    radius=.7; center=np.full(2,4*radius)
    ring=circle_polyline(center,np.sqrt(free_area(1.)/np.pi),.025)
    foam=BiologyFoam.from_disc(DiscFoam.from_closed_cells(radius,[ring],center=center,
        rho=1.,w=1.,dt=.001,edpc=.025,seed=1),gain=0.,growth_rate=0.)
    protocol.validate(foam)
    events=[];watch(foam,events)
    checks=[];stages=[];controls=[]
    evolve(foam,2.,events,baseline(foam,events),[],checks)
    for stage,time in enumerate([2.,6.,10.,14.,18.]):
        controls.append(clone(foam))
        before=foam.nFa
        assert foam.divide_synchronously(),foam.division_log[-1]
        ref=baseline(foam,events)
        records=[sample(foam,events,ref)]
        end=time+4 if stage<4 else 42.
        # Preserve the movie endpoint for an exact geometry replay check.
        evolve(foam,min(end,22.),events,ref,records,checks)
        if stage==4:
            original=json.load(gzip.open(ROOT/'circle/data/division_1_to_32_20261114/simulation.json.gz','rt'))
            for c,p in enumerate(original['frames'][-1]['polygons']):
                np.testing.assert_array_equal(foam._cell_polyline(c),p)
            evolve(foam,end,events,ref,records,checks)
        stages.append(dict(before=before,after=foam.nFa,time=time,end=end,
            reference_pairs=sorted(ref['pairs']),records=records,
            events=events[ref['event_start']:].copy()))
        print('division',before,'→',foam.nFa,'events',len(stages[-1]['events']),flush=True)
    control_results=[]
    for control,stage in zip(controls,stages):
        changes=[];watch(control,changes)
        ref=baseline(control,changes);records=[sample(control,changes,ref)];quality=[]
        evolve(control,stage['end'],changes,ref,records,quality)
        control_results.append(dict(n=control.nFa,time=stage['time'],end=stage['end'],
            records=records,events=changes,checks=quality))
        print('skip division, N=',control.nFa,'events',len(changes),flush=True)
    return serializable(dict(stages=stages,controls=control_results,checks=checks,
        exact_movie_replay=True,source_sha256=source_hash(ROOT),
        note='Contact-changing committed topology transactions only; septum creation and '
             'mother/daughter ID replacement are excluded. Controls fork from the identical '
             'pre-division state and omit that division and all later divisions. '
             'This changes cell number; per-cell normalization does not remove that difference.'))


def summarize(result):
    rows=[]
    for stage,control in zip(result['stages'],result['controls']):
        def window(events,start,stop):
            selected=[e for e in events if start-1e-8<e['time']<=stop+1e-8]
            return dict(transactions=len(selected),lost=sum(len(e['lost']) for e in selected),
                        gained=sum(len(e['gained']) for e in selected),
                        types=dict(Counter(e['kind'] for e in selected)))
        row=dict(round=f"{stage['before']}→{stage['after']}",
            first_4=window(stage['events'],stage['time'],stage['time']+4),
            skip_first_4=window(control['events'],stage['time'],stage['time']+4),
            survival_4=min(stage['records'],key=lambda s:abs(s['lag']-4))['survival'])
        if stage['after']==32:
            row['late_14_to_24']=window(stage['events'],32.,42.)
            row['final']=stage['records'][-1]
        rows.append(row)
    return rows


def plot(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    rows=summarize(result);x=np.arange(5)
    rate=[r['first_4']['transactions']/(s['after']*4) for r,s in zip(rows,result['stages'])]
    control=[r['skip_first_4']['transactions']/(s['before']*4) for r,s in zip(rows,result['stages'])]
    axes[0,0].bar(x-.18,rate,width=.36,label='Divide',color='#278f91')
    axes[0,0].bar(x+.18,control,width=.36,label='Skip this division',color='#d89a43')
    axes[0,0].set(xticks=x,xticklabels=[r['round'] for r in rows],
        ylabel='Contact-changing events / cell / τR',title='First 4 τR after each scheduled division')
    axes[0,0].legend(fontsize=9)
    stage=result['stages'][-1];control=result['controls'][-1];time=stage['time']
    bins=np.arange(0,24.01,1.)
    for run,n,label,color in [(stage,32,'Divide: 32 cells','#278f91'),
                              (control,16,'Skip: 16 cells','#d89a43')]:
        hist,_=np.histogram([e['time']-time for e in run['events']],bins)
        axes[0,1].stairs(hist/n,bins,label=label,color=color,lw=2)
        records=run['records'];lag=[r['lag'] for r in records]
        axes[1,0].plot(lag,[r['survival'] for r in records],label=label,color=color)
        axes[1,1].plot(lag,[r['residual_msd_over_area'] for r in records],label=label,color=color)
    axes[0,1].set(title='After the final division: no further driving',
        ylabel='Contact-changing events / cell / τR')
    axes[0,1].legend(fontsize=9)
    axes[1,0].set(title='Survival of contacts present at lag zero',ylabel='Fraction never lost',ylim=(-.03,1.03))
    axes[1,1].set(title='Centroid motion after removing global contraction',
        ylabel='Residual MSD / initial mean cell area')
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:
        ax.set_xlabel('Time since last scheduled division / τR')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Division-driven rearrangements, without growth or tension noise\n'
                 'Division itself is excluded from the event counts',fontsize=15)
    fig.savefig(out/'division_dynamics.png',dpi=180)
    fig.savefig(out/'division_dynamics.pdf')


if __name__=='__main__':
    data=ROOT/'circle/data/division_dynamics_20261114';data.mkdir(parents=True,exist_ok=True)
    out=ROOT/'figures/circle/division_dynamics_20261114';out.mkdir(parents=True,exist_ok=True)
    path=data/'results.json.gz'
    if '--plot-only' in sys.argv:
        result=json.load(gzip.open(path,'rt'))
    else:
        result=run()
        with gzip.open(path,'wt') as handle:json.dump(result,handle)
    (out/'summary.json').write_text(json.dumps(summarize(result),indent=2)+'\n')
    plot(result,out)
    print(json.dumps(summarize(result),indent=2),flush=True)
