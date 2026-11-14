"""Distinguish persistent contact changes from reversible topology toggles."""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def persistent_changes(initial, events, start, end, dwell):
    """Debounce each contact independently using a minimum continuous dwell.

    Initial states are the post-division graph. A switch is reported only after
    the new raw state has lasted dwell time; shorter reversals leave the filtered
    graph unchanged. Times are confirmation times, not backdated event times.
    The last dwell interval is right-censored by the end of observation.
    """
    if dwell <= 0:
        raise ValueError('dwell must be positive')
    initial = {tuple(p) for p in initial}
    timelines = {pair: [(start, True)] for pair in initial}
    state = set(initial)
    for event in events:
        t = event['time']
        if t > end + 1e-8:
            break
        for value, name in [(False, 'lost'), (True, 'gained')]:
            for pair in event[name]:
                pair = tuple(pair)
                assert (pair in state) != value, ('inconsistent event stream',pair,t,value)
                timelines.setdefault(pair, [(start, False)]).append((t,value))
                if value:
                    state.add(pair)
                else:
                    state.remove(pair)
    changes=[]
    for pair, sequence in timelines.items():
        stable = pair in initial
        for (t,value),(stop,_) in zip(sequence,sequence[1:]+[(end,None)]):
            if value != stable and stop-t >= dwell-1e-9:
                stable=value
                changes.append(dict(time=t+dwell,pair=pair,present=value))
    return sorted(changes,key=lambda e:(e['time'],e['pair']))


def filtered_record(run, dwell):
    initial={tuple(p) for p in run['records'][0]['pairs']}
    changes=persistent_changes(initial,run['events'],run['time'],run['end'],dwell)
    stable=set(initial);lost_ever=set();records=[];cursor=0
    for row in run['records']:
        while cursor<len(changes) and changes[cursor]['time']<=row['time']+1e-9:
            change=changes[cursor];pair=tuple(change['pair'])
            if change['present']:
                stable.add(pair)
            else:
                stable.remove(pair);lost_ever.add(pair)
            cursor+=1
        records.append(dict(lag=row['lag'],survival=len(initial-lost_ever)/len(initial) if initial else None,
                            retention=len(initial&stable)/len(initial) if initial else None))
    return dict(changes=changes,records=records)


def analyze(result):
    summaries=[]
    for dwell in [.1,.2,.5]:
        rows=[]
        for stage,control in zip(result['stages'],result['controls']):
            row=dict(before=stage['before'],after=stage['after'],dwell=dwell)
            for label,run in [('divide',stage),('skip',control)]:
                filtered=filtered_record(run,dwell)
                first=[e for e in filtered['changes'] if e['time']<=run['time']+4+1e-8]
                late=[e for e in filtered['changes'] if e['time']>run['end']-10-1e-8]
                near4=min(filtered['records'],key=lambda r:abs(r['lag']-4))
                raw4=[e for e in run['events'] if e['time']<=run['time']+4+1e-8]
                row[label]=dict(persistent_changes_first4=len(first),
                    raw_transactions_first4=len(raw4),
                    raw_pair_changes_first4=sum(len(e['lost'])+len(e['gained']) for e in raw4),
                    persistent_survival_4=near4['survival'],
                    final_persistent_survival=filtered['records'][-1]['survival'],
                    final_instantaneous_retention=run['records'][-1]['retention'],
                    persistent_changes_last10=len(late) if run['end']-run['time']>=10 else None,
                    final_residual_msd=run['records'][-1]['residual_msd_over_area'])
            rows.append(row)
        summaries.append(dict(dwell=dwell,rounds=rows))
    return summaries


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dwell=.2;summary=analyze(result)[1]['rounds']
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    x=np.arange(5)
    for label,offset,color,title in [('divide',-.18,'#278f91','Divide'),('skip',.18,'#d89a43','Skip this division')]:
        rates=[r[label]['persistent_changes_first4']/(r['after' if label=='divide' else 'before']*4) for r in summary]
        axes[0,0].bar(x+offset,rates,width=.36,color=color,label=title)
    axes[0,0].set(xticks=x,xticklabels=[f"{r['before']}→{r['after']}" for r in summary],
        ylabel='Persistent pair changes / cell / τR',title='First 4 τR after each division')
    axes[0,0].legend(fontsize=9)
    bins=np.arange(0,24.01,1.)
    for run,n,label,color in [(result['stages'][-1],32,'Divide: 32 cells','#278f91'),
                              (result['controls'][-1],16,'Skip: 16 cells','#d89a43')]:
        filtered=filtered_record(run,dwell)
        histogram,_=np.histogram([e['time']-run['time'] for e in filtered['changes']],bins)
        axes[0,1].stairs(histogram/n,bins,color=color,label=label,lw=2)
        lag=[r['lag'] for r in filtered['records']]
        axes[1,0].plot(lag,[r['survival'] for r in filtered['records']],color=color,label=label)
        axes[1,1].plot(lag,[r['residual_msd_over_area'] for r in run['records']],color=color,label=label)
    axes[0,1].set(ylabel='Persistent pair changes / cell / τR',title='After the final division: no further driving')
    axes[0,1].legend(fontsize=9)
    axes[1,0].set(ylabel='Fraction without a persistent loss',ylim=(-.03,1.03),title='Survival of post-division contacts')
    axes[1,1].set(ylabel='Residual MSD / initial mean area',title='Motion with global contraction removed')
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:
        ax.set_xlabel('Time since last scheduled division / τR')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Do divisions cause lasting neighbor changes?\n'
                 'Division excluded; new contact state must persist ≥ 0.2 τR',fontsize=15)
    fig.savefig(out/'persistent_division_dynamics.png',dpi=180)
    fig.savefig(out/'persistent_division_dynamics.pdf')


if __name__=='__main__':
    path=ROOT/'circle/data/division_dynamics_20261114/results.json.gz'
    result=json.load(gzip.open(path,'rt'))
    out=ROOT/'figures/circle/division_dynamics_20261114'
    summary=analyze(result)
    (out/'persistent_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(result,out)
    print(json.dumps(summary[1],indent=2))
