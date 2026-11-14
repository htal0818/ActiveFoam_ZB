"""Render completed pilots and retain all failure denominators."""
import json
from pathlib import Path
from circle.scripts.division_contact_analysis import persistent_changes
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'circle/data/collective_dynamics_20261114'
OUT=ROOT/'figures/circle/collective_dynamics_20261114'
COLORS=['#007f86','#cb7835','#7653a6']


def load(group):
    return [json.loads(p.read_text()) for p in sorted((DATA/group).rglob('*.json')) if '.partial.' not in p.name and p.name!='replay_audit.json']

def save(fig,name):
    for ax in fig.axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/f'{name}.png',dpi=180);fig.savefig(OUT/f'{name}.pdf');plt.close(fig)

def divisions():
    runs=[r for r in load('division') if r['status']=='complete']
    if not runs:return []
    runs.sort(key=lambda r:r['parameters']['interval'])
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    summary=[]
    for index,r in enumerate(runs):
        interval=r['parameters']['interval'];color=COLORS[index];s=r['stages'][-1]['run'];skip=r['skip_final']
        label=f'Interval {interval:g} τR'
        rates=[len(x['changes'])/(x['n']*x['duration']) for x in [s,skip]]
        ax[0,0].bar(index-.17,rates[0],.32,color=color)
        ax[0,0].bar(index+.17,rates[1],.32,color=color,alpha=.35,hatch='//')
        for a,key in [(ax[0,1],'contact_survival'),(ax[1,0],'shape'),(ax[1,1],'residual_msd')]:
            a.plot([x['time'] for x in s['records']],[x[key] for x in s['records']],color=color,label=label)
        summary.append(dict(interval=interval,divide_changes=len(s['changes']),skip_changes=len(skip['changes']),
            divide_rate=rates[0],skip_rate=rates[1],shape_initial=s['records'][0]['shape'],
            shape_final=s['records'][-1]['shape'],survival=s['records'][-1]['contact_survival'],
            residual_msd=s['records'][-1]['residual_msd'],preferred_total=r['preferred_total'],
            dwell_sensitivity={str(d):{label:len(persistent_changes(x['initial_pairs'],x['events'],0.,x['duration'],d))
                for label,x in [('divide',s),('skip',skip)]} for d in [.1,.2,.5]},
            max_tolerated_microfolds=max(q.get('local_film_microfolds',0) for stage in r['stages'] for q in stage['run']['checks']),
            max_resolved_crossings=max(q.get('resolved_film_crossings',0) for stage in r['stages'] for q in stage['run']['checks'])))
    ax[0,0].set(xticks=range(len(runs)),xticklabels=[f"{r['parameters']['interval']:g}" for r in runs],
        xlabel='Division interval / τR',ylabel='Persistent pair changes / cell / τR',
        title='Final round: divide (solid) vs skip (hatched)')
    ax[0,1].set(ylabel='Survival of post-division contacts',ylim=(0,1.02),title='Rearrangement persists beyond septum insertion')
    ax[1,0].set(ylabel='Mean measured shape index',title='Post-division shape relaxation')
    ax[1,1].set(ylabel='Residual MSD / initial mean cell area',title='Translation, rotation and contraction removed')
    for a in [ax[0,1],ax[1,0],ax[1,1]]:a.set_xlabel('Time after final division / τR')
    ax[0,1].legend()
    fig.suptitle('Synchronous division without growth: 8 → 256\nOne matched founder state; five rounds; dwell ≥ 0.2 τR',fontsize=14)
    save(fig,'division_timing')
    # Compare identical final cell numbers at identical elapsed times after division.
    fig,axes=plt.subplots(1,len(runs),figsize=(5*len(runs),5),squeeze=False,constrained_layout=True)
    for ax,r in zip(axes[0],runs):
        frame=r['frames'][-1];center=np.array(frame['center']);radius=frame['radius']
        from matplotlib.collections import PolyCollection
        pc=PolyCollection([np.array(p)-center for p in frame['polygons']],array=np.array(frame['shape']),
                          cmap='viridis',edgecolors='#16393c',linewidths=.35,clim=(3.6,4.5))
        ax.add_collection(pc);ax.add_patch(plt.Circle((0,0),radius,fill=False,color='black',lw=1.5))
        ax.set(xlim=(-1.05*radius,1.05*radius),ylim=(-1.05*radius,1.05*radius),aspect='equal',
               title=f"Interval {r['parameters']['interval']:g} τR; N=256")
        ax.axis('off')
    fig.colorbar(pc,ax=axes.ravel().tolist(),label='Measured shape index',shrink=.7)
    fig.suptitle('Four relaxation times after the fifth division; fixed wall and total preferred area')
    save(fig,'division_snapshots')
    return summary

def activity():
    runs=load('activity');ok=[r for r in runs if r['status']=='complete' and r['parameters']['mu_reference']==.25]
    summary=[]
    if not ok:return summary
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for di,domain in enumerate(['disc','periodic']):
        for ti,tau in enumerate([2.,10.,20.]):
            chosen=[r for r in ok if r['parameters']['domain']==domain and r['parameters']['tau']==tau]
            for j,r in enumerate(chosen):
                run=r['run'];p=r['parameters'];rows=run['records'];color=COLORS[ti]
                label=f'τT={tau:g}' if j==0 else None
                time=np.array([x['time'] for x in rows])
                for ax,key in [(axes[di,0],'persistent_changes'),(axes[di,1],'contact_survival'),(axes[di,2],'msd')]:
                    vals=np.array([x[key] for x in rows])
                    if key=='persistent_changes':vals=vals/run['n']
                    ax.plot(time,vals,color=color,ls='-' if p['seed']==1 else '--',label=label,alpha=.85)
                summary.append(dict(domain=domain,seed=p['seed'],tau=tau,mu=p['mu'],reference_mu=p['mu_reference'],
                    changes=len(run['changes']),rate=len(run['changes'])/(run['n']*run['duration']),
                    survival=rows[-1]['contact_survival'],msd=rows[-1]['msd'],
                    tension_variance=float(np.mean([x['tension_variance'] for x in rows]))))
        for j,r in enumerate(x for x in runs if x['status']=='complete' and x['parameters']['domain']==domain and x['parameters']['mu_reference']==0):
            rows=r['run']['records'];time=[x['time'] for x in rows]
            for ax,key in [(axes[di,0],'persistent_changes'),(axes[di,1],'contact_survival'),(axes[di,2],'msd')]:
                values=np.array([x[key] for x in rows])
                if key=='persistent_changes':values=values/r['run']['n']
                ax.plot(time,values,color='gray',ls=':',label='Zero noise' if j==0 else None,lw=1)
        axes[di,0].set(ylabel='Persistent pair changes / cell',title=f'{domain.capitalize()}: rearrangement')
        axes[di,1].set(ylabel='Original-contact survival',ylim=(0,1.02),title='Persistent loss, not brief toggles')
        axes[di,2].set(ylabel='MSD / initial mean cell area',title='Centroid displacement\nTranslation removed; periodic unwrapped')
        for ax in axes[di]:ax.set_xlabel('Observation time / τR')
        if axes[di,0].lines:axes[di,0].legend(fontsize=8)
    fig.suptitle('Tension persistence at fixed raw OU variance (reference ΔT=0.25 at τT=10)\nSolid: seed 1; dashed: seed 2; dotted: zero noise',fontsize=14)
    save(fig,'tension_persistence')
    return summary

def mechanics():
    allruns=load('mechanics');runs=[r for r in allruns if r['status']=='complete']
    if not runs:return []
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True);summary=[]
    for r in runs:
        p=r['parameters'];zone=p['zone'];col=COLORS[['interior','wall','periodic'].index(zone)]
        label=f"{zone}, seed {p['seed']}, quench {p['quench_tauT']:g} τT"
        curve=r['curve'];s=curve['curve'][0]['separation']-np.array([x['separation'] for x in curve['curve']])
        f=np.array([x['force'] for x in curve['curve']])-curve['baseline_force']
        axes[0,0].plot(s,f,color=col,alpha=.6,label=label)
        forced=r['branches']['forced'];sham=r['branches']['sham'];direct=set(map(tuple,r['direct_pairs']))
        def secondary(run,t):return sum(x['time']<=t and tuple(x['pair']) not in direct for x in run['changes'])
        time=np.array([x['time'] for x in forced['records']])
        excess=np.array([secondary(forced,t)-secondary(sham,t) for t in time])
        axes[0,1].plot(time,excess,color=col,alpha=.6)
        delta=np.array(forced['records'][-1]['cell_displacement'])-np.array(sham['records'][-1]['cell_displacement'])
        axes[1,0].scatter(forced['distance'],np.linalg.norm(delta,axis=1),color=col,s=9,alpha=.4)
        fac=[]
        for z in r['facilitation']:
            a=z['curves']['after'];b=z['curves']['before'];c=z['curves']['sham']
            if all(v['complete'] for v in [a,b,c]):
                difference=a['excess_work_per_shortening']-b['excess_work_per_shortening']
                unforced=a['excess_work_per_shortening']-c['excess_work_per_shortening']
                axes[1,1].scatter(z['distance'],difference,color=col,marker='o',label='After − loaded' if not summary and not fac else None)
                axes[1,1].scatter(z['distance'],unforced,edgecolors=col,facecolors='none',marker='D',label='After − unforced' if not summary and not fac else None)
                fac.append(dict(zone=z['zone'],distance=z['distance'],difference=difference,difference_unforced=unforced,
                    before=b['excess_work_per_shortening'],after=a['excess_work_per_shortening'],sham=c['excess_work_per_shortening']))
        summary.append(dict(**p,outcome=r['outcome'],work=curve['work'],
            work_per_shortening=curve['work_per_shortening'],excess_work_per_shortening=curve['excess_work_per_shortening'],
            baseline_force=curve['baseline_force'],old_pair=curve['old_pair'],shape=curve['shape_mean'],
            secondary_excess=int(excess[-1]),facilitation=fac))
    axes[0,0].set(xlabel='Contact shortening',ylabel='Holding force minus baseline',title='Mechanical resistance')
    axes[0,0].legend(fontsize=6,ncol=2)
    axes[0,1].set(xlabel='Time after release / τR',ylabel='Excess persistent secondary pair changes',title='Forced contact opening minus unforced sham')
    axes[1,0].set(xlabel='Initial distance from forced contact',ylabel='Magnitude of paired displacement difference',title='Spatial extent of release response')
    axes[1,1].set(xlabel='Distance of test contact from forced contact',ylabel='Change in excess work / shortening',title='Local 10% shortening: both loading controls')
    axes[1,1].axhline(0,color='gray',lw=.8)
    if axes[1,1].get_legend_handles_labels()[0]:axes[1,1].legend(fontsize=8)
    fig.suptitle('Contact opening, local resistance and boundary controls\nSame equations; incomplete probes excluded, not assigned zero',fontsize=14)
    save(fig,'local_rearrangement')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
    for zi,zone in enumerate(['interior','wall','periodic']):
        for seed in [1,2]:
            rows=sorted([r for r in summary if r['zone']==zone and r['seed']==seed],key=lambda r:r['quench_tauT'])
            if not rows:continue
            same=len({tuple(r['old_pair']) for r in rows})==1
            for ax,key in zip(axes,['excess_work_per_shortening','baseline_force','shape']):
                ax.plot([r['quench_tauT'] for r in rows],[r[key] for r in rows],
                    marker='o' if same else 'x',ls='-' if seed==1 else '--',color=COLORS[zi],label=f'{zone}, seed {seed}')
                ax.set_xlabel('Quench duration / τT');ax.set_xticks([1,5])
    for ax,label,title in zip(axes,['Excess work / shortening','Initial holding force','Mean measured shape index'],
                             ['Boundary and preparation response','Residual preparation stress','Shape is an output']):
        ax.set(ylabel=label,title=title)
    axes[0].legend(fontsize=7)
    fig.suptitle('Selected-contact pilot: lines pair seeds; × marks changed contact identity')
    save(fig,'boundary_preparation')
    return summary

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=dict(division=divisions(),activity=activity(),mechanics=mechanics(),failures={})
    for group in ['division','activity','mechanics']:
        rows=load(group)
        summary['failures'][group]=[dict(parameters=r['parameters'],status=r['status'],error=r.get('error'))
            for r in rows if r['status']!='complete']
        summary[group+'_counts']=dict(total=len(rows),complete=sum(r['status']=='complete' for r in rows))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k.endswith('_counts')},indent=2))
if __name__=='__main__':main()
