"""Figures and transparent outcome tables for feedback and synchronous division."""
import os,sys,json,gzip,csv,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False,'savefig.dpi':150})
COLORS=['#386cb0','#e66a35','#2a9d8f','#985ca3']
OUT=ROOT/'figures/circle/biology_20261114'


def load(folder):
    manifest=json.loads((folder/'manifest.json').read_text());records=[]
    for key in manifest['tasks']:
        with gzip.open(folder/'runs'/f'{key}.json.gz','rt') as f:records.append(json.load(f))
    return records


def save(fig,name,title,caption,gallery):
    fig.suptitle(title,fontsize=14);fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(OUT/f'{name}.png');fig.savefig(OUT/f'{name}.pdf');plt.close(fig)
    gallery.append(dict(name=name,title=title,caption=caption))


def line(ax,t,curves,label,color):
    a=np.asarray(curves);mean=a.mean(axis=0);sd=a.std(axis=0,ddof=1) if len(a)>1 else np.zeros_like(mean)
    ax.plot(t,mean,label=label,color=color);ax.fill_between(t,mean-sd,mean+sd,color=color,alpha=.15)


def feedback_figures(records,gallery):
    rr=[r for r in records if r['task']['family']=='feedback'];ok=[r for r in rr if r['accepted']]
    fig,axes=plt.subplots(1,3,figsize=(13,4.3))
    for j,(gain,delay) in enumerate(((0.,0.),(.5,0.),(.5,2.),(.5,10.))):
        group=[r for r in ok if r['task']['gain']==gain and r['task']['delay']==delay]
        r=next((r for r in group if r['task']['seed']==1),None)
        if r is None:continue
        t=np.array([s['time'] for s in r['samples']]);q=np.array([s['shape'] for s in r['samples']])
        label=f'g={gain:g}, delay={delay:g}; {len(group)}/3 seeds'
        axes[0].plot(t,q.mean(axis=1),label=label,color=COLORS[j])
        axes[1].plot(t,[s['tension_rms'] for s in r['samples']],color=COLORS[j])
        blocks=[x['summary']['spectrum']['local_shape_rms_blocks'] for x in group]
        line(axes[2],np.arange(4)+1,blocks,label,COLORS[j])
    axes[0].set(xlabel='Time / τR',ylabel='Mean shape index');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Time / τR',ylabel='RMS(T − Tfixed) at contacts')
    axes[2].set(xlabel='Successive quarter of the run',ylabel='Local temporal shape RMS',yscale='log',xticks=[1,2,3,4])
    save(fig,'01_delayed_feedback','Delayed contact feedback can sustain motion after a single perturbation',
        'No ongoing stochastic forcing: μ=0. A common deterministic tension kick initializes each run. Left and middle are seed 1; right averages local shape variability over completed seeds (bands: seed SD). Gain g multiplies tanh(delayed relative contact stretch / 0.1). The zero-delay case is an essential control. This is an explicit extension of the original tension law.',gallery)
    fig,axes=plt.subplots(1,3,figsize=(12,4.3));gains=(.5,1.,2.,4.);delays=(0.,2.,10.)
    for ax,metric,title in zip(axes,('rms','rate','overlap'),('Late detrended local shape RMS','Neighbor-changing transactions / cell / τR','Final initial-neighbor overlap')):
        grid=np.full((4,3),np.nan);counts=np.zeros((4,3),int)
        for i,g in enumerate(gains):
            for j,d in enumerate(delays):
                group=[r for r in ok if r['task']['gain']==g and r['task']['delay']==d];counts[i,j]=len(group)
                if group:
                    values=[r['summary']['spectrum']['late_local_detrended_rms'] if metric=='rms' else r['summary']['event_rate'] if metric=='rate' else r['samples'][-1]['generation_neighbor_overlap'] for r in group]
                    grid[i,j]=np.mean(values)
        norm=LogNorm(vmin=max(np.nanmin(grid),1e-6),vmax=np.nanmax(grid)) if metric=='rms' else None
        im=ax.imshow(np.ma.masked_invalid(grid),origin='lower',cmap='viridis',aspect='auto',norm=norm)
        for (i,j),n in np.ndenumerate(counts):
            text=f'{grid[i,j]:.3g}\n{n}/3' if n else 'failed\n0/3'
            ax.text(j,i,text,ha='center',va='center',color=('white' if im.norm(grid[i,j])<.6 else 'black') if n else 'black',fontsize=8)
        ax.set(xticks=range(3),xticklabels=delays,yticks=range(4),yticklabels=gains,xlabel='Delay / τR',ylabel='Feedback gain g / T₀',title=title)
        fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,'02_feedback_map','Feedback response map with numerical failures kept visible',
        'N=16, ρ=1.2, W/T₀=1, τT=10τR. Each tile uses only completed runs and reports accepted/attempted seeds. Failed geometries are not assigned a physical phase. The fluctuation map uses a logarithmic color scale. A spectral peak or large RMS alone does not establish an oscillatory phase.',gallery)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for j,d in enumerate((0.,2.,10.)):
        group=[r for r in ok if r['task']['gain']==.5 and r['task']['delay']==d]
        if not group:continue
        spectra=[r['summary']['spectrum'] for r in group];frequency=np.array(spectra[0]['frequency'])
        powers=[np.array(s['power'])/max(np.sum(s['power']),1e-30) for s in spectra]
        line(axes[0],frequency[1:],np.array(powers)[:,1:],f'delay={d:g}',COLORS[j]);axes[0].set_xlim(0,.3)
        r=next((r for r in group if r['task']['seed']==1),group[0])
        if d==10:
            q=np.array([s['shape'] for s in r['samples']]);t=np.array([s['time'] for s in r['samples']]);chosen=np.argsort(q[len(q)//2:].var(axis=0))[-4:]
            for c in chosen:axes[1].plot(t,q[:,c],label=f'cell UID {r["samples"][0]["uids"][c]}',lw=1)
    axes[0].set(xlabel='Frequency × τR',ylabel='Normalized local shape power');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Time / τR',ylabel='Individual-cell shape index');axes[1].legend(fontsize=7)
    save(fig,'03_oscillation_diagnostics','Oscillation diagnostics: spectra and individual-cell histories',
        'Spectra average power over cells after linear detrending and a Hann window, then normalize per seed; the last half of each run is used. Individual histories show the four most variable cells of the seed-1, g=0.5, delay=10 run, selected for illustration. Their selection is not an unbiased estimate of tissue-wide amplitude.',gallery)


def division_figures(records,gallery):
    rr=[r for r in records if r['task']['family']=='division' and r['accepted']]
    fig,axes=plt.subplots(2,3,figsize=(13,7));conditions=((False,False),(False,True),(True,False),(True,True))
    for j,(grow,divide) in enumerate(conditions):
        group=[r for r in rr if r['task']['growth']==grow and r['task']['divide']==divide]
        if not group:continue
        label=('growth + division' if grow and divide else 'growth only' if grow else 'division only' if divide else 'neither')+f' ({len(group)}/3)'
        t=[s['time'] for s in group[0]['samples']]
        for ax,key in zip(axes.flat,('n','nominal_density','packing','shape','transactions','wall_load')):
            values=[[float(np.mean(s[key])) for s in r['samples']] for r in group]
            line(ax,t,values,label,COLORS[j]);ax.axvline(20,color='gray',ls=':',lw=.8)
    for ax,label in zip(axes.flat,('Cell count','Σ preferred area / wall area','Measured area fraction','Mean shape index','Cumulative spontaneous contact changes','Integrated wall reaction')):
        ax.set(xlabel='Time / τR',ylabel=label)
    axes[0,0].legend(fontsize=7)
    save(fig,'04_synchronous_division_controls','Synchronous division separated from growth-driven crowding',
        'Start with 16 cells at ρ=0.6; wall area stays fixed. Growth doubles every mother’s K over 0–20τR and then stops. At t=20τR, all current mothers divide in one atomic round where enabled. Daughters split K and area equally, conserving total K at cleavage. New daughter contacts start at 2T₀−W. Division itself is excluded from spontaneous contact-change counts; subsequent rearrangements are included. Shape means compare different populations across cleavage.',gallery)
    group=[r for r in rr if r['task']['growth'] and r['task']['divide']]
    if group:
        r=next((r for r in group if r['task']['seed']==1),group[0]);frames=r['frames']
        fig,axes=plt.subplots(1,4,figsize=(14,4))
        norm=plt.Normalize(3.5,4.7)
        for ax,frame in zip(axes,frames):
            center=np.array(frame['center']);p=[np.array(v)-center for v in frame['polygons']];radius=frame['radius']
            collection=PolyCollection(p,cmap='viridis',norm=norm,linewidth=.7,edgecolors='#202020');collection.set_array(np.array(frame['shape']));ax.add_collection(collection)
            ax.add_patch(Circle((0,0),radius,fill=False,color='black'))
            ax.set(xlim=(-1.08*radius,1.08*radius),ylim=(-1.08*radius,1.08*radius),title=f"{frame['label'].replace('_',' ')}\nt={frame['time']:.1f}, N={len(p)}",xlabel='x / ℓ');ax.set_aspect('equal')
        save(fig,'05_division_snapshots','All mothers divide at the same instant inside a fixed wall',
            f"Growth + division, seed {r['task']['seed']}. Colors show shape index on a common 3.5–4.7 scale. Before and after cleavage share the same timestamp and tissue outline; the new internal daughter contacts change cell shapes without moving the old boundary nodes. The simulation takes no mechanical step between mother splits.",gallery)


def refinement_figure(gallery):
    folder=ROOT/'circle/data/biology_refinement_20261114'
    if not (folder/'manifest.json').exists():return []
    records=load(folder);fig,axes=plt.subplots(1,3,figsize=(12,4.3));rows=[]
    for j,(gain,delay) in enumerate(((0.,0.),(.5,2.),(.5,10.))):
        for r in records:
            t=r['task']
            if t['family']!='feedback_refinement' or (t['gain'],t['delay'])!=(gain,delay):continue
            i=0 if t['dt']==.005 else 1
            row=dict(gain=gain,delay=delay,dt=t['dt'],spacing=t['spacing'],history_spacing=t['history_spacing'],accepted=r['accepted'],error=r.get('error',''))
            if r['accepted']:
                s=r['summary']['spectrum'];row.update(rms=s['late_local_detrended_rms'],peak_period=s['peak_period'],last_block_rms=s['local_shape_rms_blocks'][-1])
                axes[0].plot(i,row['rms'],'o',color=COLORS[j],label=f'g={gain}, delay={delay}' if i==0 else None)
                axes[1].plot(i,row['peak_period'],'o',color=COLORS[j])
                axes[2].plot(np.arange(4)+1,s['local_shape_rms_blocks'],color=COLORS[j],ls='-' if i==0 else '--')
            else:axes[0].annotate('failed',(i,1e-3*(j+1)),color=COLORS[j])
            rows.append(row)
    axes[0].set(xticks=[0,1],xticklabels=['baseline','refined'],yscale='log',ylabel='Late detrended local shape RMS');axes[0].legend(fontsize=7)
    axes[1].set(xticks=[0,1],xticklabels=['baseline','refined'],ylabel='Dominant finite-window period / τR')
    axes[2].set(xticks=[1,2,3,4],xlabel='Successive quarter of 300τR run',ylabel='Local temporal shape RMS',yscale='log')
    save(fig,'06_feedback_refinement','Longer observations with timestep, film and delay-history refinement',
        'Seed 1, selected after the pilot. Baseline dt=0.005, film spacing=0.1, history spacing=0.05; refined values are 0.0025, 0.05 and 0.025. Solid block curves are baseline; dashed are refined. Preparation is repeated at each resolution, so pointwise trajectories need not match. These are sensitivity checks, not a demonstrated convergence order. A nonoscillating control also has a largest Fourier bin; its reported period is not a physical oscillation.',gallery)
    return rows


def combined_figure(records,gallery):
    folder=ROOT/'circle/data/biology_combined_20261114'
    if not (folder/'manifest.json').exists():return
    combined=load(folder);fig,axes=plt.subplots(1,3,figsize=(12,4))
    control=[r for r in records if r['accepted'] and r['task']['family']=='division' and r['task']['growth'] and r['task']['divide']]
    for j,(group,label) in enumerate(((control,'growth + division'),([r for r in combined if r['accepted']],'+ delayed feedback'))):
        if not group:continue
        t=[s['time'] for s in group[0]['samples']]
        for ax,key in zip(axes,('shape','transactions','wall_load')):
            line(ax,t,[[float(np.mean(s[key])) for s in r['samples']] for r in group],f'{label} ({len(group)}/3)',COLORS[j])
            ax.axvline(20,color='gray',ls=':',lw=.8)
    for ax,label in zip(axes,('Mean shape index','Cumulative spontaneous contact changes','Integrated wall reaction')):ax.set(xlabel='Time / τR',ylabel=label)
    axes[0].legend(fontsize=8)
    save(fig,'07_feedback_and_division','Combining delayed feedback with synchronous proliferation',
        'Matched growth/division protocol and seeds, adding g=0.5 and delay=2τR. Both groups use μ=0.05 and divide all mothers at 20τR. Bands are seed SD. This combined three-seed pilot was selected after the component tests and has not received its own mesh-refinement study. Increased event counts may include repeated reversible contact changes.',gallery)
    with (OUT/'combined_summary.csv').open('w',newline='') as f:
        fields=['seed','accepted','event_count','postdivision_events','shape_final','packing_final','error'];writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in combined:
            writer.writerow(dict(seed=r['task']['seed'],accepted=r['accepted'],event_count=r.get('summary',{}).get('event_count'),
                postdivision_events=sum(e['time']>20+1e-8 for e in r.get('events',[])),shape_final=r.get('summary',{}).get('shape_final'),packing_final=r.get('summary',{}).get('packing_final'),error=r.get('error','')))


def main():
    OUT.mkdir(parents=True,exist_ok=True);records=load(ROOT/'circle/data/biology_20261114');gallery=[]
    feedback_figures(records,gallery);division_figures(records,gallery);refined=refinement_figure(gallery);combined_figure(records,gallery)
    rows=[]
    for r in records:
        t=r['task'];row={key:t[key] for key in ('family','seed','gain','delay','growth','divide')};row.update(accepted=r['accepted'],error=r.get('error',''))
        for key in ('n_final','nominal_density_final','packing_final','shape_final','event_rate'):row[key]=r.get('summary',{}).get(key)
        row['late_shape_rms']=r.get('summary',{}).get('spectrum',{}).get('late_local_detrended_rms');rows.append(row)
    for name,data in (('summary.csv',rows),('refinement.csv',refined)):
        if data:
            fields=list(dict.fromkeys(k for row in data for k in row))
            with (OUT/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(data)
    report=['# Delayed feedback and synchronous division','',f"Main pilot: {sum(r['accepted'] for r in records)}/{len(records)} runs completed. Full regression suite: 170 tests pass. Failures remain excluded from physical averages.",'',
        '## Model choices','',
        'Feedback changes the tension drift explicitly: τT dT/dt = −(T−Tfixed) + g tanh(ε(t−d)/0.1) + μζ. The sensor ε is relative total contact-length change for a persistent pair of cell IDs. Free films have no feedback. References are set when contacts are born; lost/reformed contacts and daughter contacts receive fresh references and zero prehistory. Raw tensions may be negative; the existing Heaviside rule is retained in the mechanical force. Delayed histories use linear interpolation on a fixed sampling grid.','',
        'Division is synchronous and atomic. Plans are constructed from the pre-round tissue; all current mothers divide with no intervening mechanics or random draws. The cleavage normal follows the mother’s major area-moment axis. Deterministic angle adjustments avoid existing junctions and coincident neighboring cleavage sites. A failed split restores the whole round, including cell count, preferred areas, lineage, feedback history, geometry, connectivity, caches and RNG.','',
        'Cleavage preserves total K and splits it in proportion to equal daughter areas, so the osmotic pressure is continuous at that instant. Each daughter keeps the original cortical-tension law; the new shared film starts at 2T₀−W. This is geometric cleavage, without a resolved mitotic rounding or contractile-ring stage. Growth changes K separately. All coordinate, mobility, pressure and time scales remain fixed to the founding cells. The circular wall never expands.','',
        '## Interpretation limits','',
        'Feedback-only runs have no ongoing tension noise; division pilots use μ=0.05. The existing seeded random ordering of eligible topology operations is retained. Growth and division are independently controlled. New daughter identities and cell count changes are not counted as spontaneous neighbor exchanges. Neighbor overlap is restarted at each division round. Finite-window spectral peaks alone do not establish sustained oscillations; amplitude persistence and numerical sensitivity must be assessed alongside them. Strong feedback can still produce invalid crossings.','',
        'Validation runs every 0.1τR during evolution, plus a selected every-timestep feedback audit; preparation keeps its established 0.5τR cadence and the existing local microfold tolerance. Three seeds per pilot point, one selected seed for resolution checks. Bands show seed SD; a single accepted seed has no estimated seed uncertainty. No physical phases are assigned to numerical failures.','',
        'Raw records are in circle/data/biology_20261114 and circle/data/biology_refinement_20261114. See circle/BIOLOGY.md for implementation, equations and reproduction.','']
    for item in gallery:report += ['## '+item['title'],'',f"![{item['title']}]({item['name']}.png)",'',item['caption'],'']
    (OUT/'REPORT.md').write_text('\n'.join(report).rstrip()+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>Feedback and synchronous division</title><style>body{max-width:1250px;margin:35px auto;padding:0 24px;font:17px/1.5 system-ui;color:#20303d}img{max-width:100%}section{margin:40px 0}</style><h1>Delayed feedback and synchronous division</h1>',f'<p>{sum(r["accepted"] for r in records)}/{len(records)} main runs completed. Fixed circular wall. Explicit model extensions; failed geometries remain unclassified.</p><p><a href="FINDINGS.md">Findings</a> · <a href="REPORT.md">Methods</a> · <a href="summary.csv">Data summary</a></p>']
    for item in gallery:page.append(f'<section><h2>{html.escape(item["title"])}</h2><a href="{item["name"]}.pdf">PDF</a><img src="{item["name"]}.png"><p>{html.escape(item["caption"])}</p></section>')
    for name in ('synchronous_division','delayed_feedback'):
        if (OUT/f'{name}.gif').exists():page.append(f'<section><h2>{name.replace("_"," ").title()}</h2><img src="{name}.gif"></section>')
    (OUT/'index.html').write_text('\n'.join(page));print(report[2])


if __name__=='__main__':main()
