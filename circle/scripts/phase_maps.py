"""Sampled contact, shape and packing maps; no interpolated phase boundaries."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from circle.sweep import aggregate

ACTIVITY=r'$\Delta T/T_0$ (code $\mu$)'
CURVATURE=r'Wall curvature $\kappa\ell$'
LABELS={'z':'Mean cell contacts z','z_bulk':'Interior cell contacts z',
        'q':'Mean shape index P/√A','q_bulk':'Interior shape index P/√A',
        'phi':'Area fraction φ','exchange':'Neighbor changes / cell / τT',
        't1':'T1 operations / cell / τT'}


def load(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    # Missing records are errors: unfinished runs must not masquerade as failed runs.
    return manifest,[json.loads((directory/'runs'/f'{key}.json').read_text()) for key in manifest['tasks']]


def normalize(records):
    out=[]
    for r in records:
        if r['task']['domain']!='disc':continue
        v={}
        if r['accepted']:
            if r['task']['mode']=='structure':
                s=r['structure']
                v=dict(z=s['z_all'],z_bulk=s['z_interior'],q=s['shape_all'],
                       q_bulk=s['shape_interior'],phi=s['packing'])
            else:
                d=r['dynamics']
                if 'neighbor_mean' not in d or 'packing_mean' not in d:
                    raise ValueError('Active maps require contact number and packing measured during dynamics.')
                v=dict(z=d['neighbor_mean']['all'],z_bulk=d['neighbor_mean']['bulk'],
                       q=d['shape_mean'],q_bulk=float(np.mean(d['shape_series']['bulk'][1:]))
                       if d['zone_counts']['bulk'] else None,phi=d['packing_mean'],
                       exchange=d['contact_change_rate']['all']*d['tauT'],t1=d['t1_rate']*d['tauT'])
        out.append(dict(task=r['task'],accepted=r['accepted'],values=v,error=r.get('error'),stage=r['stage']))
    return out


def select(records,**criteria):
    return [r for r in records if all(r['task'][k]==v for k,v in criteria.items())]


def limits(records,metric):
    keys=('z','z_bulk') if metric.startswith('z') else ('q','q_bulk') if metric.startswith('q') else (metric,)
    values=[r['values'][k] for r in records for k in keys if r['accepted'] and r['values'].get(k) is not None]
    if not values:return 0.,1.
    lo,hi=min(values),max(values);pad=max((hi-lo)*.03,.001)
    return lo-pad,hi+pad


def draw(ax,records,sizes,xvalues,xfield,fixed,metric,color_limits):
    ordered=sorted(sizes,reverse=True)
    values=np.full((len(ordered),len(xvalues)),np.nan)
    stats={}
    for i,n in enumerate(ordered):
        for j,x in enumerate(xvalues):
            group=select(records,**(fixed|{'n':n,xfield:x}))
            stat=aggregate(group,'values.'+metric);stats[i,j]=(stat,group)
            if stat['mean'] is not None:values[i,j]=stat['mean']
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e6e8eb')
    im=ax.imshow(values,origin='lower',cmap=cmap,vmin=color_limits[0],vmax=color_limits[1],aspect='auto',interpolation='nearest')
    for (i,j),(stat,group) in stats.items():
        if stat['n']:
            digits=3 if metric.startswith('q') or metric in ('phi','exchange','t1') else 2
            text=f"{stat['mean']:.{digits}f}\n{stat['n']}/{stat['attempted']}"
            ax.text(j,i,text,ha='center',va='center',fontsize=8,color='white',
                    bbox=dict(facecolor='black',alpha=.38,edgecolor='none',pad=2))
        else:
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='none',edgecolor='#b7bbc2',hatch='///',lw=0))
            text='no interior' if metric.endswith('bulk') and any(r['accepted'] for r in group) else 'failed'
            if not group:text='not sampled'
            ax.text(j,i,text,ha='center',va='center',fontsize=8,color='#333')
    ax.set(xticks=range(len(xvalues)),xticklabels=[f'{x:g}' for x in xvalues],
           yticks=range(len(ordered)),yticklabels=[f'{np.sqrt(np.pi/n):.3f}\nN={n}' for n in ordered],
           ylabel=CURVATURE,xlabel={'w':r'Adhesion $W/T_0$','mu':ACTIVITY,'rho':r'Nominal density $\rho$'}[xfield])
    return im


def save(fig,out,name,title):
    fig.suptitle(title,fontsize=14)
    fig.text(.5,.012,'Sampled-grid means; labels show accepted/attempted seeds. Hatching: no usable estimate. '
             'κℓ=√(π/N), ℓ=√(A_T/N); curvature and size are coupled.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.94))
    fig.savefig(out/f'{name}.png',dpi=180);fig.savefig(out/f'{name}.pdf');plt.close(fig)


def overview(records,out,sizes):
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for row,metric in enumerate(['z','q']):
        for col,(mode,xfield,xvalues,fixed) in enumerate([
            ('structure','w',[.5,1.,1.5],{'rho':1.2,'mu':0.}),
            ('dynamics','mu',[0.,.5,1.,1.5],{'rho':1.3,'w':1.})]):
            ax=axes[row,col]
            im=draw(ax,records,sizes,xvalues,xfield,fixed|{'mode':mode},metric,limits(records,metric))
            ax.set_title(LABELS[metric]+(' • passive, ρ=1.2' if col==0 else ' • active, ρ=1.3, W/T₀=1'))
            fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,out,'phase_diagrams_overview','Cell contacts and shape index versus wall curvature, adhesion and tension fluctuations')


def slices(records,out,sizes,densities,mode,metric):
    active=mode=='dynamics';xfield='mu' if active else 'w'
    xvalues=[0.,.5,1.,1.5] if active else [.5,1.,1.5]
    metrics=[metric,metric+'_bulk'] if metric in ('z','q') else [metric]
    fig,axes=plt.subplots(len(metrics),len(densities),figsize=(4.2*len(densities),4.2*len(metrics)),squeeze=False)
    for row,key in enumerate(metrics):
        for col,rho in enumerate(densities):
            ax=axes[row,col];fixed={'mode':mode,'rho':rho}|({'w':1.} if active else {'mu':0.})
            im=draw(ax,records,sizes,xvalues,xfield,fixed,key,limits(records,key))
            ax.set_title(f'ρ={rho:g} • '+('interior' if key.endswith('bulk') else 'all cells'))
            fig.colorbar(im,ax=ax,shrink=.8)
    prefix={'z':'contacts','q':'shape_index','phi':'packing_fraction','exchange':'neighbor_exchange'}[metric]
    control='tension_fluctuations' if active else 'adhesion'
    name=f'{prefix}_curvature_{control}'
    save(fig,out,name,LABELS[metric]+' versus wall curvature and '+control.replace('_',' ')+
         (' • W/T₀=1' if active else ' • zero activity'))
    return name


def density_maps(records,out,sizes,densities,mode):
    active=mode=='dynamics';controls=[0.,.5,1.,1.5] if active else [.5,1.,1.5]
    fig,axes=plt.subplots(3,len(controls),figsize=(4.2*len(controls),11),squeeze=False)
    for row,metric in enumerate(['z','q','phi']):
        for col,control in enumerate(controls):
            ax=axes[row,col];fixed={'mode':mode}|({'mu':control,'w':1.} if active else {'w':control,'mu':0.})
            im=draw(ax,records,sizes,densities,'rho',fixed,metric,limits(records,metric))
            ax.set_title(('ΔT/T₀' if active else 'W/T₀')+f'={control:g} • '+LABELS[metric])
            fig.colorbar(im,ax=ax,shrink=.8)
    name='density_maps_active' if active else 'density_maps_passive'
    save(fig,out,name,'Density and wall curvature • contacts, shape and measured area fraction')
    return name


def packing_correlations(records,out,sizes):
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    colors=['#277da8','#df7931','#6c55a3','#289a79'];markers=['o','s','^','D']
    for row,mode in enumerate(['structure','dynamics']):
        active=mode=='dynamics';field='mu' if active else 'w'
        controls=[0.,.5,1.,1.5] if active else [.5,1.,1.5]
        for i,n in enumerate(sizes):
            for j,control in enumerate(controls):
                group=select(records,mode=mode,n=n,**{field:control})
                for rho in sorted({r['task']['rho'] for r in group}):
                    seeds=select(group,rho=rho);phi=aggregate(seeds,'values.phi')
                    for col,metric in enumerate(['z','q']):
                        y=aggregate(seeds,'values.'+metric)
                        if phi['mean'] is None or y['mean'] is None:continue
                        axes[row,col].errorbar(phi['mean'],y['mean'],xerr=phi['sem'] or 0.,yerr=y['sem'] or 0.,
                            fmt=markers[i%len(markers)],color=colors[j],ms=5,capsize=2,alpha=.85,
                            markerfacecolor='white' if y['n']==1 else colors[j])
        for col,metric in enumerate(['z','q']):
            ax=axes[row,col];ax.set(xlabel='Measured area fraction φ',ylabel=LABELS[metric],
                                   title=('Active' if active else 'Passive')+' • density varied');ax.grid(alpha=.2)
            handles=[Line2D([],[],color=colors[j],marker='o',ls='none',label=('ΔT/T₀' if active else 'W/T₀')+f'={c:g}') for j,c in enumerate(controls) if any(r['accepted'] for r in select(records,mode=mode,**{field:c}))]
            handles += [Line2D([],[],color='#555',marker=markers[i%len(markers)],ls='none',label=f'N={n}') for i,n in enumerate(sizes)]
            ax.legend(handles=handles,fontsize=7,ncol=2)
    save(fig,out,'contacts_shape_vs_volume_fraction','Contacts and shape versus measured packing • error bars: seed SEM; open symbols: one seed')


def write_report(records,out,passive_dir,active_dir,figures):
    fields=['mode','n','w','rho','mu']
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+['curvature','curvature_L0','metric','attempted','accepted','mean','sem'],lineterminator='\n')
        writer.writeheader()
        for condition in sorted({tuple(r['task'][k] for k in fields) for r in records}):
            group=select(records,**dict(zip(fields,condition)));t=group[0]['task']
            for metric in LABELS:
                if t['mode']=='structure' and metric in ('exchange','t1'):continue
                stat=aggregate(group,'values.'+metric)
                writer.writerow(dict(zip(fields,condition))|dict(curvature=np.sqrt(np.pi/t['n']),
                    curvature_L0=np.sqrt(np.pi*t['rho']/t['n']),metric=metric,attempted=stat['attempted'],
                    accepted=stat['n'],mean=stat['mean'],sem=stat['sem']))
    lines=['# Curvature, adhesion, activity and density maps','',
           'These are maps of continuous observables on a sampled grid, not inferred fluid/solid phase boundaries. No interpolation or smoothing fills failed points. Cell contacts means distinct neighbors per cell, not the rate of neighbor exchange. A confluent trivalent interior can remain near six neighbors while cells rearrange; these maps do not impose a universal shape-index threshold for fluidity.','',
           f'Passive source: `{passive_dir}`. Active source: `{active_dir}`. Both use the same force law and geometry acceptance rules; the active measurement was extended to record contact number and packing during dynamics.','',
           'Wall curvature is κ=1/R. Axes use κℓ with ℓ=√(A_T/N), giving κℓ=√(π/N). The CSV also reports κL0=√(πρ/N): at fixed N, varying density changes curvature measured in preferred-cell units. Curvature and finite size remain coupled; the tissue surface is flat.','',
           'Adhesion maps: μ=0, W/T0={0.5,1,1.5}, ρ={1,1.2,1.4}. Activity maps: W/T0=1, ΔT/T0 (code μ)={0,0.5,1,1.5}, ρ={1,1.2,1.3,1.4}. Sizes N={36,100,196}, three seeds per point. The repeated ρ=1.3 trajectories are reproducibility checks and are not pooled as additional independent seeds. The activity label assumes unit white-noise covariance in code time units.','',
           'Passive measurements are preparation endpoints. Active quantities average 50 τR after 10 τR warmup, sampling every 0.5 τR, with τT=10 τR. These are short finite-time averages; stationarity, mesh convergence and long-time phases are not established. Interior means centroid farther than one simulation length unit from the container wall. Dynamic zones are fixed after warmup; passive zones use endpoint geometry. When the tissue detaches, this wall-distance definition can include cells on the tissue’s free surface and is not a pure tissue-bulk classification.','',
           'ρ=N A0/A_T is the control parameter. Area fraction φ=ΣA_i/A_T is measured; it is not independently prescribed. Curves against φ are conditional observations, not sweeps at controlled φ.','',
           'Means exclude failed runs and are conditional on acceptance. Labels show accepted/attempted seeds; SEM is over seed means and is unavailable for one surviving seed. Local junction microfolds remain tolerated and reported in raw data. Sampling geometry every 100 steps does not certify all intermediate states.','',
           '| Mode | N | Accepted / attempted |','|---|---|---|']
    for mode,n in sorted({(r['task']['mode'],r['task']['n']) for r in records}):
        g=select(records,mode=mode,n=n);lines.append(f"| {mode} | {n} | {sum(r['accepted'] for r in g)} / {len(g)} |")
    lines += ['','## Example: active response at N=100, rho=1.3, W/T0=1','',
              '| Activity input | Accepted | Contacts z | Shape P/√A | Area fraction | Neighbor changes / cell / tauT |',
              '|---|---|---|---|---|---|']
    for mu in [0.,.5,1.,1.5]:
        group=select(records,mode='dynamics',n=100,rho=1.3,w=1.,mu=mu)
        if not group:continue
        stats=[aggregate(group,'values.'+k) for k in ('z','q','phi','exchange')]
        values=['—' if s['mean'] is None else f"{s['mean']:.4f}" for s in stats]
        lines.append(f"| {mu:g} | {stats[0]['n']}/{len(group)} | "+' | '.join(values)+' |')
    lines += ['','These are conditional seed means. The CSV gives SEM and counts. Nearly constant mean contact number does not imply absent neighbor exchange.','']
    lines += ['','## Figures','']+[f'- [{name}]({name}.png) ([PDF]({name}.pdf))' for name in figures]
    lines += ['','## Failures','']
    for r in records:
        if not r['accepted']:
            t=r['task'];lines.append(f"- {t['mode']}, N={t['n']}, W={t['w']}, ρ={t['rho']}, μ={t['mu']}, seed={t['seed']}: {r['stage']}: {r['error']}")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'NEXT_EXPERIMENTS.md').write_text((ROOT/'circle/NEXT_EXPERIMENTS.md').read_text())
    cards=''.join(f'<section><h2>{html.escape(name.replace("_"," "))}</h2><a href="{name}.png"><img src="{name}.png" alt="{html.escape(name)}"></a><p><a href="{name}.pdf">PDF</a></p></section>' for name in figures)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Active foam parameter maps</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#203040}img{width:100%}section{margin:40px 0}a{color:#176b9a}</style><h1>Active foam: contacts, shape and packing</h1><p>Sampled response maps. Hatching denotes missing estimates; counts show accepted seeds. Curvature and system size are coupled. The tissue is flat.</p><p><a href="REPORT.md">Methods and failures</a> · <a href="summary.csv">Means, SEM and counts (CSV)</a> · <a href="NEXT_EXPERIMENTS.md">Proposed dynamics experiments</a></p>'+cards)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--passive',type=Path,default=ROOT/'circle/data/curvature_sweep_rerun_20261114')
    p.add_argument('--active',type=Path,default=ROOT/'circle/data/phase_sweep_20261114')
    p.add_argument('--out',type=Path,default=ROOT/'figures/circle/phase_maps_20261114')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    pm,pr=load(a.passive);am,ar=load(a.active)
    if pm['sizes']!=am['sizes'] or pm['seeds']!=am['seeds']:raise ValueError('Sizes and seeds differ between map sources.')
    records=normalize([r for r in pr if r['task']['mode']=='structure']+ar)
    sizes=am['sizes'];passive_density=[1.,1.2,1.4];active_density=am['densities']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    overview(records,a.out,sizes);names=['phase_diagrams_overview']
    for metric in ['z','q','phi']:
        for mode,densities in [('structure',passive_density),('dynamics',active_density)]:
            names.append(slices(records,a.out,sizes,densities,mode,metric))
    names.append(slices(records,a.out,sizes,active_density,'dynamics','exchange'))
    names.append(density_maps(records,a.out,sizes,passive_density,'structure'))
    names.append(density_maps(records,a.out,sizes,active_density,'dynamics'))
    packing_correlations(records,a.out,sizes);names.append('contacts_shape_vs_volume_fraction')
    write_report(records,a.out,a.passive,a.active,names)
    print(f'Wrote {len(names)} figure sets, CSV, report and gallery to {a.out}')


if __name__=='__main__':main()
