"""Scientific exploratory plots from saved curvature-sweep records."""
import argparse
import csv
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
from matplotlib.patches import Circle,Polygon
from matplotlib.collections import PatchCollection
from circle.sweep import aggregate,geometry_parameters

COLORS=['#277da8','#df7931','#6c55a3','#289a79']
SEEDS=(1,2,3)


def select(records,**criteria):
    return [r for r in records if all(r['task'][k]==v for k,v in criteria.items())]


def series(ax,groups,x,key,label,color,style='-',scale=1.):
    stats=[aggregate(g,key) for g in groups]
    y=np.array([s['mean'] if s['mean'] is not None else np.nan for s in stats])*scale
    err=np.array([s['sem'] if s['sem'] is not None else 0. for s in stats])*scale
    ax.errorbar(x,y,yerr=err,color=color,ls=style,marker='o' if style=='-' else 's',
                ms=4,capsize=3,lw=1.5,label=label)
    for i,stat in enumerate(stats):
        if stat['n']==1:
            ax.plot(x[i],y[i],marker='o' if style=='-' else 's',ms=5,
                    mfc='white',mec=color,ls='none',zorder=5)
            ax.annotate('n=1',(x[i],y[i]),xytext=(5,4),textcoords='offset points',
                        fontsize=7,color=color)
    return stats


def save(fig,out,name,title):
    fig.suptitle(title,fontsize=14)
    fig.text(.5,.008,f'Exploratory: {len(SEEDS)} seeds attempted per point; means ± seed SEM where ≥2 pass. '
             'Gaps are failed runs. Local microfold tolerance applies.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.95))
    fig.savefig(out/f'{name}.png',dpi=180)
    fig.savefig(out/f'{name}.pdf')
    plt.close(fig)


def structural(records,out,sizes):
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    ws=[.5,1.,1.5]
    for i,n in enumerate(sizes):
        for domain,style in [('disc','-'),('periodic','--')]:
            groups=[select(records,mode='structure',domain=domain,n=n,w=w,rho=1.2) for w in ws]
            for ax,key in zip(axes[0],['packing','z_all','shape_all']):
                series(ax,groups,ws,f'structure.{key}',f'{domain}, N={n}',COLORS[i%len(COLORS)],style)
        for j,rho in enumerate([1.,1.2,1.4]):
            groups=[select(records,mode='structure',domain='disc',n=n,w=w,rho=rho) for w in ws]
            if rho==1.2:
                phi=[aggregate(g,'structure.packing')['mean'] for g in groups]
                series(axes[1,0],groups,phi,'structure.z_all',f'N={n}',COLORS[i%len(COLORS)])
    kappas=[geometry_parameters(n,1.2)['dimensionless_curvature'] for n in sizes]
    groups=[select(records,mode='structure',domain='disc',n=n,w=1.,rho=1.2) for n in sizes]
    series(axes[1,1],groups,kappas,'structure.mean_wall_traction','W=1, ρ=1.2',COLORS[0])
    series(axes[1,2],groups,kappas,'structure.wall_fraction','W=1, ρ=1.2',COLORS[0])
    for ax,label in zip(axes[0],['Packing fraction φ','Mean distinct neighbors z','Mean shape index P/√A']):
        ax.set(xlabel='Adhesion W/T₀',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    axes[1,0].set(xlabel='Packing fraction φ (ρ=1.2)',ylabel='Mean distinct neighbors z');axes[1,0].legend()
    axes[1,1].set(xlabel='Wall curvature × √A₀',ylabel='Wall reaction / circumference')
    axes[1,2].set(xlabel='Wall curvature × √A₀',ylabel='Fraction of cells touching wall')
    for ax in axes[1]:ax.grid(alpha=.2)
    save(fig,out,'structure','Structure and wall curvature • passive states • top row: ρ=1.2')


def maps(records,out,sizes):
    fig,axes=plt.subplots(3,len(sizes),figsize=(4*len(sizes),10),squeeze=False)
    rho=[1.,1.2,1.4];ws=[.5,1.,1.5]
    for row,(key,label,limits) in enumerate([('packing','Packing φ',(.8,1.)),('z_all','Neighbors z',(3.5,6.)),('shape_all','Shape index',(3.65,4.1))]):
        for col,n in enumerate(sizes):
            ax=axes[row,col];arr=np.full((3,3),np.nan);counts=np.zeros((3,3),int)
            for i,r in enumerate(rho):
                for j,w in enumerate(ws):
                    g=aggregate(select(records,mode='structure',domain='disc',n=n,w=w,rho=r),'structure.'+key)
                    arr[i,j]=g['mean'] if g['mean'] is not None else np.nan;counts[i,j]=g['n']
            im=ax.imshow(arr,origin='lower',cmap='viridis',vmin=limits[0],vmax=limits[1],aspect='auto')
            ax.set(xticks=range(3),xticklabels=ws,yticks=range(3),yticklabels=rho,xlabel='W/T₀',ylabel='ρ',title=f'N={n} • {label}')
            for i in range(3):
                for j in range(3):
                    ax.text(j,i,('failed' if not counts[i,j] else f'{arr[i,j]:.3f}\n{counts[i,j]}/{len(SEEDS)}'),ha='center',va='center',color='white',fontsize=9,
                            bbox=dict(facecolor='black',alpha=.3,edgecolor='none',pad=2))
            fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,out,'structure_maps','Adhesion–density maps at three wall curvatures • no interpolation between sampled points')


def snapshots(records,out,sizes):
    fig,axes=plt.subplots(3,len(sizes),figsize=(4*len(sizes),11),squeeze=False)
    extent=max(geometry_parameters(n,1.2)['radius'] for n in sizes)*1.05
    last=None
    for row,w in enumerate([.5,1.,1.5]):
        for col,n in enumerate(sizes):
            ax=axes[row,col]
            candidates=[r for r in select(records,mode='structure',domain='disc',n=n,w=w,rho=1.2) if r['accepted'] and 'snapshot'in r]
            if candidates:
                snap=candidates[0]['snapshot'];center=np.array(snap['center']);radius=snap['radius']
                ax.add_patch(Circle((0,0),radius,color='#da6c62'))
                polygons=[np.array(p)-center for p in snap['polygons']]
                shapes=[]
                for p in polygons:
                    q=np.roll(p,-1,axis=0);area=.5*np.sum(p[:,0]*q[:,1]-p[:,1]*q[:,0])
                    shapes.append(np.linalg.norm(q-p,axis=1).sum()/np.sqrt(area))
                last=PatchCollection([Polygon(p) for p in polygons],array=np.array(shapes),cmap='viridis',clim=(3.5,4.5),edgecolor='#263440',linewidth=.45)
                ax.add_collection(last);ax.add_patch(Circle((0,0),radius,fill=False,color='black',lw=1.5))
            else:ax.text(0,0,'No accepted seed-1 snapshot',ha='center')
            par=geometry_parameters(n,1.2)
            ax.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect='equal',
                   title=f'N={n}, W={w}\nR={par["radius"]:.2f}, κ={par["curvature"]:.3f}')
            ax.set_xticks([]);ax.set_yticks([])
    if last is not None:
        fig.subplots_adjust(right=.9,top=.91,bottom=.06)
        cax=fig.add_axes([.92,.22,.018,.55]);fig.colorbar(last,cax=cax,label='Shape index P/√A')
    fig.suptitle('Wall curvature at fixed nominal density ρ=1.2 • all panels share a length scale',fontsize=14)
    fig.text(.5,.015,'Passive endpoints • seed 1 • red: extracellular space • flat tissue surface throughout',ha='center',fontsize=9)
    fig.savefig(out/'curvature_snapshots.png',dpi=180);fig.savefig(out/'curvature_snapshots.pdf');plt.close(fig)


def dynamics(records,out,sizes):
    mus=[0.,.5,1.,1.5]
    fig,axes=plt.subplots(4,len(sizes),figsize=(4*len(sizes),12),squeeze=False)
    for col,n in enumerate(sizes):
        for domain,style,color in [('disc','-',COLORS[0]),('periodic','--',COLORS[1])]:
            groups=[select(records,mode='dynamics',domain=domain,n=n,mu=mu) for mu in mus]
            for row,(key,label,scale) in enumerate([
                    ('dynamics.t1_rate','T1 events / cell / τT',10.),
                    ('dynamics.contact_change_rate.all','Sampled contact changes / cell / τT',10.),
                    ('dynamics.shape_mean','Time-mean shape index',1.),
                    ('dynamics.shape_change.all','Final − initial mean shape index',1.)]):
                series(axes[row,col],groups,mus,key,domain,color,style,scale)
                axes[row,col].set(xlabel='OU activity parameter μ',ylabel=label,xticks=mus,xlim=(-.05,1.55))
                axes[row,col].grid(alpha=.2)
        axes[0,col].set_title(f'N={n}, R={np.sqrt(n/np.pi):.2f}')
        axes[0,col].legend()
        counts=[' / '.join(str(sum(r['accepted'] for r in select(records,mode='dynamics',domain=d,n=n,mu=mu))) for d in ('disc','periodic')) for mu in mus]
        axes[0,col].set_xticks(mus,[f'{mu:g}\n{count}' for mu,count in zip(mus,counts)])
        axes[0,col].set_xlabel('μ; accepted seeds below: disc / periodic')
    save(fig,out,'exchange_shape','Neighbor exchange and shape index • W=1, ρ=1.3 • short observation window')
    fig,axes=plt.subplots(2,len(sizes),figsize=(4*len(sizes),8),squeeze=False)
    for col,n in enumerate(sizes):
        for domain,style in [('disc','-'),('periodic','--')]:
            for color,mu in zip(COLORS,mus):
                group=[r for r in select(records,mode='dynamics',domain=domain,n=n,mu=mu) if r['accepted']]
                for row in range(2):
                    if not group:continue
                    curves=[r['dynamics']['motion']['all'] if row==0 else r['dynamics']['com_subtracted'] for r in group]
                    values=np.array([v['msd'] for v in curves]);mean=values.mean(0)
                    lag=np.array(curves[0]['lag'])/10
                    axes[row,col].plot(lag,mean,color=color,ls=style,label=f'{domain} μ={mu} ({len(group)}/{len(SEEDS)})')
                    if len(values)>1:
                        sem=values.std(0,ddof=1)/np.sqrt(len(values))
                        axes[row,col].fill_between(lag,np.maximum(mean-sem,1e-14),mean+sem,color=color,alpha=.08)
        for row in range(2):
            axes[row,col].set(xscale='log',yscale='log',xlabel='Lag / τT',ylabel='MSD / A₀' if row==0 else 'COM-subtracted MSD / A₀')
            axes[row,col].grid(alpha=.2)
        axes[0,col].set_title(f'N={n}');axes[0,col].legend(fontsize=6)
    save(fig,out,'msd','Cell motion versus activity and wall curvature • no diffusion-exponent claim')


def shape_changes(records,out,sizes):
    fig,axes=plt.subplots(3,len(sizes),figsize=(4*len(sizes),11),squeeze=False)
    for col,n in enumerate(sizes):
        group=[r for r in select(records,mode='dynamics',domain='disc',n=n,mu=1.) if r['accepted']]
        for zone,color in [('wall',COLORS[1]),('bulk',COLORS[0])]:
            valid=[r for r in group if r['dynamics']['zone_counts'][zone]>0]
            if not valid:continue
            t=np.array(valid[0]['dynamics']['shape_time'])/10
            shape=np.array([r['dynamics']['shape_series'][zone] for r in valid]);delta=shape-shape[:,0,None]
            axes[0,col].plot(t,delta.mean(0),color=color,label=f'{zone} ({len(valid)}/{len(SEEDS)})')
            if len(valid)>1:
                sem=delta.std(0,ddof=1)/np.sqrt(len(valid));axes[0,col].fill_between(t,delta.mean(0)-sem,delta.mean(0)+sem,color=color,alpha=.15)
            rates=np.array([r['dynamics']['contact_rate_series'][zone] for r in valid])*10
            # Five-sample blocks for readability, retaining zero-rate blocks.
            blocks=(rates.shape[1]//5)*5
            if blocks:
                y=rates[:,:blocks].reshape(len(valid),-1,5).mean(2).mean(0)
                x=t[1:blocks+1].reshape(-1,5).mean(1)
                axes[1,col].plot(x,y,color=color,label=zone)
        for r in group:
            d=r['dynamics'];change=np.array(d['cell_shape_final'])-np.array(d['cell_shape_initial'])
            axes[2,col].scatter(d['cell_contact_changes'],change,s=10,alpha=.3,color=COLORS[0])
        axes[0,col].set(title=f'N={n}, μ=1 ({len(group)}/{len(SEEDS)} accepted)',xlabel='Time / τT',ylabel='Change in mean shape index')
        axes[1,col].set(xlabel='Time / τT',ylabel='Contact changes / cell / τT')
        axes[2,col].set(xlabel='Contact changes of a cell',ylabel='Final − initial shape index')
        for row in range(3):axes[row,col].grid(alpha=.2)
        if group:axes[0,col].legend(fontsize=8)
    save(fig,out,'shape_changes','Shape-index changes and neighbor exchange • wall/interior zones fixed after warmup')


def curvature_response(records,out,sizes):
    """Compare wall/interior responses across radii, with matched-size controls."""
    for r in records:
        if r['accepted'] and r['task']['mode']=='dynamics':
            r['dynamics']['shape_zone_mean']={
                zone:float(np.mean(values[1:])) if r['dynamics']['zone_counts'][zone] else None
                for zone,values in r['dynamics']['shape_series'].items()}
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    kappas=[geometry_parameters(n,1.3)['dimensionless_curvature'] for n in sizes]
    for col,mu in enumerate([0.,.5,1.]):
        for domain,zone,color,style,label in [
                ('disc','wall',COLORS[1],'-','Disc wall layer'),
                ('disc','bulk',COLORS[0],'-','Disc interior'),
                ('periodic','all',COLORS[2],'--','Periodic, matched N')]:
            groups=[select(records,mode='dynamics',domain=domain,n=n,mu=mu) for n in sizes]
            series(axes[0,col],groups,kappas,'dynamics.contact_change_rate.'+zone,label,color,style,10.)
            series(axes[1,col],groups,kappas,'dynamics.shape_zone_mean.'+zone,label,color,style)
        axes[0,col].set_title(f'μ={mu:g}')
        for row in range(2):
            axes[row,col].set_xlabel('Disk wall curvature × √A₀')
            axes[row,col].grid(alpha=.2)
        axes[0,col].set_ylabel('Contact changes / cell / τT')
        axes[1,col].set_ylabel('Time-mean shape index')
    axes[0,0].legend(fontsize=8)
    save(fig,out,'curvature_response','Wall/interior response across radii • periodic points use matched-N disk curvature')


def report(records,out,manifest):
    fields=['domain','mode','n','w','rho','mu']
    metrics=['structure.packing','structure.z_all','structure.shape_all','structure.mean_wall_traction',
             'dynamics.t1_rate','dynamics.contact_change_rate.all','dynamics.contact_change_rate.wall',
             'dynamics.contact_change_rate.bulk','dynamics.shape_mean','dynamics.shape_change.all',
             'dynamics.shape_change.wall','dynamics.shape_change.bulk',
             'dynamics.shape_zone_mean.wall','dynamics.shape_zone_mean.bulk','dynamics.shape_zone_mean.all']
    groups={tuple(r['task'][f] for f in fields) for r in records}
    with (out/'summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields+['metric','attempted','accepted_seeds','mean','sem'],lineterminator='\n')
        writer.writeheader()
        for key in sorted(groups):
            selected=select(records,**dict(zip(fields,key)))
            for metric in metrics:
                if (key[1]=='structure') != metric.startswith('structure'):continue
                if key[0]=='periodic' and (metric=='structure.mean_wall_traction' or metric.endswith(('.wall','.bulk'))):continue
                stat=aggregate(selected,metric);stat['accepted_seeds']=stat.pop('n')
                writer.writerow(dict(zip(fields,key))|{'metric':metric}|stat)
    # Paired differences use only seeds accepted in both domains.
    conditions=sorted({tuple(r['task'][f] for f in fields[1:]) for r in records})
    paired=[]
    def value(record,path):
        v=record
        for part in path.split('.'):
            v=v.get(part) if isinstance(v,dict) else None
        return v
    for condition in conditions:
        criteria=dict(zip(fields[1:],condition))
        selected=select(records,**criteria)
        lookup={(r['task']['domain'],r['task']['seed']):r for r in selected}
        mappings=([
            ('packing','structure.packing','structure.packing'),
            ('all_neighbor_number','structure.z_all','structure.z_all'),
            ('all_shape','structure.shape_all','structure.shape_all'),
            ('interior_neighbor_number','structure.z_interior','structure.z_all'),
            ('interior_shape','structure.shape_interior','structure.shape_all'),
            ('interior_area','structure.area_interior','structure.area')]
            if criteria['mode']=='structure' else [
            ('t1_rate','dynamics.t1_rate','dynamics.t1_rate'),
            ('all_contact_change_rate','dynamics.contact_change_rate.all','dynamics.contact_change_rate.all'),
            ('interior_contact_change_rate','dynamics.contact_change_rate.bulk','dynamics.contact_change_rate.all'),
            ('all_shape','dynamics.shape_mean','dynamics.shape_mean'),
            ('interior_shape','dynamics.shape_zone_mean.bulk','dynamics.shape_mean')])
        for metric,disk_key,periodic_key in mappings:
            differences=[]
            for seed in SEEDS:
                disk=lookup.get(('disc',seed));periodic=lookup.get(('periodic',seed))
                if disk and periodic and disk['accepted'] and periodic['accepted']:
                    d,v=value(disk,disk_key),value(periodic,periodic_key)
                    if d is not None and v is not None:
                        differences.append(dict(accepted=True,value=d-v))
            stat=aggregate(differences,'value')
            paired.append(criteria|dict(metric=metric,attempted_pairs=len(SEEDS),
                                        accepted_pairs=stat['n'],disc_minus_periodic=stat['mean'],sem=stat['sem']))
    with (out/'paired_differences.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields[1:]+['metric','attempted_pairs','accepted_pairs','disc_minus_periodic','sem'],lineterminator='\n')
        writer.writeheader();writer.writerows(paired)
    lines=['# Circular-wall parameter sweeps','',
           f"{sum(r['accepted'] for r in records)} of {len(records)} attempted runs passed their domain-specific geometry checks.", '',
           '**Activity convention:** the interior-contact update matches the original equation supplied by the user. μ=ΔT/T0 assumes unit white-noise covariance in code time units; an explicit covariance has not been supplied. ΔT is the forcing amplitude, not automatically the stationary RMS. See [NOISE_CONVENTION.md](../../NOISE_CONVENTION.md).', '',
           'The tissue surface is flat. κ=1/R is wall curvature. R=√(N/π), so nominal density stays fixed across sizes; curvature and finite size cannot be separated by this disk sweep alone.',
           '', f'{len(SEEDS)} independent seeds are attempted at each point. Error bars are seed SEM, not cell-wise uncertainty. Means exclude failed runs and are conditional on acceptance. Local microfolds remain tolerated and counted. Periodic runs retain their original per-cell geometry checks.',
           '', 'Structure: W={0.5,1,1.5}, ρ={1,1.2,1.4}, μ=0 after standard preparation.',
           'Dynamics: W=1, ρ=1.3, μ={0,0.5,1,1.5}; 10 τR warmup and '+str(manifest['duration'])+' τR measured (τT=10). Sampling every 0.5 τR; MSD uses lags ≤ one quarter of the measurement window. The one-τT warmup does not establish stationarity; shape changes can include residual relaxation. These short runs do not establish diffusion exponents, rigidity thresholds or a phase diagram.',
           '', 'T1 rate counts accepted T1 operations/N/time, including boundary operations and repeated reversals; it is not identical to the cell-neighbor exchange rate. Contact-change rate counts sampled contact gains and losses per cell/time; transient changes between samples can be missed. Shape index is P/√A. Wall/interior zones for dynamics are fixed after warmup. Cell-level scatter points are correlated and do not supply independent error bars.',
           '', '| Domain | Mode | N | Accepted / attempted |','|---|---|---|---|']
    for domain,mode,n in sorted({(r['task']['domain'],r['task']['mode'],r['task']['n']) for r in records}):
        selected=select(records,domain=domain,mode=mode,n=n)
        lines.append(f"| {domain} | {mode} | {n} | {sum(r['accepted'] for r in selected)} / {len(selected)} |")
    lines+=['','## Disk activity response','',
            'Means ± independent-seed SEM where at least two runs pass. Missing points are failures, not zero rates. T1 operations and sampled neighbor changes are distinct quantities.', '',
            '| N | μ | Accepted | T1 / cell / τT | Neighbor changes / cell / τT | Mean P/√A | Final − initial P/√A |',
            '|---|---|---|---|---|---|---|']
    def formatted(stat,scale=1.):
        if stat['mean'] is None:return '—'
        result=f"{stat['mean']*scale:.4f}"
        if stat['sem'] is not None:result+=f" ± {stat['sem']*scale:.4f}"
        return result
    for n in manifest['sizes']:
        for mu in [0.,.5,1.,1.5]:
            group=select(records,domain='disc',mode='dynamics',n=n,mu=mu)
            if not group:continue
            stats=[aggregate(group,key) for key in ['dynamics.t1_rate','dynamics.contact_change_rate.all',
                                                    'dynamics.shape_mean','dynamics.shape_change.all']]
            columns=[formatted(v,k) for v,k in zip(stats,[10.,10.,1.,1.])]
            lines.append(f"| {n} | {mu:g} | {stats[0]['n']} / {len(group)} | "+' | '.join(columns)+' |')
    lines+=['','## Failures','', 'Failures are retained in per-run JSON and outcomes.csv; no interpolation fills missing points. summary.csv holds separate-domain means; paired_differences.csv uses only seeds accepted in both domains for disk-minus-periodic contrasts.','']
    for r in records:
        if not r['accepted']:
            t=r['task'];lines.append(f"- {t['domain']} {t['mode']}, N={t['n']}, W={t['w']}, ρ={t['rho']}, μ={t['mu']}, seed={t['seed']}: {r['stage']}: {r.get('error')}")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    global SEEDS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'circle/data/curvature_sweep')
    p.add_argument('--out',type=Path,default=ROOT/'figures/circle/sweep')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.data/'manifest.json').read_text())
    SEEDS=manifest['seeds']
    records=[json.loads((a.data/'runs'/f'{key}.json').read_text()) for key in manifest['tasks'] if (a.data/'runs'/f'{key}.json').exists()]
    if len(records)!=len(manifest['tasks']):raise ValueError('sweep is incomplete; plotting would hide unfinished tasks')
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    count=0
    if manifest['mode'] in ('all','structure'):
        structural(records,a.out,manifest['sizes']);maps(records,a.out,manifest['sizes'])
        snapshots(records,a.out,manifest['sizes']);count+=3
    if manifest['mode'] in ('all','dynamics'):
        dynamics(records,a.out,manifest['sizes']);shape_changes(records,a.out,manifest['sizes'])
        curvature_response(records,a.out,manifest['sizes']);count+=4
    report(records,a.data,manifest)
    print(f'Wrote {count} plot sets to {a.out}')


if __name__=='__main__':main()
