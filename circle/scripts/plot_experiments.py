"""Regenerate the exploratory dynamics figures from compressed experiment records."""
import argparse,csv,json,html,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Ellipse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle.scripts.run_experiments import load_records

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':120,'savefig.dpi':160})
COLORS=['#247ba0','#e76f51','#40916c','#8956a7']


def mean_sd(values):
    values=np.asarray(values,dtype=float)
    return values.mean(axis=0),values.std(axis=0,ddof=1) if len(values)>1 else np.zeros(values.shape[1:])


def scalar(r,key):
    if key in r['summary']:return r['summary'][key]
    samples=r['samples'][1:]
    if key=='contacts':return np.mean([s['neighbors'] for s in samples])
    if key=='attached':return np.mean([s['touching'] for s in samples])
    if key=='free':return np.mean([s['free_surface'] for s in samples])
    raise KeyError(key)


def plot_line(ax,x,values,label,color,**kwargs):
    m,sd=mean_sd(values)
    ax.plot(x,m,label=label,color=color,**kwargs)
    ax.fill_between(x,m-sd,m+sd,color=color,alpha=.15)


def save(fig,out,name,title,caption,gallery):
    fig.suptitle(title,fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(out/f'{name}.png');fig.savefig(out/f'{name}.pdf');plt.close(fig)
    gallery.append(dict(name=name,title=title,caption=caption))


def response_maps(records,out,gallery):
    allruns=[r for r in records if r['task']['family']=='adhesion_activity']
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,(key,title) in enumerate([('event_rate','Transactions / cell / τR'),
                                     ('shape_mean','Shape index P / √A'),('final_neighbor_overlap','Initial-neighbor overlap at 10 τT')]):
        scale=[scalar(r,key) for r in allruns if r['accepted']]
        vmin,vmax=(min(scale),max(scale)) if scale else (0.,1.)
        for row,n in enumerate((36,100)):
            ax=axes[row,col];grid=np.full((3,3),np.nan);counts=np.zeros((3,3),int)
            for i,mu in enumerate((.25,.5,1.)):
                for j,w in enumerate((.5,1.,1.5)):
                    rr=[r for r in allruns if r['accepted'] and r['task']['n']==n and r['task']['mu']==mu and r['task']['w']==w]
                    counts[i,j]=len(rr)
                    if rr:grid[i,j]=np.mean([scalar(r,key) for r in rr])
            im=ax.imshow(np.ma.masked_invalid(grid),origin='lower',aspect='auto',cmap='viridis',vmin=vmin,vmax=vmax)
            ax.set_xticks(range(3),['0.5','1','1.5']);ax.set_yticks(range(3),['0.25','0.5','1'])
            ax.set(xlabel='Adhesion W / T₀',ylabel='Noise amplitude μ = ΔT / T₀',title=f'N={n}: {title}')
            for (i,j),count in np.ndenumerate(counts):
                label=f'{grid[i,j]:.3g}\n{count}/3' if count else 'failed\n0/3'
                ax.text(j,i,label,ha='center',va='center',color=('white' if im.norm(grid[i,j])<.6 else '#101010') if count else 'black',fontsize=9)
            fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,out,'01_adhesion_activity','Adhesion and activity: rearrangements, shape and contact memory',
         'ρ=1.2, τT=10 τR, 5 τT warmup and 10 τT observation. Each tile is a mean over completed seeds; labels show accepted/attempted seeds. Failed trajectories are excluded, never assigned zero.',gallery)


def persistence(records,out,gallery):
    fig,axes=plt.subplots(2,3,figsize=(12,7));taus=(2.,5.,10.,20.)
    for row,mode in enumerate(('fixed_amplitude','fixed_variance')):
        for col,key in enumerate(('event_rate','shape_mean','final_neighbor_overlap')):
            ax=axes[row,col];means=[];errors=[];labels=[]
            for tau in taus:
                rr=[r for r in records if r['accepted'] and r['task']['family']=='persistence_'+mode and r['task']['tau']==tau]
                vals=[scalar(r,key) for r in rr];m,sd=mean_sd(vals) if vals else (np.nan,np.nan)
                means.append(m);errors.append(sd);labels.append(len(rr))
            ax.errorbar(taus,means,yerr=errors,fmt='o-',capsize=3,color=COLORS[row])
            for x,y,n in zip(taus,means,labels):
                if np.isfinite(y):ax.annotate(f'{n}/3',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
            ax.set(xlabel='Persistence τT / τR',ylabel={'event_rate':'Transactions / cell / τR','shape_mean':'Shape index','final_neighbor_overlap':'Neighbor overlap after 10 τT'}[key],title=mode.replace('_',' '))
    save(fig,out,'02_persistence','Persistence and fluctuation variance are separate controls',
         'Mean ± seed SD. Fixed amplitude uses μ=0.5; fixed variance uses μ=0.5√[(2τT−dt)/(20−dt)]. This matches raw stationary Euler–Maruyama variance for a persistent contact, not its rectified effective force. All variants share τT=10 preparation. Durations scale with τT; τT=10 controls are duplicate runs, not independent replicas.',gallery)


def long_dynamics(records,out,gallery):
    rr=[r for r in records if r['task']['family']=='long' and r['accepted']]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for j,w in enumerate((1.,1.5)):
        group=[r for r in rr if r['task']['w']==w]
        if not group:continue
        label=f'W={w:g}, n={len(group)}/3';x=np.array([x['lag'] for x in group[0]['summary']['motion']['lags']])/10
        for key,style in (('msd','-'),('cage_msd','--')):
            plot_line(axes[0,0],x,[[l[key] for l in r['summary']['motion']['lags']] for r in group],label+(' cage-relative' if key=='cage_msd' else ''),COLORS[j],linestyle=style)
        times=np.array([s['time'] for s in group[0]['samples']])/10
        for ax,key in ((axes[0,1],'neighbor_overlap'),(axes[0,2],'continuous_survival')):
            plot_line(ax,times,[[s[key] for s in r['samples']] for r in group],label,COLORS[j])
        plot_line(axes[1,0],times,[[np.mean(s['shape']) for s in r['samples']] for r in group],label,COLORS[j])
        for col,key in ((1,'non_gaussian'),(2,'chi4')):
            plot_line(axes[1,col],x,[[l[key] for l in r['summary']['motion']['lags']] for r in group],label,COLORS[j])
    axes[0,0].set(xscale='log',yscale='log',xlabel='Lag / τT',ylabel='MSD / A₀');axes[0,0].legend(fontsize=7)
    axes[0,1].set(xlabel='Time / τT',ylabel='Initial-neighbor overlap')
    axes[0,2].set(xlabel='Time / τT',ylabel='Unbroken initial-contact fraction')
    axes[1,0].set(xlabel='Time / τT',ylabel='Shape index')
    axes[1,1].set(xlabel='Lag / τT',ylabel='2-D non-Gaussian parameter α₂')
    axes[1,2].set(xlabel='Lag / τT',ylabel='Finite-window χ₄')
    absent=[str(w) for w in (1.,1.5) if not any(r['task']['w']==w for r in rr)]
    if absent:axes[1,0].text(.02,.03,'No completed W='+', '.join(absent)+' trajectories',transform=axes[1,0].transAxes,fontsize=8)
    if not rr:
        for ax in axes.flat:ax.text(.5,.5,'No completed trajectories',transform=ax.transAxes,ha='center')
    save(fig,out,'03_long_dynamics','Long trajectories: cage motion, contact memory and stationarity',
         'N=36, ρ=1.2, μ=0.5. Burn-in 20 τT; observation 50 τT. Bands are seed SD. Cage neighbors are fixed at measurement start. χ₄ uses correlated time origins within each trajectory and is not an ensemble susceptibility. Re-formed initial contacts count in overlap but not in unbroken survival.',gallery)


def pulse_response(records,out,gallery):
    allruns=[r for r in records if r['task']['family']=='pulse'];fig,axes=plt.subplots(2,3,figsize=(13,7))
    paircounts={}
    for row,n in enumerate((36,100)):
        sham={r['task']['seed']:r for r in allruns if r['accepted'] and r['task']['n']==n and r['task']['pattern']=='uniform'}
        for j,pattern in enumerate(('pulse','wall','sector')):
            pairs=[(r,sham[r['task']['seed']]) for r in allruns if r['accepted'] and r['task']['n']==n and r['task']['pattern']==pattern and r['task']['seed'] in sham]
            paircounts[f'{n}:{pattern}']=len(pairs)
            if not pairs:continue
            t=np.array([s['time'] for s in pairs[0][0]['samples']])/10
            for col,key in enumerate(('shape','wall_load')):
                differences=[]
                for r,c in pairs:
                    a=np.array([np.mean(s[key]) for s in r['samples']]);b=np.array([np.mean(s[key]) for s in c['samples']])
                    differences.append(a-b)
                plot_line(axes[row,col],t,differences,f'{pattern}, {len(pairs)}/3 pairs',COLORS[j])
            edges=np.arange(0,70.01,5.);curves=[]
            for r,c in pairs:
                def counts(run):return np.histogram([e['time'] for e in run['events'] if e['lost'] or e['gained']],edges)[0]/(n*5.)
                curves.append(counts(r)-counts(c))
            plot_line(axes[row,2],(edges[:-1]+edges[1:])/20,curves,pattern,COLORS[j])
        for col,label in enumerate(('Δ shape index','Δ integrated wall reaction','Δ transactions / cell / τR')):
            axes[row,col].axvspan(1,2,color='gray',alpha=.15);axes[row,col].axhline(0,color='gray',lw=.5)
            axes[row,col].set(xlabel='Time / τT',ylabel=label,title=f'N={n}: paired change from sham')
        axes[row,0].legend(fontsize=7)
    save(fig,out,'04_activity_pulses','Uniform and localized activity pulses, paired against a sham',
         'Baseline μ=0.25; μ=1 during t/τT∈[1,2). Wall activation selects film centroids within one code-length unit of the circular wall; sector activation covers ±π/4. Curves subtract the same-seed sham, using only pairs whose full trajectories completed. Bands are SD across paired seeds; missing pairs can bias the surviving sample. The protocols have equal peak amplitude but are not matched for total noise input or activated film count.',gallery)
    return paircounts


def geometry(records,out,gallery):
    rr=[r for r in records if r['task']['family']=='geometry' and r['accepted']]
    fig,axes=plt.subplots(1,3,figsize=(13,4.4));norm=plt.Normalize(3.5,4.5)
    for ax,aspect in zip(axes,(1.,1.5,2.)):
        group=[r for r in rr if r['task']['aspect']==aspect];chosen=next((r for r in group if r['task']['seed']==1),group[0] if group else None)
        if chosen:
            g=chosen['final_geometry'];center=np.array(g['center']);radius=g['radius'];a=radius*np.sqrt(aspect);b=radius/np.sqrt(aspect)
            polygons=[np.array(p)-center for p in g['polygons']];pc=PolyCollection(polygons,cmap='viridis',norm=norm,linewidth=.5,edgecolors='#222222');pc.set_array(np.array(chosen['samples'][-1]['shape']));ax.add_collection(pc)
            ax.add_patch(Ellipse((0,0),2*a,2*b,fill=False,lw=1.3));ax.set_xlim(-5.1,5.1);ax.set_ylim(-3.8,3.8)
            ax.set_title(f'a/b={aspect:g}; seed {chosen["task"]["seed"]}; {len(group)}/3 accepted')
        else:ax.text(.5,.5,'No completed trajectory',transform=ax.transAxes,ha='center')
        ax.set_aspect('equal');ax.set(xlabel='x / ℓ',ylabel='y / ℓ')
    # Place the colorbar outside the shared-coordinate panels.
    cax=fig.add_axes([.93,.2,.012,.62])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=cax,label='Shape index')
    fig.suptitle('Rigid equal-area containers at fixed cell number and density',fontsize=14)
    fig.subplots_adjust(top=.83,left=.055,right=.89,wspace=.25)
    name='05_equal_area_snapshots';fig.savefig(out/f'{name}.png');fig.savefig(out/f'{name}.pdf');plt.close(fig)
    gallery.append(dict(name=name,title='Equal-area circle and ellipses',caption='N=36, ρ=1.2, W=1, μ=0.5; all walls have area 36 in code units. Final frames at 10 τT after 5 τT warmup; same-seed examples where available. Ellipse initialization is an area-preserving deformation of the circular tiling, followed by relaxation. Container shape also changes global confinement and orientation; these are not isolated local-curvature interventions.'))
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for j,aspect in enumerate((1.,1.5,2.)):
        group=[r for r in rr if r['task']['aspect']==aspect]
        for col,(key,label) in enumerate((('shape_mean','Whole-tissue shape index'),('contacts','Whole-tissue contact number'),('event_rate','Transactions / cell / τR'))):
            if group:
                m,sd=mean_sd([scalar(r,key) for r in group]);axes[col].errorbar(aspect,m,yerr=sd,fmt='o',color=COLORS[j],capsize=4)
                axes[col].annotate(f'{len(group)}/3',(aspect,m),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
            axes[col].set(xlabel='Aspect ratio a / b',ylabel=label,xticks=[1,1.5,2])
    save(fig,out,'06_geometry_response','Shape of the container at fixed area, N and preferred cell area',
         'Means ± seed SD over completed trajectories. The ellipse projection uses its actual surface normal. Local-curvature profiles are exported separately; wall distance for ellipses uses a 1024-point wall sample.',gallery)
    # Local curvature conditioned on proximity to the wall, averaged within seed.
    fig,axes=plt.subplots(1,2,figsize=(10,4));edges=np.linspace(.09,1.,8)
    localrows=[]
    for j,aspect in enumerate((1.,1.5,2.)):
        group=[r for r in rr if r['task']['aspect']==aspect]
        for col,(key,label) in enumerate((('shape','Shape index near wall'),('neighbors','Contact number near wall'))):
            vals=[];curvatures=[]
            for r in group:
                k=np.array([s['wall_curvature'] for s in r['samples'][1:]])*np.sqrt(r['task']['rho'])
                d=np.array([s['wall_distance'] for s in r['samples'][1:]])/np.sqrt(r['task']['rho']);v=np.array([s[key] for s in r['samples'][1:]])
                row=[];curvature_row=[]
                for lo,hi in zip(edges[:-1],edges[1:]):
                    mask=(k>=lo)&(k<hi)&(d<1)
                    row.append(v[mask].mean() if mask.any() else np.nan)
                    curvature_row.append(k[mask].mean() if mask.any() else np.nan)
                    if mask.any():localrows.append(dict(aspect=aspect,seed=r['task']['seed'],metric=key,curvature_low=lo,curvature_high=hi,curvature_mean=curvature_row[-1],mean=row[-1],cell_frames=int(mask.sum())))
                vals.append(row);curvatures.append(curvature_row)
            if vals:
                arr=np.array(vals)
                for b in range(arr.shape[1]):
                    good=arr[:,b][np.isfinite(arr[:,b])]
                    if len(good):
                        m,sd=mean_sd(good);axes[col].errorbar(np.nanmean(np.array(curvatures)[:,b]),m,yerr=sd,fmt='o',color=COLORS[j],label=f'a/b={aspect:g}' if b==np.flatnonzero(np.isfinite(arr).any(axis=0))[0] else None,capsize=3)
            axes[col].set(xlabel='Nearest wall curvature κ L₀',ylabel=label)
        axes[0].legend(fontsize=8)
    save(fig,out,'07_local_curvature','Near-wall structure versus local curvature in equal-area containers',
         'Centroids within L₀ of the wall. First average cell-frames within each seed and curvature bin; error bars are SD across seeds. Curvature is that of the nearest sampled wall point; markers use the seed-averaged curvature within each bin. Global shape, wall orientation and local curvature covary; this association alone does not establish causality.',gallery)
    if localrows:
        with (out/'local_curvature.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(localrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(localrows)


def spaces(records,out,gallery):
    rr=[r for r in records if r['task']['family']=='opening_spaces' and r['accepted']]
    fig,axes=plt.subplots(1,4,figsize=(14,3.9))
    for j,w in enumerate((.25,.35)):
        for col,(key,label) in enumerate((('packing_mean','Measured area fraction'),('contacts','Mean contacts'),('attached','Wall-touching cell fraction'),('event_rate','Transactions / cell / τR'))):
            x=[];y=[];sd=[]
            for rho in (.9,1.):
                group=[r for r in rr if r['task']['w']==w and r['task']['rho']==rho]
                if group:
                    m,e=mean_sd([scalar(r,key) for r in group]);x.append(rho);y.append(m);sd.append(e)
                    axes[col].annotate(f'{len(group)}/3',(rho,m),xytext=(0,8),textcoords='offset points',fontsize=8)
            axes[col].errorbar(x,y,yerr=sd,fmt='o-',label=f'W={w:g}',color=COLORS[j],capsize=3)
            axes[col].set(xlabel='Preferred area / container area, ρ',ylabel=label,xticks=[.9,1])
    missing=[f'W={w:g}: 0/6 completed' for w in (.25,.35) if not any(r['task']['w']==w for r in rr)]
    if missing:axes[0].text(.03,.03,'; '.join(missing),transform=axes[0].transAxes,fontsize=8)
    axes[0].legend(fontsize=8)
    save(fig,out,'08_opening_spaces','Lower adhesion and density: packing, contacts and wall attachment',
         'N=36, μ=0.5; 5 τT warmup and 10 τT observation. Mean ± seed SD. Nominal density ρ is a control parameter, whereas packing is measured. Failed geometries are excluded and their counts are reported.',gallery)


def event_diagnostics(records,out,gallery):
    choices=[r for r in records if r['accepted'] and r['task']['family']=='long']
    if not choices:choices=[r for r in records if r['accepted'] and r['task']['family']=='adhesion_activity']
    if not choices:return
    chosen=max(choices,key=lambda r:r['summary']['contact_changing_transactions'])
    events=[e for e in chosen['events'] if e['lost'] or e['gained']]
    fig,axes=plt.subplots(1,3,figsize=(13,4));t=chosen['task']
    if events:
        xy=np.array([e['location'] for e in events])-np.array(chosen['final_geometry']['center'])
        sc=axes[0].scatter(xy[:,0],xy[:,1],c=[e['time']/t['tau'] for e in events],s=18,cmap='plasma');fig.colorbar(sc,ax=axes[0],label='Time / τT')
        axes[0].set_aspect('equal')
        waits=np.diff([e['time'] for e in events]);axes[1].hist(waits/t['tau'],bins=20,color=COLORS[0]);axes[1].set(xlabel='Waiting time / τT',ylabel='Count')
    axes[0].set(xlabel='x / ℓ',ylabel='y / ℓ',title='Contact-changing transaction locations')
    for i,c in enumerate(chosen['summary']['cascades']):
        null=c['shuffled_largest']
        if null:axes[2].scatter(np.full(len(null),i)+np.linspace(-.07,.07,len(null)),null,color='gray',s=12,alpha=.5)
        axes[2].scatter(i,c['largest'],color=COLORS[1],s=45,zorder=3)
    axes[2].set(xticks=range(4),xticklabels=['0.1τ, L₀','0.1τ, 2L₀','0.5τ, L₀','0.5τ, 2L₀'],ylabel='Largest connected event cluster',title='Observed (red) vs spatial shuffles (gray)');axes[2].tick_params(axis='x',rotation=25)
    save(fig,out,'09_event_clusters','Resolved neighbor changes and clustering diagnostics',
         f"Illustrative completed {t['family']} run chosen for the largest event count in that family: N={t['n']}, W={t['w']}, μ={t['mu']}, seed={t['seed']}; {len(events)} contact-changing transactions. This is not an unbiased example. Events connect within both distance and time cutoffs; nulls shuffle observed positions across fixed event times (16 shuffles). Large clusters or back-to-back reversals are not by themselves evidence for physical avalanches.",gallery)


def refinement(records,out,gallery):
    rr=[r for r in records if r['task']['family']=='refinement'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,seed in zip(axes,(1,2)):
        group=[r for r in rr if r['task']['seed']==seed]
        for i,r in enumerate(group):
            t=r['task'];val=r.get('elapsed',0.)+(t['warmup'] if r['stage']=='measurement' else 0.);color=COLORS[2] if r['accepted'] else COLORS[1]
            ax.bar(i,val,color=color);ax.text(i,val+1,r['stage'],ha='center',fontsize=8,rotation=15)
        ax.set(xticks=range(len(group)),xticklabels=[f"dt={r['task']['dt']}\nh={r['task']['spacing']}" for r in group],ylabel='Time since preparation / τR',title=f'Seed {seed}')
    save(fig,out,'10_numerical_refinement','High-activity numerical gate: first detected failure under refinement',
         'N=36, W=1, ρ=1.3, μ=1.5. Geometry checked every timestep after preparation; preparation retains checks every 0.5 τR. Bars report time after preparation to the first failed check; labels identify the stage. This is not a physical instability threshold. Seed labels are shared but Brownian paths and initial meshes are not coupled under refinement. These short first-passage comparisons cannot establish stochastic convergence.',gallery)


def correlations(records,out,gallery):
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for j,n in enumerate((36,100)):
        group=[r for r in records if r['accepted'] and r['task']['family']=='adhesion_activity' and r['task']['n']==n and r['task']['w']==1 and r['task']['mu']==.5]
        if not group:continue
        lag=[s['lag'] for s in group[0]['summary']['tension_shape_lags']]
        vals=[[s['tension_to_shape_change'] for s in r['summary']['tension_shape_lags']] for r in group]
        plot_line(axes[0],lag,vals,f'N={n}, {len(group)}/3',COLORS[j])
        edges=np.linspace(0,4,9);velocity=[];nematic=[]
        for r in group:
            samples=r['samples'];centers=np.array([s['centers'] for s in samples]);q=np.array([s['nematic'] for s in samples])
            v=np.diff(centers,axis=0)/r['task']['sample'];v-=v.mean(axis=1,keepdims=True)
            vscale=np.mean(np.sum(v*v,axis=2));qscale=np.mean(np.sum(q*q,axis=2));sumv=np.zeros(8);sumq=sumv.copy();count=sumv.copy()
            i,k=np.triu_indices(n,1)
            for frame in range(0,len(v),5):
                distance=np.linalg.norm(centers[frame,i]-centers[frame,k],axis=1)/np.sqrt(r['task']['rho'])
                vv=np.sum(v[frame,i]*v[frame,k],axis=1);qq=np.sum(q[frame,i]*q[frame,k],axis=1)
                count+=np.histogram(distance,edges)[0];sumv+=np.histogram(distance,edges,weights=vv)[0];sumq+=np.histogram(distance,edges,weights=qq)[0]
            velocity.append(np.divide(sumv,count*vscale,out=np.full(8,np.nan),where=count>0));nematic.append(np.divide(sumq,count*qscale,out=np.full(8,np.nan),where=count>0))
        for ax,values in zip(axes[1:],(velocity,nematic)):
            # The nearest bin can be empty because cells exclude each other.
            arr=np.array(values)
            for b in range(8):
                good=arr[:,b][np.isfinite(arr[:,b])]
                if len(good):
                    m,sd=mean_sd(good);ax.errorbar((edges[b]+edges[b+1])/2,m,yerr=sd,fmt='o',color=COLORS[j],capsize=3)
    axes[0].set(xlabel='Lag / τR',ylabel='Corr(tension(t), q(t+lag)−q(t))');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Cell separation / L₀',ylabel='Velocity correlation (COM removed)')
    axes[2].set(xlabel='Cell separation / L₀',ylabel='Shape nematic correlation')
    save(fig,out,'11_correlations','Tension–shape lags and spatial organization',
         'W=1, ρ=1.2, μ=0.5; completed seeds, means ± seed SD. Tension is a length-weighted incident-film average per cell, not a tracked contact tension. Lag correlations are descriptive, not causal. Spatial velocities have instantaneous whole-tissue translation removed; shape tensors come from polygon area moments. Pair counts do not represent independent replicas.',gallery)


def km_curve(episodes,grid):
    """Product-limit survival for contacts whose births were observed."""
    observed=[e for e in episodes if not e['left_censored']]
    if not observed:return None
    durations=np.array([e['duration'] for e in observed]);died=np.array([not e['right_censored'] for e in observed])
    survival=1.;points=[]
    for time in np.unique(durations[died]):
        at_risk=np.sum(durations>=time);deaths=np.sum((durations==time)&died)
        survival*=1-deaths/at_risk;points.append((time,survival))
    out=np.ones(len(grid))
    for time,value in points:out[grid>=time]=value
    out[grid>durations.max()]=np.nan
    return out


def contact_and_wall(records,out,gallery):
    groups=[]
    for w in (.5,1.,1.5):
        group=[r for r in records if r['accepted'] and r['task']['family']=='adhesion_activity' and r['task']['n']==36 and r['task']['mu']==.5 and r['task']['w']==w]
        groups.append(group)
    fig,axes=plt.subplots(1,3,figsize=(13,4));rows=[];residences=[]
    for j,(w,group) in enumerate(zip((.5,1.,1.5),groups)):
        if not group:continue
        times=np.array([s['time'] for s in group[0]['samples']])/10
        gap=[[s['neighbor_overlap']-s['continuous_survival'] for s in r['samples']] for r in group]
        plot_line(axes[0],times,gap,f'W={w:g}, {len(group)}/3',COLORS[j])
        for r in group:
            changed=set();losses=[];last_loss={};returns=[]
            for event in r['pair_changes']:
                for pair in event['lost']:
                    pair=tuple(pair);changed.add(pair);last_loss[pair]=event['time']
                    losses.append((pair,event['time']))
                for pair in event['gained']:
                    pair=tuple(pair);changed.add(pair)
                    if pair in last_loss:returns.append((pair,last_loss.pop(pair),event['time']))
            eligible=[(pair,time) for pair,time in losses if time<=r['task']['duration']-10]
            quick={(pair,birth) for pair,birth,end in returns if end-birth<=10}
            row=dict(key=r['key'],w=w,seed=r['task']['seed'],transactions=r['summary']['contact_changing_transactions'],
                distinct_changed_pairs=len(changed),eligible_losses=len(eligible),returns_within_tau=sum(x in quick for x in eligible),
                newly_born_contact_episodes=sum(not e['left_censored'] for e in r['contact_episodes']))
            rows.append(row);axes[1].scatter(row['distinct_changed_pairs'],row['transactions'],color=COLORS[j],s=35)
            flags=np.array([s['touching'] for s in r['samples']]);ts=np.array([s['time'] for s in r['samples']])
            for cell in range(r['task']['n']):
                birth=ts[0] if flags[0,cell] else None;left=bool(flags[0,cell])
                for k in range(1,len(ts)):
                    if flags[k,cell] and not flags[k-1,cell]:birth=ts[k];left=False
                    if flags[k-1,cell] and not flags[k,cell]:
                        residences.append(dict(key=r['key'],w=w,seed=r['task']['seed'],cell=cell,birth=birth,end=ts[k],duration=ts[k]-birth,left_censored=left,right_censored=False));birth=None
                if birth is not None:residences.append(dict(key=r['key'],w=w,seed=r['task']['seed'],cell=cell,birth=birth,end=ts[-1],duration=ts[-1]-birth,left_censored=left,right_censored=True))
        grid=np.linspace(0,30,121);curves=[km_curve(r['contact_episodes'],grid) for r in group];curves=[x for x in curves if x is not None]
        if curves:
            arr=np.array(curves);m=[];sd=[]
            for col in arr.T:
                good=col[np.isfinite(col)];a,b=mean_sd(good) if len(good) else (np.nan,np.nan);m.append(a);sd.append(b)
            axes[2].plot(grid/10,m,color=COLORS[j],label=f'W={w:g}; {len(curves)} seeds with births')
            axes[2].fill_between(grid/10,np.array(m)-sd,np.array(m)+sd,color=COLORS[j],alpha=.15)
    axes[0].set(xlabel='Time / τT',ylabel='Re-formed fraction of initial contacts');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Distinct pairs gained or lost',ylabel='Contact-changing transactions')
    axes[2].set(xlabel='Contact age / τT',ylabel='New-contact survival estimate',ylim=(-.03,1.03));axes[2].legend(fontsize=7)
    save(fig,out,'12_contact_reversals','Frequent events need not imply lasting neighbor exchange',
         'N=36, ρ=1.2, μ=0.5. Left: initial-neighbor overlap minus unbroken survival identifies re-formed contacts. Middle: each dot is one completed seed; repeated changes can involve only a few distinct cell pairs. Right: product-limit survival of contacts born during observation, excluding initial left-censored episodes and retaining right censoring. Curves average seed estimates, not pooled contact episodes; bands are seed SD. Sampled wall-contact residence episodes are exported separately and have 0.5 τR resolution, including possible remeshing-induced flicker.',gallery)
    for name,data in (('contact_reversals.csv',rows),('wall_contact_residences.csv',residences)):
        if data:
            with (out/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)


def boundary_profiles(records,out,gallery):
    fig,axes=plt.subplots(1,3,figsize=(13,4));rows=[];kindrows=[]
    for r in records:
        if not r['accepted']:continue
        for kind in sorted({e['kind'] for e in r['events']}):
            events=[e for e in r['events'] if e['kind']==kind]
            kindrows.append(dict(key=r['key'],family=r['task']['family'],seed=r['task']['seed'],kind=kind,
                committed_transactions=len(events),contact_changing=sum(bool(e['lost'] or e['gained']) for e in events)))
    for j,n in enumerate((36,100)):
        group=[r for r in records if r['accepted'] and r['task']['family']=='adhesion_activity' and r['task']['n']==n and r['task']['w']==1 and r['task']['mu']==.5]
        per_seed=[]
        for r in group:
            samples=r['samples'];dt=r['task']['sample'];l0=np.sqrt(r['task']['rho'])
            x=np.array([s['centers'] for s in samples]);dist=np.array([s['wall_distance'] for s in samples[:-1]])/l0
            shape=np.array([s['shape'] for s in samples[:-1]]);free=np.array([s['free_surface'] for s in samples[:-1]])
            speed=np.linalg.norm(np.diff(x,axis=0),axis=2)/(dt*l0)
            participation=np.zeros_like(shape)
            for e in r['events']:
                if not e['cells']:continue
                frame=min(int(e['time']/dt),len(shape)-1)
                participation[frame,e['cells']]+=1
            values=[]
            for lo,hi in ((0,.5),(.5,1),(1,2),(2,5)):
                mask=(dist>=lo)&(dist<hi)
                values.append([shape[mask].mean(),speed[mask].mean(),participation[mask].sum()/(mask.sum()*dt)] if mask.any() else [np.nan]*3)
                for isfree in (False,True):
                    split=mask&(free==isfree)
                    if split.any():rows.append(dict(key=r['key'],n=n,seed=r['task']['seed'],distance_low=lo,distance_high=hi,
                        has_free_film=isfree,cell_time=float(split.sum()*dt),shape=float(shape[split].mean()),
                        speed=float(speed[split].mean()),event_participation_rate=float(participation[split].sum()/(split.sum()*dt))))
            per_seed.append(values)
        if per_seed:
            arr=np.array(per_seed)
            for col in range(3):
                m=[];sd=[]
                for b in range(4):
                    good=arr[:,b,col][np.isfinite(arr[:,b,col])];a,e=mean_sd(good) if len(good) else (np.nan,np.nan);m.append(a);sd.append(e)
                axes[col].errorbar(range(4),m,yerr=sd,fmt='o-',color=COLORS[j],label=f'N={n}, {len(group)}/3',capsize=3)
    for ax,label in zip(axes,('Shape index','Speed / (L₀ / τR)','Event participations / cell / τR')):
        ax.set(xticks=range(4),xticklabels=['0–0.5','0.5–1','1–2','2–5'],xlabel='Centroid distance from wall / L₀',ylabel=label)
    axes[0].legend(fontsize=8)
    save(fig,out,'13_boundary_profiles','Where deformation and neighbor changes occur',
         'W=1, ρ=1.2, μ=0.5; mean ± seed SD. Event participation counts each affected cell once per contact-changing transaction and divides by observed cell-time in the distance bin. Bins use the preceding stored cell centroids (0.5 τR resolution). The CSV also separates cells with and without any free film: distance from the container wall alone does not identify tissue interior.',gallery)
    for name,data in (('boundary_profiles.csv',rows),('transition_kinds.csv',kindrows)):
        if data:
            with (out/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'circle/data/experiments_20261114');p.add_argument('--out',type=Path,default=ROOT/'figures/circle/experiments_20261114');a=p.parse_args()
    manifest,records=load_records(a.data);a.out.mkdir(parents=True,exist_ok=True);gallery=[]
    rows=[]
    for r in records:
        t=r['task'];row={k:t[k] for k in ('family','n','w','rho','mu','tau','aspect','pattern','seed','dt','spacing','warmup','duration')};row.update(key=r['key'],accepted=r['accepted'],stage=r['stage'],error=r.get('error',''))
        for key in ('event_rate','contact_changing_transactions','shape_mean','contacts','packing_mean','attached','free','final_neighbor_overlap','final_continuous_survival'):
            row[key]=scalar(r,key) if r['accepted'] else ''
        rows.append(row)
    with (a.out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    response_maps(records,a.out,gallery);persistence(records,a.out,gallery);long_dynamics(records,a.out,gallery)
    paircounts=pulse_response(records,a.out,gallery);geometry(records,a.out,gallery);spaces(records,a.out,gallery)
    event_diagnostics(records,a.out,gallery);refinement(records,a.out,gallery);correlations(records,a.out,gallery);contact_and_wall(records,a.out,gallery);boundary_profiles(records,a.out,gallery)
    counts=defaultdict(lambda:[0,0])
    for r in records:counts[r['task']['family']][0]+=r['accepted'];counts[r['task']['family']][1]+=1
    report=['# Exploratory dynamics experiments','',f"{sum(r['accepted'] for r in records)} of {len(records)} runs completed their scheduled geometry checks. Passing sampled checks is not proof of continuous validity or numerical convergence.",'','| Family | Completed / attempted |','|---|---|']
    report.extend(f'| {k} | {v[0]} / {v[1]} |' for k,v in counts.items())
    audit_path=ROOT/'circle/data/experiments_cadence_audit_20261114/REPORT.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text());report+=['',f"Additional validation audit: {audit['completed']}/{audit['attempted']} selected baseline runs pass checks at every timestep after preparation. Their stored trajectories and event records match the corresponding ordinary runs exactly. This does not certify the unmonitored preparation intervals or every other experiment."]
    report+=['','## Protocol and interpretation','','Three seeds per scientific point (two in the refinement gate). Errors and bands are seed SD, not uncertainty from treating cell-frames as independent. A point with one accepted seed has no estimated seed SD. Failed trajectories are omitted from observable averages and remain in the outcome table; complete-case selection can bias the surviving sample.','',
             'Preparation retains the established checks every 0.5 τR; transient unresolved crossings were found when testing a finer preparation check interval. Measurement and warmup are checked every 0.5 τR except the high-activity refinement gate, which checks every timestep. Microfold tolerance and transition guards were not loosened.','',
             'The original force law and OU normalization are unchanged. μ is ΔT/T₀ under unit white-noise covariance in code time units. The persistence variance match uses the Euler–Maruyama stationary variance. Measurements distinguish transactions, lost/gained contacts, initial-neighbor overlap and unbroken survival.','',
             'The observer records outer committed topology transactions, suppressing nested events rolled back by a parent. Internal suboperations within one outer transaction are coalesced; pair changes are also checked every scheduled timestep. Raw event and right/left-censored contact-episode records are saved in each compressed run.','',
             'The ellipse preserves container area and cell count. It uses Euclidean closest-point wall projection and an affine-deformed seed network followed by relaxation; global anisotropy and local curvature still covary. Circle-only traction profiles must not be reused as ellipse arclength profiles.','',
             f'Complete same-seed pulse/sham pair counts: `{json.dumps(paircounts,sort_keys=True)}`.','',
             'Contact-specific tension–length lag correlations need persistent film lineage IDs across transitions and remain deferred. Current tension correlations are cell-averaged. Mechanosensitive feedback, spatially correlated noise, growth/division and osmotic forcing remain separate extensions.','',
             f"Core source SHA256: `{manifest['source_hash']}`.",'','## Figures','']
    for item in gallery:report+=['### '+item['title'],'',f"![{item['title']}]({item['name']}.png)",'',item['caption'],'']
    (a.out/'REPORT.md').write_text('\n'.join(report).rstrip()+'\n')
    content=['<!doctype html><meta charset="utf-8"><title>Active foam dynamics experiments</title><style>body{max-width:1250px;margin:35px auto;padding:0 24px;font:17px/1.5 system-ui;color:#20303d}img{width:100%}section{margin:45px 0}p{max-width:1000px}a{color:#176b94}</style><h1>Active foam: exploratory dynamics experiments</h1>',f'<p>{sum(r["accepted"] for r in records)} / {len(records)} completed runs. Fixed walls; original force law. Shading/error bars show seed SD. Missing runs are not physical phases.</p><p><a href="FINDINGS.md">Main findings</a> · <a href="REPORT.md">Methods and limitations</a> · <a href="summary.csv">Run summary CSV</a></p>']
    for item in gallery:content.append(f'<section><h2>{html.escape(item["title"])}</h2><a href="{item["name"]}.pdf">PDF</a><img src="{item["name"]}.png"><p>{html.escape(item["caption"])}</p></section>')
    (a.out/'index.html').write_text('\n'.join(content));print('\n'.join(report[:16]))


if __name__=='__main__':main()
