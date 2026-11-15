"""Actual ring simulation snapshots; no division, ECM or inferred energy."""
from pathlib import Path
import json
import numpy as np
from circle.invagination import PolarFoam,TopologyLimit,LumenRupture
from circle import protocol
from circle.collective import save_result
from circle.force_probe import isolated_copy
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'circle/data/invagination_preview_20261115'
OUT=ROOT/'figures/circle/invagination_preview_20261115'


def evolve(f,duration):
    frames=[f.frame()];status='complete';reason=None;count=round(duration/f.dt)
    start=round(f.morph_time/f.dt);every=max(1,round(.5/f.dt))
    for k in range(1,count+1):
        try:
            f.advance_scheduled(start+k,0.)
            if k%every==0:
                protocol.validate(f);frames.append(f.frame())
        except (TopologyLimit,LumenRupture,ValueError) as exc:
            status='stopped';reason=f'{type(exc).__name__}: {exc}';break
    return dict(frames=frames,status=status,reason=reason,last_valid_time=frames[-1]['time'])


def simulate():
    f=PolarFoam.ring();initial=f.frame();protocol.validate(f)
    baseline=evolve(f,30.)
    if baseline['status']!='complete':return dict(initial=initial,baseline=baseline,runs={})
    runs={}
    for name,apical,lateral,delay in [('control',0.,0.,0.),('apical',2.,0.,0.),('lateral',0.,2.,0.),('combined',2.,2.,10.)]:
        work=isolated_copy(f);work.activation_start=work.morph_time;work.apical_amplitude=apical
        work.lateral_amplitude=lateral;work.lateral_delay=delay
        runs[name]=evolve(work,30.)
        print(name,runs[name]['status'],runs[name]['last_valid_time'],runs[name]['reason'],flush=True)
    return dict(initial=initial,baseline=baseline,runs=runs,parameters=dict(n=24,dt=.0025,spacing=.075,
        tauT=10.,P0=10.,W=1.,mu=0.,lumen_bulk_modulus=5.,lumen_initial_pressure=f.lumen_initial_pressure,
        lumen_reference_area=f.lumen_reference_area,wall_radius=f.radius,division=False,ECM=False),
        limitation='Preview stops before the first proposed topology operation; polarized inheritance across rearrangements is not yet validated.')


def draw(ax,frame,title):
    from matplotlib.collections import PolyCollection
    center=np.array(frame['center']);radius=frame['wall_radius']
    polygons=[np.array(p)-center for p in frame['polygons']]
    colors=['#f6c2ae' if active else '#ece9db' for active in frame['active_cells']]
    ax.add_collection(PolyCollection(polygons,facecolors=colors,edgecolors='#514f4c',linewidths=.8))
    lumen=np.array(frame['inner'])-center
    ax.fill(lumen[:,0],lumen[:,1],color='#d8edf7',zorder=-1)
    for name,col in [('outer','#c65339'),('inner','#367eaa')]:
        p=np.array(frame[name])-center;p=np.vstack((p,p[0]));ax.plot(p[:,0],p[:,1],color=col,lw=1.8)
    ax.add_patch(__import__('matplotlib').patches.Circle((0,0),radius,fill=False,ls='--',lw=1,color='#777777'))
    ax.set(xlim=(-radius*1.03,radius*1.03),ylim=(-radius*1.03,radius*1.03),aspect='equal',title=title)
    ax.axis('off')


def render(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True)
    if not r['runs']:raise RuntimeError(f"baseline did not complete: {r['baseline']['reason']}")
    selected=r['runs']['apical']['frames'][-1]
    # Compare the control at precisely the same elapsed time as the active case.
    control=min(r['runs']['control']['frames'],key=lambda f:abs(f['time']-selected['time']))
    start=r['baseline']['frames'][-1]
    fig,axes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
    draw(axes[0],start,'Before activation · 24 epithelial cells')
    draw(axes[1],control,f"Control · {selected['time']-start['time']:.1f} τR later")
    draw(axes[2],selected,f"Apical activation · {selected['time']-start['time']:.1f} τR later")
    fig.suptitle('Hollow epithelial ring with a closed, compliant lumen\nOrange: apical exterior · blue: basal lumen boundary · pink: vegetal cells',fontsize=13)
    fig.savefig(OUT/'initial_and_snapshot.png',dpi=170);fig.savefig(OUT/'initial_and_snapshot.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    draw(axes[0],r['initial'],'Constructed annular geometry · t=0')
    draw(axes[1],start,'After passive preparation · t=30 τR')
    fig.savefig(OUT/'initial_geometry.png',dpi=170);plt.close(fig)
    return dict(selected_time=selected['time'],elapsed=selected['time']-start['time'],status=r['runs']['apical']['status'],
                reason=r['runs']['apical']['reason'],lumen_area_initial=start['lumen_area'],
                lumen_area_final=selected['lumen_area'],lumen_pressure_initial=start['lumen_pressure'],
                lumen_pressure_final=selected['lumen_pressure'])

if __name__=='__main__':
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/'simulation.json'
    if path.exists():result=json.loads(path.read_text())
    else:result=simulate();save_result(path,result)
    summary=render(result);save_result(OUT/'summary.json',summary);print(summary)
