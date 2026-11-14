"""Replay and verify one experiment while rendering actual film geometry to a GIF."""
import argparse,gzip,json,sys,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from circle import biology_experiments as experiments
from circle.biology import BiologyFoam


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=['division','feedback']);a=p.parse_args()
    task=experiments.BioTask('division',1,rho=.6,duration=60.,mu=.05,growth=True,divide=True) if a.case=='division' else experiments.BioTask('feedback',1,gain=.5,delay=10.)
    path=ROOT/'circle/data/biology_20261114/runs'/f'{task.key}.json.gz'
    with gzip.open(path,'rt') as f:baseline=json.load(f)
    if not baseline['accepted']:raise ValueError('selected reference did not complete')
    captured=[]
    def capture(foam,label=''):
        if a.case=='feedback' and foam.bio_time<60-1e-8:return
        captured.append(experiments.geometry(foam,label))
    class FilmedBiologyFoam(BiologyFoam):
        def advance_scheduled(self,*args,**kwargs):
            result=super().advance_scheduled(*args,**kwargs)
            if round(self.bio_time/self.dt)%round(.5/self.dt)==0:capture(self)
            return result
        def divide_synchronously(self):
            capture(self,'before synchronous division')
            result=super().divide_synchronously()
            capture(self,'after synchronous division')
            return result
    experiments.BiologyFoam=FilmedBiologyFoam
    replay=experiments.run(task)
    assert replay['accepted'],replay.get('error')
    for key in ('samples','events','division_log','frames','summary'):
        assert replay[key]==baseline[key],f'replay differs: {key}'
    out=ROOT/'figures/circle/biology_20261114';out.mkdir(parents=True,exist_ok=True)
    images=[];durations=[]
    fig,ax=plt.subplots(figsize=(6,6),dpi=100);norm=plt.Normalize(3.5,4.7)
    for frame in captured:
        ax.clear();center=np.array(frame['center']);radius=frame['radius']
        polygons=[np.array(p)-center for p in frame['polygons']]
        collection=PolyCollection(polygons,cmap='viridis',norm=norm,edgecolors='#172329',linewidth=.9)
        collection.set_array(np.asarray(frame['shape']));ax.add_collection(collection)
        ax.add_patch(Circle((0,0),radius,fill=False,color='black',lw=1.8))
        for polygon,uid in zip(polygons,frame['uids']):
            mid=polygon.mean(axis=0);ax.text(*mid,str(uid),fontsize=7,ha='center',va='center',color='white')
        rho=np.sum(frame['preferred'])/(np.pi*radius**2)
        label='Synchronous growth + division' if a.case=='division' else 'Delayed feedback: g=0.5, delay=10, μ=0'
        ax.set(xlim=(-1.08*radius,1.08*radius),ylim=(-1.08*radius,1.08*radius),xlabel='x / ℓ',ylabel='y / ℓ',
               title=f'{label}\nt={frame["time"]:.1f} τR   N={len(polygons)}   ρ={rho:.2f}\n{frame["label"] or "Rigid, frictionless circular wall"}')
        ax.set_aspect('equal');fig.tight_layout();fig.canvas.draw()
        image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=128)
        images.append(image);durations.append(700 if frame['label'] else 80)
    name='synchronous_division' if a.case=='division' else 'delayed_feedback'
    images[0].save(out/f'{name}.gif',save_all=True,append_images=images[1:],duration=durations,loop=0,optimize=False)
    plt.close(fig)
    (out/f'{name}_replay.json').write_text(json.dumps(dict(reference_key=task.key,exact_replay=True,frames=len(images),
        verified_fields=['samples','events','division_log','frames','summary'],note='Numbers label persistent cell IDs; colors show shape index on the fixed range 3.5–4.7. Longer pauses around division are for display only.'),indent=2)+'\n')
    print(name,len(images),'frames; exact replay verified',flush=True)


if __name__=='__main__':main()
