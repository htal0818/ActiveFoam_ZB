"""Eight full-size founding cells to 256, with division but no growth."""
import gzip
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from circle import protocol
from circle.biology import BiologyFoam
from circle import biology_experiments
from circle.biology_experiments import geometry,watch
from circle.experiments import source_hash
from circle.sweep import serializable
from circle.scripts.division_lineage import render
from circle.scripts.division_dynamics import baseline,sample
from circle.scripts.division_contact_analysis import filtered_record

NAME='division_8_to_256_20261114'


def vectorized_uid_pairs(foam):
    faces=foam.e_f[np.all(foam.e_f>=0,axis=1)]
    return set(map(tuple,np.sort(foam.cell_uid[faces],axis=1).tolist()))


def plot_dynamics(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    summary=result['dynamics_summary'];x=np.arange(len(summary))
    axes[0,0].bar(x,[r['rate'] for r in summary],color='#278f91')
    axes[0,0].set(xticks=x,xticklabels=[f"{r['n']//2}→{r['n']}" for r in summary],
        ylabel='Persistent pair changes / cell / τR',title='First 4 τR after each division')
    for stage in result['stages']:
        filtered=filtered_record(stage,.2)
        label=f"N = {stage['n']}"
        axes[0,1].plot([r['lag'] for r in filtered['records']],
            [r['survival'] for r in filtered['records']],label=label)
        axes[1,1].plot([r['lag'] for r in stage['records']],
            [r['residual_msd_over_area'] for r in stage['records']],label=label)
    axes[0,1].set(xlabel='Time after division / τR',ylabel='Fraction without a persistent loss',
                  title='Survival of post-division contacts',ylim=(0,1.02))
    axes[0,1].legend(fontsize=8)
    frames=result['frames']
    axes[1,0].plot([f['time'] for f in frames],[np.mean(f['shape']) for f in frames],color='#278f91')
    for row in result['rounds']:axes[1,0].axvline(row['time'],color='gray',lw=.8,ls=':')
    axes[1,0].set(xlabel='Time / τR',ylabel='Mean shape index',title='Shape changes and relaxation')
    axes[1,1].set(xlabel='Time after division / τR',ylabel='Residual MSD / initial mean area',
                  title='Motion with global contraction removed')
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('8 → 256: neighbor dynamics without growth\n'
        'Division excluded from counts; new contact state must persist ≥ 0.2 τR',fontsize=15)
    fig.savefig(out/'neighbor_dynamics.png',dpi=180)
    fig.savefig(out/'neighbor_dynamics.pdf')
    plt.close(fig)


def run():
    radius=.7*np.sqrt(8)
    print('Preparing eight full-size cells at zero noise.',flush=True)
    base=protocol.equilibrated(8,w=1.,rho=1.,seed=1,radius=radius,dt=.001,
                              edpc=.025,anneal_mu=0.,anneal_tauT=.5,
                              quench_tauT=1.,relax_steps=1500)
    foam=BiologyFoam.from_disc(base,gain=0.,growth_rate=0.)
    events=[];watch(foam,events)
    frames=[];checks=[];rounds=[];stages=[];ref=None;records=[]
    print('Prepared; starting five synchronous rounds.',flush=True)

    def capture(label=''):
        frame=geometry(foam,label);frame['area']=foam.f_area.copy()
        frames.append(serializable(frame))

    def check():
        quality=protocol.validate(foam)
        np.testing.assert_allclose(foam.preferred.sum(),8.,rtol=1e-13)
        assert foam.radius==radius and foam.growth_rate==foam.feedback_gain==0.
        checks.append(dict(time=foam.bio_time,n=foam.nFa,**serializable(quality)))

    check();capture('Eight founding cells')
    for step in range(1,22001):
        foam.advance_scheduled(step,mu=0.)
        if step%100==0:
            check()
            if ref is not None:records.append(sample(foam,events,ref))
        if step in {2000,6000,10000,14000,18000}:
            if ref is not None:
                stages.append(dict(time=ref['time'],end=foam.bio_time,n=foam.nFa,
                    records=records,events=events[ref['event_start']:].copy()))
            before=foam.nFa;area=float(foam.f_area.sum())
            capture('Before synchronous division')
            if not foam.divide_synchronously():
                raise RuntimeError(f'Division rejected: {foam.division_log[-1]}')
            assert foam.nFa==before*2
            np.testing.assert_allclose(foam.f_area.sum(),area,rtol=1e-10)
            check();capture(f'Synchronous division: {before} → {foam.nFa}')
            rounds.append(dict(time=foam.bio_time,before=before,after=foam.nFa,
                preferred_before=8.,preferred_after=float(foam.preferred.sum()),
                area_before=area,area_after=float(foam.f_area.sum())))
            ref=baseline(foam,events);records=[sample(foam,events,ref)]
            print(f't={foam.bio_time:.1f}: {before} → {foam.nFa}; total K/P0={foam.preferred.sum():.15g}',flush=True)
        elif step%100==0:capture()
        if step%1000==0:print(f'  t={foam.bio_time:.1f}, N={foam.nFa}',flush=True)
    stages.append(dict(time=ref['time'],end=foam.bio_time,n=foam.nFa,
        records=records,events=events[ref['event_start']:].copy()))
    check()
    assert [r['after'] for r in rounds]==[16,32,64,128,256]
    np.testing.assert_allclose(foam.preferred,np.full(256,1/32),rtol=1e-8)
    summary=[]
    for stage in stages:
        filtered=filtered_record(stage,.2)
        summary.append(dict(n=stage['n'],raw_transactions=len(stage['events']),
            persistent_pair_changes=len(filtered['changes']),
            rate=len(filtered['changes'])/(stage['n']*4),
            contact_survival=filtered['records'][-1]['survival'],
            residual_msd=stage['records'][-1]['residual_msd_over_area']))
    return serializable(dict(source_sha256=source_hash(ROOT),
        parameters=dict(dt=.001,spacing=.025,interval=4.,wall_radius=radius,
            W=1.,P0=10.,tauT=10.,mu=0.,growth_rate=0.,feedback_gain=0.,
            initial_preferred_area=8.,seed=1),
        preparation='Eight-cell clipped Voronoi; zero-adhesion relaxation, zero-noise '
                    'anneal at W=1, then zero-noise relaxation. See script for exact durations.',
        rounds=rounds,division_log=foam.division_log,checks=checks,frames=frames,
        stages=stages,dynamics_summary=summary))


if __name__=='__main__':
    fast_contacts='--reference-contacts' not in sys.argv
    if fast_contacts:
        # Only the observer is replaced; integration and topology use their
        # existing methods. Record this choice alongside the parameters.
        biology_experiments.uid_pairs=vectorized_uid_pairs
    data=ROOT/'circle/data'/NAME;data.mkdir(parents=True,exist_ok=True)
    out=ROOT/'figures/circle'/NAME;out.mkdir(parents=True,exist_ok=True)
    path=data/'simulation.json.gz'
    if '--render-only' in sys.argv:result=json.load(gzip.open(path,'rt'))
    else:
        result=run()
        result['measurement_implementation']=('vectorized_uid_pairs' if fast_contacts
                                               else 'original uid_pairs')
        with gzip.open(path,'wt') as handle:json.dump(result,handle)
    render(result,out)
    plot_dynamics(result,out)
    (out/'dynamics_summary.json').write_text(json.dumps(result['dynamics_summary'],indent=2)+'\n')
    print(json.dumps(result['dynamics_summary'],indent=2),flush=True)
