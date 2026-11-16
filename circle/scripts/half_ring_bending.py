"""Activate 24 of 48 cell profiles without prescribing their displacement."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from circle.bending import compare_midlines, smooth_patch
from circle.collective import save_result
from circle.scripts import bending_experiments as experiment
from circle.scripts.invagination_preview import draw

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'circle/data/bending_48_20261115'
DATA = ROOT/'circle/data/half_ring_bending_20261115'
OUT = ROOT/'figures/circle/half_ring_bending_20261115'


def plateau_weights(n=48, cells=24, taper_cells=4):
    """Flat central activation with cubic-smooth taper on both ends of a half-ring."""
    theta = 2*np.pi*(np.arange(n)+.5)/n
    distance = np.abs(np.angle(np.exp(1j*(theta+np.pi/2))))
    x = np.clip((np.pi*cells/n-distance)/(2*np.pi*taper_cells/n), 0., 1.)
    return x*x*(3-2*x)


def configs():
    common = dict(apical=2., bulk=5., lateral=1., patch=24, ramp=5., delay=20.,
                  duration=100., dt=.0025, spacing=.075, collapse_fraction=.005)
    return [dict(common, name='half_cosine'),
            dict(common, name='half_plateau', activation_weights=plateau_weights().tolist()),
            dict(common, name='half_A4_L1', apical=4.),
            dict(common, name='half_A8_L1', apical=8.),
            dict(common, name='half_A4_L0', apical=4., lateral=0.)]


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def analyze(run, control):
    reference = {round(m['elapsed'], 8): m for m in control['history']}
    history = []
    mask = np.asarray(run['activation_weights']) > 0
    for m in run['history']:
        c = reference[round(m['elapsed'], 8)]
        result = compare_midlines(m, c, 24)
        result['elapsed'] = m['elapsed']
        if result['radial_profile_valid']:
            cp = np.asarray(c['midline'])-c['lumen_center']
            angles = np.arctan2(cp[:, 1], cp[:, 0])
            response = np.interp(angles, result['theta'], result['radial_residual'], period=2*np.pi)
            result['selected_profiles_moving_inward'] = int(np.sum(mask & (response < -result['displacement_tolerance'])))
            result['selected_profiles_with_curvature_reversal'] = int(np.sum(mask &
                (np.asarray(m['curvature']) < 0) & (np.asarray(m['curvature_wide']) < 0)))
        history.append(result)
    last = history[-1]
    summary = dict(run['summary'], selected_profiles=int(mask.sum()),
                   total_apical_target_increment=float(run['parameters']['apical']*np.sum(run['activation_weights'])),
                   final_diagnostics=last,
                   initial_control_match=True,
                   interpretation='Selection of 24 profiles does not impose inward motion on all 24.')
    save_result(DATA/f"{run['parameters']['name']}_analysis.json", dict(summary=summary,
        history=[{k:v for k,v in h.items() if not isinstance(v, np.ndarray)} for h in history]))
    return summary


def render(runs, control, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    OUT.mkdir(parents=True, exist_ok=True)
    # Show both prescribed profiles and all valid outcomes, without selecting by appearance.
    fig, axes = plt.subplots(2, 3, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()
    control_frame = dict(control['frames'][-1], active_cells=np.zeros(48, dtype=bool))
    draw(axes[0], control_frame, 'Unactivated control · +100 τR')
    labels = {'half_cosine':'Cosine · apical +2, lateral +1',
              'half_plateau':'Plateau · apical +2, lateral +1',
              'half_A4_L1':'Cosine · apical +4, lateral +1',
              'half_A8_L1':'Cosine · apical +8, lateral +1',
              'half_A4_L0':'Cosine · apical +4, lateral unchanged'}
    for ax, run in zip(axes[1:], runs):
        draw(ax, run['frames'][-1], f"{labels[run['parameters']['name']]}\n+{run['summary']['last_valid_elapsed']:.1f} τR · {run['summary']['status']}")
    fig.suptitle('Half-ring activation · 24 of 48 profiles selected (pink)\n'
                 'All tested protocols at their last valid sampled times · fixed topology and wall')
    fig.savefig(OUT/'half_ring_snapshots.png', dpi=200)
    fig.savefig(OUT/'half_ring_snapshots.pdf'); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    theta = (360*(np.arange(48)+.5)/48-270+180) % 360-180
    order = np.argsort(theta)
    for run in runs:
        name = run['parameters']['name']; summary = summaries[name]
        axes[0].plot(theta[order], np.asarray(run['activation_weights'])[order], 'o-', ms=3, label=labels[name])
        last = summary['final_diagnostics']
        if last['radial_profile_valid']:
            angle = (np.asarray(last['theta'])*180/np.pi+270) % 360-180
            index = np.argsort(angle)
            axes[1].plot(angle[index], np.asarray(last['radial_residual'])[index], label=labels[name])
        axes[2].plot(theta[order], np.asarray(run['history'][-1]['curvature_wide'])[order], 'o-', ms=3, label=labels[name])
    for ax in axes:
        ax.axhline(0, c='k', ls=':', lw=1); ax.axvspan(-90, 90, color='#f6c2ae', alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set(xlabel='Initial material angle from vegetal pole (degrees)', ylabel='Prescribed activation weight')
    axes[1].set(xlabel='Aligned polar angle from vegetal pole (degrees)', ylabel='Radial response without uniform contraction')
    axes[2].set(xlabel='Initial material angle from vegetal pole (degrees)', ylabel='Wider-scale signed midline curvature')
    fig.suptitle('Half-ring activation and measured response at each last valid sampled time')
    fig.savefig(OUT/'half_ring_diagnostics.png', dpi=180); fig.savefig(OUT/'half_ring_diagnostics.pdf'); plt.close(fig)

    # Display the completed branch with the most selected profiles reversing curvature;
    # shoulder depth breaks ties. Retain every branch above, including any stopped ones.
    completed = [r for r in runs if r['summary']['status'] == 'complete']
    def score(run):
        d = summaries[run['parameters']['name']]['final_diagnostics']
        return (d.get('selected_profiles_with_curvature_reversal', 0), d.get('shoulder_depth', 0.))
    selected = max(completed, key=score)
    name = selected['parameters']['name']
    save_result(OUT/'selection.json', dict(name=name,
        criterion='Completed run with most selected profiles reversing both curvature estimates, then greatest shoulder depth.',
        summary=summaries[name]))
    end = selected['frames'][-1]['time']
    reference = min(control['frames'], key=lambda f:abs(f['time']-end))
    assert abs(reference['time']-end) < 1e-7
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    draw(axes[0], selected['frames'][0], 'Before activation · 24 selected profiles')
    draw(axes[1], dict(reference, active_cells=np.zeros(48, dtype=bool)), 'Matched unactivated control')
    draw(axes[2], selected['frames'][-1], f"{labels[name]}\n+{selected['summary']['last_valid_elapsed']:.1f} τR")
    fig.suptitle('Actual simulated geometry · rigid wall, closed compliant lumen\n'
                 'Pink marks the prescribed half-ring activation; displacement is not imposed')
    fig.savefig(OUT/'half_ring_before_after.png', dpi=200)
    fig.savefig(OUT/'half_ring_before_after.pdf'); plt.close(fig)

    # Match physical times for the side-by-side movie.
    movie_runs = [control, selected]
    frame_maps = [{round(f['time']-r['frames'][0]['time'], 8):f for f in r['frames']} for r in movie_runs]
    times = sorted(set(frame_maps[0]) & set(frame_maps[1]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.3), constrained_layout=True)
    fig.suptitle('24 selected profiles · cortical activation only\nActual saved states; no imposed inward displacement')
    def animate(i):
        t = times[min(i, len(times)-1)]
        for j, (ax, run, lookup) in enumerate(zip(axes, movie_runs, frame_maps)):
            frame = lookup[t]
            if j == 0:
                frame = dict(frame, active_cells=np.zeros(48, dtype=bool))
            label = 'Unactivated control' if j == 0 else labels[run['parameters']['name']]
            ax.clear(); draw(ax, frame, f"{label}\n+{t:.1f} τR")
        return []
    movie = FuncAnimation(fig, animate, frames=len(times)+3, interval=250, blit=False)
    movie.save(OUT/'half_ring_motion.gif', writer=PillowWriter(fps=4), dpi=110); plt.close(fig)


def main():
    experiment.PREPARED = experiment.prepare()
    experiment.DATA = DATA; DATA.mkdir(parents=True, exist_ok=True)
    settings = configs()
    assert np.count_nonzero(smooth_patch(48, 24)) == 24
    assert np.count_nonzero(plateau_weights()) == 24
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context('fork')) as pool:
        for summary in pool.map(experiment.run_case, settings): print(json.dumps(summary), flush=True)
    runs = [read(DATA/f"{c['name']}.json.gz") for c in settings]
    control = read(BASE/'control_B5.json.gz')
    for run in runs:
        for initial, reference in zip(run['frames'][0]['polygons'], control['frames'][0]['polygons']):
            np.testing.assert_array_equal(initial, reference)
        assert run['initial_lumen_pressure'] == control['initial_lumen_pressure']
        assert run['initial_lumen_area'] == control['initial_lumen_area']
    summaries = {r['parameters']['name']:analyze(r, control) for r in runs}
    save_result(DATA/'summary.json', summaries)
    render(runs, control, summaries)
    for name, summary in summaries.items():
        print(name, {k:v for k,v in summary['final_diagnostics'].items() if not isinstance(v, np.ndarray)}, flush=True)


if __name__ == '__main__':
    main()
