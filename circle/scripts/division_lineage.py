"""Run and render five synchronous divisions of one cell, without growth.

Run from the repository root with .venv/bin/python circle/scripts/division_lineage.py.
All times and lengths retain the founding-cell units. No model parameters are
rescaled as daughters get smaller. Geometry frames are actual simulation output.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from circle import protocol
from circle.benchmarks import free_area
from circle.biology import BiologyFoam
from circle.biology_experiments import geometry
from circle.disc import DiscFoam, circle_polyline
from circle.experiments import source_hash
from circle.sweep import serializable


def simulate(dt=.001, spacing=.025, interval=4.):
    radius = .7
    center = np.full(2, 4 * radius)
    ring = circle_polyline(center, np.sqrt(free_area(1.) / np.pi), spacing)
    base = DiscFoam.from_closed_cells(radius, [ring], center=center, rho=1.,
                                     w=1., dt=dt, edpc=spacing, seed=1)
    foam = BiologyFoam.from_disc(base, gain=0., growth_rate=0.)
    initial_preferred = float(foam.preferred.sum())
    frames, checks, rounds = [], [], []
    division_steps = {round((2. + i * interval) / dt) for i in range(5)}
    capture_every = max(1, round(.1 / dt))
    check_every = max(1, round(.05 / dt))

    def capture(label=''):
        frame = geometry(foam, label)
        frame['area'] = foam.f_area.copy()
        frames.append(serializable(frame))

    def check():
        quality = protocol.validate(foam)
        np.testing.assert_allclose(foam.preferred.sum(), initial_preferred, rtol=1e-13)
        assert foam.growth_rate == foam.feedback_gain == 0.
        assert foam.radius == radius
        checks.append(dict(time=foam.bio_time, n=foam.nFa, **serializable(quality)))

    check()
    capture('Initial cell')
    final_step = max(division_steps) + round(interval / dt)
    for step in range(1, final_step + 1):
        foam.advance_scheduled(step, mu=0.)
        if step % check_every == 0:
            check()
        if step in division_steps:
            before_n = foam.nFa
            before_area = float(foam.f_area.sum())
            before_preferred = float(foam.preferred.sum())
            capture('Before synchronous division')
            assert foam.divide_synchronously(), foam.division_log[-1]
            assert foam.nFa == 2 * before_n
            np.testing.assert_allclose(foam.f_area.sum(), before_area, rtol=1e-10)
            np.testing.assert_allclose(foam.preferred.sum(), before_preferred, rtol=1e-13)
            check()
            capture(f'Synchronous division: {before_n} → {foam.nFa}')
            rounds.append(dict(time=foam.bio_time, before=before_n, after=foam.nFa,
                               preferred_before=before_preferred,
                               preferred_after=float(foam.preferred.sum()),
                               area_before=before_area, area_after=float(foam.f_area.sum())))
            print(f't={foam.bio_time:.3f}: {before_n} → {foam.nFa}; '
                  f'total preferred area={foam.preferred.sum():.15g}', flush=True)
        elif step % capture_every == 0:
            capture()
        if step % round(1. / dt) == 0:
            print(f'  relaxed to t={foam.bio_time:.1f}, N={foam.nFa}', flush=True)
    check()
    assert [r['after'] for r in rounds] == [2, 4, 8, 16, 32]
    assert len(set(foam.cell_uid)) == 32
    return serializable(dict(source_sha256=source_hash(ROOT),
        parameters=dict(dt=dt, spacing=spacing, interval=interval, wall_radius=radius,
                        W=1., P0=10., tauT=10., mu=0., growth_rate=0., feedback_gain=0.,
                        initial_preferred_area=1., seed=1),
        description='Five synchronous equal-area divisions; daughter K partitions mother K. '
                    'The original scheduled foam transitions remain enabled between divisions. '
                    'No growth, tension noise, feedback, or wall motion.',
        rounds=rounds, division_log=foam.division_log, checks=checks, frames=frames))


def render(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    frames = result['frames']
    # Daughters inherit their first-generation ancestor's color; new septa are
    # visible as dark boundaries. Colors do not encode an inferred measurement.
    ancestry = {0: 0}
    for entry in result['division_log']:
        if entry['accepted']:
            for event in entry['events']:
                for daughter in event['daughters']:
                    ancestry[daughter] = daughter if event['parent'] == 0 else ancestry[event['parent']]
    palette = {0: '#98cec5', 1: '#56b5ad', 2: '#eeb46b'}

    def draw(ax, frame, label=True):
        center = np.asarray(frame['center'])
        radius = frame['radius']
        polygons = [np.asarray(p) - center for p in frame['polygons']]
        colors = [palette[ancestry[uid]] for uid in frame['uids']]
        ax.add_collection(PolyCollection(polygons, facecolors=colors,
                                        edgecolors='#243746', linewidth=1.1))
        ax.add_patch(Circle((0, 0), radius, fill=False, color='#697580', lw=1.4))
        ax.set(xlim=(-1.06 * radius, 1.06 * radius), ylim=(-1.06 * radius, 1.06 * radius))
        ax.set_aspect('equal')
        ax.axis('off')
        if label:
            ax.set_title(f'{len(polygons)} cell' + ('s' if len(polygons) != 1 else '') +
                         f'   ·   t = {frame["time"]:.1f} τR', fontsize=13)

    fig, ax = plt.subplots(figsize=(6.4, 6.8), dpi=100)
    fig.subplots_adjust(left=.03, right=.97, bottom=.1, top=.86)
    fig.suptitle('1 → 32 cells\nSynchronous division · no growth', fontsize=17, y=.98)
    footer = fig.text(.5, .045, '', ha='center', fontsize=11)
    images, durations = [], []
    for frame in frames:
        ax.clear()
        draw(ax, frame)
        footer.set_text(frame['label'] or 'Fixed wall · constant total K · mechanical relaxation')
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert(
            'P', palette=Image.Palette.ADAPTIVE, colors=128))
        durations.append(1000 if frame['label'] else 60)
    durations[-1] = 1800
    images[0].save(out / 'division_1_to_32.gif', save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False)
    plt.close(fig)

    selected = [next(f for f in reversed(frames) if len(f['uids']) == n)
                for n in [1, 2, 4, 8, 16, 32]]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.7), constrained_layout=True)
    for ax, frame in zip(axes.flat, selected):
        draw(ax, frame)
    fig.suptitle('Five synchronous division rounds, without growth\n'
                 'Same spatial scale and fixed wall in every panel', fontsize=17)
    fig.savefig(out / 'division_snapshots.png', dpi=180)
    fig.savefig(out / 'division_snapshots.pdf')
    plt.close(fig)

    times = np.array([f['time'] for f in frames])
    counts = np.array([len(f['uids']) for f in frames])
    preferred = np.array([sum(f['preferred']) for f in frames])
    areas = np.array([sum(f['area']) for f in frames])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    axes[0].step(times, counts, where='post', color='#267f7d', lw=2)
    axes[0].set(ylabel='Cell count', yticks=[1, 2, 4, 8, 16, 32])
    axes[1].plot(times, preferred / preferred[0], color='#267f7d', label='Total K / initial K')
    axes[1].plot(times, areas / areas[0], color='#d28b31', label='Total area / initial area')
    axes[1].legend(fontsize=8)
    axes[1].set(ylabel='Normalized total', ylim=(0, 1.1))
    axes[2].plot(times, areas / counts, color='#267f7d', label='Actual mean area')
    axes[2].step(times, preferred / counts, where='post', color='#d28b31', label='Mean preferred area')
    axes[2].set(ylabel='Area (founding-cell units)')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('Time / τR')
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out / 'division_conservation.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt', type=float, default=.001)
    parser.add_argument('--spacing', type=float, default=.025)
    parser.add_argument('--interval', type=float, default=4.)
    parser.add_argument('--name', default='division_1_to_32_20261114')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    data = ROOT / 'circle/data' / args.name
    out = ROOT / 'figures/circle' / args.name
    data.mkdir(parents=True, exist_ok=True)
    path = data / 'simulation.json.gz'
    if args.render_only:
        with gzip.open(path, 'rt') as handle:
            result = json.load(handle)
    else:
        result = simulate(args.dt, args.spacing, args.interval)
        with gzip.open(path, 'wt') as handle:
            json.dump(result, handle)
    render(result, out)
    print(f'Animation and snapshots: {out}', flush=True)


if __name__ == '__main__':
    main()
