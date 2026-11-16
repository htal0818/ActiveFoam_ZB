"""Render the 48-profile planar blastula cross-section, with measured controls.

Reads actual validated frames from invagination_size; no generated geometry or
interpolated dynamics. The 48 profiles are a provisional cross-sectional model,
not the total number of cells on the embryo's surface.
"""
import gzip
import json
from pathlib import Path

import numpy as np

from circle.collective import save_result
from circle.scripts.invagination_preview import draw

ROOT = Path(__file__).resolve().parents[2]
TAG = 'invagination_blastula_crosssection_48_20261115'


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    data = ROOT/'circle/data'/TAG
    out = ROOT/'figures/circle'/TAG
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(data/'simulation.json.gz', 'rt') as stream:
        result = json.load(stream)
    if result['baseline']['status'] != 'complete' or not result['runs']:
        raise ValueError('preparation did not complete; no activated state to display')
    baseline = result['baseline']['frames'][-1]
    active = result['runs']['apical']['frames']
    control = result['runs']['control']['frames']
    start = baseline['time']
    elapsed = active[-1]['time']-start
    if not np.isclose(control[-1]['time'], active[-1]['time'], atol=1e-8):
        raise ValueError('matched control did not reach the selected snapshot time')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), constrained_layout=True)
    draw(axes[0], baseline, 'Prepared ring · 48 cell profiles')
    draw(axes[1], control[-1], f'Control · +{elapsed:.1f} τR')
    draw(axes[2], active[-1], f'Apical activation · +{elapsed:.1f} τR')
    fig.suptitle('Planar cross-section of a sealed blastula · closed, compliant lumen\n'
                 'Orange: outward apical surface · blue: inward basal surface · pink: designated vegetal cells')
    fig.savefig(out/'cross_section_snapshots.png', dpi=200)
    fig.savefig(out/'cross_section_snapshots.pdf')
    plt.close(fig)

    # Include the constructed geometry, then several actually stored active frames.
    indices = sorted(set(np.linspace(0, len(active)-1, min(5, len(active)), dtype=int)))
    panels = [(result['initial'], 'Constructed annulus · before preparation')]
    panels += [(active[i], f"After activation · +{active[i]['time']-start:.1f} τR")
               for i in indices]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4), constrained_layout=True)
    for ax, (frame, title) in zip(axes.flat, panels):
        draw(ax, frame, title)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis('off')
    fig.suptitle('48-profile planar ring · actual simulation snapshots\n'
                 'Fixed wall; no division or growth. Last frame precedes the first proposed T1.')
    fig.savefig(out/'time_sequence.png', dpi=180)
    fig.savefig(out/'time_sequence.pdf')
    plt.close(fig)

    # Match each frame by time, rather than assuming identical stored-frame indices.
    paired = []
    for frame in active:
        matched = min(control, key=lambda c: abs(c['time']-frame['time']))
        if not np.isclose(matched['time'], frame['time'], atol=1e-8):
            raise ValueError('animation contains an unmatched control frame')
        paired.append((matched, frame))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), constrained_layout=True)
    fig.suptitle('48-profile planar cross-section · control and apical activation\n'
                 'Discrete saved frames; playback time is not physical time')
    def animate(i):
        left, right = paired[min(i, len(paired)-1)]
        for ax, frame, title in zip(axes, (left, right), ('Control', 'Apical activation')):
            ax.clear()
            draw(ax, frame, f"{title} · +{frame['time']-start:.1f} τR")
        return []
    animation = FuncAnimation(fig, animate, frames=len(paired)+3, interval=650, blit=False)
    animation.save(out/'cross_section.gif', writer=PillowWriter(fps=1.5), dpi=100)
    plt.close(fig)

    measurements = result['baseline']['measurements']
    tail = [m for m in measurements if m['time'] >= start-10-1e-8]
    areas = np.array([m['lumen_area'] for m in tail])
    pressures = np.array([m['lumen_pressure'] for m in tail])
    stability = dict(window_tauR=10.,
        relative_lumen_area_range=float(np.ptp(areas)/np.mean(areas)),
        lumen_pressure_range_over_P0=float(np.ptp(pressures)/result['parameters']['P0']),
        interpretation='Geometry stationarity diagnostic only, not a force-residual or convergence certificate.')
    summary = dict(cross_section_profiles=result['parameters']['n'],
                   biological_surface_cell_count_reference=512,
                   uniform_packing_estimate=float(np.sqrt(np.pi*512)),
                   geometry_calibrated_to_microscopy=False,
                   prepared_time=start, elapsed_after_activation=elapsed,
                   active_profiles=result['parameters']['active_cells'],
                   preparation_stability=stability,
                   status=result['runs']['apical']['status'],
                   reason=result['runs']['apical']['reason'])
    save_result(out/'cross_section_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    render()
