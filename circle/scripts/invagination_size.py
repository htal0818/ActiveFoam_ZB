"""Size-scaled ring snapshots using the existing polarized-foam equations.

Keeps constructed cell area and epithelial thickness fixed, enlarging the ring.
The wall is stationary within every run. Stops before an unsupported topology
operation, exactly as the 24-cell preview does.
"""
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np

from circle import protocol
from circle.collective import save_result
from circle.force_probe import isolated_copy
from circle.invagination import PolarFoam, TopologyLimit, LumenRupture
from circle.scripts.invagination_preview import draw
from circle.sweep import serializable

ROOT = Path(__file__).resolve().parents[2]


def inner_radius_for_size(n):
    thickness = np.sqrt(9 + 24 / np.pi) - 3
    return (n / np.pi - thickness**2) / (2 * thickness)


def save_compressed(path, result):
    temporary = path.with_suffix('.writing')
    with gzip.open(temporary, 'wt') as stream:
        json.dump(serializable(result), stream, separators=(',', ':'))
    temporary.replace(path)


def rotate_view(frame, angle):
    """Rotate all plotted coordinates together about the fixed ring center."""
    center = np.asarray(frame['center'])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    result = dict(frame)
    def transform(points):
        return (np.asarray(points)-center) @ rotation.T + center
    result['polygons'] = [transform(p) for p in frame['polygons']]
    result['outer'] = transform(frame['outer'])
    result['inner'] = transform(frame['inner'])
    return result


def evolve(foam, duration, label, checkpoint):
    start_time = foam.morph_time
    start_step = round(start_time / foam.dt)
    sample = max(1, round(.5 / foam.dt))
    store = max(sample, round(2.5 / foam.dt))
    report = max(store, round(5 / foam.dt))
    count = round(duration / foam.dt)
    latest = foam.frame()
    result = dict(frames=[latest], measurements=[], status='running', reason=None,
                  start_time=start_time, last_valid_time=start_time)
    wall_start = time.monotonic()
    for k in range(1, count + 1):
        try:
            foam.advance_scheduled(start_step + k, 0.)
            if k % sample == 0 or k == count:
                quality = protocol.validate(foam)
                latest = foam.frame()
                result['last_valid_time'] = latest['time']
                result['measurements'].append(dict(
                    time=latest['time'], lumen_area=latest['lumen_area'],
                    lumen_pressure=latest['lumen_pressure'],
                    cell_area_min=float(foam.f_area.min()),
                    cell_area_max=float(foam.f_area.max()), quality=quality))
                if k % store == 0 or k == count:
                    result['frames'].append(latest)
            if k % report == 0:
                save_compressed(checkpoint, result)
                print(f'{label}: elapsed {foam.morph_time-start_time:.1f} tauR; '
                      f'wall time {time.monotonic()-wall_start:.1f}s', flush=True)
        except (TopologyLimit, LumenRupture, ValueError) as exc:
            result.update(status='stopped', reason=f'{type(exc).__name__}: {exc}')
            break
    else:
        result['status'] = 'complete'
    if result['frames'][-1]['time'] != latest['time']:
        result['frames'].append(latest)
    result['wall_seconds'] = time.monotonic() - wall_start
    save_compressed(checkpoint, result)
    print(label, result['status'], 'last validated time', result['last_valid_time'],
          result['reason'], flush=True)
    return result


def simulate(args, data):
    radius = inner_radius_for_size(args.n)
    foam = PolarFoam.ring(n=args.n, inner_radius=radius, spacing=args.spacing, dt=args.dt)
    initial_quality = protocol.validate(foam)
    result = dict(initial=foam.frame(), initial_quality=initial_quality, runs={},
                  parameters=dict(n=args.n, inner_radius=radius,
                      outer_radius=float(np.sqrt(radius**2+args.n/np.pi)),
                      wall_radius=foam.radius, active_cells=int(foam.active_cells.sum()),
                      scaling='fixed constructed cell area and layer thickness; larger ring and fixed wall',
                      dt=args.dt, spacing=args.spacing, mu=0., W=1., P0=10., rho=1.,
                      tauT=foam.tauT, apical_mean_increment=2.,
                      lumen_bulk_modulus=foam.lumen_bulk_modulus,
                      lumen_reference_area=foam.lumen_reference_area,
                      lumen_initial_pressure=foam.lumen_initial_pressure,
                      preparation=args.prep, requested_activation_duration=args.duration,
                      validation_interval=.5, stored_frame_interval=2.5),
                  limitation='Stop before first proposed topology operation; polarized transition inheritance is not validated.')
    save_compressed(data/'initial.json.gz', result)
    result['baseline'] = evolve(foam, args.prep, 'preparation', data/'preparation.json.gz')
    if result['baseline']['status'] == 'complete':
        active = isolated_copy(foam)
        active.activation_start = active.morph_time
        active.apical_amplitude = 2.
        result['runs']['apical'] = evolve(active, args.duration, 'apical', data/'apical.json.gz')
        elapsed = result['runs']['apical']['last_valid_time'] - foam.morph_time
        result['runs']['control'] = evolve(foam, elapsed, 'matched control', data/'control.json.gz')
    save_compressed(data/'simulation.json.gz', result)
    return result


def render(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    n = result['parameters']['n']
    baseline = result['baseline']['frames'][-1]
    active_run = result['runs'].get('apical', result['baseline'])
    selected = active_run['frames'][-1]
    elapsed = selected['time'] - baseline['time']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    draw(axes[0], result['initial'], f'Constructed geometry · {n} epithelial cells')
    draw(axes[1], selected, f'Validated snapshot · t = {selected["time"]:.1f} τR')
    fig.suptitle('Closed, compliant lumen · fixed outer wall\n'
                 'Orange: apical · blue: basal · pink: designated activation sector')
    fig.savefig(output/'initial_and_snapshot.png', dpi=220)
    fig.savefig(output/'initial_and_snapshot.pdf')
    plt.close(fig)

    summary = dict(parameters=result['parameters'], elapsed_after_activation=elapsed,
                   selected_time=selected['time'], status=active_run['status'],
                   reason=active_run['reason'], limitation=result['limitation'])
    if result['runs']:
        control_run = result['runs']['control']
        control = control_run['frames'][-1]
        if not np.isclose(control['time'], selected['time'], atol=1e-8):
            raise ValueError('control did not reach the matched snapshot time')
        thickness = result['parameters']['outer_radius']-result['parameters']['inner_radius']
        half_width = 9 * thickness
        for name, angle, label in [('vegetal', 0., 'vegetal center'),
                                   ('activation_edge', -np.pi/6, 'sector edge (view rotated 30°)')]:
            frames = [rotate_view(f, angle) for f in (baseline, control, selected)]
            # A common physical window, without independently centering each state.
            center = np.asarray(baseline['center'])
            bottom = min(np.min(np.asarray(f['outer'])[:, 1]-center[1]) for f in frames)
            fig, axes = plt.subplots(1, 3, figsize=(15, 3), constrained_layout=True)
            for ax, frame, title in zip(axes, frames,
                    ('Before activation', f'Control · +{elapsed:.1f} τR', f'Apical activation · +{elapsed:.1f} τR')):
                draw(ax, frame, title)
                ax.set(xlim=(-half_width, half_width),
                       ylim=(bottom-thickness, bottom+4*thickness))
                ax.plot([half_width-6, half_width-1], [bottom-.55*thickness]*2, color='#333333', lw=2)
                ax.text(half_width-3.5, bottom-.8*thickness, '5 length units', ha='center', va='top', fontsize=9)
            fig.suptitle(f'{n}-cell ring · {label} at the same physical scale\n'
                         'Apical mean tension increment +2; actual tension follows the original OU relaxation')
            fig.savefig(output/f'{name}_closeup.png', dpi=220)
            fig.savefig(output/f'{name}_closeup.pdf')
            plt.close(fig)
        summary.update(lumen_area_prepared=baseline['lumen_area'],
                       lumen_area_active=selected['lumen_area'], lumen_area_control=control['lumen_area'],
                       lumen_pressure_active=selected['lumen_pressure'],
                       lumen_pressure_control=control['lumen_pressure'])
    save_result(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--prep', type=float, default=30.)
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--dt', type=float, default=.0025)
    parser.add_argument('--spacing', type=float, default=.075)
    parser.add_argument('--tag', default='512_20261115')
    args = parser.parse_args()
    if args.n < 3 or min(args.prep, args.duration, args.dt, args.spacing) <= 0:
        parser.error('n must be at least 3 and time/spacing parameters positive')
    data = ROOT/'circle/data'/f'invagination_{args.tag}'
    output = ROOT/'figures/circle'/f'invagination_{args.tag}'
    data.mkdir(parents=True, exist_ok=True)
    path = data/'simulation.json.gz'
    if path.exists():
        with gzip.open(path, 'rt') as stream:
            result = json.load(stream)
        for key, value in [('n', args.n), ('preparation', args.prep),
                           ('requested_activation_duration', args.duration), ('dt', args.dt), ('spacing', args.spacing)]:
            if result['parameters'][key] != value:
                raise ValueError('saved run has different parameters; choose a new --tag')
    else:
        result = simulate(args, data)
    print(json.dumps(render(result, output), indent=2))


if __name__ == '__main__':
    main()
