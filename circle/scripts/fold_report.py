"""Diagnostic report for the 48-profile ring: stored half-ring runs and activation profiles.

    PYTHONPATH=. python -m circle.scripts.fold_report

Reads saved trajectories only.  Writes figures and ``summary.json`` to
``figures/circle/fold_mechanics_20261116/``; the prose is in ``REPORT.md`` there.
"""
import gzip
import json
from pathlib import Path

import numpy as np

from circle import fold_diagnostics as fd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'circle/data/bending_48_20261115'
HALF = ROOT/'circle/data/half_ring_bending_20261115'
DATA = ROOT/'circle/data/fold_mechanics_20261116'
OUT = ROOT/'figures/circle/fold_mechanics_20261116'

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
PROFILES = [('uniform_peak4', 'uniform, peak 4 (sum 96)', '#2a78d6'),
            ('uniform_sum48', 'uniform, peak 2 (sum 48)', '#6250d6'),
            ('cosine_peak4', 'cosine, peak 4 (sum 48.2)', '#eb6834'),
            ('center_peak4', 'centre-weighted, peak 4 (sum 48)', '#1baf7a')]
STORED = [('half_cosine', 'cosine +2, lateral +1'), ('half_plateau', 'plateau +2, lateral +1'),
          ('half_A4_L1', 'cosine +4, lateral +1'), ('half_A8_L1', 'cosine +8, lateral +1'),
          ('half_A4_L0', 'cosine +4, lateral 0')]


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def angle_from_pole(n=48):
    theta = 360*(np.arange(n)+.5)/n
    return (theta+90+180) % 360-180


def series(run, control, domain):
    """Time series of the fold measures, up to the last valid sample of the run."""
    rows = []
    for row in run['history']:
        reference = fd.at_time(control['history'], row['elapsed'])
        depth = fd.fold_depth(row, reference, domain)
        curve = fd.curvature_summary(row, reference, domain)
        radial = fd.radial_response(row, reference)
        rows.append(dict(elapsed=row['elapsed'], patch_depth=depth['patch_depth'],
                         domain_depth=depth['domain_depth'],
                         min_curvature=curve['spans'][1]['minimum'],
                         min_curvature_wide=curve['spans'][3]['minimum'],
                         longest_negative_run=curve['longest_negative_run'],
                         domain_turning_change=curve['domain_turning_change'],
                         uniform=radial.get('uniform', np.nan),
                         pole=radial.get('pole_value', np.nan),
                         lumen_area=row['lumen_area'], lumen_pressure=row['lumen_pressure'],
                         lumen_area_vs_control=row['lumen_area']/reference['lumen_area']-1,
                         lumen_pressure_vs_control=row['lumen_pressure']-reference['lumen_pressure']))
    return rows


def summarise(name, label, run, control, domain):
    final = run['history'][-1]
    reference = fd.at_time(control['history'], final['elapsed'])
    first = run['history'][0]
    curve = fd.curvature_summary(final, reference, domain)
    depth = fd.fold_depth(final, reference, domain)
    radial = fd.radial_response(final, reference)
    changes = fd.cell_changes(final, reference)
    labels = fd.classify_cells(changes)
    rest = np.setdiff1d(np.arange(len(labels)), domain)
    lumen = fd.lumen_accounting(final, reference, first)
    frame = run['frames'][-1]
    elapsed = frame['time']-run['frames'][0]['time']
    actual, target = fd.apical_tension(run['parameters'], run['activation_weights'], [elapsed])
    force = fd.incremental_force(frame, actual[0]-1.)
    cls = fd.classify_run(run, control, domain)
    height = float(np.mean(reference['height']))
    mean = lambda values, cells: float(np.mean(np.asarray(values)[cells]))
    return dict(
        name=name, label=label, status=run['summary']['status'], reason=run['summary']['reason'],
        last_valid=final['elapsed'], cls=cls['cls'], class_label=cls['label'],
        peak_increment=run['parameters']['apical'],
        summed_increment=float(run['parameters']['apical']*np.sum(run['activation_weights'])),
        lateral=run['parameters']['lateral'], mean_height=height,
        patch_depth=depth['patch_depth'], patch_depth_in_heights=depth['patch_depth']/height,
        domain_depth=depth['domain_depth'],
        shoulder_separation_change=depth['domain_shoulder_separation_change'],
        domain_turning=curve['domain_turning'], control_domain_turning=curve['control_domain_turning'],
        domain_turning_change=curve['domain_turning_change'], total_turning=curve['total_turning'],
        min_curvature={str(s): v['minimum'] for s, v in curve['spans'].items()},
        negative_profiles={str(s): v['negative_profiles'] for s, v in curve['spans'].items()},
        negative_at_every_span=curve['negative_at_every_span'],
        longest_negative_run=curve['longest_negative_run'],
        uniform_radial=radial.get('uniform'), pole_radial=radial.get('pole_value'),
        residual_rms=radial.get('residual_rms'), modes=radial.get('mode_amplitudes'),
        activated=dict(apical=mean(changes['apical'], domain), basal=mean(changes['basal'], domain),
                       height=mean(changes['height'], domain), area=mean(changes['area'], domain),
                       wedge=mean(changes['wedge'], domain),
                       wedge_control=mean(changes['wedge_control'], domain)),
        unactivated=dict(apical=mean(changes['apical'], rest), basal=mean(changes['basal'], rest),
                         height=mean(changes['height'], rest), area=mean(changes['area'], rest)),
        responses={key: int(np.sum(labels[domain] == key)) for key in sorted(set(labels[domain]))},
        lumen_area_vs_control=lumen['lumen_area_vs_control'],
        lumen_pressure_vs_control=lumen['lumen_pressure_vs_control'],
        cell_area_vs_control=lumen['cell_area_vs_control'],
        wall=fd.wall_clearance(frame),
        force=dict(elapsed=elapsed, normal_inward_sum=float(force['normal_inward'][domain].sum()),
                   tangential_abs_sum=float(np.abs(force['tangential'][domain]).sum()),
                   normal_at_pole=float(force['normal_inward'][domain[10:14]].mean()),
                   max_tangential=float(np.abs(force['tangential']).max()),
                   max_moment=float(np.abs(force['moment']).max()),
                   mean_couple_arm=float(np.mean(force['height'][domain]))/2),
        per_cell=dict(apical=changes['apical'].tolist(), basal=changes['basal'].tolist(),
                      height=changes['height'].tolist(), wedge=changes['wedge'].tolist(),
                      curvature=np.asarray(final['curvature']).tolist(),
                      curvature_span3=fd.curvature(np.asarray(final['midline']), 3).tolist(),
                      tension=actual[0].tolist(), normal=force['normal_inward'].tolist(),
                      tangential=force['tangential'].tolist(), moment=force['moment'].tolist()))


def style(ax, xlabel, ylabel, title=None):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=.6); ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK, fontsize=10); ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=10)


def figures(runs, summaries, timeseries, control, refinement):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    angle = angle_from_pole(); order = np.argsort(angle)
    shade = lambda ax: ax.axvspan(-90, 90, color='#f3ede4', zorder=0)
    present = [(n, l, c) for n, l, c in PROFILES if n in summaries]

    # 1. prescribed profile and resulting tension; depth against time
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for name, label, color in present:
        s = summaries[name]
        axes[0].plot(angle[order], np.asarray(s['per_cell']['tension'])[order], color=color, lw=2,
                     marker='o', ms=3, label=label)
        t = [r['elapsed'] for r in timeseries[name]]
        for ax, key in ((axes[1], 'patch_depth'), (axes[2], 'domain_depth')):
            ax.plot(t, [r[key]/s['mean_height'] for r in timeseries[name]], color=color, lw=2,
                    label=label)
            if s['status'] != 'complete':
                ax.plot(t[-1], timeseries[name][-1][key]/s['mean_height'], marker='x', ms=9,
                        color=color, mew=2.5)
    shade(axes[0])
    style(axes[0], 'cell position from the vegetal pole (degrees)', 'apical tension at the last valid time (T0)',
          'Apical tension actually reached')
    style(axes[1], 'time since activation (tau_R)', 'depth / mean cell height',
          'Depth of the central 8 cells below their shoulders')
    style(axes[2], 'time since activation (tau_R)', 'depth / mean cell height',
          'Depth of the pole below the domain shoulders')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Activation profiles on 24 vegetal profiles of 48; fixed connectivity; x = run stopped on invalid geometry',
                 color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(OUT/'depth_vs_time.png', dpi=150); plt.close(fig)

    # 2. curvature against position, two smoothing spans
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    final_control = control['history'][-1]
    for ax, key, span in ((axes[0], 'curvature', 1), (axes[1], 'curvature_span3', 3)):
        reference = fd.curvature(np.asarray(final_control['midline']), span)
        ax.plot(angle[order], reference[order], color=MUTED, lw=1.5, ls='--', label='control')
        for name, label, color in present:
            ax.plot(angle[order], np.asarray(summaries[name]['per_cell'][key])[order], color=color,
                    lw=2, marker='o', ms=3, label=label)
        ax.axhline(0, color=INK, lw=.8); shade(ax)
        style(ax, 'cell position from the vegetal pole (degrees)', 'signed midline curvature',
              f'Curvature from points {span} cell{"s" if span > 1 else ""} apart')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Midline curvature at the last valid time (negative = bent inward); shaded = activated domain',
                 color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(OUT/'curvature_profiles.png', dpi=150); plt.close(fig)

    # 3. what the cells did
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, key, ylabel, title in ((axes[0], 'apical', 'relative change vs control', 'Apical length'),
                                   (axes[1], 'basal', 'relative change vs control', 'Basal length'),
                                   (axes[2], 'height', 'relative change vs control', 'Cell height'),
                                   (axes[3], 'wedge', 'apical length / basal length', 'Wedging ratio')):
        for name, label, color in present:
            ax.plot(angle[order], np.asarray(summaries[name]['per_cell'][key])[order], color=color,
                    lw=2, marker='o', ms=3, label=label)
        if key == 'wedge':
            ax.plot(angle[order], np.asarray(final_control['wedge_ratio'])[order], color=MUTED,
                    lw=1.5, ls='--', label='control')
        else:
            ax.axhline(0, color=INK, lw=.8)
        shade(ax); style(ax, 'cell position from the vegetal pole (degrees)', ylabel, title)
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle('Cell shape changes at the last valid time; shaded = activated domain', color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(OUT/'cell_shapes.png', dpi=150); plt.close(fig)

    # 4. incremental force of the activation
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, key, ylabel, title in (
            (axes[0], 'normal', 'force per cell (T0)', 'Along the inward normal'),
            (axes[1], 'tangential', 'force per cell (T0)', 'Along the surface (counter-clockwise +)'),
            (axes[2], 'moment', 'moment per cell (T0 x length)', 'Moment about the cell midline point')):
        for name, label, color in present:
            ax.plot(angle[order], np.asarray(summaries[name]['per_cell'][key])[order], color=color,
                    lw=2, marker='o', ms=3, label=label)
        ax.axhline(0, color=INK, lw=.8); shade(ax)
        style(ax, 'cell position from the vegetal pole (degrees)', ylabel, title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Force added by the extra apical tension, on the geometry at the last saved frame',
                 color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(OUT/'incremental_force.png', dpi=150); plt.close(fig)

    # 5. lumen and uniform contraction
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for name, label, color in present:
        t = [r['elapsed'] for r in timeseries[name]]
        axes[0].plot(t, [100*r['lumen_area_vs_control'] for r in timeseries[name]], color=color, lw=2, label=label)
        axes[1].plot(t, [r['lumen_pressure_vs_control'] for r in timeseries[name]], color=color, lw=2)
        axes[2].plot(t, [r['uniform'] for r in timeseries[name]], color=color, lw=2)
    style(axes[0], 'time since activation (tau_R)', 'lumen area vs control (%)', 'Lumen area')
    style(axes[1], 'time since activation (tau_R)', 'lumen pressure minus control', 'Lumen pressure')
    style(axes[2], 'time since activation (tau_R)', 'mean radial shift of the midline',
          'Uniform contraction of the ring')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Lumen response and uniform contraction, relative to the matched control', color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(OUT/'lumen.png', dpi=150); plt.close(fig)

    # 6. refinement
    if refinement:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        shades = {'dt 0.0025, spacing 0.075': '#f4b08f', 'dt 0.00125, spacing 0.075': '#eb6834',
                  'dt 0.00125, spacing 0.0375': '#8a3310'}
        for label, rows in refinement.items():
            t = [r['elapsed'] for r in rows]
            axes[0].plot(t, [r['patch_depth'] for r in rows], color=shades[label], lw=2, label=label)
            axes[1].plot(t, [r['min_curvature_wide'] for r in rows], color=shades[label], lw=2)
        style(axes[0], 'time since activation (tau_R)', 'depth of the central 8 cells',
              'Fold depth')
        style(axes[1], 'time since activation (tau_R)', 'minimum curvature (3-cell span)',
              'Most negative curvature')
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle('Cosine profile, peak 4: time-step and spacing refinement over the shared 50 tau_R',
                     color=INK, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(OUT/'refinement.png', dpi=150); plt.close(fig)


def main():
    domain = fd.vegetal_order()
    stored_control = read(BASE/'control_B5.json.gz')
    summaries, timeseries, runs = {}, {}, {}
    for name, label in STORED:
        run = read(HALF/f'{name}.json.gz')
        summaries[name] = summarise(name, label, run, stored_control, domain)
        timeseries[name] = series(run, stored_control, domain)
    control = read(DATA/'control.json.gz') if (DATA/'control.json.gz').exists() else stored_control
    for name, label, _ in PROFILES:
        path = DATA/f'{name}.json.gz'
        if path.exists():
            runs[name] = read(path)
            summaries[name] = summarise(name, label, runs[name], control, domain)
            timeseries[name] = series(runs[name], control, domain)
    refinement = {}
    levels = [('dt 0.0025, spacing 0.075', DATA/'cosine_peak4.json.gz', control),
              ('dt 0.00125, spacing 0.075', DATA/'cosine_peak4_dt_half.json.gz', BASE/'control_B5_dt_half.json.gz'),
              ('dt 0.00125, spacing 0.0375', DATA/'cosine_peak4_mesh_half.json.gz', BASE/'control_B5_mesh_half.json.gz')]
    refinement_status = {}
    for label, path, reference in levels:
        if not Path(path).exists():
            continue
        run = read(path)
        reference = reference if isinstance(reference, dict) else read(reference)
        rows = [r for r in series(run, reference, domain) if r['elapsed'] <= 50.+1e-9]
        refinement[label] = rows
        refinement_status[label] = dict(status=run['summary']['status'], reason=run['summary']['reason'],
                                        last_valid=run['summary']['last_valid_elapsed'])
    comparison = {}
    if len(refinement) == 3:
        shared = min(rows[-1]['elapsed'] for rows in refinement.values())
        pick = lambda rows, key: next(r[key] for r in rows if abs(r['elapsed']-shared) < 1e-9)
        labels = list(refinement)
        for key in ('patch_depth', 'domain_depth', 'domain_turning_change', 'min_curvature_wide',
                    'lumen_area_vs_control'):
            values = [pick(refinement[label], key) for label in labels]
            comparison[key] = dict(values=values,
                                   two_finest_relative_difference=abs(values[2]-values[1])/abs(values[2])
                                   if values[2] else None)
        comparison['shared_time'] = shared
    figures(runs, summaries, timeseries, control, refinement)
    reproduction = (json.loads((DATA/'control_reproduction.json').read_text())
                    if (DATA/'control_reproduction.json').exists() else None)
    payload = dict(summaries=summaries, refinement=dict(status=refinement_status, comparison=comparison),
                   control_reproduction=reproduction,
                   timeseries={k: v[::10] for k, v in timeseries.items()})
    (OUT/'summary.json').write_text(json.dumps(payload, indent=1, default=float))

    f = lambda x, d=3: 'n/a' if x is None else f'{x:.{d}f}'
    print('| run | status | class | central depth / height | domain depth | turning change | min curvature (1,3) | negative at all spans | wedge (control) | apical | basal | height | lumen area | uniform shift |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for name, s in summaries.items():
        a = s['activated']
        print(f"| {s['label']} | {s['status']} @ {s['last_valid']:.0f} | {s['cls']} | {f(s['patch_depth_in_heights'])} | "
              f"{f(s['domain_depth'])} | {f(s['domain_turning_change'])} | {f(s['min_curvature']['1'])}, {f(s['min_curvature']['3'])} | "
              f"{len(s['negative_at_every_span'])} (run {s['longest_negative_run']}) | {f(a['wedge'])} ({f(a['wedge_control'])}) | "
              f"{100*a['apical']:+.1f}% | {100*a['basal']:+.1f}% | {100*a['height']:+.1f}% | {100*s['lumen_area_vs_control']:+.2f}% | {f(s['uniform_radial'])} |")
    print('\nforces:')
    for name, s in summaries.items():
        q = s['force']
        print(f"  {s['label']}: inward normal sum {q['normal_inward_sum']:.3f}, at pole {q['normal_at_pole']:.3f}, "
              f"sum |tangential| {q['tangential_abs_sum']:.3f}, max tangential {q['max_tangential']:.3f}, max moment {q['max_moment']:.3f}; responses {s['responses']}")
    print('\nrefinement:', json.dumps(dict(status=refinement_status, comparison=comparison), default=float))
    print('control reproduction:', reproduction)


if __name__ == '__main__':
    main()
