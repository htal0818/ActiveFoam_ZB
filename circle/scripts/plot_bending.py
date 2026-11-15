"""Plot actual cell-level diagnostics and controlled bending pilot outcomes."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from circle.bending import measure_frame, compare_midlines
from circle.collective import save_result
from circle.scripts.invagination_preview import draw

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'circle/data/bending_48_20261115'
OUT = ROOT/'figures/circle/bending_48_20261115'


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def original_diagnostics():
    old = read(ROOT/'circle/data/invagination_blastula_crosssection_48_20261115/simulation.json.gz')
    active_frames = old['runs']['apical']['frames']; control_frames = old['runs']['control']['frames']
    active = [measure_frame(f) for f in active_frames]
    control = [measure_frame(f) for f in control_frames]
    ids = np.flatnonzero(old['initial']['active_cells'])
    elapsed = np.array([m['time']-active[0]['time'] for m in active])
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    for ax, key, label in zip(axes.flat, ('apical', 'basal', 'height', 'area', 'wedge_ratio', 'curvature'),
            ('Apical arclength', 'Basal arclength', 'Apicobasal height', 'Cell area',
             'Apical / basal arclength', 'Signed midline curvature')):
        for j, cell in enumerate(ids):
            ax.plot(elapsed, [m[key][cell] for m in active], 'o-', ms=3,
                    color=plt.cm.viridis(j/7), label=f'Cell {cell}')
        if key == 'wedge_ratio': ax.axhline(1, c='k', ls=':', lw=1)
        if key == 'curvature': ax.axhline(0, c='k', ls=':', lw=1)
        ax.set(xlabel='Time after activation (τR)', ylabel=label)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 2].plot(elapsed, [m['lumen_area'] for m in active], 'o-', label='Activated')
    axes[1, 2].plot(elapsed, [m['lumen_area'] for m in control], '--', label='Control')
    axes[1, 2].set(xlabel='Time after activation (τR)', ylabel='Lumen area')
    axes[1, 2].legend()
    compared = compare_midlines(active[-1], control[-1])
    theta = (compared['theta']+np.pi/2+np.pi) % (2*np.pi)-np.pi
    order = np.argsort(theta)
    axes[1, 3].plot(theta[order]*180/np.pi, compared['radial_residual'][order])
    axes[1, 3].axhline(0, c='k', ls=':', lw=1)
    axes[1, 3].axvspan(-30, 30, color='orange', alpha=.15)
    axes[1, 3].set(xlabel='Angle from vegetal center (degrees)', ylabel='Radial shift after uniform component removed', xlim=(-90, 90))
    fig.suptitle('Original sharp activation: all eight vegetal cell profiles\n'
                 'The two patch-edge apical surfaces nearly collapse; single-cell negative curvature is not a broad inward fold')
    fig.savefig(OUT/'original_diagnostics.png', dpi=180)
    fig.savefig(OUT/'original_diagnostics.pdf'); plt.close(fig)
    save_result(DATA/'original_diagnostics.json', dict(cell_ids=ids, active=active, control=control, comparison=compared))


def analyze_runs():
    runs = {}
    for path in sorted(DATA.glob('*.json.gz')):
        if path.name == 'preparation.json.gz': continue
        result = read(path)
        if 'parameters' in result: runs[result['parameters']['name']] = result
    analyzed = {}
    for name, run in runs.items():
        if name.startswith('control_'): continue
        control_name = run['parameters'].get('control_name', f"control_B{run['parameters']['bulk']:g}")
        if control_name not in runs: continue
        control = runs[control_name]
        reference = {round(m['elapsed'], 8):m for m in control['history']}
        comparisons = []
        for m in run['history']:
            t = round(m['elapsed'], 8)
            if t not in reference: raise ValueError('missing matched-time control')
            comparison = compare_midlines(m, reference[t], run['parameters']['patch'])
            comparisons.append(dict(comparison, elapsed=t))
        valid = [x for x in comparisons if x['radial_profile_valid']]
        last = comparisons[-1]
        summary = dict(run['summary'], last_valid_morphology=last['classification'],
            max_inward_displacement=max((x['inward_displacement'] for x in valid), default=None),
            max_shoulder_depth=max((x['shoulder_depth'] for x in valid), default=None),
            min_patch_curvature=min((x['min_patch_curvature'] for x in valid), default=None),
            min_patch_curvature_wide=min((x['min_patch_curvature_wide'] for x in valid), default=None),
            total_target_increment=float(run['parameters']['apical']*np.sum(run['activation_weights'])),
            parameters=run['parameters'])
        m = run['history'][-1]; matched = reference[round(m['elapsed'], 8)]
        summary.update(final_inward_displacement=last.get('inward_displacement'),
                       final_shoulder_depth=last.get('shoulder_depth'),
                       final_lumen_area_percent_vs_control=100*(m['lumen_area']/matched['lumen_area']-1),
                       final_min_apical_length=m.get('min_length_by_role', {}).get('apical'))
        inversion_times = [c['elapsed'] for c in comparisons if c['classification'] == 'curvature_inversion']
        summary['first_curvature_inversion'] = min(inversion_times) if inversion_times else None
        tail = []
        for c in reversed(comparisons):
            if c['classification'] != 'curvature_inversion': break
            tail.append(c['elapsed'])
        summary['final_contiguous_inversion_duration'] = max(tail)-min(tail) if tail else 0.
        analyzed[name] = dict(summary=summary, comparisons=comparisons)
    compact = {name: dict(summary=item['summary'],
                   comparisons=[{key:value for key,value in c.items() if not isinstance(value, np.ndarray)}
                                for c in item['comparisons']],
                   last_profile=item['comparisons'][-1]) for name,item in analyzed.items()}
    save_result(DATA/'analysis.json', compact)
    fields = ['name', 'status', 'last_valid_elapsed', 'last_valid_morphology',
              'final_inward_displacement', 'final_shoulder_depth', 'first_curvature_inversion',
              'final_contiguous_inversion_duration', 'final_lumen_area_percent_vs_control',
              'final_min_apical_length', 'total_target_increment']
    with (DATA/'outcomes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for item in analyzed.values(): writer.writerow(item['summary'])
    return runs, analyzed


def render_sweeps(runs, analyzed):
    groups = [(['apical_A0.5', 'apical_A1', 'apical_A2', 'apical_A4'], 'Apical amplitude; B=5, 8 profiles'),
              (['lumen_B0.5', 'apical_A2', 'lumen_B50', 'lumen_B500'], 'Lumen compression modulus; apical +2'),
              (['apical_A2', 'lateral_L0.5', 'lateral_L1', 'weak_combined'], 'Delayed lateral activation at 20 τR'),
              (['patch_P4', 'apical_A2', 'patch_P12', 'smooth_equal_total', 'sharp_ramped_A2', 'sharp_A2'], 'Patch width and activation boundary')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for group, (ax, (names, title)) in enumerate(zip(axes.flat, groups)):
        for name in names:
            if name not in analyzed: continue
            data = analyzed[name]
            valid = [c for c in data['comparisons'] if c['radial_profile_valid']]
            if not valid: continue
            t = [c['elapsed'] for c in valid]; y = [c['inward_displacement'] for c in valid]
            p = runs[name]['parameters']
            if group == 0:
                label = f"Apical mean +{p['apical']:g}"
            elif group == 1:
                label = f"Lumen modulus {p['bulk']:g}"
            elif group == 2:
                label = f"Apical +{p['apical']:g}, lateral +{p['lateral']:g}"
            else:
                label = {'sharp_A2':'8 profiles, sharp instantaneous target',
                         'sharp_ramped_A2':'8 profiles, sharp spatial profile',
                         'smooth_equal_total':'8 profiles, smooth matched total'}.get(
                             name, f"{p['patch']} profiles, smooth")
            line, = ax.plot(t, y, label=label)
            ax.plot(t[-1], y[-1], 'o' if data['summary']['status'] == 'complete' else 'x',
                    color=line.get_color(), ms=7)
        ax.axhline(0, c='k', ls=':', lw=1)
        ax.set(xlabel='Time after activation (τR)', ylabel='Localized inward displacement (length units)', title=title)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed-topology bending pilot · matched controls, global contraction removed\n'
                 '×: last sampled valid state before a guard/failure; ○: completed requested hold. Truncated curves are censored.')
    fig.savefig(OUT/'controlled_sweeps.png', dpi=180); fig.savefig(OUT/'controlled_sweeps.pdf'); plt.close(fig)
    if not analyzed: return
    selected = max(analyzed, key=lambda k: analyzed[k]['summary']['max_inward_displacement'] or -np.inf)
    run = runs[selected]
    control = runs[run['parameters'].get('control_name', f"control_B{run['parameters']['bulk']:g}")]
    active_frame = run['frames'][-1]
    # Controls store geometry every 5 units; reconstruct the figure only at a common saved time.
    common = [(a, c) for a in run['frames'] for c in control['frames'] if abs(a['time']-c['time']) < 1e-8]
    a, c = common[-1]
    elapsed = a['time']-run['frames'][0]['time']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    draw(axes[0], run['frames'][0], 'Prepared ring')
    draw(axes[1], c, f'Matched control · +{elapsed:.1f} τR')
    p = run['parameters']
    draw(axes[2], a, f"Apical +{p['apical']:g}, lateral +{p['lateral']:g} · +{elapsed:.1f} τR")
    fig.suptitle('Largest measured localized inward response in this exploratory pilot\n'
                 f"Fixed topology · spacing {p['spacing']:g} · dt {p['dt']:g} · {run['summary']['status']}")
    fig.savefig(OUT/'selected_snapshots.png', dpi=180); fig.savefig(OUT/'selected_snapshots.pdf'); plt.close(fig)
    last = analyzed[selected]['comparisons'][-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    m = run['history'][-1]; ids = np.arange(48)
    theta = 360*(ids+.5)/48-270; theta = (theta+180) % 360-180
    order = np.argsort(theta)
    axes[0].plot(theta[order], np.asarray(m['wedge_ratio'])[order], 'o-')
    axes[0].axhline(1, c='k', ls=':'); axes[0].set(ylabel='Apical / basal arclength')
    for key, label in [('curvature', 'Adjacent-cell estimate'), ('curvature_wide', 'Wider estimate')]:
        axes[1].plot(theta[order], np.asarray(m[key])[order], 'o-', label=label)
    axes[1].axhline(0, c='k', ls=':'); axes[1].legend(fontsize=8); axes[1].set(ylabel='Signed midline curvature')
    angle = (np.asarray(last['theta'])*180/np.pi+270) % 360-180
    order2 = np.argsort(angle)
    axes[2].plot(angle[order2], np.asarray(last['radial_residual'])[order2])
    axes[2].axhline(0, c='k', ls=':'); axes[2].set(ylabel='Radial response without uniform contraction')
    for ax in axes: ax.set(xlim=(-90, 90))
    for ax in axes[:2]: ax.set(xlabel='Initial material angle from vegetal center (degrees)')
    axes[2].set(xlabel='Aligned polar angle from vegetal center (degrees)')
    fig.suptitle(f"{selected}: last valid sampled diagnostics at +{m['elapsed']:.1f} τR")
    fig.savefig(OUT/'selected_geometry_diagnostics.png', dpi=180)
    fig.savefig(OUT/'selected_geometry_diagnostics.pdf'); plt.close(fig)
    save_result(OUT/'selection.json', dict(name=selected, snapshot_elapsed=elapsed,
        diagnostic_elapsed=m['elapsed'], summary=analyzed[selected]['summary']))
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.suptitle(f"Apical mean +{p['apical']:g}; lateral mean +{p['lateral']:g} from {p['delay']:g} τR\n"
                 f"Actual saved frames · fixed topology · exploratory spacing {p['spacing']:g}")
    def animate(index):
        active_frame, control_frame = common[min(index, len(common)-1)]
        t = active_frame['time']-run['frames'][0]['time']
        for ax in axes: ax.clear()
        draw(axes[0], active_frame, f'Activated ring · +{t:.1f} τR')
        draw(axes[1], active_frame, 'Vegetal region · dashed gray: control')
        center = np.asarray(active_frame['center'])
        for boundary in ('inner', 'outer'):
            points = np.asarray(control_frame[boundary])-center
            axes[1].plot(points[:, 0], points[:, 1], '--', color='#777777', lw=1)
        axes[1].set(xlim=(-3.6, 3.6), ylim=(-8.2, -4.3))
        return []
    animation = FuncAnimation(fig, animate, frames=len(common)+3, interval=250, blit=False)
    animation.save(OUT/'bending_motion.gif', writer=PillowWriter(fps=4), dpi=110)
    plt.close(fig)
    ids = np.flatnonzero(np.asarray(run['activation_weights']) > 0)
    history = run['history']; times = [m['elapsed'] for m in history]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    for ax, key, label in zip(axes.flat, ('apical', 'basal', 'height', 'area', 'wedge_ratio', 'curvature'),
            ('Apical arclength', 'Basal arclength', 'Apicobasal height', 'Cell area',
             'Apical / basal arclength', 'Signed midline curvature')):
        for j, cell in enumerate(ids):
            ax.plot(times, [m[key][cell] for m in history], lw=1.4,
                    color=plt.cm.viridis(j/max(1, len(ids)-1)), label=f'Cell {cell}')
        ax.axvline(p['delay'], c='#888888', ls=':', lw=1)
        if key == 'wedge_ratio': ax.axhline(1, c='k', ls=':', lw=1)
        if key == 'curvature': ax.axhline(0, c='k', ls=':', lw=1)
        ax.set(xlabel='Time after activation (τR)', ylabel=label)
    axes[0, 0].legend(fontsize=7, ncol=2)
    bytime = {round(m['elapsed'], 8):m for m in control['history']}
    axes[1, 2].plot(times, [100*(m['lumen_area']/bytime[round(m['elapsed'], 8)]['lumen_area']-1) for m in history])
    axes[1, 2].set(xlabel='Time after activation (τR)', ylabel='Lumen area change versus control (%)')
    for role, color in [('apical', '#c65339'), ('lateral', '#367eaa')]:
        for field, style, kind in [('actual_tensions', '-', 'Actual'), ('target_tensions', '--', 'Target')]:
            axes[1, 3].plot(times, [m.get(field, {}).get(role, 1.) for m in history], style,
                           color=color, label=f'{kind} {role}')
    axes[1, 3].set(xlabel='Time after activation (τR)', ylabel='Maximum interface tension')
    axes[1, 3].legend(fontsize=8)
    fig.suptitle(f"Smooth apical +{p['apical']:g}, delayed lateral +{p['lateral']:g} · individual vegetal cell diagnostics\n"
                 'Vertical dotted lines mark lateral activation; symmetric cell pairs overlap')
    fig.savefig(OUT/'selected_cell_diagnostics.png', dpi=180)
    fig.savefig(OUT/'selected_cell_diagnostics.pdf'); plt.close(fig)


def render_refinement(runs, analyzed):
    names = ['apical_A2', 'refine_dt_A2', 'refine_mesh_A2']
    labels = ['dt=0.0025, spacing=0.075', 'dt=0.00125, spacing=0.075',
              'dt=0.00125, spacing=0.0375']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    rows = []
    for name, label in zip(names, labels):
        if name not in analyzed: continue
        comparison = [m for m in analyzed[name]['comparisons'] if m['elapsed'] <= 50]
        history = [m for m in runs[name]['history'] if m['elapsed'] <= 50]
        times = [m['elapsed'] for m in comparison]
        axes[0, 0].plot(times, [m['inward_displacement'] for m in comparison], label=label)
        axes[0, 1].plot(times, [m['min_patch_curvature_wide'] for m in comparison], label=label)
        recorded = [m for m in history if 'min_length_by_role' in m]
        axes[1, 0].plot([m['elapsed'] for m in recorded],
                        [m['min_length_by_role']['apical'] for m in recorded], label=label)
        mask = np.asarray(runs[name]['activation_weights']) > 0
        axes[1, 1].plot(times, [np.min(np.asarray(m['wedge_ratio'])[mask]) for m in history], label=label)
        for t in (20., 40., 50.):
            candidate = [c for c in comparison if c['elapsed'] == t]
            if candidate:
                rows.append(dict(run=name, elapsed=t,
                    inward_displacement=candidate[0]['inward_displacement'],
                    min_patch_curvature_wide=candidate[0]['min_patch_curvature_wide'],
                    morphology=candidate[0]['classification']))
    for ax, ylabel in zip(axes.flat, ('Localized inward displacement', 'Minimum wider-scale patch curvature',
                                    'Shortest apical film', 'Minimum apical / basal width ratio')):
        ax.set(xlabel='Time after activation (τR)', ylabel=ylabel); ax.legend(fontsize=8)
    axes[0, 1].axhline(0, c='k', ls=':', lw=1)
    axes[1, 1].axhline(1, c='k', ls=':', lw=1)
    fig.suptitle('Resolution sensitivity · smooth apical mean increment +2\nMatched controls at each resolution; no mobility rescaling')
    fig.savefig(OUT/'resolution_check.png', dpi=180); fig.savefig(OUT/'resolution_check.pdf'); plt.close(fig)
    save_result(DATA/'resolution_comparison.json', rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-only', action='store_true')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    original_diagnostics()
    if not args.original_only:
        runs, analyzed = analyze_runs()
        render_sweeps(runs, analyzed)
        render_refinement(runs, analyzed)
