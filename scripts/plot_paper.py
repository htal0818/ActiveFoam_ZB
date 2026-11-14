"""Render accelerated foam checkpoints and overlay publisher source data."""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

from plot_requested import fit_stress
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from activefoam.paper_measure import sampled_neighbor_changes


ACTIVITY = (.25, .5, .75, 1., 1.25, 1.5)
COLORS = dict(zip(ACTIVITY, ('#406fa9', '#df951d', '#86a844', '#df5e38', '#8576ad', '#bc7029')))
W_LABEL = r'$W/T_0$'
MU_LABEL = r'$\Delta T/T_0$'
TIME = r'$t/\tau_R$'
MSD = r'MSD / $L_0^2$'
PHI = r'$\phi$'
STRESS = r'$\sigma_{xy}/\sigma_0$'
NE = r'Cellular NE rate ($\tau_R^{-1}$)'


def write_text(path, content):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def source_tables(folder):
    if folder is None:
        return {}
    from openpyxl import load_workbook

    tables = {}
    for figure, attachment in {2: 11, 3: 12, 4: 13, 'ED1': 15, 'ED2': 16}.items():
        for name in (f'publisher_fig{figure}.xlsx', f'moesm{attachment}.xlsx'):
            path = folder / name
            if path.is_file():
                with path.open('rb') as stream:
                    workbook = load_workbook(stream, read_only=True, data_only=True)
                    tables[figure] = list(workbook.active.values)[5 if figure != 'ED2' else 4:]
                    workbook.close()
                break
    return tables


def columns(rows, *indices):
    data = [[row[i] for i in indices] for row in rows
            if all(isinstance(row[i], (int, float)) for i in indices)]
    return np.asarray(data, dtype=float).reshape(-1, len(indices)).T


def grouped(records, *keys):
    groups = defaultdict(list)
    for params, data in records:
        groups[tuple(params[key] for key in keys)].append(data)
    return dict(sorted(groups.items()))


def line_stats(ax, x, ys, label=None, color=None, marker=None):
    y = np.mean(ys, axis=0)
    line, = ax.plot(x, y, label=label, color=color, marker=marker, markersize=3)
    if len(ys) > 1:
        sd = np.std(ys, axis=0)
        ax.fill_between(x, y - sd, y + sd, color=line.get_color(), alpha=.15)
    return y


def scan(ax, records, xkey, metric, label=None, color=None, grid=None):
    samples = grouped(records, xkey)
    x = [key[0] for key in samples] if grid is None else grid
    values = [np.asarray([float(metric(data)) for data in samples.get((point,), ())])
              for point in x]
    values = [vals[np.isfinite(vals)] for vals in values]
    mean = [vals.mean() if len(vals) else np.nan for vals in values]
    sd = [vals.std() if len(vals) else np.nan for vals in values]
    if x:
        line, = ax.plot(x, mean, 'o-', markersize=3, label=label, color=color)
        if max(map(len, values)) > 1:
            ax.fill_between(x, np.subtract(mean, sd), np.add(mean, sd),
                            color=line.get_color(), alpha=.15)
    return np.asarray(x), np.asarray(mean)


def draw_foam(ax, data, prefix='', color_by_shape=False, color_by_contacts=False, frame='0.35',
              highlight=None):
    points, offsets = (data[prefix + k] for k in ('points', 'offsets'))
    box, strain = (float(data[prefix + k]) for k in ('box', 'strain'))
    cells, colors = [], []
    domain = np.array([[0., 0.], [box, 0.], [(1 + strain) * box, box], [strain * box, box]])
    clip = Polygon(domain, facecolor='#d6a5a1', edgecolor=frame, linewidth=2)
    ax.add_patch(clip)
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        polygon = points[start:stop]
        coord = polygon.mean(axis=0) / box
        m, n = np.floor([coord[0] - strain * coord[1], coord[1]])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                copies = polygon + box * np.array([dx - m + (dy - n) * strain, dy - n])
                cells.append(copies)
                if color_by_shape:
                    colors.append(data[prefix + 'cell_shape'][i])
                elif color_by_contacts:
                    colors.append(data['cell_z'][i])
                elif highlight is not None:
                    colors.append('#e5b956' if i in highlight else '#e0e3e5')
    collection = PolyCollection(cells, facecolors=colors if highlight is not None else '#e0e3e5',
                                edgecolors='0.25', linewidths=.4)
    if color_by_shape or color_by_contacts:
        collection.set_array(np.asarray(colors))
        collection.set_cmap('RdYlBu_r' if color_by_shape else 'viridis')
        collection.set_clim(*( (3.6, 4.8) if color_by_shape else (0, 6)))
    collection.set_clip_path(clip)
    ax.add_collection(collection)
    ax.set(xlim=(domain[:, 0].min(), domain[:, 0].max()), ylim=(0, box), aspect='equal')
    ax.axis('off')
    return collection


def long_msd(d):
    if 'msd' in d and np.isclose(d['tlag'][-1], 100.):
        return d['msd'][-1]
    return np.nan


def ne_bins(data, times, cells):
    # Figure 4b's workbook labels forward 1 tau_R intervals on an integer grid.
    if 'neighbor_exchange_rate' in data:
        grid = data['neighbor_exchange_time']
        indices = np.searchsorted(grid, times)
        if np.any(indices >= len(grid)) or not np.array_equal(grid[indices], times):
            raise ValueError('Neighbor exchange samples do not cover the requested time grid')
        return data['neighbor_exchange_rate'][indices]
    samples, changes, _ = sampled_neighbor_changes(
        data['contacts'], data['time'][0], data['time'][-1], stride=200
    )
    return np.histogram(samples - 1., bins=np.r_[times, times[-1] + 1], weights=changes)[0] / cells


def ne_rate(data, cells):
    if 'sampled_neighbor_change_count' in data:
        duration = float(data['sampled_neighbor_change_duration'])
        return float(data['sampled_neighbor_change_count']) / (cells * duration) if duration else np.nan
    _, changes, duration = sampled_neighbor_changes(data['contacts'], 0., data['active_stress'][-1, 0])
    return changes.sum() / (cells * duration) if duration else np.nan


def fit_relaxation(records, active_noise):
    t = records[0]['time']
    stress = np.mean([d['stress'] for d in records], axis=0)
    fit = fit_stress(t, stress)
    offset, amplitude, tau, beta = fit[0]
    fraction = (active_noise - offset) / amplitude if amplitude > 0 else np.nan
    if fraction <= 0:
        crossing = np.inf
    elif fraction >= 1:
        crossing = 0.
    else:
        crossing = tau * (-np.log(fraction)) ** (1 / beta)
    return t, stress, fit[0], crossing


def verify_equilibrium_checkpoint(job, raw, data):
    metadata = json.loads(raw['metadata'].item())
    if any(metadata[key] != job[key] for key in ('id', 'kind', 'parameters', 'profile', 'source_hash')):
        raise ValueError(f'Checkpoint metadata mismatch for {job["id"]}')
    if not np.all((data['phi'] > 0.) & (data['phi'] <= 1.01)):
        raise ValueError(f'Invalid volume fraction in {job["id"]}')
    if job['kind'] == 'equilibrium':
        if not (all(np.ndim(data[key]) == 0 for key in ('phi', 'z', 'shape_mean'))
                and np.isfinite(data['z']) and data['z'] >= 0.
                and np.isfinite(data['shape_mean']) and data['shape_mean'] > 0.):
            raise ValueError(f'Invalid contact number or shape factor in {job["id"]}')
        return
    w = np.linspace(0., 1., 51)
    if job['parameters']['direction'] == 'down':
        w = w[::-1]
    if (not np.array_equal(data['w'], w) or data['phi'].shape != w.shape
            or data['steps'].shape != w.shape or data['max_force'].shape != w.shape
            or data['steps'][0] != 0
            or not np.all((data['steps'][1:] >= 1) & (data['steps'][1:] <= 20008))
            or not np.all(np.isfinite(data['max_force'][1:]) & (data['max_force'][1:] >= 0.))):
        raise ValueError(f'Invalid adhesion sweep in {job["id"]}')


def verify_figure3_checkpoint(job, raw, data):
    ident = job['id']
    if json.loads(raw['metadata'].item()) != job:
        raise ValueError(f'Checkpoint metadata mismatch for {ident}')
    sample_dt = float(data['sample_dt'])
    frames = int(data['sample_frames'])
    duration_tauT = job['parameters']['duration_tauT']
    lag = data['tlag']
    count = int(data['sampled_neighbor_change_count'])
    if (sample_dt != .05 or frames != round(duration_tauT * 10 / sample_dt)
            or not np.isclose(data['sampled_neighbor_change_duration'], (frames - 1) * sample_dt)
            or count < 0
            or count != 2 * (int(data['sampled_contact_gains']) + int(data['sampled_contact_losses']))):
        raise ValueError(f'Invalid measurement cadence or neighbor counts for {ident}')
    if (lag.ndim != 1 or not len(lag) or not np.all(np.isfinite(lag))
            or not np.all(np.diff(lag) > 0.) or not np.isclose(lag[0], .005)
            or not np.isclose(lag[-1], min(100., (frames - 1) * sample_dt / 10))):
        raise ValueError(f'Invalid MSD lag grid for {ident}')
    msd, centered = data['msd'], data['msd_com']
    if (msd.shape != lag.shape or centered.shape != lag.shape
            or not np.all(np.isfinite(msd) & np.isfinite(centered))
            or np.any((centered < 0.) | (centered > msd))
            or data['msd_center'].item() != 'area'
            or not np.isfinite(data['max_crossing_depth'])
            or not 0 <= data['resolved_crossing_checks'] <= data['quality_checks']):
        raise ValueError(f'Invalid MSD or geometry diagnostics for {ident}')


def verify_passive_checkpoint(job, raw, data):
    t, stress = data['time'], data['stress']
    duration = job['parameters']['duration']
    if (json.loads(raw['metadata'].item()) != job
            or t.ndim != 1 or stress.shape != t.shape or len(t) < 2
            or not np.isfinite(t).all() or not np.isfinite(stress).all()
            or np.any(np.diff(t) <= 0) or not np.isclose(t[0], -20.)
            or not np.isclose(t[-1], duration)
            or not np.isclose(data['yield_stress'], stress[t >= duration - 20.].mean())):
        raise ValueError(f'Invalid passive shear checkpoint {job["id"]}')


class Render:
    def __init__(self, data, out, source, additional_data=()):
        self.manifest = json.loads((data / 'manifest.json').read_text())
        job_dirs = {job['id']: data for job in self.manifest['jobs']}
        for directory in additional_data:
            other = json.loads((directory / 'manifest.json').read_text())
            for field in ('paper', 'model', 'profile', 'source_hash', 'n_side',
                          'geometry_policy', 'pressure_normals'):
                if self.manifest.get(field) != other.get(field):
                    raise ValueError(f'{directory} has a different {field}')
            for job in other['jobs']:
                if job['id'] in job_dirs:
                    raise ValueError(f'Duplicate job {job["id"]} in {directory}')
                job_dirs[job['id']] = directory
                self.manifest['jobs'].append(job)
            self.manifest['panels'] = list(dict.fromkeys(self.manifest['panels'] + other['panels']))
            self.manifest['limitations'].update(other['limitations'])
        self.out = out
        self.source = source
        self.records = defaultdict(list)
        self.jobs = defaultdict(list)
        self.phase = {}
        self.produced = []
        self.comparisons = []
        self.geometry = {}
        self.figure3_solvers = defaultdict(int)
        self.passive_solvers = defaultdict(int)
        self.crossing_panels = set()
        self.undefined_spaces = []
        self.cells = self.manifest['n_side']**2
        self.verify_2efg = (
            bool(additional_data) and self.manifest['profile'] == 'full'
            and set(self.manifest['panels']) == {'2e', '2f', '2g'}
            and self.cells == 100 and len(self.manifest['jobs']) == 1840
        )
        self.verify_3acd = (
            self.manifest.get('protocol', {}).get('msd_center') == 'polygon area centroid'
            and set(self.manifest['panels']) == {'3a', '3b', '3c', '3d'}
        )
        self.verify_2ik = (self.manifest.get('replicates_per_setting') == 1 and self.cells == 100
                           and {'2i', '2j', '2k'}.issubset(self.manifest['panels']))
        for job in self.manifest['jobs']:
            for panel in job['panels']:
                self.jobs[panel].append(job)
            checkpoint = job_dirs[job['id']] / (job['id'] + '.npz')
            if not checkpoint.exists():
                continue
            with np.load(checkpoint, allow_pickle=False) as raw:
                d = {key: raw[key] for key in raw.files if key != 'metadata'}
                if self.verify_2efg:
                    verify_equilibrium_checkpoint(job, raw, d)
                if self.verify_2ik:
                    verify_passive_checkpoint(job, raw, d)
                    solver = (str(d['solver_source_hash'].item()) if 'solver_source_hash' in d
                              else job['source_hash'])
                    self.passive_solvers[solver] += 1
                if self.verify_3acd:
                    verify_figure3_checkpoint(job, raw, d)
                    solver = (str(d['solver_source_hash'].item()) if 'solver_source_hash' in d
                              else job['source_hash'])
                    self.figure3_solvers[solver] += 1
            self.geometry[job['id']] = int(d['resolved_crossing_checks'])
            if self.geometry[job['id']]:
                self.crossing_panels.update(job['panels'])
            if 'space_count' in d and not np.isfinite(d['space_count']):
                self.undefined_spaces.append(job['parameters'])
            for panel in job['panels']:
                self.records[panel].append((job['kind'], job['parameters'], d))
        self.phase_results()

    def entries(self, panel, kind=None):
        return [(p, d) for k, p, d in self.records[panel] if kind is None or k == kind]

    def ensemble_coverage(self, panel):
        groups = defaultdict(list)
        for job in self.jobs[panel]:
            parameters = job['parameters']
            coordinate = tuple((key, parameters[key]) for key in ('rho', 'w', 'mu', 'direction')
                               if key in parameters)
            groups[job['kind'], coordinate].append(job)
        return [
            dict(kind=kind, **dict(coordinate), requested=len(jobs),
                 completed=sum(job['id'] in self.geometry for job in jobs))
            for (kind, coordinate), jobs in groups.items()
        ]

    def verify_2efg_figures(self, summary):
        coverage = summary['ensemble_coverage']
        for panel, groups, seeds, comparisons in (('2e', 129, 10, 9), ('2f', 16, 20, 1), ('2g', 13, 20, 1)):
            if (panel not in self.produced or len(coverage[panel]) != groups
                    or any(item['completed'] != seeds or item['requested'] != seeds
                           for item in coverage[panel])):
                raise ValueError(f'Incomplete ensemble coverage for Figure {panel}')
            metrics = [item for item in self.comparisons if item['panel'] == panel]
            if len(metrics) != comparisons or any(not np.isfinite(item['rms']) for item in metrics):
                raise ValueError(f'Missing publisher comparisons for Figure {panel}')
            pixels = plt.imread(self.out / f'fig{panel}.png')
            if pixels.ndim != 3 or np.ptp(pixels[..., :3]) == 0.:
                raise ValueError(f'Invalid Figure {panel} image')
        return dict(panels=['2e', '2f', '2g'], checkpoints=len(self.geometry),
                    publisher_comparisons=len(self.comparisons))

    def verify_figure3_figures(self, summary):
        expected = ('3a', '3a_alpha', '3c', '3d')
        if not set(expected).issubset(self.produced) or 3 not in self.source:
            raise ValueError('Figure 3 output or publisher workbook is missing')
        seeds = self.manifest.get('replicates_per_setting', 10)
        for panel, settings in (('3a', 6), ('3c', 78), ('3d', 78)):
            coverage = summary['ensemble_coverage'][panel]
            if len(coverage) != settings or any(
                    row['completed'] != seeds or row['requested'] != seeds for row in coverage):
                raise ValueError(f'Incomplete Figure {panel} parameter coverage')
        for panel in expected:
            pixels = plt.imread(self.out / f'fig{panel}.png')
            if pixels.ndim != 3 or np.ptp(pixels[..., :3]) == 0.:
                raise ValueError(f'Invalid Figure {panel} image')
        comparisons = summary['comparisons']
        for panel, count in (('3a', 12), ('3a exponent', 6), ('3c', 6), ('3d', 6)):
            scores = [entry for entry in comparisons if entry['panel'] == panel]
            value = 'model' if panel == '3a exponent' else 'rms'
            if len(scores) != count or any(not np.isfinite(entry[value]) for entry in scores):
                raise ValueError(f'Missing finite publisher comparisons for Figure {panel}')
            if panel in ('3c', '3d') and any(entry['points'] != 13 for entry in scores):
                raise ValueError(f'Incomplete adhesion grid in Figure {panel} comparisons')
        return dict(panels=list(expected), checkpoints=len(self.geometry),
                    publisher_comparisons=len(comparisons))

    def verify_passive_figures(self, summary):
        for panel, settings, scores in (('2i', 5, 5), ('2j', 20, 1), ('2k', 320, 1)):
            rows = summary['ensemble_coverage'][panel]
            if (len(rows) != settings or panel not in self.produced
                    or any(r['requested'] != 1 or r['completed'] != 1 for r in rows)):
                raise ValueError(f'Incomplete single-run grid for Figure {panel}')
            comparisons = [e for e in self.comparisons if e['panel'] == panel]
            if len(comparisons) != scores or any(not np.isfinite(e['rms']) for e in comparisons):
                raise ValueError(f'Missing publisher comparison for Figure {panel}')
            if panel in ('2j', '2k') and comparisons[0]['points'] != settings:
                raise ValueError(f'Incomplete publisher grid for Figure {panel}')
            pixels = plt.imread(self.out / f'fig{panel}.png')
            if pixels.ndim != 3 or np.ptp(pixels[..., :3]) == 0.:
                raise ValueError(f'Invalid Figure {panel} image')
        return dict(panels=['2i', '2j', '2k'], checkpoints=len(self.geometry),
                    publisher_comparisons=len(self.comparisons))

    def target(self, figure, *indices):
        return columns(self.source.get(figure, []), *indices)

    def overlay(self, ax, panel, target, model=None, color='0.38', label=None,
                log_score=False, log_x=False, **metadata):
        x, y = target
        if not len(x):
            return
        ax.plot(x, y, '--', color=color, alpha=.6, lw=1, label=label)
        if model is not None:
            self.score(panel, target, model, log_score=log_score, log_x=log_x, **metadata)

    def score(self, panel, target, model, log_score=False, log_x=False, **metadata):
        x, y = target
        if not len(x):
            return
        mx, my = model
        order = np.argsort(x)
        x, y = x[order], y[order]
        valid = (mx >= x[0]) & (mx <= x[-1]) & np.isfinite(my)
        observed = my[valid]
        if log_x:
            reference = np.exp(np.interp(np.log(mx[valid]), np.log(x), np.log(y)))
        else:
            reference = np.interp(mx[valid], x, y)
        residual = observed - reference
        if not residual.size:
            return
        metrics = dict(panel=panel, points=len(residual), **metadata,
                       bias=float(residual.mean()), rms=float(np.sqrt(np.mean(residual**2))))
        if log_score:
            positive = (observed > 0.) & (reference > 0.)
            log_error = np.log10(observed[positive]) - np.log10(reference[positive])
            metrics.update(log10_points=int(positive.sum()),
                           log10_bias=float(log_error.mean()) if log_error.size else None,
                           log10_rms=float(np.sqrt(np.mean(log_error**2))) if log_error.size else None,
                           model_zeros=int((observed == 0.).sum()),
                           reference_zeros=int((reference == 0.).sum()))
        self.comparisons.append(metrics)

    def save(self, fig, panel):
        completed = len(self.records[panel])
        scheduled = len(self.jobs[panel])
        run_type = {'pilot': 'short pilot', 'spot': 'publication-duration spot runs', 'coarse': 'coarse parameter grid',
                    'full': 'publication-duration runs'}[self.manifest['profile']]
        extra = {job['parameters'].get('extra_relaxation_tauT', 0.) for job in self.jobs[panel]}
        if extra - {0.}:
            run_type = 'extended relaxation sensitivity'
        panel_label = '3a exponent' if panel == '3a_alpha' else panel
        if self.manifest.get('replicates_per_setting') == 1:
            run_type = 'one run per setting'
        title = f'{panel_label}  •  {run_type}  •  {self.cells} cells  •  {completed}/{scheduled} checkpoints'
        if panel in ('3b', '3a_alpha'):
            title = f'{panel_label}  •  {self.cells} cells  •  {completed}/{scheduled} checkpoints'
        note = ''
        if panel in self.crossing_panels:
            note = 'Boundary crossings recorded. See run_summary.json.'
        if panel == '2d' and self.undefined_spaces:
            note = f'{len(self.undefined_spaces)} space counts undefined. See run_summary.json.'
        if (panel in ('3a', '3c', '3d') and 3 in self.source
                or panel in ('4a', '4b', '4f', '4g') and 4 in self.source):
            legend = 'Simulation solid · Published dashed'
            if panel == '3d':
                legend += ' · Zero rates: filled triangle for simulation, open triangle for published data'
            note = f'{legend}\n{note}' if note else legend
        if note:
            if fig.get_constrained_layout():
                title += '\n' + note
            else:
                fig.text(.01, .012, note, fontsize=7, color='0.35')
        fig.suptitle(title, fontsize=10)
        if not fig.get_constrained_layout():
            fig.tight_layout(rect=(0, .035, 1, .95))
        temporary = self.out / f'.fig{panel}.{os.getpid()}.tmp.png'
        fig.savefig(temporary, dpi=170)
        temporary.replace(self.out / f'fig{panel}.png')
        plt.close(fig)
        self.produced.append(panel)

    def phase_results(self):
        shear, dynamics = defaultdict(list), defaultdict(list)
        for records in self.records.values():
            for kind, p, d in records:
                if kind not in ('shear', 'dynamics') or p['mu'] < .5 or p['rho'] != 1.:
                    continue
                group = (p['w'], p['mu'])
                target = shear[group] if kind == 'shear' else dynamics[group]
                if all(d is not old for old in target):
                    target.append(d)
        for key in shear.keys() & dynamics.keys():
            ds = shear[key]
            if min(d['time'][-1] for d in ds) < 1000. - 1e-6:
                continue
            noise = np.mean([d['sigma_A'] for d in dynamics[key]])
            if not np.isfinite(noise) or noise <= 0:
                continue
            try:
                *_, crossing = fit_relaxation(ds, noise)
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            if not np.isnan(crossing):
                self.phase[key] = crossing

    def maps(self):
        for panel, quantity, start, label in (
            ('2b', 'phi', 0, PHI), ('2c', 'z', 4, '$z$'), ('2k', 'yield_stress', 60, r'$\sigma_Y/\sigma_0$'),
        ):
            if not self.records[panel]:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(8., 3.5), sharex=True, sharey=True)
            groups = grouped(self.entries(panel), 'w', 'rho')
            source_rho, source_w, source_z = self.target(2, start, start + 1, start + 2)
            truth = {(round(w, 4), round(rho, 4)): z
                     for rho, w, z in zip(source_rho, source_w, source_z)}
            model = {key: np.mean([d[quantity] for d in ds]) for key, ds in groups.items()}
            grid = self.manifest.get('grid')
            if grid is not None:
                grid_w = np.asarray(grid['adhesions'])
                grid_rho = np.asarray(grid['densities'])
                truth = {key: z for key, z in truth.items()
                         if key[0] in grid_w and key[1] in grid_rho}
            residuals = [y - truth[key] for key, y in model.items() if key in truth]
            if residuals:
                self.comparisons.append(dict(panel=panel, points=len(residuals),
                                             bias=float(np.mean(residuals)),
                                             rms=float(np.sqrt(np.mean(np.square(residuals))))))
            values = list(truth.values()) or list(model.values())
            for ax, points, title in zip(axes, (truth, model), ('Published', 'Simulation')):
                if grid is not None and points:
                    field = [[points.get((w, rho), np.nan) for w in grid_w] for rho in grid_rho]
                    tiles = ax.pcolormesh(grid_w, grid_rho, field, shading='nearest',
                                          vmin=min(values + list(model.values())),
                                          vmax=max(values + list(model.values())))
                    ax.set_xticks(grid_w, [f'{w:g}' for w in grid_w])
                    ticks = grid_rho if len(grid_rho) <= 8 else grid_rho[::2]
                    ax.set_yticks(ticks, [f'{rho:g}' for rho in ticks])
                    ax.tick_params(labelsize=8)
                    fig.colorbar(tiles, ax=ax, label=label)
                elif points:
                    x, y = np.asarray(list(points)).T
                    scatter = ax.scatter(x, y, c=list(points.values()), marker='s', cmap='viridis',
                                         vmin=min(values), vmax=max(values))
                    fig.colorbar(scatter, ax=ax, label=label)
                ax.set(title=title, xlabel=W_LABEL, ylabel=r'$\rho$')
            self.save(fig, panel)

    def equilibrium(self):
        if self.records['2a']:
            fig, axes = plt.subplots(3, 3, figsize=(7, 7), squeeze=False)
            data = grouped(self.entries('2a'), 'rho', 'w')
            for ax, ((rho, w), ds) in zip(axes.flat, sorted(data.items(), key=lambda row: (-row[0][0], row[0][1]))):
                draw_foam(ax, ds[0])
                ax.set_title(f'ρ={rho:g}, W/T₀={w:g}', fontsize=8)
            for ax in list(axes.flat)[len(data):]:
                ax.axis('off')
            self.save(fig, '2a')
        if self.records['2d']:
            fig, axes = plt.subplots(2, 1, figsize=(5, 5.5), sharex=True)
            for (rho,), _ in grouped(self.entries('2d'), 'rho').items():
                i = (.81, 1., 1.44).index(rho)
                selected = [(p, d) for p, d in self.entries('2d') if p['rho'] == rho]
                target = self.target(2, 8 + 4*i, 9 + 4*i)
                for j, (key, factor, metric) in enumerate((
                    ('space_density', 1., 'N_E/(2N)'),
                    ('space_mean_area', rho, 'mean space area / box area per cell'),
                )):
                    model = scan(axes[j], selected, 'w', lambda d: factor*d[key], f'ρ={rho:g}', f'C{i}')
                    self.overlay(axes[j], '2d', np.split(target, 2, axis=1)[j], model, f'C{i}',
                                 density=rho, quantity=metric)
            axes[0].set(ylabel=r'$N_E/(2N)$')
            axes[1].set(ylabel=r'$\bar A_E/(A_T/N)$', xlabel=W_LABEL)
            axes[0].legend(fontsize=8)
            self.save(fig, '2d')
        if self.records['2e']:
            fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, layout='constrained')
            for i, (ax, rho) in enumerate(zip(axes, (.81, 1., 1.44))):
                target = self.target(2, 20 + 3*i, 21 + 3*i)
                qs = [(p, d) for p, d in self.entries('2e', 'equilibrium') if p['rho'] == rho]
                model = scan(ax, qs, 'w', lambda d: d['phi'], 'Quenched', 'C2')
                self.overlay(ax, '2e', target[:, 102:], model, 'C2',
                             density=rho, branch='quenched')
                for direction, color, label, ref in (
                    ('up', 'C0', 'Increasing', target[:, :51]),
                    ('down', 'C1', 'Decreasing', target[:, 51:102]),
                ):
                    ds = [d for p, d in self.entries('2e', 'hysteresis')
                          if p['rho'] == rho and p['direction'] == direction]
                    model = None
                    if ds:
                        w = ds[0]['w']
                        model = w, line_stats(ax, w, [d['phi'] for d in ds], label, color)
                    self.overlay(ax, '2e', ref, model, color, density=rho, branch=direction)
                ax.set(title=fr'$\rho={rho:g}$', xlabel=W_LABEL)
            axes[0].set_ylabel(PHI)
            branches = [('Quenched', 'C2'), ('Increasing', 'C0'), ('Decreasing', 'C1')]
            fig.legend(handles=[Line2D([], [], color=color, label=label) for label, color in branches],
                       fontsize=8, ncol=3, loc='outside lower center', frameon=False,
                       title='Solid lines show simulations. Dashed lines show published curves.')
            self.save(fig, '2e')
        if self.records['2f']:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            by_rho = grouped(self.entries('2f'), 'rho')
            x, y = np.asarray([(np.mean([d['phi'] for d in ds]), np.mean([d['z'] for d in ds]))
                               for ds in by_rho.values()]).T
            dx, dy = np.asarray([(np.std([d['phi'] for d in ds]), np.std([d['z'] for d in ds]))
                                 for ds in by_rho.values()]).T
            ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='o-', markersize=4,
                        capsize=2, elinewidth=.8, label='Simulation')
            self.overlay(ax, '2f', self.target(2, 29, 30), (x, y), label='Published')
            if x.min() <= .83 and x.max() > .83 and (x > .83).sum() >= 3:
                above = x > .83
                delta = x[above] - .83
                basis = np.column_stack((np.sqrt(delta), delta))
                coeff = np.linalg.lstsq(basis, y[above] - 4., rcond=None)[0]
                ax.plot(x[above], 4. + basis @ coeff, ':', label=r'Fit at $\phi_c=0.83$')
                coeff = np.linalg.lstsq(basis[:, :1], y[above] - 4., rcond=None)[0]
                ax.plot(x[above], 4. + basis[:, :1] @ coeff, '-.', color='0.5',
                        label='Square-root fit')
            _, d = min(self.entries('2f'), key=lambda entry: abs(entry[1]['phi'] - .83))
            if 'cell_z' in d:
                draw_foam(ax.inset_axes([.61, .12, .32, .36]), d, color_by_contacts=True)
            ax.set(xlabel=PHI, ylabel='$z$')
            ax.legend(fontsize=8)
            self.save(fig, '2f')
        if self.records['2g']:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            model = scan(ax, self.entries('2g'), 'w', lambda d: d['shape_mean'], 'Simulation')
            self.overlay(ax, '2g', self.target(2, 32, 33), model, label='Published')
            selected = grouped(self.entries('2g'), 'w')
            for i, (w,) in enumerate((next(iter(selected)), next(reversed(selected)))):
                inset = ax.inset_axes([.11 + i*.28, .42, .26, .36])
                draw_foam(inset, selected[w,][0], color_by_shape=True)
                inset.set_title(fr'$W/T_0={w:g}$', fontsize=8, pad=2)
            ax.set(xlabel=W_LABEL, ylabel=r'$\bar s$')
            ax.legend(loc='upper left', fontsize=8)
            self.save(fig, '2g')

    def mechanics(self):
        if self.records['2h']:
            _, d = self.entries('2h')[0]
            fig, axes = plt.subplots(2, 3, figsize=(9, 4.5))
            for ax, prefix, name in zip(axes[0], ('initial_', 'affine_', ''),
                                        ('Before shear', 'Affine step', 'After relaxation')):
                draw_foam(ax, d, prefix)
                ax.set_title(name, fontsize=9)
            axes[1, 0].plot(d['time'], np.where(d['time'] < 0, 0., 1.5))
            axes[1, 0].set(xlabel=TIME, ylabel=r'$\epsilon_{xy}$', xlim=(-15, 15))
            axes[1, 1].plot(d['time'], d['stress'])
            axes[1, 1].set(xlabel=TIME, ylabel=STRESS, xlim=(-15, 15))
            axes[1, 2].axis('off')
            self.save(fig, '2h')
        for panel, key, values, figure, start in (
            ('2i', 'w', (0., .2, .5, 1., 1.5), 2, 36),
            ('4a', 'mu', (0., .5, 1., 1.5), 4, 0),
            ('4b', 'mu', (0., .5, 1., 1.5), 4, 16),
        ):
            if not self.records[panel]:
                continue
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ds = grouped(self.entries(panel, 'shear'), key)
            for i, value in enumerate(values):
                runs = ds.get((value,), [])
                target = self.target(figure, start + i*(3 if panel == '4b' else 4),
                                     start + 1 + i*(3 if panel == '4b' else 4))
                target = target[:, target[0] > 0]
                model = None
                if runs:
                    t = runs[0]['time']
                    color = COLORS.get(value, '#37988f') if key == 'mu' else f'C{i}'
                    if panel == '4b':
                        end = min(t[-1], target[0, -1] + 1) if target.size else t[-1]
                        t = np.arange(1, end, dtype=float)
                        ys = [ne_bins(d, t, self.cells) for d in runs] if len(t) else []
                    else:
                        t, ys = t[t > 0], [d['stress'][t > 0] for d in runs]
                    if ys:
                        mean = line_stats(ax, t, ys, f'{value:g}', color)
                        model = (t, mean)
                else:
                    color = COLORS.get(value, '#37988f') if key == 'mu' else f'C{i}'
                coordinate = {'activity': value} if key == 'mu' else {'adhesion': value}
                self.overlay(ax, panel, target, model, color, **coordinate)
            if panel == '4a' and 4 in self.source:
                ax.set_xlim(.05, 580.)
            ax.set(xlabel=TIME, ylabel=NE if panel == '4b' else STRESS, xscale='log')
            if panel == '4b' and ax.get_ylim()[1] > 0:
                ax.set_yscale('log', nonpositive='mask')
            ax.legend(title=MU_LABEL if key == 'mu' else W_LABEL, fontsize=8)
            self.save(fig, panel)

        for panel, key, ylabel in (('2j', 'yield_stress', r'$\sigma_Y/\sigma_0$'),
                                   ('S5', 'stress_jump', r'$\Delta\sigma_{xy}/\sigma_0$')):
            if not self.records[panel]:
                continue
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for (rho,), _ in grouped(self.entries(panel), 'rho').items():
                model = scan(ax, [(p, d) for p, d in self.entries(panel) if p['rho'] == rho],
                             'w', lambda d: d[key], f'ρ={rho:g}')
                if panel == '2j':
                    self.overlay(ax, panel, self.target(2, 56, 57), model, label='Published')
            ax.set(xlabel=W_LABEL, ylabel=ylabel)
            if len(ax.lines) < 9:
                ax.legend(fontsize=8)
            self.save(fig, panel)

    def active(self):
        if self.records['1f']:
            _, d = self.entries('1f')[0]
            t, length, tension = d['junction'].T
            end = t <= 20.
            fig, ax = plt.subplots(figsize=(6, 3.8))
            twin = ax.twinx()
            ax.plot(t[end], length[end], color='C0')
            twin.plot(t[end], tension[end], color='C1')
            ax.set(xlabel=TIME, ylabel=r'Contact length / $L_0$')
            twin.set_ylabel(r'Tension / $T_0$', color='C1')
            self.save(fig, '1f')
        if self.records['3a']:
            fig, ax = plt.subplots(figsize=(6, 4.6))
            fig_alpha, ax_alpha = plt.subplots(figsize=(5.2, 3.8))
            self.jobs['3a_alpha'] = self.jobs['3a']
            self.records['3a_alpha'] = self.records['3a']
            if '3a' in self.crossing_panels:
                self.crossing_panels.add('3a_alpha')
            groups = grouped(self.entries('3a'), 'mu')
            x, alpha_ref = self.target(3, 18, 19)
            if len(x):
                ax_alpha.plot(x, alpha_ref, '--o', color='.5', lw=1.1, ms=4, label='Published')
            alpha_points = []
            for mu in ACTIVITY:
                ref = self.target(3, 3*ACTIVITY.index(mu), 3*ACTIVITY.index(mu)+1)
                ref[0] /= 10.
                ds = groups.get((mu,), [])
                if not ds:
                    self.overlay(ax, '3a', ref, color=COLORS[mu])
                    continue
                t = ds[0]['tlag']
                y = line_stats(ax, t, [d['msd'] for d in ds], f'{mu:g}', COLORS[mu])
                self.overlay(ax, '3a', ref, (t, y), COLORS[mu], log_score=True,
                             log_x=True, activity=mu, seeds=len(ds), frame='lab')
                if all('msd_com' in d for d in ds):
                    com = np.mean([d['msd_com'] for d in ds], axis=0)
                    self.score('3a', ref, (t, com), log_score=True, log_x=True,
                               activity=mu, seeds=len(ds), frame='COM-subtracted')
                select = (t >= 10.) & (t <= 100.) & (y > 0.)
                if select.sum() >= 3:
                    alpha = float(np.polyfit(np.log(t[select]), np.log(y[select]), 1)[0])
                    alpha_points.append((mu, alpha))
                    expected = alpha_ref[x == mu]
                    self.comparisons.append(dict(panel='3a exponent', activity=mu, seeds=len(ds),
                                                 model=alpha, published=float(expected[0]) if len(expected) else None,
                                                 fit_window_tauT=[10., 100.]))
            if alpha_points:
                activity, slopes = np.asarray(alpha_points).T
                ax_alpha.scatter(activity, slopes, c=[COLORS[mu] for mu in activity],
                                 s=27, zorder=3, label='Simulation')
            ax_alpha.set(xlabel=MU_LABEL, ylabel=r'MSD exponent $\alpha$', xticks=ACTIVITY)
            ax_alpha.legend(frameon=False)
            ax_alpha.set_title(r'Simulation fit over $10$–$100\,\tau_T$', fontsize=9)
            ax.set(xlabel=r'$t/\tau_T$', ylabel=MSD, xscale='log', yscale='log')
            handles = [Line2D([], [], color=COLORS[mu], label=f'{mu:g}')
                       for mu in ACTIVITY if 3 in self.source or (mu,) in groups]
            ax.legend(handles=handles, title=MU_LABEL, fontsize=8, loc='lower right')
            self.save(fig, '3a')
            self.save(fig_alpha, '3a_alpha')
        for panel in ('3b', '4h'):
            groups = grouped(self.entries(panel), 'mu')
            if not groups:
                continue
            fig, axes = plt.subplots(1, len(groups), figsize=(3.3*len(groups), 3.8), squeeze=False,
                                     layout='constrained')
            for ax, ((mu,), ds) in zip(axes.flat, groups.items()):
                d = ds[0]
                phase = self.phase.get((1., mu))
                frame = '.35' if phase is None else ('#c32f24' if phase < 1000 else '#24578e')
                collection = draw_foam(ax, d, color_by_shape=panel == '4h', frame=frame)
                if panel == '3b':
                    box = float(d['box'])
                    trajectory = d['trajectory']
                    n = min(len(trajectory), round(1000. / float(d['sample_dt'])) + 1)
                    if 'trajectory_cells' not in d:
                        indices = np.linspace(0, self.cells - 1, min(self.cells, 7), dtype=int)
                        trajectory = trajectory[:, indices]
                    for i, track in enumerate(trajectory[-n:].transpose(1, 0, 2)):
                        track = track - box*np.floor(track[-1]/box)
                        ax.plot(*track.T, color=f'C{i}', linewidth=.7)
                ax.set_title(f'ΔT/T₀ = {mu:g}', fontsize=9)
            if panel == '4h':
                fig.colorbar(collection, ax=axes.ravel().tolist(), location='bottom',
                             shrink=.55, pad=.03, label=r'$P/\sqrt{A}$')
            self.save(fig, panel)
        for panel, metric, ylabel, figure, xcol, ycol, step in (
            ('3c', long_msd, r'MSD at $100\tau_T$ / $L_0^2$', 3, 21, 22, 2),
            ('3d', lambda d: ne_rate(d, self.cells), NE, 3, 35, 36, 1),
            ('4f', lambda d: d['phi_mean'], PHI, 4, 32, 33, 2),
            ('4g', lambda d: d['shape_mean'], r'$\bar s$', 4, 46, 47, 2),
        ):
            runs = self.entries(panel, 'dynamics')
            if not runs or (panel == '3c' and not any(np.isfinite(long_msd(d)) for _, d in runs)):
                continue
            fig, ax = plt.subplots(figsize=(6, 4.5))
            activities = ACTIVITY if panel in ('3c', '3d') else [mu for (mu,) in grouped(runs, 'mu')]
            if panel == '3d':
                ax.set_yscale('log', nonpositive='mask')
                source_rates = self.target(3, 36, 37, 38, 39, 40, 41)
                positive_rates = source_rates[source_rates > 0.]
                floor = positive_rates.min() / 3 if positive_rates.size else 1e-6
            grid = sorted({j['parameters']['w'] for j in self.jobs[panel]}) if panel in ('3c', '3d') else None
            if panel in ('4f', '4g') and 'active_adhesions' in self.manifest:
                grid = self.manifest['active_adhesions']
            for mu in activities:
                color = COLORS[mu]
                samples = [(p, d) for p, d in runs if p['mu'] == mu]
                model = scan(ax, samples, 'w', metric, f'{mu:g}', color, grid=grid)
                source_col = ycol + step*ACTIVITY.index(mu)
                ref = self.target(figure, xcol, source_col)
                if panel in ('3c', '4f', '4g'):
                    w, mean, sd = self.target(figure, xcol, source_col, source_col + 1)
                    ax.fill_between(w, np.maximum(0., mean - sd), mean + sd, color=color, alpha=.09)
                self.overlay(ax, panel, ref, model, color, activity=mu,
                             log_score=panel == '3d')
                if panel == '3d':
                    for (x_values, rates), filled in ((model, True), (ref, False)):
                        zeros = x_values[rates == 0.]
                        if len(zeros):
                            ax.scatter(zeros, np.full(len(zeros), floor), marker='v', s=16,
                                       edgecolors=color, facecolors=color if filled else 'none')
                if panel == '4g':
                    for w, value in zip(*model):
                        tau = self.phase.get((w, mu))
                        if tau is not None:
                            ax.scatter(w, value, marker='s', s=12, zorder=5,
                                       color='#c32f24' if tau < 1000 else '#24578e')
            ax.set(xlabel=W_LABEL, ylabel=ylabel)
            if panel == '3c':
                ax.set_ylim(bottom=0.)
            if panel in ('3c', '3d') and 3 in self.source:
                handles = [Line2D([], [], color=COLORS[mu], label=f'{mu:g}') for mu in ACTIVITY]
                ax.legend(handles=handles, title=MU_LABEL, fontsize=8, ncol=2)
            else:
                ax.legend(title=MU_LABEL, fontsize=8)
            self.save(fig, panel)
        if self.records['ED1']:
            fig, ax = plt.subplots(figsize=(5, 4))
            x, y = self.target('ED1', 0, 1)
            if len(x):
                ax.scatter(x, y, color='.75', s=9, label='Published')
            runs = [(ne_rate(d, self.cells), long_msd(d)) for _, d in self.entries('ED1')]
            xy = np.asarray(runs)
            keep = np.all(np.isfinite(xy) & (xy > 0.), axis=1)
            if keep.any():
                ax.scatter(*xy[keep].T, s=15, color='C0', label='Simulation')
            if keep.sum() >= 3 and np.ptp(xy[keep, 0]) > 0.:
                x, y = xy[keep].T
                exponent, intercept = np.polyfit(np.log(x), np.log(y), 1)
                fit_x = np.geomspace(x.min(), x.max(), 50)
                ax.plot(fit_x, np.exp(intercept) * fit_x**exponent, color='C0',
                        label=f'Fit slope {exponent:.2f}')
            if keep.any():
                ax.set(xlabel=NE, ylabel=MSD, xscale='log', yscale='log')
                ax.legend(fontsize=8)
                self.save(fig, 'ED1')
            else:
                plt.close(fig)

    def phases(self):
        for panel in ('4d', '4e', 'ED2'):
            has_msd = panel == 'ED2' and any(
                np.isfinite(long_msd(d)) for _, d in self.entries(panel, 'dynamics')
            )
            if (not self.phase and not has_msd) or not self.records[panel]:
                continue
            fig, axes = plt.subplots(1, 3 if panel == 'ED2' else 2,
                                     figsize=(10.5 if panel == 'ED2' else 7.5, 3.5),
                                     squeeze=False, sharex=True, sharey=True)
            axes = axes.ravel()
            w, mu, times = self.target(4, 28, 29, 30)
            key = sorted(self.phase)
            model_w, model_mu = np.asarray(key).reshape(-1, 2).T
            model_tau = np.array([self.phase[k] for k in key])
            if panel in ('4d', '4e') and 'active_adhesions' in self.manifest:
                selected = np.isin(w, self.manifest['active_adhesions'])
                w, mu, times = w[selected], mu[selected], times[selected]
                published = {(x, y): tau for x, y, tau in zip(w, mu, times)}
                compared = [(tau, published[k]) for k, tau in zip(key, model_tau) if k in published]
                if compared:
                    simulated, reference = np.asarray(compared).T
                    if panel == '4d':
                        finite = np.isfinite(simulated) & (simulated > 0.) & (reference > 0.)
                        residual = np.log10(simulated[finite] / reference[finite])
                        if len(residual):
                            self.comparisons.append(dict(panel=panel, points=len(residual),
                                                         log10_bias=float(residual.mean()),
                                                         log10_rms=float(np.sqrt(np.mean(residual**2)))))
                    else:
                        agreement = (simulated < 1000.) == (reference < 1000.)
                        self.comparisons.append(dict(panel=panel, points=len(agreement),
                                                     agreement=float(agreement.mean())))
            if panel != 'ED2':
                datasets = [(w, mu, times, 'Published'), (model_w, model_mu, model_tau, 'Simulation')]
            else:
                datasets = [(model_w, model_mu, model_tau, 'Stress relaxation')]
            for ax, (x, y, tau, title) in zip(axes, datasets):
                if len(x):
                    if panel == '4d':
                        scatter = ax.scatter(x, y, c=np.clip(tau, 10., 1e6), marker='s', s=42,
                                             cmap='viridis_r', norm=LogNorm(10., 1e6))
                        fig.colorbar(scatter, ax=ax, label=r'$\tau_{SR}/\tau_R$')
                    else:
                        ax.scatter(x, y, c=np.where(tau < 1000., '#c32f24', '#24578e'), marker='s', s=36)
                ax.set_title(title, fontsize=9)
            if panel == 'ED2':
                groups = grouped(self.entries('ED2', 'dynamics'), 'w', 'mu')
                for ax, threshold in zip(axes[1:], (.5, .25)):
                    for (w, mu), ds in groups.items():
                        msd = np.mean([long_msd(d) for d in ds])
                        if np.isfinite(msd):
                            ax.scatter(w, mu, c='#c32f24' if msd >= threshold else '#24578e',
                                       marker='s', s=36)
                    ax.set_title(f'MSD threshold {threshold:g}', fontsize=9)
            if panel != '4d':
                fig.text(.5, .015, 'Red = fluid    Blue = solid', ha='center', fontsize=8)
            for ax in axes:
                ax.set(xlabel=W_LABEL, ylabel=MU_LABEL)
            self.save(fig, panel)

    def supplement(self):
        if self.records['S2']:
            fig, (ax, ax_sd) = plt.subplots(1, 2, figsize=(9, 3.8))
            groups = grouped(self.entries('S2', 'dynamics'), 'w')
            for (w,), ds in groups.items():
                scan(ax_sd, [(p, d) for p, d in self.entries('S2', 'dynamics') if p['w'] == w],
                     'mu', lambda d: d['sigma_A'], f'{w:g}')
            selected = [(p, d) for p, d in self.entries('S2', 'dynamics')
                        if p['w'] == max(k[0] for k in groups)]
            for (mu,), ds in grouped(selected, 'mu').items():
                d = ds[0]
                t, s = d['active_stress'].T
                ax.plot(t, s, lw=.7, color=COLORS[mu], label=f'{mu:g}')
            ax.set(xlabel=TIME, ylabel=STRESS)
            ax_sd.set(xlabel=MU_LABEL, ylabel=r'$\sigma_A/\sigma_0$')
            ax.legend(title=MU_LABEL, fontsize=8)
            ax_sd.legend(title=W_LABEL, fontsize=6)
            self.save(fig, 'S2')
        if self.records['S3']:
            groups = grouped(self.entries('S3'), 'rho')
            fig, axes = plt.subplots(1, len(groups), figsize=(4.2*len(groups), 3.7), squeeze=False)
            for ax, ((rho,), _) in zip(axes.flat, groups.items()):
                subset = [(p, d) for p, d in self.entries('S3') if p['rho'] == rho]
                for (mu,), _ in grouped(subset, 'mu').items():
                    scan(ax, [(p, d) for p, d in subset if p['mu'] == mu], 'w',
                         lambda d: d['phi_mean'], f'{mu:g}')
                ax.set(title=f'ρ={rho:g}', xlabel=W_LABEL, ylabel=PHI)
                ax.legend(title=MU_LABEL, fontsize=8)
            self.save(fig, 'S3')
        if self.records['S4']:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
            runs = self.entries('S4')
            for (mu,), _ in grouped(runs, 'mu').items():
                subset = [(p, d) for p, d in runs if p['mu'] == mu]
                if mu == 1.5:
                    scan(axes[0], subset, 'w', lambda d: d['film_tension'], f'{mu:g}', COLORS[mu])
                if mu >= .75:
                    scan(axes[1], subset, 'w', lambda d: 100*d['zero_tension_fraction'],
                         f'{mu:g}', COLORS[mu])
            axes[0].set(xlabel=W_LABEL, ylabel=r'Mean film tension / $T_0$')
            axes[1].set(xlabel=W_LABEL, ylabel='Zero-tension films (%)')
            axes[1].legend(title=MU_LABEL, fontsize=8)
            self.save(fig, 'S4')


def gallery(out, summary, panels):
    ordered = []
    for panel in panels:
        if panel in summary['rendered']:
            ordered.append(panel)
            if panel == '3a' and '3a_alpha' in summary['rendered']:
                ordered.append('3a_alpha')

    links = ' · '.join(f'<a href="#panel-{p}">{"3a exponent" if p == "3a_alpha" else p}</a>' for p in ordered)
    images = '\n'.join(f'<figure id="panel-{p}"><a href="fig{p}.png">'
                       f'<img src="fig{p}.png" loading="lazy" alt="Figure {p}"></a></figure>'
                       for p in ordered)
    unavailable = ', '.join(summary['unavailable'])
    write_text(out / 'index.html', f"""<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Active foam simulation figures</title>
<style>
body {{ margin: 2rem auto; width: min(94%, 74rem); font: 16px system-ui, sans-serif; color: #212529; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 29rem), 1fr)); gap: 1rem; }}
figure {{ margin: 0; padding: .5rem; border: 1px solid #dfe2e5; border-radius: .4rem; }}
img {{ display: block; width: 100%; height: auto; }}
nav {{ margin: 1rem 0; line-height: 1.9; }}
</style>
<h1>Active foam simulation figures</h1>
<p>{escape(summary['profile'].title())} run · {summary['completed']} of {summary['requested']} checkpoints.
<a href="run_summary.json">Run summary</a> ·
<a href="https://doi.org/10.1038/s41567-021-01215-1">Original paper</a></p>
{f'<p>Simulation data are pending for {escape(unavailable)}.</p>' if unavailable else ''}
<nav aria-label="Figures">{links}</nav>
<main>{images}</main>
</html>
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--additional-data', nargs='+', type=Path, default=(),
                        help='Join disjoint manifests from the same simulation configuration')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-cache', type=Path)
    parser.add_argument('--skip-panel', action='append', default=[],
                        help='Omit a panel from the figures and gallery')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    render = Render(args.data, args.out, source_tables(args.reference_cache), args.additional_data)
    for panel in args.skip_panel:
        render.records.pop(panel, None)
        render.jobs.pop(panel, None)
    render.manifest['panels'] = [p for p in render.manifest['panels'] if p not in args.skip_panel]
    render.maps()
    render.equilibrium()
    render.mechanics()
    render.active()
    render.phases()
    render.supplement()
    summary = dict(source_hash=render.manifest['source_hash'], profile=render.manifest['profile'],
                   completed=len(render.geometry), requested=len(render.manifest['jobs']),
                   rendered=render.produced,
                   unavailable=[p for p in render.manifest['panels'] if p not in render.produced],
                   limits=render.manifest['limitations'], comparisons=render.comparisons,
                   ensemble_coverage={panel: render.ensemble_coverage(panel)
                                      for panel in ('2e', '2f', '2g', '2i', '2j', '2k', '3a', '3b', '3c', '3d')
                                      if render.jobs[panel]},
                   resolved_crossing_checks=render.geometry,
                   undefined_space_checkpoints=render.undefined_spaces,
                   relaxation_tau_R={f'{w:g},{mu:g}': tau if np.isfinite(tau) else 'no fitted crossing'
                                     for (w, mu), tau in render.phase.items()})
    if render.verify_2efg and summary['completed'] == summary['requested']:
        summary['verification'] = render.verify_2efg_figures(summary)
        print('Verified complete ensembles and images for Figures 2e–g.')
    if (render.verify_3acd and summary['requested'] == 78 * render.manifest.get('replicates_per_setting', 10)
            and summary['completed'] == summary['requested']):
        summary['verification'] = render.verify_figure3_figures(summary)
        print('Verified complete ensembles and images for Figures 3a, 3c, and 3d.')
    if render.verify_2ik and summary['completed'] == summary['requested']:
        summary['verification'] = render.verify_passive_figures(summary)
        print('Verified the single-run grid and images for Figures 2i–k.')
    if 'protocol' in render.manifest:
        summary['protocol'] = render.manifest['protocol']
    for property_name in ('replicates_per_setting', 'seed_selection'):
        if property_name in render.manifest:
            summary[property_name] = render.manifest[property_name]
    if render.figure3_solvers:
        summary['figure3_checkpoints_per_solver'] = dict(render.figure3_solvers)
    if render.passive_solvers:
        summary['passive_checkpoints_per_solver'] = dict(render.passive_solvers)
    if 3 in render.source and any(panel in render.manifest['panels'] for panel in ('3a', '3c', '3d')):
        workbook = next((args.reference_cache / name for name in ('publisher_fig3.xlsx', 'moesm12.xlsx')
                         if (args.reference_cache / name).is_file()), None)
        if workbook is not None:
            summary['figure3_workbook'] = dict(name=workbook.name,
                                               sha256=hashlib.sha256(workbook.read_bytes()).hexdigest())
    write_text(args.out / 'run_summary.json', json.dumps(summary, indent=2))
    gallery(args.out, summary, render.manifest['panels'])
    print(f'Rendered {len(render.produced)} panels from {len(render.geometry)} checkpoints.')


if __name__ == '__main__':
    main()
