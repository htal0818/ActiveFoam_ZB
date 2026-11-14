"""Compare paired high-adhesion shear trajectories and their active stress baselines."""
import argparse
import json
from pathlib import Path

import numpy as np

from plot_paper import STRESS, TIME, columns, fit_relaxation, plt, source_tables, stretched


def load_run(directory, kind, parameters, convention, seed):
    manifest = json.loads((directory / 'manifest.json').read_text())
    wanted = ('w', 'rho', 'mu', 'n_side', 'geometry_policy')
    job, = [job for job in manifest['jobs'] if job['kind'] == kind
            and job['parameters']['seed'] == seed
            and job['parameters'].get('pressure_normals', 'reference') == convention
            and all(job['parameters'][key] == parameters[key] for key in wanted)]
    path = directory / (job['id'] + '.npz')
    complete = path.is_file()
    if not complete:
        path = path.with_suffix('.partial.npz')
    with np.load(path, allow_pickle=False) as raw:
        data = {key: raw[key] for key in raw.files if key != 'metadata'}
    return job, data, complete


def crossing_counts(geometry, horizon):
    last = geometry[geometry[:, 0] <= horizon + 1e-9][-1]
    count, resolved, microfold = (last[1:4] - geometry[0, 1:4]).astype(int)
    return dict(time_tauR=float(last[0]), checks=int(count),
                resolved=int(resolved), microfold=int(microfold))


def compare(args):
    runs = {'reference': args.reference, 'boundary': args.boundary}
    palette = {'reference': '#636678', 'boundary': '#047b76'}
    panels = {'reference': 'Centroid normals', 'boundary': 'Boundary normals'}
    parameters = dict(w=1.2, rho=1., mu=1.5, n_side=4, geometry_policy='reference')
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.2), layout='constrained', sharex='col')
    fig.suptitle(r'High-adhesion shear at $W/T_0=1.2$, $\Delta T/T_0=1.5$, $\rho=1$')
    w, mu, tau = columns(source_tables(args.source).get(4, []), 28, 29, 30)
    published = tau[np.isclose(w, parameters['w']) & np.isclose(mu, parameters['mu'])]
    result = dict(parameters=parameters, strain=1.5, seeds={},
                  published_relaxation_tauR=float(published.item()) if len(published) else None)
    for col, seed in enumerate((1, 2)):
        curves, traces, stats = {}, {}, {}
        for convention, directory in runs.items():
            job, shear, complete = load_run(directory, 'shear', parameters, convention, seed)
            baseline = directory if convention == 'boundary' else args.reference_dynamics[col]
            noise_job, dynamics, _ = load_run(baseline, 'dynamics', parameters, convention, seed)
            if job['parameters']['duration'] != 1000. or noise_job['parameters']['duration_tauT'] != 120.:
                raise ValueError('expected the full 1000 tau_R shear and 120 tau_T dynamics windows')
            geometry = shear['geometry_trace']
            t, stress = shear['time'], shear['stress']
            end = float(t[-1] if complete else geometry[-1, 0])
            if convention == 'reference' and seed == 1 and not complete and t[-1] >= 208.26:
                # The per-step replay identified the first invalid area between validation samples.
                end = float(t[t < 208.26][-1])
            keep = (t >= 0.) & (t <= end + 1e-9)
            t, stress = t[keep], stress[keep]
            noise = float(dynamics['sigma_A'])
            row = dict(complete=complete, last_stress_time_tauR=float(t[-1]), shear_job=job['id'],
                       dynamics_job=noise_job['id'], sigma_A=noise, stress_at_zero=float(stress[0]),
                       crossings=crossing_counts(geometry, end),
                       max_depth_including_preparation_L0=float(geometry[-1, 4]),
                       unstrained_resolved_checks=int(dynamics['resolved_crossing_checks']),
                       unstrained_checks=int(dynamics['quality_checks']))
            if complete:
                _, _, fit, crossing = fit_relaxation([shear], noise)
                row.update(fit=dict(zip(('offset', 'amplitude', 'tau_tauR', 'beta'), fit.tolist())),
                           relaxation_tauR=float(crossing) if np.isfinite(crossing) else None,
                           relaxation_within_window=bool(np.isfinite(crossing) and crossing <= end),
                           mean_stress_last_20_tauR=float(stress[t >= end - 20.].mean()))
                fit_time = t[t >= 1.]
                axes[0, col].plot(fit_time, stretched(fit_time, *fit), color=palette[convention],
                                  lw=1.2, ls='--')
            label = panels[convention] + ('' if complete else ' (failed)')
            axes[0, col].plot(t, stress, color=palette[convention], alpha=.72, lw=.65, label=label)
            axes[0, col].axhline(noise, color=palette[convention], lw=.8, ls=':', alpha=.8)
            axes[1, col].plot(geometry[:, 0], geometry[:, 2] - geometry[0, 2],
                              color=palette[convention], lw=1.4)
            curves[convention] = (t, stress)
            traces[convention] = geometry
            stats[convention] = row
        end = min(t[-1] for t, _ in curves.values())
        t, stress = curves['reference']
        select = t <= end + 1e-9
        residual = np.interp(t[select], *curves['boundary']) - stress[select]
        for convention, row in stats.items():
            row['crossings_matched_window'] = crossing_counts(traces[convention], end)
            t, stress = curves[convention]
            tail = (t >= max(0., end - 20.)) & (t <= end + 1e-9)
            row['matched_mean_stress_last_20_tauR'] = float(stress[tail].mean())
        result['seeds'][seed] = dict(runs=stats, matched_stress_window_tauR=float(end),
                                     matched_stress_rms=float(np.sqrt(np.mean(residual**2))))
        if end < 1000.:
            for ax in axes[:, col]:
                ax.axvline(end, color='0.48', lw=.75, ls='-.')
        axes[0, col].set(title=f'Seed {seed}', ylabel=STRESS)
        axes[0, col].legend(loc='upper right', fontsize=8, frameon=False)
        axes[1, col].set(xlabel=TIME, ylabel='Checks with resolved crossings')
        axes[1, col].set_xlim(0, 1000)
    fig.supxlabel('Dashed lines show stress fits. Dotted lines show unstrained stress SD. '
                  'Crossing counts start at shear.', fontsize=9)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / 'pressure_normals.png', dpi=180)
    plt.close(fig)
    (args.out / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('boundary', type=Path)
    parser.add_argument('--reference-dynamics', type=Path, nargs=2, required=True,
                        metavar=('SEED1', 'SEED2'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    compare(parser.parse_args())
