"""Compare the 100-cell boundary-normal diagnostic with Figure 4 source data."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_paper import columns, fit_relaxation, source_tables, write_text
from plot_requested import stretched


def read_run(folder, job):
    path = folder / (job['id'] + '.npz')
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(saved['metadata'].item()) != job:
            raise ValueError(f'Metadata differs from the manifest for {path}')
        return {key: saved[key] for key in saved.files if key != 'metadata'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--reference-cache', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    jobs = {j['kind']: j for j in manifest['jobs']}
    w, mu = 1.2, 1.5
    runs = {}
    report = {'w': w, 'activity': mu, 'rho': 1.0, 'seed': 1, 'cells': 100, 'runs': {}}
    for kind, job in jobs.items():
        p = job['parameters']
        if (p['w'], p['mu'], p['rho'], p['seed'], p['n_side'], p['pressure_normals']) != (
                w, mu, 1.0, 1, 10, 'boundary'):
            raise ValueError('Unexpected parameters in the boundary-normal diagnostic')
        data = read_run(args.data, job)
        error = args.data / (job['id'] + '.error.json')
        info = {'id': job['id'], 'status': 'pending'}
        if data is not None:
            if kind == 'shear':
                time, stress = data['time'], data['stress']
                if (not np.isclose(time[-1], p['duration']) or not np.isfinite(stress).all()
                        or not np.all(np.diff(time) > 0.0)):
                    raise ValueError('Invalid or incomplete shear record')
            else:
                time, stress = data['active_stress'].T
                if (not np.isclose(time[-1], 10 * p['duration_tauT']) or
                        not np.isfinite(stress).all() or not np.all(np.diff(time) > 0.0)):
                    raise ValueError('Invalid or incomplete unstrained record')
            runs[kind] = data
            info.update(status='completed', duration_tau_R=float(time[-1]),
                        geometry_checks=int(data['quality_checks']),
                        crossing_checks=int(data['resolved_crossing_checks']))
        elif error.is_file():
            info.update(status='failed',
                        error=json.loads(error.read_text())['traceback'].splitlines()[-1])
        report['runs'][kind] = info
    source = source_tables(args.reference_cache)[4]
    ref_w, ref_mu, times = columns(source, 28, 29, 30)
    published_tau = times[np.isclose(ref_w, w) & np.isclose(ref_mu, mu)].item()
    report['published_relaxation_tau_R'] = published_tau
    args.out.mkdir(parents=True, exist_ok=True)
    title = r'$W/T_0=1.2$, $\Delta T/T_0=1.5$ · 100 cells · seed 1'
    unstrained = runs.get('dynamics')
    if unstrained is not None:
        noise = float(unstrained['sigma_A'])
        report['sigma_A'] = noise
        fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.3), layout='constrained')
        fig.suptitle(title)
        for ax, (key, label, indices) in zip(axes, (
                ('phi_mean', r'Volume fraction $\phi$ · 4f', (32, 43, 44)),
                ('shape_mean', r'Mean shape factor $\bar{s}$ · 4g', (46, 57, 58)))):
            ref_w, ref_value, ref_band = columns(source, *indices)
            target = np.isclose(ref_w, w)
            value, band = ref_value[target].item(), ref_band[target].item()
            obs = float(unstrained[key])
            report[key] = {'simulation': obs, 'published': value, 'published_band': band}
            ax.errorbar(0, value, yerr=band, fmt='o', color='#66788d', capsize=4)
            ax.scatter(1, obs, s=40, color='#ba632d', zorder=3)
            for position, height in [(0, value), (1, obs)]:
                ax.annotate(f'{height:.5f}', (position, height), xytext=(8, 0),
                            textcoords='offset points', va='center', fontsize=9)
            ax.set(title=label, xlim=(-.35, 1.65),
                   xticks=[0, 1], xticklabels=['Published', 'Boundary normal'])
            ax.grid(axis='y', alpha=.23)
            ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        fig.savefig(args.out / 'structure.png', dpi=170)
        plt.close(fig)
    shear = runs.get('shear')
    if shear is not None:
        t, stress = shear['time'], shear['stress']
        positive_time = t > 0.
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, layout='constrained',
                                 gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(title)
        stress_ax, crossing_ax = axes
        stress_ax.semilogx(t[positive_time], stress[positive_time], color='#ba632d',
                          lw=.9, label='Boundary-normal shear')
        stress_ax.axvline(published_tau, color='#426a8d', ls=':', lw=1.6,
                          label=f'Published time (4d), {published_tau:.1f} τR')
        if unstrained is not None:
            _, _, fit, tau = fit_relaxation([shear], noise)
            selected = t >= 1.
            stress_ax.semilogx(t[selected], stretched(t[selected], *fit), color='#663919',
                              ls='--', lw=1.1, label='Stretched exponential fit')
            stress_ax.axhline(noise, color='#737861', ls='-.', lw=1.,
                              label=r'Unstrained fluctuations $\sigma_A$')
            if np.isfinite(tau) and tau > 0:
                stress_ax.scatter(tau, noise, s=28, color='#ba632d', zorder=4,
                                  label=f'Fitted relaxation time, {tau:.1f} τR')
            report['fit_parameters'] = dict(zip(('offset', 'amplitude', 'time_constant', 'beta'),
                                                map(float, fit)))
            report['relaxation_tau_R'] = float(tau) if np.isfinite(tau) else None
            report['simulation_phase'] = 'fluid' if tau < 1000 else 'solid'
            report['published_phase'] = 'fluid' if published_tau < 1000 else 'solid'
        stress_ax.set_ylabel(r'$\sigma_{xy}/\sigma_0$')
        stress_ax.legend(fontsize=8)
        time, _checks, crossings, _microfolds, _depth = shear['geometry_trace'].T
        crossing = np.diff(crossings) > 0
        crossing_ax.step(time[1:], np.cumsum(crossing), where='post', color='#5e7467')
        crossing_ax.set(xlabel=r'Time after shear, $t/\tau_R$',
                        ylabel='Post-shear checks\nwith crossings',
                        xlim=(t[positive_time][0], t[-1]))
        report['post_shear_crossing_checks'] = int(crossing.sum())
        report['post_shear_geometry_checks'] = int(len(crossing))
        fig.savefig(args.out / 'stress_and_crossings.png', dpi=170)
        plt.close(fig)
    write_text(args.out / 'comparison.json', json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
