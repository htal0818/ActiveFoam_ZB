"""Compare the two author-code MATLAB traces with Figure 2 source data."""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
from openpyxl import load_workbook

matplotlib.use('Agg')
from matplotlib import pyplot as plt


SPOTS = (('w02', .2, 41), ('w05', .5, 45))


def wait_for_traces(directories):
    waiting = list(directories)
    while waiting:
        for directory in waiting[:]:
            log = directory / 'run.log'
            if 'phase=5 ' in log.read_text():
                waiting.remove(directory)
                continue
            pid = int((directory / 'run.pid').read_text())
            try:
                os.kill(pid, 0)
            except PermissionError:
                pass
            except ProcessLookupError as exc:
                if 'phase=5 ' not in log.read_text():
                    raise RuntimeError(f'MATLAB exited before finishing {directory}. See {log}.') from exc
                waiting.remove(directory)
        if waiting:
            time.sleep(30)


def source_columns(rows, first, count):
    block = [row[first-1:first-1+count] for row in rows]
    return np.array([row for row in block if all(isinstance(x, (int, float)) for x in row)]).T


def compare(data, reference, out):
    workbook = load_workbook(reference / 'moesm11.xlsx', read_only=True, data_only=True)
    try:
        rows = list(workbook['figure2'].iter_rows(min_row=6, max_col=63, values_only=True))
    finally:
        workbook.close()
    j_w, j_yield = source_columns(rows, 57, 2)
    k_rho, k_w, k_yield = source_columns(rows, 61, 3)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, layout='constrained')
    summary = []
    for ax, (key, w, column) in zip(axes, SPOTS):
        path = data / key / 'trace.csv'
        raw = np.genfromtxt(path, delimiter=',', names=True)
        t, strain, stress = raw['time_tauR'], raw['strain'], raw['stress_over_sigma0']
        if (t.ndim != 1 or t.size != 6000 or not np.allclose(np.diff(t), .05)
                or not np.all(np.isfinite(stress)) or t[0] >= 0. or t[-1] < 279.9
                or not np.allclose(strain, np.where(t > 0., 1.5, 0.))):
            raise ValueError(f'Invalid or incomplete shear trace in {path}')
        rt, rs, sd = source_columns(rows, column, 3)
        keep = rt > 0.
        order = np.argsort(rt[keep])
        rt, rs, sd = rt[keep][order], rs[keep][order], sd[keep][order]
        compare_at = (t >= rt[0]) & (t <= rt[-1])
        error = stress[compare_at] - np.interp(t[compare_at], rt, rs)
        tail = (t >= 260.) & (t <= 280.)
        ref_tail = (rt >= 260.) & (rt <= 280.)
        summary.append(dict(
            w=w, rho=1., seed=1, cells=100, samples=int(compare_at.sum()),
            rms=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()),
            matlab_mean_260_to_280=float(stress[tail].mean()),
            published_2i_mean_260_to_280=float(rs[ref_tail].mean()),
            published_2j_yield=float(j_yield[np.isclose(j_w,w)].item()),
            published_2k_yield=float(k_yield[np.isclose(k_rho,1.) & np.isclose(k_w,w)].item()),
        ))
        positive = t > 0.
        ax.fill_between(rt, rs-sd, rs+sd, color='.86', linewidth=0,
                        label='Published mean ± SD')
        ax.plot(rt, rs, '--', color='.35', linewidth=1.2)
        ax.plot(t[positive], stress[positive], color='#176c76', linewidth=1.2, label='MATLAB R2026a')
        ax.set(title=fr'$W/T_0={w:g}$', xlabel=r'Time after shear $t/\tau_R$', xscale='log')
    axes[0].set_ylabel(r'Shear stress $\sigma_{xy}/\sigma_0$')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(r'Passive shear reference at $\rho=1$ · 100 cells · seed 1')
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'comparison.png', dpi=180)
    plt.close(fig)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f'Compared both complete MATLAB traces in {out}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.wait:
        wait_for_traces(args.data/key for key, _, _ in SPOTS)
    compare(args.data, args.reference_cache, args.out)


if __name__ == '__main__':
    main()
