"""Activation-profile and refinement runs for the 48-profile epithelial ring.

    PYTHONPATH=. python -m circle.scripts.fold_mechanics

Experiment I of the invagination plan: the activated domain stays at 24 cells
and only the spatial distribution of the extra apical mean tension changes.
The force law, OU equation, pressure laws, wall and fixed connectivity are
those of ``circle.bending``; this script only chooses ``activation_weights``.

Profiles (j = 0..23 across the vegetal half, strongest at its centre):
  uniform   g = 1 on all 24 cells
  cosine    g = sin^2(pi (j + 1/2) / 24)                (existing reference)
  center    g = 1 on the central 8 cells, 0.25 on the other 16

The cosine weights are divided by their largest value so the peak is exactly one
(the stored reference profile), which makes their sum 12.05 instead of 12.

Normalisations: equal peak increment (4 T0) and equal summed increment
sum_i dT g_i.  At peak 4 the sums are 48.21 (cosine), 48.00 (center) and 96.00
(uniform); the uniform profile at peak 2 has sum 48.00.  Cosine and center
therefore match in peak exactly and in sum to 0.4%.

Refinement: the cosine case repeated at half the time step, and at half the
time step and half the film spacing, for 50 tau_R, against the stored controls
at the same resolutions.
"""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from circle.bending import smooth_patch
from circle.scripts import bending_experiments as experiment

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'circle/data/bending_48_20261115'
DATA = ROOT/'circle/data/fold_mechanics_20261116'


def vegetal_order(n=48, cells=24):
    """Indices of the activated cells ordered across the vegetal half."""
    theta = 2*np.pi*(np.arange(n)+.5)/n
    delta = np.angle(np.exp(1j*(theta+np.pi/2)))
    selected = np.flatnonzero(np.abs(delta) < np.pi*cells/n)
    return selected[np.argsort(delta[selected])]


def profiles(n=48, cells=24):
    order = vegetal_order(n, cells)
    uniform = np.zeros(n); uniform[order] = 1.
    center = np.zeros(n); center[order] = .25
    center[order[cells//2-4:cells//2+4]] = 1.
    # ``smooth_patch`` is the stored reference profile: sin^2(pi (j+1/2)/24) divided by
    # its largest value (0.99572), so that the peak weight is exactly one.
    cosine = smooth_patch(n, cells)
    expected = np.sin(np.pi*(np.arange(cells)+.5)/cells)**2
    np.testing.assert_allclose(cosine[order], expected/expected.max(), atol=1e-12)
    return dict(uniform=uniform, cosine=cosine, center=center)


def configs():
    common = dict(bulk=5., lateral=0., patch=24, ramp=5., delay=20., duration=100.,
                  dt=.0025, spacing=.075, collapse_fraction=.005)
    g = profiles()
    cases = [dict(common, name='uniform_peak4', apical=4., activation_weights=g['uniform'].tolist()),
             dict(common, name='uniform_sum48', apical=2., activation_weights=g['uniform'].tolist()),
             dict(common, name='cosine_peak4', apical=4., activation_weights=g['cosine'].tolist()),
             dict(common, name='center_peak4', apical=4., activation_weights=g['center'].tolist()),
             dict(common, name='control', apical=0., activation_weights=g['cosine'].tolist())]
    for label, spacing in (('dt', .075), ('mesh', .0375)):
        cases.append(dict(common, name=f'cosine_peak4_{label}_half', apical=4., dt=.00125,
                          spacing=spacing, duration=50., activation_weights=g['cosine'].tolist(),
                          control_name=f'control_B5_{label}_half'))
    return cases


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def main():
    experiment.PREPARED = experiment.prepare()
    experiment.DATA = DATA; DATA.mkdir(parents=True, exist_ok=True)
    cases = configs()
    for case in cases:
        weights = np.asarray(case['activation_weights'])
        assert np.count_nonzero(weights) == 24
    order = np.argsort([-c['duration']*(2 if c['spacing'] < .07 else 1)/c['dt'] for c in cases])
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('fork')) as pool:
        for summary in pool.map(experiment.run_case, [cases[i] for i in order]):
            print(json.dumps(summary), flush=True)
    # How closely does this machine reproduce the stored prepared state and control?
    mine, stored = read(DATA/'control.json.gz'), read(BASE/'control_B5.json.gz')
    worst = max(np.abs(np.asarray(a)-np.asarray(b)).max()
                for fa, fb in zip(mine['frames'], stored['frames'])
                for a, b in zip(fa['polygons'], fb['polygons']) if len(a) == len(b))
    same = all(len(a) == len(b) for fa, fb in zip(mine['frames'], stored['frames'])
               for a, b in zip(fa['polygons'], fb['polygons']))
    print(json.dumps(dict(control_reproduction=dict(same_node_counts=same,
                                                    largest_difference=float(worst)))))
    (DATA/'control_reproduction.json').write_text(json.dumps(
        dict(same_node_counts=same, largest_difference=float(worst))))


if __name__ == '__main__':
    main()
