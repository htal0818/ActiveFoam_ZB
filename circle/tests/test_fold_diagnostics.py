"""Diagnostics of epithelial-ring runs: exact cases and stored trajectories."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from circle import fold_diagnostics as fd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'circle/data/bending_48_20261115'
HALF = ROOT / 'circle/data/half_ring_bending_20261115'


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


@pytest.fixture(scope='module')
def control():
    return read(BASE / 'control_B5.json.gz')


@pytest.fixture(scope='module')
def cosine():
    return read(HALF / 'half_A4_L0.json.gz')


def ring(n=48, radius=7.0, dent=0.0, width=0.25):
    """Counter-clockwise ring with an optional inward dent at the vegetal pole."""
    theta = 2 * np.pi * (np.arange(n) + .5) / n
    delta = np.angle(np.exp(1j * (theta + np.pi / 2)))
    r = radius - dent * np.exp(-(delta / width) ** 2)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def row(points, height=1.0):
    from circle.bending import curvature
    return dict(midline=points, lumen_center=np.zeros(2), height=np.full(len(points), height),
                curvature=curvature(points), curvature_wide=curvature(points, 2))


def test_turning_sums_to_two_pi_and_circle_curvature_is_exact():
    circle, dented = ring(), ring(dent=1.5)
    assert fd.turning_angles(circle).sum() == pytest.approx(2 * np.pi)
    assert fd.turning_angles(dented).sum() == pytest.approx(2 * np.pi)
    summary = fd.curvature_summary(row(dented), row(circle), fd.vegetal_order())
    assert summary['total_turning'] == pytest.approx(2 * np.pi)
    for span, values in fd.curvature_summary(row(circle), row(circle), fd.vegetal_order())['spans'].items():
        assert values['minimum'] == pytest.approx(1 / 7.0) and values['maximum'] == pytest.approx(1 / 7.0)
        assert values['negative_profiles'] == 0
    # A dent is negative curvature at its centre and extra positive curvature at its rim.
    assert summary['spans'][1]['minimum'] < 0 < 1 / 7.0 < summary['spans'][1]['maximum']
    assert summary['longest_negative_run'] >= 2
    assert set(summary['negative_at_every_span']) <= set(fd.vegetal_order().tolist())


def test_fold_depth_is_zero_without_a_dent_and_equals_the_dent_depth():
    domain = fd.vegetal_order()
    circle = row(ring())
    flat = fd.fold_depth(circle, circle, domain)
    assert flat['domain_depth'] == 0 and flat['patch_depth'] == 0
    # exact sagitta of a circular arc between the two central-patch end cells
    half_angle = np.pi * 7 / 48
    assert flat['patch_sagitta'] == pytest.approx(7.0 * (np.cos(np.pi / 48) - np.cos(half_angle)), rel=1e-9)
    dented = fd.fold_depth(row(ring(dent=1.0, width=0.2)), circle, domain)
    # The two central cells sit pi/48 either side of the pole, where the dent is exp(-(pi/48/0.2)^2)
    # deep; the patch shoulders, 7 pi/48 away, have barely moved.
    at_centre = np.exp(-(np.pi / 48 / 0.2) ** 2)
    assert dented['patch_depth'] == pytest.approx(at_centre, rel=0.02)
    assert dented['domain_depth'] == pytest.approx(at_centre, rel=0.02)
    assert fd.fold_depth(row(ring(dent=-1.0, width=0.2)), circle, domain)['patch_depth'] < 0


def test_radial_response_separates_uniform_contraction_from_a_local_dent():
    circle = row(ring())
    shrunk = fd.radial_response(row(0.95 * ring()), circle)
    assert shrunk['uniform'] == pytest.approx(-0.35, rel=1e-3) and shrunk['residual_rms'] < 1e-6
    dented = fd.radial_response(row(ring(dent=1.0)), circle)
    assert dented['pole_value'] < -0.5 and dented['vegetal_mean'] < 0 < dented['animal_mean']
    assert abs(np.mean(dented['residual'])) < 1e-12


def test_cell_response_labels():
    changes = dict(apical=np.array([-.1, -.1, -.1, 0., 0., 0.]),
                   basal=np.array([0., -.1, .1, -.1, .1, 0.]))
    assert fd.classify_cells(changes).tolist() == [
        'apical shorter, basal unchanged', 'apical and basal shorter',
        'apical shorter, basal longer', 'basal shorter only', 'basal longer only', 'little change']


def test_reconstructed_tensions_match_the_stored_run(cosine):
    """Tension does not depend on geometry, so replaying the OU update must reproduce it."""
    times = [m['elapsed'] for m in cosine['history'][1:]]
    actual, target = fd.apical_tension(cosine['parameters'], cosine['activation_weights'], times)
    stored = np.array([m['actual_tensions']['apical'] for m in cosine['history'][1:]])
    stored_target = np.array([m['target_tensions']['apical'] for m in cosine['history'][1:]])
    assert np.abs(actual.max(axis=1) - stored).max() < 1e-10
    assert np.abs(target.max(axis=1) - stored_target).max() < 1e-10
    weights = np.asarray(cosine['activation_weights'])
    assert np.allclose(target[-1], 1 + 4 * weights) and np.all(actual[-1] <= target[-1] + 1e-12)
    assert np.all(actual[:, weights == 0] == 1.0)


def test_incremental_force_of_a_uniform_increment_on_the_initial_ring(control):
    frame = control['frames'][0]
    n = frame['n']
    result = fd.incremental_force(frame, np.full(n, 0.5))
    assert np.abs(result['net_force']).max() < 1e-10                 # a film cannot push itself
    # Belt tension on a closed convex ring presses every cell inward by the same amount.
    # Summed along the film the pressing force is increment * turning (2 pi in total); the
    # resultant on each cell, taken along that cell's own normal, is smaller because the
    # free films bulge (about 62 degrees of turning per cell here against 7.5 for the ring).
    normal = result['normal_inward']
    assert np.all(normal > 0) and np.ptp(normal) < 1e-3 * normal.mean()
    assert 0.8 * 0.5 * 2 * np.pi < normal.sum() < 0.5 * 2 * np.pi
    turning = result['apical_turning'].mean()
    kink = turning - 2 * np.pi / n
    per_cell = 0.5 * (2 * np.sin(turning / 2) - 2 * np.sin(kink / 2) * np.cos(np.pi / n))
    assert normal.mean() == pytest.approx(per_cell, rel=1e-2)
    assert np.abs(result['tangential']).max() < 1e-6 * normal.mean()
    assert np.abs(result['moment']).max() < 1e-8
    # Vanishes with the increment.
    zero = fd.incremental_force(frame, np.zeros(n))
    assert not zero['force'].any() and not zero['moment'].any()


def test_incremental_force_of_a_graded_increment_is_tangential(control):
    """A tension gradient pulls each junction toward the more contractile side."""
    frame = control['frames'][0]
    n = frame['n']
    gradient = np.zeros(n)
    gradient[10:20] = 0.1 * np.arange(10)                             # rises counter-clockwise
    result = fd.incremental_force(frame, gradient)
    inside = result['tangential'][12:18]
    # Each junction is pulled by the tension difference of its two films, 0.1, times the
    # cosine of the angle at which those (outward-bulging) films meet the surface direction.
    assert np.ptp(inside) < 1e-3 and 0.08 < inside.mean() < 0.1
    assert np.abs(result['net_force']).max() < 1e-10


def test_wall_and_pressure_bookkeeping(control, cosine):
    assert fd.cell_pressure([1.0, 0.5, 2.0]).tolist() == [0.0, 10.0, -5.0]
    clearance = fd.wall_clearance(cosine['frames'][-1])
    assert clearance['min_gap'] > 1.0 and clearance['contact_nodes'] == 0
    final = cosine['history'][-1]
    reference = fd.at_time(control['history'], final['elapsed'])
    lumen = fd.lumen_accounting(final, reference, cosine['history'][0])
    assert lumen['lumen_area_vs_control'] < 0 < lumen['lumen_pressure_vs_control']
    with pytest.raises(KeyError):
        fd.at_time(control['history'], 12.34)


def test_stored_runs_are_classified_without_overstating(control, cosine):
    domain = fd.vegetal_order()
    assert len(domain) == 24 and np.all(np.asarray(cosine['activation_weights'])[domain] > 0)
    shallow = fd.classify_run(cosine, control, domain)
    assert shallow['cls'] == 2 and shallow['patch_depth_in_heights'] < 1     # a shallow fold
    assert fd.classify_run(read(HALF / 'half_cosine.json.gz'), control, domain)['cls'] == 1
    failed = fd.classify_run(read(HALF / 'half_A8_L1.json.gz'), control, domain)
    assert failed['cls'] == 4 and 'crossing' in failed['reason']
    assert fd.classify_run(control, control, domain)['cls'] == 0
