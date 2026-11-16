import numpy as np
import pytest

from circle.bending import (BendingFoam, measure_frame, compare_midlines, curvature,
                            smooth_patch, edge_roles, t1_candidates, T1_LENGTH,
                            validate_closed_ring)
from circle.invagination import PolarFoam
from circle.force_probe import isolated_copy
from circle import state


def circle_measure(radius=5., rotation=0., translation=(0., 0.)):
    theta = 2*np.pi*(np.arange(48)+.5)/48+rotation
    center = np.asarray(translation)
    points = center+radius*np.column_stack((np.cos(theta), np.sin(theta)))
    return dict(midline=points, lumen_center=center, height=np.ones(48),
                curvature=curvature(points), curvature_wide=curvature(points, 2))


def test_curvature_and_radial_measure_remove_rigid_motion_and_uniform_shrinkage():
    c = circle_measure()
    a = circle_measure(4.8, .19, (20., -8.))
    np.testing.assert_allclose(c['curvature'], .2, atol=1e-13)
    np.testing.assert_allclose(c['curvature_wide'], .2, atol=1e-13)
    result = compare_midlines(a, c)
    assert result['uniform_radial_difference'] == pytest.approx(-.2)
    np.testing.assert_allclose(result['radial_residual'], 0, atol=1e-12)
    assert result['classification'] == 'no_fold'


def test_inward_dimple_with_reversed_curvature_is_detected():
    c = circle_measure()
    theta = 2*np.pi*(np.arange(48)+.5)/48
    delta = np.angle(np.exp(1j*(theta+np.pi/2)))
    radius = 5-1.5*np.exp(-(delta/.4)**2)
    points = radius[:, None]*np.column_stack((np.cos(theta), np.sin(theta)))
    a = dict(c, midline=points, curvature=curvature(points), curvature_wide=curvature(points, 2))
    result = compare_midlines(a, c, 12)
    assert result['classification'] == 'curvature_inversion'
    assert result['inward_displacement'] > .5


def test_saved_frame_measurements_recover_real_cell_areas_and_boundary_lengths():
    f = PolarFoam.ring(n=48, inner_radius=6.539573317412917)
    m = measure_frame(f.frame())
    np.testing.assert_allclose(m['area'], f.f_area, atol=1e-11)
    outer = np.sqrt(6.539573317412917**2+48/np.pi)
    np.testing.assert_allclose(m['wedge_ratio'], outer/6.539573317412917, rtol=1e-5)
    assert np.all(m['height'] > 1.)


def test_sharp_target_and_ou_update_match_original_polarized_model():
    p = PolarFoam.ring(n=16)
    p.apical_amplitude = 2.
    b = isolated_copy(p); b.__class__ = BendingFoam
    b.activation_weights = b.active_cells.astype(float); b.activation_ramp = 0.
    for _ in range(8):
        p.step(0.); b.step(0.)
    np.testing.assert_array_equal(p.vpos, b.vpos)
    np.testing.assert_array_equal(p.e_t, b.e_t)
    assert not state.unclassified(b)


def test_smooth_targets_leave_basal_tension_unchanged_and_delay_lateral_input():
    f = BendingFoam.ring(n=48)
    f.activation_weights = smooth_patch(48, 8); f.activation_ramp = 5.
    f.apical_amplitude = 2.; f.lateral_amplitude = .5; f.lateral_delay = 20.
    roles = edge_roles(f)
    assert np.count_nonzero(f.activation_weights) == 8
    f.morph_time = 10.
    fixed = f.interface_data()[5]
    assert np.all(fixed[roles == 'basal'] == 1.)
    assert np.all(fixed[roles == 'lateral'] == 1.)
    assert fixed[roles == 'apical'].max() == 3.
    f.morph_time = 25.
    fixed = f.interface_data()[5]
    assert np.all(fixed[roles == 'basal'] == 1.)
    assert fixed[roles == 'lateral'].max() > 1.


def test_candidate_audit_includes_apical_basal_and_lateral_without_rng_draws():
    f = BendingFoam.ring(n=16)
    roles = edge_roles(f); previous = f.edge_lengths(); now = previous.copy()
    selected = [int(np.flatnonzero(roles == name)[0]) for name in ('apical', 'basal', 'lateral')]
    now[selected] = T1_LENGTH/2
    rng = f.rng.bit_generator.state
    assert set(t1_candidates(f, previous, now)) == set(selected)
    assert f.rng.bit_generator.state == rng
    validate_closed_ring(f)
