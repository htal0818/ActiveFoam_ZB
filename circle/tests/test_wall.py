"""Step 2: the disc integrator and the wall constraint."""
import copy

import numpy as np
import pytest
from scipy.optimize import brentq

from activefoam.foam_full import FoamTissue
from circle import state
from circle.disc import DiscFoam, circle_polyline
from circle.measure import wall_load


def planar_copy(disc, accelerated):
    """A planar ``FoamTissue`` holding the same state and pressure convention."""
    foam = FoamTissue.__new__(FoamTissue)
    for name in state.PARAMETERS:
        if hasattr(disc, name) and name not in {"radius", "center", "wall_tol", "delta_max",
                                                "audit", "remesh"}:
            setattr(foam, name, copy.deepcopy(getattr(disc, name)))
    snap = state.snapshot(disc)
    for name, value in snap.physical.items():
        setattr(foam, name, value)
    foam.e_mid = [np.array(m) for m in snap.physical["e_mid"]]
    foam.accelerated = accelerated
    foam.rng = np.random.default_rng(11)
    return foam


@pytest.mark.parametrize("accelerated", [False, True])
def test_wall_at_infinity_reproduces_the_planar_step(accelerated):
    disc = DiscFoam.random(16, seed=4, w=0.7)
    for _ in range(50):
        disc.step(0.0)
    planar = planar_copy(disc, accelerated)
    disc.rng = np.random.default_rng(11)
    disc.radius = 1e9                       # the wall never acts
    for _ in range(40):
        disc.step(0.8)
        planar.step(0.8)
    assert len(disc.wall.node_edge) == 0 and not disc.wall.junction_flag.any()
    assert np.array_equal(disc.e_rfn, planar.e_rfn)
    tolerance = 0.0 if accelerated else 1e-11
    assert np.allclose(disc.vpos, planar.vpos, rtol=0, atol=tolerance)
    assert np.allclose(disc.e_t, planar.e_t, rtol=0, atol=tolerance)
    assert np.allclose(disc.f_area, planar.f_area, rtol=0, atol=tolerance)
    for a, b in zip(disc.e_mid, planar.e_mid):
        assert np.allclose(a, b, rtol=0, atol=tolerance)


def test_no_node_leaves_the_disc_during_an_active_run():
    foam = DiscFoam.random(16, seed=1, w=0.6)
    limit = foam.radius * (1 + 1e-12)
    for k in range(1, 1501):
        foam.advance_scheduled(k, 1.0)
        assert foam.max_radius() <= limit
    assert foam.wall.excursion < 0.05 * foam.radius


def free_area(rho, pressure=10.0, tension=1.0):
    """Area of an isolated cell: p_n = T_eff / r with p_n = P0 (rho/A - 1)/sqrt(rho)."""
    radius = brentq(lambda r: pressure * (rho / (np.pi * r * r) - 1) / np.sqrt(rho) - tension / r,
                    1e-2, 5.0)
    return np.pi * radius ** 2


def test_free_cell_far_from_the_wall_matches_the_planar_free_state():
    center = np.array([12.0, 12.0])
    ring = circle_polyline(center + [0.4, -0.3], 0.6, 0.15)
    foam = DiscFoam.from_closed_cells(3.0, [ring], center=center)
    reaction = 0.0
    for _ in range(8000):
        foam.step(0.0)
        reaction += wall_load(foam.wall)["total"]
    assert reaction == 0.0
    assert foam.f_area[0] == pytest.approx(free_area(1.0), rel=2e-3)

    planar = planar_copy(DiscFoam.from_closed_cells(3.0, [ring], center=center), True)
    for _ in range(8000):
        planar.step(0.0)
    assert np.allclose(foam.vpos, planar.vpos, rtol=0, atol=0)
    assert foam.f_area[0] == planar.f_area[0]


def squeezed(spacing, rho=1.5):
    radius = np.sqrt(1.0 / np.pi)
    foam = DiscFoam.from_closed_cells(
        radius, [circle_polyline((4 * radius, 4 * radius), radius, spacing)],
        rho=rho, edpc=spacing)
    for _ in range(200):
        foam.step(0.0)
    return foam


def test_squeezed_cell_traction_matches_pressure_minus_laplace_term():
    """Integrated traction magnitude 2 pi R_w (p_n - T_eff/R_w), in code units."""
    errors = []
    for spacing in (0.15, 0.075, 0.0375):
        foam = squeezed(spacing)
        load = wall_load(foam.wall)
        p_n = foam.P0 * (foam.rho / foam.f_area[0] - 1) / np.sqrt(foam.rho)
        t_eff = max(foam.e_t[0], 0.0)
        reference = 2 * np.pi * foam.radius * (p_n - t_eff / foam.radius)
        assert reference > 0
        errors.append(abs(load["total"] / reference - 1))
        # Axisymmetric case: the resultant vanishes although the total does not.
        assert np.hypot(*load["resultant"]) < 1e-10 * load["total"]
        assert abs(load["torque"]) < 1e-10 * load["total"] * foam.radius
        # Every node is on the wall, so the projection estimate is the constraint value.
        assert load["total"] == pytest.approx(load["direct"], rel=1e-9)
        assert load["arrivals"] == 0
    assert errors[0] < 0.03
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.0 < coarse / fine < 5.0          # second order in the spacing


def test_cell_that_no_longer_fills_the_disc_detaches():
    radius = np.sqrt(1.0 / np.pi)
    foam = DiscFoam.from_closed_cells(
        radius, [circle_polyline((4 * radius, 4 * radius), radius, 0.1)], rho=0.5, edpc=0.1)
    foam.step(0.0)
    assert wall_load(foam.wall)["total"] == 0.0      # inward force: no reaction at all
    for _ in range(6000):
        foam.step(0.0)
    assert len(foam.wall.node_edge) == 0 and not foam.wall.junction_flag.any()
    assert foam.max_radius() < 0.9 * foam.radius
    assert foam.f_area[0] == pytest.approx(free_area(0.5), rel=3e-3)


def test_projection_reaction_converges_to_the_constraint_on_a_fixed_mesh():
    """One step from one state at dt, dt/2, dt/4: the estimate approaches max(0, f.n)."""
    foam = DiscFoam.random(16, seed=1, w=1.0)
    for _ in range(400):
        foam.step(0.0)
    foam.remesh = False
    for _ in range(5):                      # every wall node is now a start-of-step contact
        foam.step(0.0)
    start = state.snapshot(foam)
    errors = []
    for dt in (0.005, 0.0025, 0.00125):
        state.restore(foam, start)
        foam.dt = dt
        foam.step(0.0)
        wall = foam.wall
        contact = wall.node_flag == 1
        assert contact.sum() > 10
        estimate = np.linalg.norm(wall.node_reaction[contact], axis=1)
        errors.append(np.abs(estimate - wall.node_direct[contact]).max())
        # The constraint value itself does not depend on dt.
        if dt == 0.005:
            direct = wall.node_direct[contact].copy()
        else:
            assert np.array_equal(wall.node_direct[contact], direct)
    assert errors[0] > 0
    for coarse, fine in zip(errors, errors[1:]):
        assert 1.8 < coarse / fine < 2.2          # first order in dt


def relaxed_three_cells(scale):
    """Three-cell disc relaxed at zero activity; refinement level ``scale``."""
    foam = DiscFoam.random(3, seed=0, w=1.0, dt=0.005 / scale, edpc=0.15 / scale)
    steps = int(round(30.0 / foam.dt))
    window = steps // 5
    loads, resampled = [], 0
    for k in range(steps):
        before = foam.e_rfn.copy()
        foam.step(0.0)
        resampled += int(np.any(foam.e_rfn != before))
        if k >= steps - window:
            loads.append(wall_load(foam.wall)["total"])
    return np.mean(loads), np.std(loads), resampled


def test_time_averaged_traction_converges_with_remeshing_on():
    """Joint refinement of dt and spacing, with films remeshing along the way."""
    results = [relaxed_three_cells(scale) for scale in (1, 2, 4)]
    means = [r[0] for r in results]
    assert all(r[2] > 0 for r in results)                  # remeshing did happen
    assert all(m > 0 for m in means)
    coarse, fine = abs(means[0] - means[1]), abs(means[1] - means[2])
    assert fine < coarse
    assert fine < 0.02 * means[2]


def test_wall_reactions_are_radial():
    foam = DiscFoam.random(36, seed=2, w=0.8)
    for k in range(1, 801):
        foam.advance_scheduled(k, 0.7)
    load = wall_load(foam.wall)
    assert load["total"] > 0
    assert abs(load["torque"]) < 1e-9 * load["total"] * foam.radius
    # An asymmetric confined tissue generally has a nonzero resultant.
    assert np.hypot(*load["resultant"]) <= load["total"]
