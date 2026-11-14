"""Step 2: the disc integrator and the wall constraint.

The benchmarks live in ``circle.benchmarks``; each returns its time step, film
spacing, duration, equilibrium criterion and measured errors.  Run with
``pytest -rP`` to see them, or ``python circle/scripts/wall_benchmarks.py`` to
regenerate ``circle/BENCHMARKS.md``.
"""
import copy
import json

import numpy as np
import pytest

from activefoam.foam_full import FoamTissue
from circle import benchmarks, state
from circle.disc import DiscFoam, circle_polyline
from circle.measure import wall_load


def report(result):
    def tidy(value):
        return float(f"{value:.6g}") if isinstance(value, float) else value
    rows = result.get("rows", [])
    print(json.dumps({k: tidy(v) for k, v in result.items() if k != "rows"}))
    for row in rows:
        print("   ", json.dumps({k: tidy(v) for k, v in row.items()}))
    return result


def planar_copy(disc, accelerated):
    """A planar ``FoamTissue`` holding the same state and pressure convention."""
    foam = FoamTissue.__new__(FoamTissue)
    disc_only = {"radius", "center", "wall_tol", "delta_max", "audit", "remesh", "wall_resample"}
    for name in state.PARAMETERS - disc_only:
        if hasattr(disc, name):
            setattr(foam, name, copy.deepcopy(getattr(disc, name)))
    snap = state.snapshot(disc)
    for name, value in snap.physical.items():
        setattr(foam, name, value)
    foam.e_mid = [np.array(m) for m in snap.physical["e_mid"]]
    foam.accelerated = accelerated
    foam.rng = np.random.default_rng(11)
    return foam


# --------------------------------------------------------------------------- #
#  the planar force law is preserved
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("accelerated", [False, True])
def test_wall_at_infinity_reproduces_the_planar_step(accelerated):
    """dt 0.005, spacing 0.15, 40 steps at mu = 0.8 after 50 relaxation steps."""
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
    difference = max(np.abs(disc.vpos - planar.vpos).max(),
                     max(np.abs(a - b).max() for a, b in zip(disc.e_mid, planar.e_mid)),
                     np.abs(disc.e_t - planar.e_t).max(), np.abs(disc.f_area - planar.f_area).max())
    print(f"planar path accelerated={accelerated}: largest difference {difference:.3g}")
    assert difference <= (0.0 if accelerated else 1e-11)


def test_no_node_leaves_the_disc_during_an_active_run():
    """16 cells, W = 0.6, mu = 1.0, dt 0.005, spacing 0.15, 7.5 tau_R with transitions."""
    foam = DiscFoam.random(16, seed=1, w=0.6)
    limit = foam.radius * (1 + 1e-12)
    worst = excursion = 0.0
    for k in range(1, 1501):
        foam.advance_scheduled(k, 1.0)
        worst = max(worst, foam.max_radius() / foam.radius - 1)
        excursion = max(excursion, foam.wall.excursion)
        assert foam.max_radius() <= limit
    print(f"largest r/R_w - 1 = {worst:.3g}; largest trial overshoot = {excursion:.3g}")
    assert excursion < 0.05 * foam.radius


# --------------------------------------------------------------------------- #
#  single-cell benchmarks
# --------------------------------------------------------------------------- #
def test_free_cell_far_from_the_wall_matches_the_planar_free_state():
    result = report(benchmarks.free_cell(spacing=0.15, dt=0.005, remesh=True))
    assert result["converged"]
    assert result["wall_reaction"] == 0.0 and result["arrival_steps"] == 0
    assert abs(result["area_error"]) < 1e-3

    # Same trajectory as the planar kernel, node for node.
    center = np.array([12.0, 12.0])
    ring = circle_polyline(center + [0.4, -0.3], 0.6, 0.15)
    foam = DiscFoam.from_closed_cells(3.0, [ring], center=center)
    planar = planar_copy(DiscFoam.from_closed_cells(3.0, [ring], center=center), True)
    for _ in range(4000):
        foam.step(0.0)
        planar.step(0.0)
    assert np.array_equal(foam.vpos, planar.vpos) and foam.f_area[0] == planar.f_area[0]


def test_squeezed_cell_traction_on_a_fixed_mesh():
    """Remeshing off, every node on the wall from the start: pure discretisation error."""
    errors = []
    for spacing in benchmarks.SPACINGS:
        result = report(benchmarks.squeezed_cell(spacing, dt=0.005, remesh=False,
                                                 start="on_wall"))
        assert result["converged"] and result["resample_events"] == 0
        assert result["reference"] > 0
        errors.append(abs(result["traction_error"]))
        # Axisymmetric: zero resultant although the integrated magnitude is not zero.
        assert result["resultant"] < 1e-10 and result["torque"] < 1e-10
        # All nodes are start-of-step contacts: the estimate equals the constraint value.
        assert result["projection_vs_constraint"] < 1e-9
        # Per-length traction from tributary lengths, against p_n - T_eff/R_w.
        assert result["traction_mean"] == pytest.approx(result["traction_reference"],
                                                        rel=errors[-1] + 1e-6)
        assert result["traction_spread"] < 1e-9
        assert result["tributary_sum"] == pytest.approx(2 * np.pi * np.sqrt(1 / np.pi), rel=5e-3)
    assert errors[0] < 0.02
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.5 < coarse / fine < 4.5              # second order in the spacing


@pytest.mark.parametrize("remesh", [False, True])
def test_squeezed_cell_that_grows_onto_the_wall(remesh):
    """A smaller off-centre cell expands, meets the wall on one side and fills the disc.

    Fixed mesh: the node count stays that of the small cell, so the final
    spacing is stretched and the error is larger.  Remeshing on: films are
    resampled as they lengthen and the on-wall discretisation error is recovered.

    On a fixed mesh the nodes keep sliding along the frictionless wall under
    forces of third order in the segment angle, so the time to meet the
    equilibrium criterion grows quickly as the spacing shrinks: 75 tau_R at
    spacing 0.15, more than 400 tau_R at 0.075.  The fixed-mesh case is
    therefore run at 0.15 and 0.1 only.
    """
    spacings = (0.15, 0.1)          # finer fixed meshes equilibrate too slowly (see docstring)
    errors, stretched = [], []
    for spacing in spacings:
        result = report(benchmarks.squeezed_cell(spacing, dt=0.005, remesh=remesh,
                                                 start="grow"))
        assert result["converged"]
        assert (result["resample_events"] > 0) == remesh
        assert result["arrival_steps"] > 0
        assert result["projection_vs_constraint"] < 1e-6
        assert result["resultant"] < 1e-3
        errors.append(abs(result["traction_error"]))
        stretched.append(result["nodes"])
    reference = [abs(benchmarks.squeezed_cell(s, remesh=False, start="on_wall")["traction_error"])
                 for s in spacings]
    if remesh:
        assert errors == pytest.approx(reference, rel=0.1)
    else:
        assert all(1.5 * r < e < 2.5 * r for e, r in zip(errors, reference))
    assert errors[1] < errors[0] / 1.8            # (0.15/0.1)^2 = 2.25 at second order


def test_cell_that_no_longer_fills_the_disc_detaches():
    result = report(benchmarks.detaching_cell(spacing=0.1, dt=0.005, remesh=True))
    assert result["converged"]
    assert result["first_step_contacts"] > 0 and result["first_step_reaction"] == 0.0
    assert result["final_contacts"] == 0 and result["max_radius"] < 0.9
    assert abs(result["area_error"]) < 1e-3


def test_projection_reaction_converges_to_the_constraint_on_a_fixed_mesh():
    """One step from one state at dt, dt/2, dt/4: the estimate approaches max(0, f.n)."""
    result = report(benchmarks.projection_vs_constraint(dts=(0.005, 0.0025, 0.00125)))
    rows = result["rows"]
    assert all(row["contacts"] > 10 and row["constraint_unchanged"] for row in rows)
    assert rows[0]["max_error"] > 0
    for coarse, fine in zip(rows, rows[1:]):
        assert 1.9 < coarse["max_error"] / fine["max_error"] < 2.1      # first order in dt


# --------------------------------------------------------------------------- #
#  remeshing enabled
# --------------------------------------------------------------------------- #
def test_time_averaged_traction_under_joint_refinement_with_remeshing():
    """Three cells in partial wall contact; dt and spacing refined together."""
    result = report(benchmarks.three_cell_refinement(scales=(1, 2, 4), duration=150.0))
    rows = result["rows"]
    assert all(row["converged"] for row in rows)
    assert all(row["resample_events"] > 0 for row in rows)
    assert rows[2]["change_from_coarser"] < rows[1]["change_from_coarser"]
    # Partial contact converges more slowly than full contact: bounds, not an order.
    assert rows[1]["change_from_coarser"] < 0.10 and rows[2]["change_from_coarser"] < 0.04
    assert all(row["intermittency"] < 2e-3 for row in rows)


def test_wall_aware_resampling_removes_the_transient_contact_loss():
    """Optional ``wall_resample``: same equilibrium, smaller instantaneous dips."""
    plain = benchmarks.three_cell_refinement(scales=(4,), duration=40.0)["rows"][0]
    aware = benchmarks.three_cell_refinement(scales=(4,), duration=40.0,
                                             wall_resample=True)["rows"][0]
    print(f"transient dip plain {plain['transient_dip']:.3f}, wall-aware {aware['transient_dip']:.3f}")
    assert plain["transient_dip"] < 0.8 < aware["transient_dip"]
    assert aware["mean"] == pytest.approx(plain["mean"], rel=1e-2)


def test_wall_reactions_are_radial():
    """36 cells, W = 0.8, mu = 0.7, dt 0.005, spacing 0.15, 4 tau_R with transitions."""
    foam = DiscFoam.random(36, seed=2, w=0.8)
    for k in range(1, 801):
        foam.advance_scheduled(k, 0.7)
    load = wall_load(foam.wall)
    print(f"total {load['total']:.4g}, resultant {np.hypot(*load['resultant']):.4g}, "
          f"torque {load['torque']:.3g}")
    assert load["total"] > 0
    assert abs(load["torque"]) < 1e-9 * load["total"] * foam.radius
    # An asymmetric confined tissue generally has a nonzero resultant.
    assert np.hypot(*load["resultant"]) <= load["total"]
