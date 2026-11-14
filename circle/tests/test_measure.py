"""Step 5: packing fraction, wall reaction and traction, wall layer vs interior."""
import numpy as np
import pytest

from circle import protocol
from circle.disc import DiscFoam
from circle.measure import (DynamicsRecorder, cell_centroids, neighbor_pairs, packing_fraction,
                            touching_wall, traction_profile, wall_layer, wall_load,
                            wall_traction, zone_statistics)


@pytest.fixture(scope="module")
def pressed():
    """16 cells at rho = 1.6: the tiling is pushed against the wall everywhere."""
    foam = DiscFoam.random(16, seed=2, w=1.0, rho=1.6, remesh=False)
    wall_points = np.vstack([np.asarray(m) for m in foam.e_mid] + [foam.vpos])
    on_wall = np.hypot(*(wall_points - foam.center).T) >= foam.radius - foam.wall_tol
    unique = np.unique(np.round(wall_points[on_wall], 12), axis=0)
    foam.step(0.0)
    return foam, len(unique)


def test_each_physical_node_is_counted_once(pressed):
    """Junctions end several films but are one node each.

    Contacts are counted from positions alone, so their number must equal the
    number of distinct points on the wall.  Not every contact reacts: cells
    larger than their preferred area pull their films inward.
    """
    foam, unique_wall_points = pressed
    record = foam.wall
    load = wall_load(record)
    assert load["contacts"] == unique_wall_points and load["arrivals"] == 0
    junctions = np.count_nonzero(np.linalg.norm(record.junction_reaction, axis=1))
    nodes = np.count_nonzero(np.linalg.norm(record.node_reaction, axis=1))
    assert junctions > 0 and nodes > 0
    assert junctions + nodes <= unique_wall_points
    # Every reacting node is a start-of-step contact, so the projection estimate equals the
    # constraint value up to the first-order-in-dt term (unrelaxed tiling: large sliding forces).
    assert load["total"] == pytest.approx(load["direct"], rel=1e-3)


def test_traction_is_force_per_length_and_integrates_back(pressed):
    foam, _ = pressed
    load = wall_load(foam.wall)
    angle, traction, tributary = wall_traction(foam)
    assert np.sum(traction * tributary) == pytest.approx(load["total"], rel=1e-12)
    assert 0 < tributary.sum() <= 2 * np.pi * foam.radius
    centers, profile, arclength = traction_profile(foam, bins=24)
    assert np.sum(profile * arclength) == pytest.approx(load["total"], rel=1e-12)
    assert arclength.sum() == pytest.approx(2 * np.pi * foam.radius)
    # Per-length traction is of the order of p_n - T/R_w, not of the nodal force.
    p_n = foam.P0 * (foam.rho / foam.f_area - 1) / np.sqrt(foam.rho)
    assert 0.5 * (p_n.min() - 1 / foam.radius) < traction.mean() < 1.5 * p_n.max()


def test_packing_fraction_uses_the_disc_area():
    foam = DiscFoam.random(36, seed=1)
    assert packing_fraction(foam) == pytest.approx(foam.f_area.sum() / (np.pi * foam.radius ** 2))
    assert foam.f_area.sum() / foam.bs ** 2 < 0.06        # the box area would be meaningless


def test_zone_statistics_partition_the_cells():
    foam = protocol.equilibrated(36, 1.0, 1.3, seed=1)
    stats = zone_statistics(foam)
    layer = wall_layer(foam)
    assert stats["n_wall_layer"] + stats["n_interior"] == foam.nFa
    assert stats["n_wall_layer"] == layer.sum() > 0 and stats["n_interior"] > 0
    assert np.all(layer[touching_wall(foam)])              # touching cells are in the layer
    assert stats["z_wall"] < stats["z_interior"] <= 6.5
    for key in ("shape_wall", "shape_interior", "area_wall", "area_interior"):
        assert np.isfinite(stats[key]) and stats[key] > 0
    centers = cell_centroids(foam)
    assert np.all(np.hypot(*(centers - foam.center).T) < foam.radius)
    assert all(a < b for a, b in neighbor_pairs(foam))


def test_dynamics_recorder_splits_motion_and_exchange_by_zone():
    foam = protocol.initialize(16, 0.5, 1.2, seed=1, anneal_mu=1.0)
    recorder = DynamicsRecorder(foam)
    protocol.evolve(foam, 10.0, 1.0, sample_every=20, observer=recorder)
    out = recorder.results()
    assert out["n_wall"] + out["n_interior"] == 16
    assert out["duration"] == pytest.approx(10.0)
    for name in ("wall", "interior"):
        assert np.all(np.isfinite(out[f"msd_{name}"])) and np.all(out[f"msd_{name}"] >= 0)
        assert out[f"exchange_rate_{name}"] >= 0
    assert out["msd_interior"][-1] > out["msd_interior"][0]
    assert out["wall_load_mean"] > 0

    still = DiscFoam.random(16, seed=1, w=1.0)
    quiet = DynamicsRecorder(still)
    quiet(0.1, still)                                      # no step taken: nothing moved
    result = quiet.results()
    assert result["exchange_rate_wall"] == 0 and np.all(result["msd_interior"] == 0)
