"""Step 4: geometry diagnostics, Euclidean audit and box independence."""
import numpy as np
import pytest
from numba.typed import List

from circle import protocol, state
from circle.disc import DiscFoam, circle_polyline
from circle.kernels import contained_cells, film_crossings, films_finite


def typed(lines):
    return List([np.array(line, dtype=float, order="C") for line in lines])


# --------------------------------------------------------------------------- #
#  crossing and overlap diagnostics
# --------------------------------------------------------------------------- #
def test_clean_tiling_has_no_crossings_and_no_nested_cells():
    foam = DiscFoam.random(36, seed=4)
    mids, loops = foam._fast_geometry_inputs()
    assert film_crossings(mids, foam.e_v) == 0
    assert contained_cells(mids, loops, foam.e_f) == 0
    assert films_finite(foam.vpos, mids)


def test_adjacent_films_crossing_away_from_their_shared_junction_are_counted():
    """Two films share junction 0 and cross again at (1, 1/3): one crossing."""
    a = [[0, 0], [1.5, 0.5], [2, 0]]               # junction 0 -> junction 1
    b = [[0, 0], [1, 0.0], [1, 1.0]]               # junction 0 -> junction 2
    ends = np.array([[0, 1], [0, 2]], dtype=np.int64)
    assert film_crossings(typed([a, b]), ends) == 1
    # Reversing a film changes which end is shared, not the answer.
    ends_reversed = np.array([[0, 1], [2, 0]], dtype=np.int64)
    assert film_crossings(typed([a, b[::-1]]), ends_reversed) == 1


def test_shared_endpoint_alone_is_not_a_crossing():
    a = [[0, 0], [1.5, 0.5], [2, 0]]
    b = [[0, 0], [0.5, -0.5], [1, -1.0]]
    ends = np.array([[0, 1], [0, 2]], dtype=np.int64)
    assert film_crossings(typed([a, b]), ends) == 0
    # Two films joining the same two junctions (a two-sided cell) do not cross either.
    c = [[0, 0], [1, -0.5], [2, 0]]
    both = np.array([[0, 1], [0, 1]], dtype=np.int64)
    assert film_crossings(typed([a, c]), both) == 0


def test_films_without_a_common_junction_and_self_crossings_are_counted():
    a = [[0, 0], [2, 2]]
    b = [[0, 2], [2, 0]]
    ends = np.array([[0, 1], [2, 3]], dtype=np.int64)
    assert film_crossings(typed([a, b]), ends) == 1
    loop = [[0, 0], [2, 0], [2, 2], [1, -1], [0, 1]]        # one film crossing itself
    assert film_crossings(typed([loop]), np.array([[0, 1]], dtype=np.int64)) == 2


def test_nonfinite_film_node_is_detected():
    foam = DiscFoam.random(16, seed=4)
    mids, _ = foam._fast_geometry_inputs()
    assert foam._invalid_reason() is None
    e = int(np.argmax([len(m) for m in foam.e_mid]))
    line = np.array(foam.e_mid[e]); line[1, 1] = np.inf
    foam.e_mid[e] = line
    assert np.all(np.isfinite(foam.vpos)) and np.all(np.isfinite(foam.f_area))
    assert "nonfinite" in foam._invalid_reason()


def test_packing_fraction_does_not_detect_overlap():
    """Overlapping cells below the disc area: the area check passes, the explicit ones do not."""
    radius = 3.0
    center = np.array([12.0, 12.0])
    crossing = DiscFoam.from_closed_cells(
        radius, [circle_polyline(center + [-0.4, 0], 0.6, 0.1),
                 circle_polyline(center + [0.4, 0], 0.6, 0.1)], center=center)
    mids, loops = crossing._fast_geometry_inputs()
    assert crossing.volume_fraction() < 0.1
    assert film_crossings(mids, crossing.e_v) == 2
    assert contained_cells(mids, loops, crossing.e_f) == 0

    nested = DiscFoam.from_closed_cells(
        radius, [circle_polyline(center, 1.0, 0.1), circle_polyline(center + [0.1, 0], 0.4, 0.1)],
        center=center)
    mids, loops = nested._fast_geometry_inputs()
    assert nested.volume_fraction() < 0.2
    assert film_crossings(mids, nested.e_v) == 0
    assert contained_cells(mids, loops, nested.e_f) == 1
    with pytest.raises(ValueError, match="nested"):
        protocol.validate(nested)


def test_validate_reports_diagnostics_and_enforces_the_wall():
    foam = DiscFoam.random(16, seed=1, w=0.6)
    for k in range(1, 401):
        foam.advance_scheduled(k, 0.5)
    report = protocol.validate(foam)
    assert 0 < report["packing"] <= 1 and report["nested"] == 0
    assert report["max_radius"] <= foam.radius * (1 + 1e-12)
    assert state.unclassified(foam) == []
    foam.vpos[0] = foam.center + [1.5 * foam.radius, 0.0]
    with pytest.raises(ValueError, match="outside the wall"):
        protocol.validate(foam)


# --------------------------------------------------------------------------- #
#  Euclidean audit and box independence over an active run
# --------------------------------------------------------------------------- #
def active_run(box_factor, audit=False, steps=2000):
    foam = DiscFoam.random(16, seed=3, w=0.3, audit=audit,
                           box=None if box_factor is None else box_factor * np.sqrt(16 / np.pi))
    for k in range(1, steps + 1):
        foam.advance_scheduled(k, 1.5)
        if k % 100 == 0:
            protocol.validate(foam, strict=False)
    return foam


def test_audit_mode_finds_no_periodic_shift_in_an_active_run():
    """Every Python-level unwrap and wrap is compared with its periodic version."""
    foam = active_run(None, audit=True)
    assert foam.audit_calls > 1000
    kinds = foam.log.counts()
    # All five kinds under audit are covered by the 36-cell run in test_transitions.
    assert {"T1", "T2r", "T3r"} <= set(kinds)
    assert foam.wall.excursion < foam.delta_max


def test_active_run_is_independent_of_the_auxiliary_box(record_property):
    """Fixed absolute disc coordinates, three box sizes, same seed."""
    reference = active_run(8.0)
    assert len(reference.log.accepted) > 30
    for factor in (16.0, 1000.0):
        other = active_run(factor)
        assert np.array_equal(other.center, reference.center)
        assert other.log.accepted == reference.log.accepted
        assert other.log.rejected == reference.log.rejected
        assert other.log.declined == reference.log.declined
        assert other.e_v.shape == reference.e_v.shape and np.array_equal(other.e_v, reference.e_v)
        difference = max(np.abs(other.vpos - reference.vpos).max(),
                         np.abs(other.e_t - reference.e_t).max(),
                         max(np.abs(np.asarray(a) - np.asarray(b)).max()
                             for a, b in zip(other.e_mid, reference.e_mid)))
        bitwise = difference == 0.0
        record_property(f"box_{factor:g}R_bitwise_identical", bool(bitwise))
        print(f"box {factor:g} R_w: largest difference {difference:.3g}, bitwise identical: {bitwise}")
        assert difference < 1e-9
