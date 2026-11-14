"""Local contact probes: separation-controlled contraction and force-controlled pulses.

Prepared states: 16 cells, W/T0 = 1, rho = 1.3, dt 0.005, film spacing 0.15,
equilibrated at zero activity by ``protocol.equilibrated`` (disc) and
``protocol.periodic_equilibrated`` (4 x 4 periodic box).
"""
import copy

import numpy as np
import pytest

from activefoam.foam_full import SPACE
from circle import probe, protocol, state


@pytest.fixture(scope="module")
def disc():
    return protocol.equilibrated(16, 1.0, 1.3, seed=1)


@pytest.fixture(scope="module")
def periodic():
    return protocol.periodic_equilibrated(4, 1.0, 1.3, seed=1)


def longest_interior(foam):
    lengths = foam.edge_lengths()
    eligible = [e for e in probe.contacts(foam) if probe.zone(foam, e) != "wall"]
    return max(eligible, key=lambda e: lengths[e])


@pytest.fixture(scope="module")
def disc_curve(disc):
    return probe.contract_contact(disc, longest_interior(disc))


def test_probes_leave_the_disc_untouched(disc):
    before = state.snapshot(disc)
    iteration, accepted = disc.log.iteration, len(disc.log.accepted)
    e = longest_interior(disc)
    probe.contract_contact(disc, e, levels=3, max_hold=3.0)
    probe.dipole_pulse(disc, e, 0.05, hold=2.0, release=2.0)
    assert state.differences(disc, before) == []
    assert disc.log.iteration == iteration
    assert state.unclassified(disc) == []
    assert len(disc.log.accepted) >= accepted          # the log is a diagnostic, never rewound


def test_probes_leave_the_periodic_foam_untouched(periodic):
    e = longest_interior(periodic)
    positions, tensions = periodic.vpos.copy(), periodic.e_t.copy()
    films = [np.array(m) for m in periodic.e_mid]
    generator = copy.deepcopy(periodic.rng.bit_generator.state)
    for _ in range(2):                                  # the saved state is reusable
        result = probe.contract_contact(periodic, e, levels=3, max_hold=3.0)
        probe.dipole_pulse(periodic, e, 0.05, hold=2.0, release=2.0)
        assert np.array_equal(periodic.vpos, positions) and np.array_equal(periodic.e_t, tensions)
        assert all(np.array_equal(a, b) for a, b in zip(periodic.e_mid, films))
        assert periodic.rng.bit_generator.state == generator
    assert result["work"] > 0 and probe.zone(periodic, e) == "periodic"


def test_stepwise_contraction_is_quasi_static(disc, disc_curve):
    """Levels 8 vs 16 and hold 15 vs 40 tau_R: the work to the T1 length is stable."""
    e = disc_curve["edge"]
    assert disc_curve["collapsed"] and disc_curve["all_converged"]
    assert disc_curve["final_separation"] == pytest.approx(probe.T1_LENGTH)
    assert abs(disc_curve["baseline_force"]) < 0.02
    assert np.all(np.diff(disc_curve["force"]) > 0)       # resistance grows as the contact shortens
    assert disc_curve["stiffness"] > 0 and disc_curve["work"] > 0
    finer = probe.contract_contact(disc, e, levels=16)
    longer = probe.contract_contact(disc, e, max_hold=40.0, tolerance=5e-4)
    print(f"work: 8 levels {disc_curve['work']:.4f}, 16 levels {finer['work']:.4f}, "
          f"long holds {longer['work']:.4f}; stiffness {disc_curve['stiffness']:.3f}, "
          f"{longer['stiffness']:.3f}")
    assert finer["work"] == pytest.approx(disc_curve["work"], rel=0.03)
    assert longer["work"] == pytest.approx(disc_curve["work"], rel=0.03)
    assert longer["stiffness"] == pytest.approx(disc_curve["stiffness"], rel=0.10)


def test_time_step_convergence_of_the_holding_force(disc, disc_curve):
    saved = probe.save(disc)
    try:
        disc.dt = 0.0025
        halved = probe.contract_contact(disc, disc_curve["edge"])
    finally:
        disc.dt = 0.005
        probe.load(disc, saved)
    print(f"work at dt 0.005 {disc_curve['work']:.4f}, at dt 0.0025 {halved['work']:.4f}")
    assert halved["work"] == pytest.approx(disc_curve["work"], rel=0.03)
    assert halved["peak_force"] == pytest.approx(disc_curve["peak_force"], rel=0.03)


def test_force_control_agrees_with_separation_control(disc, disc_curve):
    """A held force of 0.1 must sit on the stepwise force-separation curve.

    The sham run (zero force) removes the slow drift of the prepared state.
    """
    e, force = disc_curve["edge"], 0.1
    sham = probe.dipole_pulse(disc, e, 0.0, hold=30.0, release=30.0)
    pulse = probe.dipole_pulse(disc, e, force, hold=30.0, release=30.0)
    shortening = sham["loaded_separation"] - pulse["loaded_separation"]
    on_curve = np.interp(shortening, disc_curve["d0"] - disc_curve["separation"],
                         disc_curve["force"])
    unrecovered = (sham["final_separation"] - pulse["final_separation"]) / shortening
    print(f"held force {force}, shortening {shortening:.4f}, stepwise force there {on_curve:.4f}, "
          f"unrecovered {100 * unrecovered:.1f}%")
    assert shortening > 0
    assert on_curve == pytest.approx(force, rel=0.10)
    assert abs(unrecovered) < 0.08                      # elastic: the deformation recovers
    assert not pulse["lost"] and pulse["pairs_lost"] == [] and pulse["pairs_gained"] == []
    assert not sham["lost"] and sham["compliance"] != sham["compliance"]      # nan for zero force


def test_forced_exchange_uses_the_models_own_t1_operation(disc, disc_curve):
    """Contract, apply the model's T1 operation, let go: the outcome is classified."""
    e = disc_curve["edge"]
    pair = tuple(sorted(map(int, disc.e_f[e])))
    before = state.snapshot(disc)
    accepted = len(disc.log.accepted)
    result = probe.force_exchange(disc, e, release=10.0)
    kinds = [entry[1] for entry in disc.log.accepted[accepted:]]
    print(f"work {result['work']:.4f}, outcome {result['outcome']}, lost {result['pairs_lost']}, "
          f"gained {result['pairs_gained']}, transitions {sorted(set(kinds))}")
    assert state.differences(disc, before) == []
    assert result["work"] == pytest.approx(disc_curve["work"])        # same contraction
    assert result["outcome"] in {"exchange", "reversal", "space"}
    assert "T1" in kinds
    assert (result["outcome"] == "reversal") != (pair in result["pairs_lost"])
    if result["outcome"] == "exchange":
        assert result["pairs_gained"]


def test_force_control_does_not_trigger_the_schedules_t1(disc, disc_curve):
    """A strong held force closes the contact below the T1 length, yet the schedule keeps it.

    The schedule's T1 trigger needs the contact to shrink across a free step; a
    contact held shut from outside lengthens across each free step.  This is why
    ``force_exchange`` applies the T1 operation itself.
    """
    e = disc_curve["edge"]
    strong = probe.dipole_pulse(disc, e, 1.5 * disc_curve["peak_force"], hold=5.0, release=1.0)
    assert strong["min_separation"] < probe.T1_LENGTH and not strong["lost"]
    weak = probe.dipole_pulse(disc, e, 0.1, hold=5.0, release=1.0)
    assert weak["min_separation"] > 0.8 * weak["d0"] and not weak["lost"]


def test_invalid_targets_are_refused(disc):
    free = int(np.flatnonzero(disc.e_f[:, 0] == SPACE)[0])
    with pytest.raises(probe.ProbeInvalid):
        probe.contract_contact(disc, free)
    with pytest.raises(probe.ProbeInvalid):
        probe.dipole_pulse(disc, free, 0.1)
    before = state.snapshot(disc)
    e = longest_interior(disc)
    with pytest.raises(probe.ProbeInvalid, match="wall"):
        probe.shift_junction(disc, int(disc.e_v[e, 0]), np.array([10.0 * disc.radius, 0.0]))
    assert state.differences(disc, before) == []
    assert all(disc.e_f[c, 0] >= 0 and disc.e_f[c, 1] >= 0 for c in probe.contacts(disc))


def test_agrees_with_the_independent_force_probe_implementation(disc, disc_curve):
    """Compare complete curves at matched resolution and tight settling tolerances."""
    from circle import force_probe
    config = force_probe.ProbeConfig(increments=24, apply_t1=False)
    reference = probe.contract_contact(force_probe.isolated_copy(disc), disc_curve["edge"],
                                       levels=24, window=5., max_hold=300., tolerance=1e-4)
    other = force_probe.probe(disc, disc_curve["edge"], config)
    assert reference["all_converged"] and other["complete"]
    disc_curve = reference
    separation = np.array([row["separation"] for row in other["curve"]])
    force = np.array([row["force"] for row in other["curve"]])
    force = force - force[0]
    assert len(separation) == 25
    shortening = disc_curve["d0"] - separation
    here = np.interp(shortening, disc_curve["d0"] - disc_curve["separation"], disc_curve["force"])
    compare = shortening > 0.08 * disc_curve["d0"]         # past the first, noisiest levels
    worst = np.abs(here[compare] / force[compare] - 1).max()
    print(f"{int(compare.sum())} common levels, largest relative difference {worst:.4f}")
    assert worst < 0.03
