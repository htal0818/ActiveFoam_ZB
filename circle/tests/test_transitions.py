"""Step 3: transitions near the wall, rejection and complete rollback."""
import numpy as np
import pytest

from activefoam.foam_full import FoamTissue, SPACE
from circle import state
from circle.disc import DiscFoam

KINDS = {            # log name -> (planar method, argument, value returned on rejection)
    "T1": ("t1_foam", False),
    "T2r": ("t2_reverse", None),
    "T3r": ("_merge_space_edge", None),
    "T4a": ("t4_adjacent", False),
    "T4": ("t4_transition", False),
}


def call(foam, kind, ident=0):
    return getattr(foam, KINDS[kind][0])(ident)


def push_junction_outside(foam, vertex=0, factor=1.2):
    """Move one junction, and the film ends attached to it, beyond the wall."""
    target = foam.center + factor * foam.radius * np.array([1.0, 0.0])
    foam.vpos[vertex] = target
    for e in foam.v_e[vertex]:
        if e < 0:
            continue
        line = np.array(foam.e_mid[e])
        if foam.e_v[e, 0] == vertex:
            line[0] = target
        if foam.e_v[e, 1] == vertex:
            line[-1] = target
        foam.e_mid[e] = line


def scramble(foam):
    """Touch every kind of state a transition may modify, then leave the disc."""
    push_junction_outside(foam)
    foam.e_t[0] += 3.0
    foam.f_v[0] = foam.f_v[0][::-1]
    foam.f_area[0] *= 0.5
    foam._prev_lens = None
    foam._adjacent_revision = getattr(foam, "_adjacent_revision", 0) + 1
    foam._t4_boxes = np.zeros((len(foam.e_v), 4))
    foam._add_vertex(foam.vpos[0].copy())           # T2-reverse reads this as acceptance


@pytest.fixture
def relaxed():
    foam = DiscFoam.random(16, seed=1, w=0.6)
    for k in range(1, 301):
        foam.advance_scheduled(k, 0.5)
    return foam


def test_every_instance_attribute_is_classified(relaxed):
    assert state.unclassified(relaxed) == []
    assert set(state.PHYSICAL) <= set(vars(relaxed))


def test_snapshot_detects_and_restores_each_physical_field(relaxed):
    snap = state.snapshot(relaxed)
    assert state.differences(relaxed, snap) == []
    mutations = {
        "vpos": lambda f: f.vpos.__setitem__((0, 0), f.vpos[0, 0] + 1e-9),
        "v_e": lambda f: f.v_e.__setitem__((0, 0), f.v_e[0, 0] + 1),
        "v_f": lambda f: f.v_f.__setitem__((0, 0), f.v_f[0, 0] + 1),
        "e_v": lambda f: f.e_v.__setitem__((0, 0), f.e_v[0, 0] + 1),
        "e_f": lambda f: f.e_f.__setitem__((0, 1), f.e_f[0, 1] + 1),
        "e_t": lambda f: f.e_t.__setitem__(0, f.e_t[0] + 1e-9),
        "e_rfn": lambda f: f.e_rfn.__setitem__(0, f.e_rfn[0] + 1),
        "e_mid": lambda f: f.e_mid[0].__setitem__((0, 0), f.e_mid[0][0, 0] + 1e-9),
        "f_v": lambda f: f.f_v[0].append(99),
        "f_e": lambda f: f.f_e[0].append(99),
        "f_area": lambda f: f.f_area.__setitem__(0, f.f_area[0] + 1e-9),
        "f_center": lambda f: f.f_center.__setitem__((0, 0), f.f_center[0, 0] + 1e-9),
        "rng": lambda f: f.rng.random(),
        "_prev_lens": lambda f: setattr(f, "_prev_lens", np.ones(3)),
        "_adjacent_revision": lambda f: setattr(f, "_adjacent_revision", 17),
        "_t4_boxes": lambda f: setattr(f, "_t4_boxes", np.ones((2, 4))),
    }
    for name, mutate in mutations.items():
        mutate(relaxed)
        assert state.differences(relaxed, snap) == [name]
        state.restore(relaxed, snap)
        assert state.differences(relaxed, snap) == []
    # The snapshot survives being restored from and stays independent of the foam.
    relaxed.step(0.5)
    assert state.differences(relaxed, snap) != []
    state.restore(relaxed, snap)
    assert state.differences(relaxed, snap) == []


@pytest.mark.parametrize("kind", list(KINDS))
def test_output_outside_the_wall_is_rejected_and_rolled_back(relaxed, kind, monkeypatch):
    method, rejected_value = KINDS[kind]
    accepted_value = {"T2r": None, "T3r": {}}.get(kind, True)

    def invalid(self, ident):
        scramble(self)
        return accepted_value

    monkeypatch.setattr(FoamTissue, method, invalid)
    before = state.snapshot(relaxed)
    accepted_before = list(relaxed.log.accepted)
    result = call(relaxed, kind)
    assert result is rejected_value or result == rejected_value
    assert state.differences(relaxed, before) == []
    # The attempt is logged outside the restored state; nothing was projected.
    iteration, logged, ident, reason = relaxed.log.rejected[-1]
    assert logged == kind and "outside the wall" in reason
    assert relaxed.log.accepted == accepted_before
    assert relaxed.max_radius() <= relaxed.radius * (1 + 1e-12)
    # Caches are consistent with the restored state.
    mids, loops = relaxed._fast_geometry_inputs()
    assert [list(loop) for loop in loops] == [list(loop) for loop in relaxed.f_e]
    monkeypatch.undo()
    relaxed.advance_scheduled(301, 0.5)


@pytest.mark.parametrize("kind", list(KINDS))
def test_exception_inside_a_transition_is_rolled_back(relaxed, kind, monkeypatch):
    def broken(self, ident):
        scramble(self)
        raise ValueError("half-finished transition")

    monkeypatch.setattr(FoamTissue, KINDS[kind][0], broken)
    before = state.snapshot(relaxed)
    call(relaxed, kind)
    assert state.differences(relaxed, before) == []
    assert "exception: ValueError" in relaxed.log.rejected[-1][3]


@pytest.mark.parametrize("kind", list(KINDS))
def test_declined_attempt_leaves_no_trace_in_the_state(relaxed, kind, monkeypatch):
    """Even if the planar code declines after touching state, the disc restores it."""
    def declines(self, ident):
        self.e_t[0] += 1.0
        self._prev_lens = None
        return KINDS[kind][1]

    monkeypatch.setattr(FoamTissue, KINDS[kind][0], declines)
    before = state.snapshot(relaxed)
    rejected = len(relaxed.log.rejected)
    call(relaxed, kind)
    assert state.differences(relaxed, before) == []
    assert len(relaxed.log.rejected) == rejected
    assert relaxed.log.declined[kind] >= 1


def test_an_attempt_that_draws_random_numbers_is_an_error(relaxed, monkeypatch):
    def draws(self, ident):
        self.rng.random()
        return True

    monkeypatch.setattr(FoamTissue, "t1_foam", draws)
    before = state.snapshot(relaxed)
    with pytest.raises(RuntimeError, match="random draws"):
        relaxed.t1_foam(0)
    assert state.differences(relaxed, before) == []      # generator state restored too


@pytest.mark.parametrize("kind", list(KINDS))
def test_nonfinite_film_node_is_rejected(relaxed, kind, monkeypatch):
    """A NaN in an intermediate film node, with junctions and areas left finite."""
    accepted_value = {"T2r": None, "T3r": {}}.get(kind, True)

    def poisons_a_film(self, ident):
        e = int(np.argmax([len(m) for m in self.e_mid]))
        line = np.array(self.e_mid[e])
        assert len(line) > 2
        line[1, 0] = np.nan
        self.e_mid[e] = line
        self._add_vertex(self.vpos[0].copy())           # T2-reverse reads this as acceptance
        return accepted_value

    monkeypatch.setattr(FoamTissue, KINDS[kind][0], poisons_a_film)
    before = state.snapshot(relaxed)
    call(relaxed, kind)
    assert state.differences(relaxed, before) == []
    assert "nonfinite" in relaxed.log.rejected[-1][3]
    assert all(np.all(np.isfinite(m)) for m in relaxed.e_mid)


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("error", [RuntimeError, TypeError, ZeroDivisionError, KeyboardInterrupt])
def test_unexpected_exception_is_restored_then_reraised(relaxed, kind, error, monkeypatch):
    def broken(self, ident):
        scramble(self)
        raise error("not a planar rejection")

    monkeypatch.setattr(FoamTissue, KINDS[kind][0], broken)
    before = state.snapshot(relaxed)
    rejected = len(relaxed.log.rejected)
    with pytest.raises(error):
        call(relaxed, kind)
    assert state.differences(relaxed, before) == []
    assert len(relaxed.log.rejected) == rejected          # not disguised as a rejection


def test_failure_inside_acceptance_validation_is_restored(relaxed, monkeypatch):
    """The validity check itself runs under the same protection as the operation."""
    def mutates(self, ident):
        self.e_t[0] += 1.0
        return True

    def failing_check(self):
        raise RuntimeError("validation failed to run")

    monkeypatch.setattr(FoamTissue, "t1_foam", mutates)
    monkeypatch.setattr(DiscFoam, "_invalid_reason", failing_check)
    before = state.snapshot(relaxed)
    with pytest.raises(RuntimeError, match="validation failed"):
        relaxed.t1_foam(0)
    assert state.differences(relaxed, before) == []


def test_real_space_creation_pushed_outside_is_undone(monkeypatch):
    foam = DiscFoam.random(16, seed=1, w=0.6)
    vertex = int(np.flatnonzero(np.all(foam.v_f != SPACE, axis=1))[0])
    real = FoamTissue.t2_reverse

    def real_then_outside(self, ident):
        real(self, ident)
        push_junction_outside(self, ident)

    before = state.snapshot(foam)
    monkeypatch.setattr(FoamTissue, "t2_reverse", real_then_outside)
    foam.t2_reverse(vertex)
    assert state.differences(foam, before) == []
    monkeypatch.undo()
    foam.t2_reverse(vertex)                               # the unmodified event is accepted
    assert foam.log.accepted[-1][1:] == ("T2r", vertex)
    assert len(foam.vpos) == len(before.physical["vpos"]) + 2
    assert foam.max_radius() <= foam.radius * (1 + 1e-12)


def test_real_neighbor_exchange_pushed_outside_is_undone(monkeypatch):
    foam = DiscFoam.random(36, seed=2, w=1.0)
    real = FoamTissue.t1_foam
    moved = {}

    def real_then_outside(self, e):
        done = real(self, e)
        if done:
            push_junction_outside(self, int(self.e_v[e, 0]))
            moved["yes"] = True
        return done

    monkeypatch.setattr(FoamTissue, "t1_foam", real_then_outside)
    contacts = np.flatnonzero(np.all(foam.e_f != SPACE, axis=1))
    for e in contacts:
        before = state.snapshot(foam)
        rejected = len(foam.log.rejected)
        assert foam.t1_foam(int(e)) is False
        assert state.differences(foam, before) == []
        if len(foam.log.rejected) > rejected:
            break
    assert moved and foam.log.rejected[-1][1] == "T1"


def test_wall_adjacent_transitions_of_every_kind_succeed():
    """Representative accepted events within one cell length of the wall."""
    foam = DiscFoam.random(36, seed=2, w=0.2, audit=True)
    limit = foam.radius * (1 + 1e-12)
    for k in range(1, 6001):
        foam.advance_scheduled(k, 1.5)
        if k % 50 == 0:
            assert foam.max_radius() <= limit
    near_wall = {}
    for (iteration, kind, ident), site in zip(foam.log.accepted, foam.log.sites):
        if site > 1 - 1.0 / foam.radius:
            near_wall[kind] = near_wall.get(kind, 0) + 1
    assert set(near_wall) == set(KINDS)
    assert all(count >= 3 for count in near_wall.values())
    # Out-of-wall rejections need not occur naturally; exceptions must not.
    assert not any("exception" in entry[3] for entry in foam.log.rejected)
    assert np.all(np.isfinite(foam.f_area)) and np.all(foam.f_area > 0)
    assert np.all(np.sum(foam.v_e >= 0, axis=1) >= 1)
    assert foam.audit_calls > 0
    assert state.unclassified(foam) == []


def test_same_seed_gives_identical_runs():
    runs = []
    for _ in range(2):
        foam = DiscFoam.random(16, seed=3, w=0.3)
        for k in range(1, 1501):
            foam.advance_scheduled(k, 1.5)
        runs.append(foam)
    a, b = runs
    assert a.log.accepted == b.log.accepted and len(a.log.accepted) > 20
    assert a.log.declined == b.log.declined
    assert np.array_equal(a.vpos, b.vpos) and np.array_equal(a.e_t, b.e_t)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
