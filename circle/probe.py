"""Local mechanical probes: forcing one cell-cell contact.

Thomas & Hopyan (Biophys. J. 122, 4264, 2023) measure tissue rigidity through
the barrier to a neighbour exchange: one edge is shortened to a point in small
steps while the rest of the tissue relaxes, and the change of energy is the
barrier.  They compare a periodic sheet with a bounded flat disc and with
curved surfaces.

This module is the force-balance version of that measurement for the foam
model, which has no energy functional.  An external agent applies equal and
opposite forces to the two junctions of a contact, along the line joining them.
What is recorded is mechanical: the force the agent needs and the work it does.

``contract_contact``  separation controlled, connectivity frozen.  The agent
                      shortens the junction separation in steps down to the
                      model's own T1 length, and the tissue relaxes at each
                      step.  Gives the quasi-static stiffness, the
                      force-separation curve and the work to reach the T1
                      length (the barrier analogue).
``force_exchange``    the same contraction, then the model's own T1 operation
                      on the shortened contact, then the agent lets go and the
                      foam evolves freely.  Reports whether the contact change
                      persists, reverses, or leaves a space.
``dipole_pulse``      force controlled.  A constant inward force is held and
                      then released.  Gives the deformation and its recovery.
                      Meant for weak forces.  A strong force closes the contact
                      but does not make the transition schedule remove it: the
                      schedule's T1 trigger asks that the contact shrink across
                      a free step, and a contact held shut from outside
                      lengthens across each free step.  Use ``force_exchange``.

Both work on a ``DiscFoam`` and on a periodic ``FoamTissue``.  The probe is an
operator split: after each ordinary step the two junctions are displaced by
``M*dt*f`` (``M = psi``), so the external force enters the same overdamped
dynamics as every other force.  Nothing in the force law, the tension dynamics
or the transition rules is changed.
"""
import copy

import numpy as np

from . import state
from .disc import DiscFoam
from .measure import neighbor_pairs

T1_LENGTH = 0.01 * 2 * np.sqrt(np.pi)       # the model's short-edge threshold


class ProbeInvalid(ValueError):
    """The probe cannot be applied to this contact."""


# --------------------------------------------------------------------------- #
#  state handling: a probe never leaves a trace in the foam it was given
# --------------------------------------------------------------------------- #
class Saved:
    def __init__(self, foam):
        self.disc = isinstance(foam, DiscFoam)
        if self.disc:
            self.snap = state.snapshot(foam)
            self.iteration = foam.log.iteration
        else:
            self.snap = foam._snapshot()
            self.rng = copy.deepcopy(foam.rng.bit_generator.state)
            self.extra = {name: copy.deepcopy(getattr(foam, name)) for name in
                          ("_prev_lens", "_adjacent_revision") if hasattr(foam, name)}


def save(foam):
    return Saved(foam)


def load(foam, saved):
    """Return ``foam`` to a saved state; ``saved`` stays valid."""
    if saved.disc:
        state.restore(foam, saved.snap)
        foam.log.iteration = saved.iteration
        return
    foam._restore(saved.snap)
    saved.snap = foam._snapshot()               # ``_restore`` hands over the arrays
    foam.rng.bit_generator.state = copy.deepcopy(saved.rng)
    for name in ("_prev_lens", "_adjacent_revision"):
        if name in saved.extra:
            setattr(foam, name, copy.deepcopy(saved.extra[name]))
        elif hasattr(foam, name):
            delattr(foam, name)


# --------------------------------------------------------------------------- #
#  geometry helpers
# --------------------------------------------------------------------------- #
def contacts(foam, min_length=3 * T1_LENGTH):
    """Cell-cell films that join two distinct junctions and are not already short."""
    lengths = foam.edge_lengths()
    return [int(e) for e, (a, b) in enumerate(foam.e_f)
            if a >= 0 and b >= 0 and a != b and foam.e_v[e, 0] != foam.e_v[e, 1]
            and lengths[e] >= min_length]


def chord(foam, e):
    """Vector from the first to the second junction of film ``e`` and its length."""
    line = np.asarray(foam.e_mid[e])
    vector = line[-1] - line[0]
    return vector, float(np.hypot(*vector))


def film_between(foam, pair):
    """Longest film between two cells, or None if they are not in contact."""
    a, b = pair
    faces = foam.e_f
    hits = np.flatnonzero(((faces[:, 0] == a) & (faces[:, 1] == b))
                          | ((faces[:, 0] == b) & (faces[:, 1] == a)))
    hits = [int(e) for e in hits if foam.e_v[e, 0] != foam.e_v[e, 1]]
    if not hits:
        return None
    lengths = foam.edge_lengths()
    return max(hits, key=lambda e: lengths[e])


def shift_junction(foam, v, delta):
    """Displace junction ``v`` and every film end attached to it by ``delta``."""
    target = foam.vpos[v] + delta
    if isinstance(foam, DiscFoam):
        if getattr(foam, "aspect", 1.0) != 1.0:
            raise ProbeInvalid("the probe supports circular walls only")
        if np.hypot(*(target - foam.center)) > foam.radius:
            raise ProbeInvalid("the probe would push a junction through the wall")
    foam.vpos[v] = target
    for e in foam.v_e[v]:
        if e < 0:
            continue
        line = np.array(foam.e_mid[e], dtype=float, order="C")
        if foam.e_v[e, 0] == v:
            line[0] += delta
        if foam.e_v[e, 1] == v:
            line[-1] += delta
        foam.e_mid[e] = line


def _squeeze(foam, e, amount):
    """Move the two junctions of film ``e`` toward each other by ``amount`` in total."""
    vector, length = chord(foam, e)
    if length < 1e-12:
        return
    unit = vector / length
    v0, v1 = int(foam.e_v[e, 0]), int(foam.e_v[e, 1])
    shift_junction(foam, v0, 0.5 * amount * unit)
    shift_junction(foam, v1, -0.5 * amount * unit)
    foam._wrap_vertices()
    foam._update_faces()


def zone(foam, e, depth=1.0):
    """'periodic', or for a disc 'wall' if either cell of the contact is in the wall layer."""
    if not isinstance(foam, DiscFoam):
        return "periodic"
    from .measure import wall_layer
    layer = wall_layer(foam, depth)
    return "wall" if layer[foam.e_f[e, 0]] or layer[foam.e_f[e, 1]] else "interior"


# --------------------------------------------------------------------------- #
#  separation-controlled contraction (barrier analogue)
# --------------------------------------------------------------------------- #
def _hold(foam, e, start, target, ramp_steps, window_steps, max_windows, tolerance):
    """Bring the separation from ``start`` to ``target`` and hold it until the force settles.

    The force on each junction is the correction that restores the separation
    after a free step, over ``2*M*dt``.  It is averaged over windows; the hold
    ends when two consecutive window means agree to ``tolerance`` (absolute plus
    the same fraction of the force), or after ``max_windows``.
    """
    mobility = foam.psi * foam.dt
    for k in range(1, ramp_steps + 1):
        foam.step(0.0)
        _squeeze(foam, e, chord(foam, e)[1] - (start + (target - start) * k / ramp_steps))
    means = []
    for _ in range(max_windows):
        total = 0.0
        for _ in range(window_steps):
            foam.step(0.0)
            correction = chord(foam, e)[1] - target
            _squeeze(foam, e, correction)
            total += 0.5 * correction / mobility
        means.append(total / window_steps)
        if len(means) >= 2 and abs(means[-1] - means[-2]) <= tolerance * (1.0 + abs(means[-1])):
            return means[-1], True, len(means)
    return means[-1], False, len(means)


def contract_contact(foam, e, levels=8, first=0.03, stop=T1_LENGTH, ramp=0.5, window=1.0,
                     max_hold=15.0, tolerance=2e-3, restore=True):
    """Shorten contact ``e`` in steps down to the T1 length, relaxing at each step.

    This is the protocol of Thomas & Hopyan with energy replaced by mechanical
    work.  Connectivity is frozen (plain ``step(0)``, no transitions).  The
    junction separation is first held at its initial value ``d0`` to measure the
    ``baseline_force`` of the prepared state (zero at a true equilibrium), then
    shortened by the fraction ``first`` and on through ``levels`` equal steps to
    ``stop``.  At each level the tissue relaxes with the separation held until
    the holding force settles (see ``_hold``; times in tau_R).

    Returns ``separation`` and ``force`` at each level (force relative to the
    baseline: the tissue's resistance to shortening), a ``converged`` flag per
    level, the quasi-static ``stiffness`` (force over shortening at the first
    level), ``peak_force`` and ``work``, the trapezoidal integral of force over
    shortening down to ``stop``.  ``preempted`` is set if another contact
    that started above 1.5 T1 lengths reached the T1 length first; ``work`` then
    covers the levels completed.
    """
    if foam.e_f[e, 0] < 0 or foam.e_f[e, 1] < 0:
        raise ProbeInvalid("not a cell-cell contact")
    if foam.e_v[e, 0] == foam.e_v[e, 1]:
        raise ProbeInvalid("closed film")
    saved = save(foam) if restore else None
    try:
        steps = lambda duration: max(1, int(round(duration / foam.dt)))
        settings = (steps(ramp), steps(window), max(2, int(round(max_hold / window))), tolerance)
        _, d0 = chord(foam, e)
        if d0 * (1 - first) <= stop:
            raise ProbeInvalid("contact already near the T1 length")
        # Contacts that start clear of the T1 length; one of them reaching it first
        # preempts the probe.  Contacts already that short are part of the prepared state.
        others = np.array([c for c in contacts(foam, min_length=1.5 * T1_LENGTH) if c != e],
                          dtype=int)
        baseline, settled, windows = _hold(foam, e, d0, d0, 1, *settings[1:])
        targets = np.concatenate([[d0 * (1 - first)],
                                  np.linspace(d0 * (1 - first), stop, levels + 1)[1:]])
        separation, force, converged, hold_time = [d0], [0.0], [settled], [windows * window]
        preempted, current = False, d0
        for target in targets:
            value, settled, windows = _hold(foam, e, current, target, *settings)
            current = target
            separation.append(target)
            force.append(value - baseline)
            converged.append(settled)
            hold_time.append(windows * window)
            if len(others) and foam.edge_lengths()[others].min() < T1_LENGTH:
                preempted = True
                break
        separation, force = np.array(separation), np.array(force)
        shortening = d0 - separation
        return dict(edge=int(e), d0=d0, final_separation=float(current), dt=foam.dt,
                    tension=float(foam.e_t[e]), baseline_force=float(baseline),
                    separation=separation, force=force, converged=np.array(converged),
                    hold_time=np.array(hold_time), duration=float(np.sum(hold_time)),
                    stiffness=float(force[1] / shortening[1]),
                    peak_force=float(force.max()),
                    work=float(np.sum(0.5 * (force[1:] + force[:-1]) * np.diff(shortening))),
                    collapsed=not preempted, preempted=preempted,
                    all_converged=bool(np.all(converged)))
    finally:
        if restore:
            load(foam, saved)


# --------------------------------------------------------------------------- #
#  forced neighbour exchange: contraction, the model's own T1, release
# --------------------------------------------------------------------------- #
def force_exchange(foam, e, release=10.0, persistence=0.2, start=0, restore=True,
                   **contraction):
    """Contract contact ``e`` to the T1 length, apply the model's T1 operation, then let go.

    After ``contract_contact`` the model's own guarded T1 operation is applied
    to the shortened contact, followed by the removal of any space-space film it
    leaves, exactly as the transition schedule would do.  The operation is
    called directly because the schedule's trigger also asks that the contact be
    shrinking across a free step, which a contact held by an external agent is
    not.  The agent then lets go and the foam evolves freely with the full
    schedule for ``release``.

    ``outcome`` is one of
      'exchange'     the pair stays apart and a new pair is in contact;
      'reversal'     the original contact has come back;
      'space'        the pair stays apart and no new contact has formed;
      'T1 declined'  the model's T1 operation refused the contact;
      'preempted'    another contact reached the T1 length first.
    Contact changes are counted only if they have lasted ``persistence``.
    The contraction results (``work``, ``stiffness`` ...) are returned as well.
    """
    pair = (int(foam.e_f[e, 0]), int(foam.e_f[e, 1]))
    saved = save(foam) if restore else None
    try:
        initial = neighbor_pairs(foam)
        result = contract_contact(foam, e, restore=False, **contraction)
        result.update(pair=pair, pairs_lost=[], pairs_gained=[])
        if result["preempted"]:
            return dict(result, outcome="preempted")
        if not foam.t1_foam(int(e)):
            return dict(result, outcome="T1 declined")
        foam.resolve_space_space_edges()
        foam._update_faces()
        iteration = max(0, foam.log.iteration) if isinstance(foam, DiscFoam) else int(start)
        changed_since, previous = {}, neighbor_pairs(foam)
        steps = int(round(release / foam.dt))
        for k in range(1, steps + 1):
            iteration += 1
            foam.advance_scheduled(iteration, 0.0)
            if k % 20 == 0 or k == steps:
                current = neighbor_pairs(foam)
                for changed in current ^ previous:
                    changed_since[changed] = k * foam.dt
                previous = current
        final = neighbor_pairs(foam)
        settled = lambda p: release - changed_since.get(p, 0.0) >= persistence
        key = tuple(sorted(pair))
        result["pairs_lost"] = sorted(p for p in initial - final if settled(p))
        result["pairs_gained"] = sorted(p for p in final - initial if settled(p))
        if key in final:
            outcome = "reversal"
        elif result["pairs_gained"]:
            outcome = "exchange"
        else:
            outcome = "space"
        return dict(result, outcome=outcome)
    finally:
        if restore:
            load(foam, saved)


# --------------------------------------------------------------------------- #
#  force-controlled pulse (weak forces: deformation and recovery)
# --------------------------------------------------------------------------- #
def dipole_pulse(foam, e, force, hold=10.0, release=10.0, persistence=0.2, start=0,
                 record_every=20, transitions=True, restore=True):
    """Hold a constant inward force on the junctions of contact ``e``, then release.

    With ``transitions=True`` the full transition schedule runs throughout.
    The force follows the contact between the same two cells and stops if that
    contact disappears.  The schedule will not normally remove a contact that
    is being held shut (see the module docstring); ``min_separation`` reports
    how far it closed.
    With ``transitions=False`` connectivity is frozen, as in
    ``contract_contact``.  ``force=0`` is the sham run that measures the drift of
    the prepared state itself.

    Returns the equilibrated shortening under load and the ``compliance``
    (shortening over force), the ``residual`` shortening after release, whether
    the contact was ``lost`` and whether it was ``restored``, and the cell pairs
    ``pairs_lost`` / ``pairs_gained`` between the start and the end of the run.
    A change only counts if it has lasted at least ``persistence`` at the end.
    """
    pair = (int(foam.e_f[e, 0]), int(foam.e_f[e, 1]))
    if min(pair) < 0:
        raise ProbeInvalid("not a cell-cell contact")
    saved = save(foam) if restore else None
    try:
        mobility = foam.psi * foam.dt
        _, d0 = chord(foam, e)
        initial = neighbor_pairs(foam)
        iteration = max(0, foam.log.iteration) if isinstance(foam, DiscFoam) else int(start)
        n_hold, n_release = int(round(hold / foam.dt)), int(round(release / foam.dt))
        times, separation = [0.0], [d0]
        lost_at = None
        changed_since = {}                     # pair -> time its current status began
        previous = set(initial)
        for k in range(1, n_hold + n_release + 1):
            iteration += 1
            if transitions:
                foam.advance_scheduled(iteration, 0.0)
            else:
                foam.step(0.0)
            film = film_between(foam, pair)
            if film is None and lost_at is None:
                lost_at = k * foam.dt
            if film is not None and k <= n_hold and lost_at is None:
                _squeeze(foam, film, 2.0 * mobility * force)
            if k % record_every == 0 or k == n_hold:
                current = neighbor_pairs(foam)
                for changed in current ^ previous:
                    changed_since[changed] = k * foam.dt
                previous = current
                times.append(k * foam.dt)
                separation.append(chord(foam, film)[1] if film is not None else np.nan)
        end = (n_hold + n_release) * foam.dt
        final = neighbor_pairs(foam)
        settled = lambda p: end - changed_since.get(p, 0.0) >= persistence
        times, separation = np.array(times), np.array(separation)
        loaded = separation[(times > 0.8 * hold) & (times <= hold)]
        loaded = float(np.nanmean(loaded)) if np.any(np.isfinite(loaded)) else np.nan
        last = separation[-1]
        return dict(edge=int(e), pair=pair, force=float(force), hold=hold, release=release,
                    d0=d0, loaded_separation=loaded,
                    compliance=(d0 - loaded) / force if force else np.nan,
                    final_separation=float(last), residual=float((d0 - last) / d0),
                    min_separation=float(np.nanmin(separation)),
                    lost=lost_at is not None, lost_at=lost_at,
                    restored=film_between(foam, pair) is not None,
                    pairs_lost=sorted(p for p in initial - final if settled(p)),
                    pairs_gained=sorted(p for p in final - initial if settled(p)),
                    times=times, separation=separation)
    finally:
        if restore:
            load(foam, saved)
