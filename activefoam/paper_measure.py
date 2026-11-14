"""Observables for the original paper, using the accelerated curved-edge model."""
import numpy as np

from .foam_full import SPACE, crd_local, edge_len
from .foam_measure import dynamics, equilibrated_foam, evolve, initialize, quality, validate

SPACE_AREA_CUTOFF = 1e-4


def snapshot(foam):
    """Store unwrapped curved polygons in a pickle-free checkpoint."""
    polygons = [foam._cell_polyline(c) for c in range(foam.nFa)]
    return dict(points=np.concatenate(polygons), offsets=np.r_[0, np.cumsum(list(map(len, polygons)))],
                box=foam.bs, strain=foam.sstn, cell_shape=foam.shape_factors())


def exterior_spaces(foam, min_area_fraction=SPACE_AREA_CUTOFF):
    """Count connected extracellular regions on the periodic box.

    The reversed free films trace the boundary with space to the left.
    Positive contractible contours enclose a space. Opposite-winding
    noncontractible boundaries enclose bands. With contractible contours
    only, area closure identifies any space that spans the torus.
    """
    space_area = max(0., foam.bs**2 - foam.f_area.sum())
    free = np.flatnonzero(foam.e_f[:, 0] == SPACE)
    if not len(free):
        return 0, 0.
    if np.any(foam.e_f[free, 1] == SPACE):
        return np.nan, np.nan
    directed = {}
    for cell_edges in foam.f_e:
        for signed in cell_edges:
            edge = signed if signed >= 0 else -signed - 1
            if foam.e_f[edge, 0] != SPACE:
                continue
            reverse = -signed - 1
            start, end = foam.e_v[edge][::(-1 if reverse < 0 else 1)]
            if start in directed:
                return np.nan, np.nan
            directed[start] = (end, reverse)
    if len(directed) != len(free) or {end for end, _ in directed.values()} != set(directed):
        return np.nan, np.nan

    areas, winding = [], []
    while directed:
        start = current = next(iter(directed))
        pieces = []
        while current in directed:
            current, signed = directed.pop(current)
            points = foam._signed_edge_pts(signed)
            pieces.append(points[:-1])
        contour = crd_local(np.vstack([*pieces, points[-1:]]), foam.bs, foam.sstn)
        delta = contour[-1] - contour[0]
        periods = np.rint((delta - [foam.sstn * delta[1], 0.]) / foam.bs).astype(int)
        if current != start or not np.allclose(delta, foam.bs * (periods + [foam.sstn * periods[1], 0])):
            return np.nan, np.nan
        if np.any(periods):
            winding.append(periods)
        else:
            x, y = (contour - contour[0]).T
            areas.append(.5 * np.sum(x[:-1] * y[1:] - y[:-1] * x[1:]))

    cutoff = min_area_fraction * foam.A0
    seed_areas = [area for area in areas if 0. < area <= cutoff]
    count = sum(area > cutoff for area in areas)
    if winding:
        # Disjoint essential simple loops must all have the same primitive
        # winding up to orientation, in alternating cell/space pairs.
        basis = winding[0]
        if (len(winding) % 2 or np.any(np.sum(winding, axis=0)) or
                any(not (np.array_equal(p, basis) or np.array_equal(p, -basis)) for p in winding)):
            return np.nan, np.nan
        count += len(winding) // 2
    else:
        background = (space_area - sum(areas)) / foam.bs**2
        if abs(background - round(background)) > 1e-7 or round(background) not in (0, 1):
            return np.nan, np.nan
        count += round(background)
    return count, max(0., space_area - sum(seed_areas)) / count if count else 0.


def static_paper(w, rho, seed, n_side, geometry_policy, record_space=False, record_snapshot=False,
                 extra_relaxation_tauT=0., pressure_normals='reference'):
    foam = equilibrated_foam(w, rho, seed, n_side, geometry_policy=geometry_policy,
                             pressure_normals=pressure_normals)
    if extra_relaxation_tauT:
        evolve(foam, extra_relaxation_tauT * foam.tauT, 0., start=round(4 * foam.tauT / foam.dt))
    result = dict(phi=foam.volume_fraction(), z=foam.neighbor_number().mean(),
                  shape_mean=foam.shape_factors().mean(), **quality(foam))
    if record_space:
        count, area = exterior_spaces(foam)
        result.update(space_count=count, space_density=count / (2 * foam.nFa),
                      space_mean_area=area / foam.A0, space_area_cutoff=SPACE_AREA_CUTOFF)
    if record_snapshot:
        result.update(snapshot(foam))
        result['cell_z'] = foam.neighbor_number()
    return result


class ContactEvents:
    """Log changed cell pairs after each accepted transition."""

    def __init__(self, foam, start=0, example=None):
        self.previous = self.pairs(foam)
        self.start = start
        self.rows = []
        self.example = example
        self.frames = {}
        if example is not None:
            self.before_transition = self._prepare_example

    @staticmethod
    def pairs(foam):
        return {(a, b) for a, b in foam.e_f[foam.e_f[:, 0] >= 0].tolist() if a != b}

    def __call__(self, step, transition, foam):
        current = self.pairs(foam)
        lost, gained = self.previous - current, current - self.previous
        self.previous = current
        swap = transition == "T1" and bool(lost and gained)
        captured = (self.example == "swap" and swap
                    or self.example == "gain" and transition == "T4" and gained)
        if captured:
            self.frames.update({f'event_before_{key}': val for key, val in self.before.items()})
            self.frames.update({f'event_after_{key}': val for key, val in snapshot(foam).items()})
            self.frames['event_lost'] = np.asarray(sorted(lost), dtype=int).reshape(-1, 2)
            self.frames['event_gained'] = np.asarray(sorted(gained), dtype=int).reshape(-1, 2)
            self.example = None
            del self.before, self.before_transition
        for direction, pairs in ((-1, lost), (1, gained)):
            self.rows.extend(((step - self.start) * foam.dt, a, b, direction, int(swap))
                             for a, b in sorted(pairs))

    def array(self):
        return np.asarray(self.rows, dtype=float).reshape(-1, 5)

    def _prepare_example(self, transition, foam):
        if (transition, self.example) in (('T1', 'swap'), ('T4', 'gain')):
            self.before = snapshot(foam)


def sampled_neighbor_changes(events, start, stop, dt=.005, stride=10):
    """Reconstruct changes between MATLAB frames or a regular subsample."""
    start_step, stop_step = np.rint(np.array([start, stop]) / dt).astype(np.int64)
    first = (start_step - 3 + stride - 1) // stride
    last = (stop_step - 3) // stride
    step = np.rint(events[:, 0] / dt).astype(np.int64)
    frame = (step - 3 + stride - 1) // stride
    take = (frame > first) & (frame <= last)
    keys = np.column_stack((frame[take], events[take, 1:3].astype(np.int64)))
    unique, index = np.unique(keys, axis=0, return_inverse=True)
    net = np.bincount(index, weights=events[take, 3])
    changed = net != 0
    times = (stride * unique[changed, 0] + 3) * dt
    return times, 2 * np.abs(net[changed]), max(0, last - first) * (stride * dt)


class DynamicMeasurements:
    def __init__(self, record_snapshot=False, record_junction=False, example=None):
        self.record_snapshot = record_snapshot
        self.record_junction = record_junction
        self.example = example
        self.stress = []
        self.tensions = []
        self.junction = []

    def begin(self, foam, start):
        self.contacts = ContactEvents(foam, start, self.example)
        if self.record_snapshot:
            self.initial = snapshot(foam)
        if self.record_junction:
            contacts = np.flatnonzero(foam.e_f[:, 0] >= 0)
            if contacts.size:
                lengths = foam.edge_lengths()[contacts]
                self.pair = foam.e_f[contacts[np.argsort(lengths)[len(contacts)//2]]].copy()
            else:
                self.pair = None

    def sample(self, t, foam):
        self.stress.append((t, foam.shear_stress() / np.sqrt(foam.rho)))
        tension = np.maximum(foam.e_t, 0.)
        self.tensions.append((tension.sum(), tension @ tension, (tension == 0.).sum(), tension.size))
        if self.record_junction:
            edges = [] if self.pair is None else np.flatnonzero(np.all(foam.e_f == self.pair, axis=1))
            lengths = np.array([edge_len(foam.e_mid[e]).sum() for e in edges])
            length = lengths.sum()
            tension = np.average(np.maximum(foam.e_t[edges], 0.), weights=lengths) if length else np.nan
            self.junction.append((t, length / foam.lsc if length else np.nan, tension))

    def finish(self, foam):
        stress = np.asarray(self.stress)
        tail = stress[stress[:, 0] > stress[-1, 0] - 1000., 1]
        tension_sum, tension_squares, zero_count, film_count = np.sum(self.tensions, axis=0)
        tension_mean = tension_sum / film_count
        events = self.contacts.array()
        result = dict(active_stress=stress, sigma_A=np.std(tail), contacts=events,
                      ne_rate=2 * len(events) / (foam.nFa * stress[-1, 0]),
                      film_tension=tension_mean,
                      film_tension_sd=np.sqrt(max(0., tension_squares / film_count - tension_mean**2)),
                      zero_tension_fraction=zero_count / film_count)
        if self.record_snapshot:
            result.update(snapshot(foam))
            result.update({f"initial_{key}": val for key, val in self.initial.items()})
        if self.record_junction:
            result.update(junction=np.asarray(self.junction),
                          junction_pair=self.pair if self.pair is not None else np.array([-1, -1]))
        result.update(self.contacts.frames)
        return result


def dynamics_paper(record_snapshot=False, record_junction=False, example=None, **parameters):
    return dynamics(**parameters, measurements=DynamicMeasurements(record_snapshot, record_junction, example))


class ShearRunError(ValueError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


def shear_paper(w, rho, mu, seed, n_side, geometry_policy, duration, pilot=False,
                record_snapshot=False, pressure_normals='reference', record_diagnostics=False):
    # ts_simStepZero prepares a passive sample with a noisy anneal and five
    # additional tension times of passive quench; ts_simStep uses active preparation.
    anneal_mu = mu if mu else .25 * (2 - w)
    anneal = (1. if not mu else 4.004) if not pilot else .25
    foam = initialize(w, rho, seed, n_side, anneal_mu=anneal_mu, anneal_tauT=anneal,
                      geometry_policy=geometry_policy, pressure_normals=pressure_normals)
    if not mu:
        foam.e_t = np.array([foam.fixed_tension(e) for e in range(len(foam.e_t))])
        evolve(foam, 50.04 if not pilot else 1., 0., start=round(anneal * foam.tauT / foam.dt))
    baseline = 20. if not pilot else 2.
    scale = np.sqrt(rho)
    time, stress = [-baseline], [foam.shear_stress() / scale]
    events = ContactEvents(foam, start=round(baseline / foam.dt))

    def pre(t, f):
        if t < baseline:
            time.append(t - baseline)
            stress.append(f.shear_stress() / scale)

    steps = evolve(foam, baseline, mu, sample_every=10, observer=pre, contact_observer=events)
    before = foam.shear_stress() / scale
    data = {}
    if record_snapshot:
        data.update({f"initial_{key}": val for key, val in snapshot(foam).items()})
    foam.apply_affine_shear(1.5)
    jump = foam.shear_stress() / scale - before
    if record_snapshot:
        data.update({f"affine_{key}": val for key, val in snapshot(foam).items()})

    geometry = []

    def observe(t, f):
        time.append(t)
        stress.append(f.shear_stress() / scale)
        if record_diagnostics and round(t / f.dt) % 100 == 0:
            q = quality(f)
            geometry.append((t, q['quality_checks'], q['resolved_crossing_checks'],
                             q['microfold_checks'], q['max_crossing_depth']))

    observe(0., foam)
    try:
        evolve(foam, duration, mu, start=steps, sample_every=10, observer=observe,
               post_shear=bool(mu), contact_observer=events)
    except ValueError as exc:
        if record_diagnostics:
            raise ShearRunError(str(exc), dict(time=np.asarray(time), stress=np.asarray(stress),
                                                geometry_trace=np.asarray(geometry))) from exc
        raise
    time, stress = np.asarray(time), np.asarray(stress)
    if record_diagnostics:
        data['geometry_trace'] = np.asarray(geometry)
    if record_snapshot:
        data.update(snapshot(foam))
    return dict(data, time=time, stress=stress, stress_jump=jump,
                yield_stress=stress[time >= duration - min(20., duration / 10)].mean(),
                contacts=events.array(), **quality(foam))


def hysteresis_paper(rho, seed, n_side, geometry_policy, direction, pilot=False,
                     pressure_normals='reference'):
    ws = np.linspace(0., 1., 51 if not pilot else 6)
    if direction == 'down':
        ws = ws[::-1]
    foam = equilibrated_foam(ws[0], rho, seed, n_side, geometry_policy=geometry_policy,
                             pressure_normals=pressure_normals)
    foam._measure_vertex_force = True
    iteration = round(4 * foam.tauT / foam.dt)
    max_steps = round((10 * foam.tauT + 8 * foam.dt) / foam.dt) if not pilot else 200
    phis, forces, steps = [foam.volume_fraction()], [np.nan], [0]
    for w in ws[1:]:
        foam.w = foam.gam = w
        for count in range(1, max_steps + 1):
            iteration += 1
            foam.advance_scheduled(iteration, 0.)
            if count % 100 == 0:
                validate(foam)
            if foam._max_vertex_force <= 1e-3:
                break
        phis.append(validate(foam))
        forces.append(foam._max_vertex_force)
        steps.append(count)
    return dict(w=ws, phi=phis, max_force=forces, steps=steps, **quality(foam))
