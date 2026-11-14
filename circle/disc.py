"""Active foam inside a rigid, frictionless, nonadhesive circular wall.

``DiscFoam`` is ``activefoam.foam_full.FoamTissue`` with

* a disc-specific integration path (forces, trial step, wall projection,
  wall records, then remeshing);
* Euclidean geometry: the periodic unwrap and wrap hooks are identities;
* every topological transition guarded by a complete snapshot, with rollback
  if the result leaves the disc or is invalid.

Pressure forces use boundary-orientation normals and current-step areas.
The planar code is not modified.

Units follow the planar code: a node moves by ``psi*dt*f`` per step, so the
mobility is ``M = psi = sqrt(rho)`` and nodal forces ``f`` are built from
``T_eff = max(T, 0)`` along each film segment and
``p_n = P0*(rho/A - 1)/sqrt(rho)`` times half the segment length along its
normal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from activefoam.foam_full import FoamTissue, SPACE, crd_local, crd_domain
from activefoam.topology import Tissue
from activefoam._kernels import boundary_normal_signs

from . import state
from .kernels import advance_disc, films_finite, max_radius
from .tiling import clipped_voronoi


class AuditError(AssertionError):
    """A geometric operation depended on the auxiliary periodic box."""


@dataclass
class WallRecord:
    """Wall reactions of one step, on the node identities before remeshing.

    ``junction_*`` arrays have one row per junction; ``node_*`` arrays have one
    row per intermediate film node that touched the wall.  Reactions are nodal
    forces in the code's force units.  ``flag`` is 1 for a node on the wall at
    the start of the step and 2 for one that arrived during it.
    """
    junction_pos: np.ndarray
    junction_reaction: np.ndarray
    junction_direct: np.ndarray
    junction_flag: np.ndarray
    junction_tributary: np.ndarray
    node_edge: np.ndarray
    node_index: np.ndarray
    node_pos: np.ndarray
    node_reaction: np.ndarray
    node_direct: np.ndarray
    node_flag: np.ndarray
    node_tributary: np.ndarray
    excursion: float
    residual: float
    resampled: int
    center: np.ndarray
    radius: float


@dataclass
class TransitionLog:
    """Accepted and rejected transition attempts.  Lives outside restored state."""
    iteration: int = -1
    accepted: list = field(default_factory=list)      # (iteration, kind, id)
    sites: list = field(default_factory=list)         # r/R_w of each accepted event
    rejected: list = field(default_factory=list)      # (iteration, kind, id, reason)
    declined: dict = field(default_factory=dict)      # kind -> count (planar code said no)

    def counts(self, which="accepted"):
        out = {}
        for entry in getattr(self, which):
            out[entry[1]] = out.get(entry[1], 0) + 1
        return out


class _ClosedCells:
    """Stand-in for a ``Tissue``: detached cells, each one closed film (fixtures)."""

    def __init__(self, box, polylines):
        self.L = box
        self.polylines = polylines


@dataclass
class _ClosedTiling:
    radius: float
    center: np.ndarray
    polylines: list
    arcs: dict = field(default_factory=dict)


def circle_polyline(center, radius, spacing):
    """Counter-clockwise ring of nodes on a circle (open: last node != first)."""
    count = max(8, int(np.ceil(2.0 * np.pi * radius / spacing)))
    angles = 2.0 * np.pi * np.arange(count) / count
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])


class DiscFoam(FoamTissue):
    """Curved-edge foam confined to a disc of fixed radius."""

    REJECT_TOL = 1e-9          # relative to the wall radius

    def __init__(self, tiling, w=0.0, rho=1.0, P0=10.0, tauT=10.0, dt=0.005, edpc=0.15,
                 seed=0, box=None, audit=False, remesh=True, wall_resample=False):
        self.radius = float(tiling.radius)
        self.center = np.array(tiling.center, dtype=float)
        self.wall_tol = 1e-9 * self.radius
        self.delta_max = 0.5 * self.radius
        self.audit = bool(audit)
        self.remesh = bool(remesh)
        self.wall_resample = bool(wall_resample)
        self.audit_calls = 0
        self.wall = None
        self.log = TransitionLog()
        box = 8.0 * self.radius if box is None else float(box)
        reach = self.radius + self.delta_max
        if np.any(self.center - reach < 0.0) or np.any(self.center + reach > box) or 2 * reach >= box / 2:
            raise ValueError("disc does not fit the auxiliary box with the required margin")
        self._arcs = tiling.arcs
        base = (_ClosedCells(box, tiling.polylines) if isinstance(tiling, _ClosedTiling)
                else Tissue(box, tiling.pos, tiling.cells))
        try:
            super().__init__(base, w=w, rho=rho, P0=P0,
                             tauT=tauT, dt=dt, edpc=edpc, beta=0.0, seed=seed,
                             accelerated=True, reference_pressure_lag=False,
                             pressure_normals="boundary")
        finally:
            del self._arcs

    @classmethod
    def random(cls, n_cells, seed=0, radius=None, box=None, edpc=0.15, **kwargs):
        """Disc of ``n_cells`` cells from a clipped Poisson-Voronoi tiling.

        The disc sits at fixed absolute coordinates ``(4R, 4R)`` whatever the box,
        and the tiling uses generator ``100 + seed`` as the planar protocol does.
        """
        radius = float(np.sqrt(n_cells / np.pi) if radius is None else radius)
        tiling = clipped_voronoi(n_cells, np.random.default_rng(100 + seed), radius=radius,
                                 center=(4.0 * radius, 4.0 * radius), spacing=edpc)
        return cls(tiling, seed=seed, box=box, edpc=edpc, **kwargs)

    @classmethod
    def from_closed_cells(cls, radius, polylines, center=None, **kwargs):
        """Detached cells in a disc, each given as a counter-clockwise ring of nodes.

        Every cell is one closed film with a single junction (its first node).
        Used for single-cell and few-cell validation cases.
        """
        center = (4.0 * radius, 4.0 * radius) if center is None else center
        rings = [np.array(p, dtype=float, order="C") for p in polylines]
        return cls(_ClosedTiling(float(radius), np.array(center, dtype=float), rings), **kwargs)

    # ------------------------------------------------------------------ #
    #  construction: outer films follow the wall from the first evaluation
    # ------------------------------------------------------------------ #
    def _build_closed(self, polylines):
        n = len(polylines)
        self.nFa = n
        self.vpos = np.array([p[0] for p in polylines], dtype=float)
        self.v_e = np.full((n, 3), -1, np.int64)
        self.v_f = np.full((n, 3), -1, np.int64)
        self.v_e[:, 2] = np.arange(n)
        self.v_f[:, 2] = np.arange(n)
        self.e_v = np.column_stack([np.arange(n), np.arange(n)]).astype(np.int64)
        self.e_f = np.column_stack([np.full(n, SPACE), np.arange(n)]).astype(np.int64)
        self.e_mid = [np.vstack([p, p[:1]]) for p in polylines]
        self.e_rfn = np.array([len(m) - 2 for m in self.e_mid], np.int64)
        self.f_v = [[i] for i in range(n)]
        self.f_e = [[i] for i in range(n)]
        self.e_t = np.array([self.fixed_tension(e) for e in range(n)])
        self._update_faces()
        if np.any(self.f_area <= 0.0):
            raise ValueError("closed cells must be given counter-clockwise")

    def _build_from(self, T):
        if isinstance(T, _ClosedCells):
            return self._build_closed(T.polylines)
        super()._build_from(T)
        for e in range(len(self.e_v)):
            if self.e_f[e, 0] != SPACE:
                continue
            a, b = int(self.e_v[e, 0]), int(self.e_v[e, 1])
            line = self._arcs[(a, b)] if (a, b) in self._arcs else self._arcs[(b, a)][::-1]
            self.e_mid[e] = np.array(line, dtype=float, order="C")
            self.e_rfn[e] = len(line) - 2
        self._update_faces()

    # ------------------------------------------------------------------ #
    #  Euclidean geometry
    # ------------------------------------------------------------------ #
    def _local(self, points):
        out = np.array(points, dtype=float, order="C")
        if self.audit:
            self.audit_calls += 1
            if not np.array_equal(crd_local(out, self.bs, self.sstn), out):
                raise AuditError("periodic unwrap shifted a point inside the disc")
        return out

    def _wrap_vertices(self):
        if self.audit:
            self.audit_calls += 1
            if not np.array_equal(crd_domain(self.vpos, self.bs, self.sstn), self.vpos):
                raise AuditError("periodic wrap moved a junction inside the disc")

    def apply_affine_shear(self, strain):
        raise NotImplementedError("a frictionless circular wall cannot impose a shear step")

    # ------------------------------------------------------------------ #
    #  integration with the wall constraint
    # ------------------------------------------------------------------ #
    def step(self, mu=0.0):
        """One Euler-Maruyama step with projection onto the wall."""
        mids, loops = self._fast_geometry_inputs()
        outward = boundary_normal_signs(self.e_f, loops)
        before = self.vpos
        (vertices, new, counts, excursion, residual, resampled, v_react, v_direct, v_flag,
         v_trib,
         n_edge, n_index, n_pos, n_react, n_direct, n_flag, n_trib) = advance_disc(
            self.vpos, self.e_v, self.e_f, self.e_t, mids, self.f_area, self.psi, self.dt,
            self.P0, self.lsc, self.edpc, outward, self.center[0], self.center[1],
            self.radius, self.wall_tol, self.remesh, self.wall_resample)
        if not excursion <= self.delta_max:
            raise ValueError(f"trial position {excursion:.3g} beyond the wall; "
                             "the Euclidean guarantee of the auxiliary box no longer holds")
        if getattr(self, "_measure_vertex_force", False):
            self._max_vertex_force = (np.linalg.norm(vertices - before - self.psi * self.dt * v_react,
                                                     axis=1).max() / (self.psi * self.dt))
        self.vpos, self.e_mid, self.e_rfn = vertices, new, counts
        self.wall = WallRecord(vertices.copy(), v_react, v_direct, v_flag, v_trib,
                               n_edge, n_index, n_pos, n_react, n_direct, n_flag, n_trib,
                               float(excursion), float(residual), int(resampled),
                               self.center.copy(),
                               self.radius)
        count = np.sum(self.e_f != SPACE, axis=1).astype(float)
        fixed = count - self.gam * np.all(self.e_f != SPACE, axis=1)
        xi = self.rng.standard_normal(len(count))
        self.e_t = (self.e_t - self.dt / self.tauT * (self.e_t - fixed)
                    + mu / self.tauT * np.sqrt(self.dt) / 2 * count * xi)
        self._wrap_vertices()
        self._update_faces()

    def advance_scheduled(self, iteration, mu=0.0, post_shear=False, contact_observer=None):
        if post_shear:
            raise NotImplementedError("no shear protocol in the disc")
        self.log.iteration = int(iteration)
        return super().advance_scheduled(iteration, mu, contact_observer=contact_observer)

    # ------------------------------------------------------------------ #
    #  guarded transitions
    # ------------------------------------------------------------------ #
    def max_radius(self):
        mids, _ = self._fast_geometry_inputs()
        return float(max_radius(self.vpos, mids, self.center[0], self.center[1]))

    def _invalid_reason(self):
        """Why the current state may not be accepted after a transition, or None."""
        mids, _ = self._fast_geometry_inputs()
        if not films_finite(self.vpos, mids):
            return "nonfinite junction or film node"
        if not np.all(np.isfinite(self.f_area)) or np.any(self.f_area <= 0.0):
            return "nonpositive or nonfinite cell area"
        radius = float(max_radius(self.vpos, mids, self.center[0], self.center[1]))
        if not radius <= self.radius * (1.0 + self.REJECT_TOL):
            return f"node outside the wall by {radius - self.radius:.3g}"
        return None

    EXPECTED_FAILURES = (ValueError, StopIteration, IndexError, KeyError)

    def _guard(self, kind, ident, site, operation, accepted_of, rejected_value):
        """Run one transition attempt; leave the state untouched unless it is accepted.

        The planar operation, the acceptance test and the validity check all run
        under the same protection.  A failure of an expected type is a rejected
        attempt.  Anything else is restored first and then re-raised.
        """
        site = float(np.hypot(*(np.asarray(site, dtype=float) - self.center)) / self.radius)
        snap = state.snapshot(self)
        reason = None
        accepted = False
        try:
            result = operation()
            accepted = bool(accepted_of(result, snap))
            if accepted:
                reason = self._invalid_reason()
        except self.EXPECTED_FAILURES as exc:
            result, accepted = rejected_value, False
            reason = f"exception: {type(exc).__name__}: {exc}"
        except BaseException:
            state.restore(self, snap)
            raise
        if self.rng.bit_generator.state != snap.rng:
            state.restore(self, snap)
            raise RuntimeError(f"{kind} consumed random draws; attempts must not")
        if accepted and reason is None:
            self.log.accepted.append((self.log.iteration, kind, ident))
            self.log.sites.append(site)
            return result
        state.restore(self, snap)
        if reason is None:
            self.log.declined[kind] = self.log.declined.get(kind, 0) + 1
        else:
            self.log.rejected.append((self.log.iteration, kind, ident, reason))
        return rejected_value

    def t1_foam(self, e):
        return self._guard("T1", int(e), self.vpos[self.e_v[e]].mean(axis=0),
                           lambda: FoamTissue.t1_foam(self, e),
                           lambda result, snap: bool(result), False)

    def t2_reverse(self, t2Id):
        return self._guard("T2r", int(t2Id), self.vpos[t2Id],
                           lambda: FoamTissue.t2_reverse(self, t2Id),
                           lambda result, snap: len(self.vpos) > len(snap.physical["vpos"]),
                           None)

    def t4_adjacent(self, v):
        return self._guard("T4a", int(v), self.vpos[v],
                           lambda: FoamTissue.t4_adjacent(self, v),
                           lambda result, snap: bool(result), False)

    def t4_transition(self, t4Id):
        film = self.e_mid[t4Id]
        return self._guard("T4", int(t4Id), film[len(film) // 2],
                           lambda: FoamTissue.t4_transition(self, t4Id),
                           lambda result, snap: bool(result), False)

    def _merge_space_edge(self, deId):
        return self._guard("T3r", int(deId), self.vpos[self.e_v[deId]].mean(axis=0),
                           lambda: FoamTissue._merge_space_edge(self, deId),
                           lambda result, snap: result is not None, None)

    # ------------------------------------------------------------------ #
    #  measurements that used the box area
    # ------------------------------------------------------------------ #
    @property
    def disc_area(self):
        return np.pi * self.radius ** 2

    def volume_fraction(self):
        """Cell area over disc area.  A necessary check, not an overlap test."""
        return float(np.sum(np.clip(self.f_area, 0, None)) / self.disc_area)

    def stress_tensor(self):
        return FoamTissue.stress_tensor(self) * self.bs ** 2 / self.disc_area

    def shear_stress(self):
        return self.stress_tensor()[0, 1]
