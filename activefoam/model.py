"""Dynamic active-foam vertex model (confluent regime, periodic box).

Faithful re-implementation of the confluent limit of the dynamic vertex model of

    S. Kim, M. Pochitaloff, G. A. Stooke-Vaughan & O. Campas,
    "Embryonic tissues as active foams", Nature Physics 17, 859-866 (2021),

matching the reference MATLAB code (``ts_*.m``).  The straight-edge tension and
pressure forces implemented here are algebraically identical to the code's
per-edge forces (see the note in ``topology.py``).

Parameter dictionary (paper symbol -> code symbol used here)
------------------------------------------------------------
    W/T0        w        relative cell-cell adhesion
    Delta T/T0  dT       magnitude of tension fluctuations
    P0 L0 / T0  P0       relative normal (pressure) force        (paper: 10)
    tau_T/tau_R tauT     tension persistence time                (paper: 10)
    dt          dt       Euler-Maruyama step (units of tau_R)    (paper: 0.005)
    rho         rho      density (=1 in the confluent runs)

Fixed-point tension at a cell-cell contact:  T0_ij = 2 - W/T0   (beta = 0).
Tension enters the force through a Heaviside:  T_eff = max(0, T_ij).
Cell pressure (osmotic):  Delta p_i = P0 * (A0/A_i - 1),  A0 = rho.

The force computation is fully vectorised.  Flat (ragged) arrays describing the
topology are cached and rebuilt only when a T1 transition changes connectivity.
"""

from __future__ import annotations

import numpy as np

from .topology import Tissue, build_periodic_voronoi, min_image, wrap


class ActiveFoam:
    """One realisation of the confluent dynamic vertex model."""

    def __init__(
        self,
        n_side: int = 6,
        w: float = 1.0,
        dT: float = 0.5,
        rho: float = 1.0,
        P0: float = 10.0,
        tauT: float = 10.0,
        dt: float = 0.005,
        l_t1: float = 0.05,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.w = w
        self.dT = dT
        self.rho = rho
        self.A0 = rho
        self.P0 = P0
        self.tauT = tauT
        self.dt = dt
        self.l_t1 = l_t1
        self.T0_int = 2.0 - w
        self.shear = 0.0

        self.tissue = build_periodic_voronoi(n_side, self.rng)
        self.tension: dict[tuple[int, int], float] = {
            e: self.T0_int for e in self.tissue.edges()
        }
        self.time = 0.0
        self._build_arrays()

    # ------------------------------------------------------------------ #
    # cached flat arrays (rebuilt on topology change)
    # ------------------------------------------------------------------ #
    def _build_arrays(self):
        T = self.tissue
        # --- interior edges + aligned tension array -----------------------
        edges = T.edges()
        self.E = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), np.int64)
        self.Tvals = np.array([self.tension[e] for e in edges], dtype=float)
        self._edge_index = {e: i for i, e in enumerate(edges)}

        # --- flat ragged cell arrays for pressure forces ------------------
        fv, fcell, fnext, fprev, fref = [], [], [], [], []
        for ci, loop in enumerate(T.cells):
            n = len(loop)
            base = len(fv)
            ref = loop[0]
            for k in range(n):
                fv.append(loop[k])
                fcell.append(ci)
                fnext.append(base + (k + 1) % n)
                fprev.append(base + (k - 1) % n)
                fref.append(ref)
        self.fv = np.array(fv, np.int64)
        self.fcell = np.array(fcell, np.int64)
        self.fnext = np.array(fnext, np.int64)
        self.fprev = np.array(fprev, np.int64)
        self.fref = np.array(fref, np.int64)
        self.n_cells = T.n_cells

    def _rebuild_after_t1(self):
        self.tissue._rebuild_edges()
        self._resync_tensions()
        self._build_arrays()

    def _resync_tensions(self):
        new = {e: self.tension.get(e, self.T0_int) for e in self.tissue.edges()}
        self.tension = new

    # ------------------------------------------------------------------ #
    # vectorised geometry
    # ------------------------------------------------------------------ #
    def cell_areas(self, pos: np.ndarray | None = None) -> np.ndarray:
        """Signed cell areas (CCW loops -> positive), fully vectorised.

        Uses displacements relative to each cell's first vertex (min-image), which
        is exact as long as a cell's diameter is < L/2 -- always true here.
        """
        T = self.tissue
        pos = T.pos if pos is None else pos
        L = T.L
        rel = min_image(pos[self.fv] - pos[self.fref], L)
        rel_n = rel[self.fnext]
        cross = rel[:, 0] * rel_n[:, 1] - rel_n[:, 0] * rel[:, 1]
        return 0.5 * np.bincount(self.fcell, weights=cross, minlength=self.n_cells)

    # ------------------------------------------------------------------ #
    # vectorised forces
    # ------------------------------------------------------------------ #
    def compute_forces(self, pos: np.ndarray | None = None) -> np.ndarray:
        T = self.tissue
        pos = T.pos if pos is None else pos
        L = T.L
        F = np.zeros((T.n_vertices, 2))

        # ---- pressure / area forces:  F += P_i * dA_i/dR_alpha -----------
        area = self.cell_areas(pos)
        p_cell = self.P0 * (self.A0 / area - 1.0)
        dn = min_image(pos[self.fv[self.fnext]] - pos[self.fv], L)   # next - self
        dp = min_image(pos[self.fv[self.fprev]] - pos[self.fv], L)   # prev - self
        grad = 0.5 * np.stack([dn[:, 1] - dp[:, 1], dp[:, 0] - dn[:, 0]], axis=1)
        fp = p_cell[self.fcell][:, None] * grad
        np.add.at(F, self.fv, fp)

        # ---- junctional tension forces (Heaviside) -----------------------
        if len(self.E):
            a, b = self.E[:, 0], self.E[:, 1]
            teff = np.maximum(self.Tvals, 0.0)
            dr = min_image(pos[b] - pos[a], L)
            Ln = np.hypot(dr[:, 0], dr[:, 1])
            good = Ln > 1e-12
            f = np.zeros_like(dr)
            f[good] = (teff[good] / Ln[good])[:, None] * dr[good]
            np.add.at(F, a, f)
            np.add.at(F, b, -f)
        return F

    # ------------------------------------------------------------------ #
    # one Euler-Maruyama step
    # ------------------------------------------------------------------ #
    def step(self):
        T = self.tissue
        F = self.compute_forces()
        T.pos = self._wrap_shear(T.pos + self.dt * F)
        # Ornstein-Uhlenbeck tension update
        if len(self.Tvals):
            xi = self.rng.standard_normal(len(self.Tvals))
            self.Tvals = (
                self.Tvals
                - self.dt / self.tauT * (self.Tvals - self.T0_int)
                + self.dT / self.tauT * np.sqrt(self.dt) * xi
            )
        self.time += self.dt

    def _sync_tension_dict(self):
        for e, i in self._edge_index.items():
            self.tension[e] = float(self.Tvals[i])

    # ------------------------------------------------------------------ #
    # periodic wrap with Lees-Edwards shear
    # ------------------------------------------------------------------ #
    def _wrap_shear(self, pos: np.ndarray) -> np.ndarray:
        L = self.tissue.L
        pos = pos.copy()
        over = pos[:, 1] >= L
        under = pos[:, 1] < 0
        pos[over, 0] -= self.shear * L
        pos[over, 1] -= L
        pos[under, 0] += self.shear * L
        pos[under, 1] += L
        pos[:, 0] = np.mod(pos[:, 0], L)
        return pos

    # ------------------------------------------------------------------ #
    # T1 transitions
    # ------------------------------------------------------------------ #
    def do_t1_transitions(self) -> int:
        T = self.tissue
        self._sync_tension_dict()
        n_flips = 0
        if len(self.E) == 0:
            return 0
        a, b = self.E[:, 0], self.E[:, 1]
        dr = min_image(T.pos[b] - T.pos[a], T.L)
        Ln = np.hypot(dr[:, 0], dr[:, 1])
        order = np.argsort(Ln)
        touched: set[int] = set()
        did = False
        for idx in order:
            if Ln[idx] >= self.l_t1:
                break
            va, vb = int(a[idx]), int(b[idx])
            if va in touched or vb in touched:
                continue
            if self._flip_edge(va, vb):
                n_flips += 1
                touched.update((va, vb))
                T._rebuild_edges()
                did = True
        if did:
            self._rebuild_after_t1()
        return n_flips

    def _find_cell_with_directed_edge(self, u: int, w: int):
        for ci in self.tissue.vertex_cells.get(u, ()):
            loop = self.tissue.cells[ci]
            n = len(loop)
            for k in range(n):
                if loop[k] == u and loop[(k + 1) % n] == w:
                    return ci, k
        return None, None

    def _flip_edge(self, v1: int, v2: int) -> bool:
        T = self.tissue
        cL, kL = self._find_cell_with_directed_edge(v1, v2)
        cR, kR = self._find_cell_with_directed_edge(v2, v1)
        if cL is None or cR is None or cL == cR:
            return False
        loopL, loopR = T.cells[cL], T.cells[cR]
        nL, nR = len(loopL), len(loopR)
        if nL <= 3 or nR <= 3:
            return False
        a = loopL[(kL - 1) % nL]
        b = loopL[(kL + 2) % nL]
        c = loopR[(kR - 1) % nR]
        d = loopR[(kR + 2) % nR]
        c1_set = T.vertex_cells[v1] - {cL, cR}
        c2_set = T.vertex_cells[v2] - {cL, cR}
        if len(c1_set) != 1 or len(c2_set) != 1:
            return False
        c1 = next(iter(c1_set))
        c2 = next(iter(c2_set))
        if len({cL, cR, c1, c2}) != 4:
            return False
        if c2 in _cell_neighbors(T, c1):
            return False

        p1 = T.pos[v1]
        p2 = p1 + min_image(T.pos[v2] - p1, T.L)
        mid = 0.5 * (p1 + p2)
        e = p2 - p1
        perp = np.array([-e[1], e[0]])
        nperp = np.hypot(*perp)
        perp = perp / nperp if nperp > 1e-12 else np.array([1.0, 0.0])
        cLc = _cell_centroid(T, cL)
        cRc = _cell_centroid(T, cR)
        if np.dot(min_image(cRc - cLc, T.L), perp) < 0:
            perp = -perp
        half = 0.5 * self.l_t1 * 1.6
        T.pos[v1] = wrap(mid + half * perp, T.L)
        T.pos[v2] = wrap(mid - half * perp, T.L)

        T.cells[cL] = [v for v in loopL if v != v1]
        T.cells[cR] = [v for v in loopR if v != v2]
        T.cells[c1] = _insert_between(T.cells[c1], v1, a, v2)
        T.cells[c2] = _insert_between(T.cells[c2], v2, c, v1)
        return True

    # ------------------------------------------------------------------ #
    # observables
    # ------------------------------------------------------------------ #
    def shear_stress(self, pos: np.ndarray | None = None) -> float:
        """sigma_xy = rho/AT * sum_edges t_ij l_x l_y / |l|   (paper Eq. 5)."""
        T = self.tissue
        pos = T.pos if pos is None else pos
        if len(self.E) == 0:
            return 0.0
        a, b = self.E[:, 0], self.E[:, 1]
        teff = np.maximum(self.Tvals, 0.0)
        dr = min_image(pos[b] - pos[a], T.L)
        Ln = np.hypot(dr[:, 0], dr[:, 1])
        good = Ln > 1e-12
        s = np.sum(teff[good] * dr[good, 0] * dr[good, 1] / Ln[good])
        return self.rho * s / (T.L * T.L)

    def cell_centroids(self) -> np.ndarray:
        T = self.tissue
        L = T.L
        rel = min_image(T.pos[self.fv] - T.pos[self.fref], L)
        cx = np.bincount(self.fcell, weights=rel[:, 0], minlength=self.n_cells)
        cy = np.bincount(self.fcell, weights=rel[:, 1], minlength=self.n_cells)
        cnt = np.bincount(self.fcell, minlength=self.n_cells)
        ref_pos = np.zeros((self.n_cells, 2))
        # reference position per cell = pos of its first vertex
        first_flat = np.zeros(self.n_cells, np.int64)
        seen = np.zeros(self.n_cells, bool)
        for i, ci in enumerate(self.fcell):
            if not seen[ci]:
                first_flat[ci] = i
                seen[ci] = True
        ref_pos = T.pos[self.fref[first_flat]]
        cen = ref_pos + np.stack([cx / cnt, cy / cnt], axis=1)
        return cen

    def shape_factors(self) -> np.ndarray:
        return self.tissue.shape_factors()


# --------------------------------------------------------------------------- #
# combinatorial helpers
# --------------------------------------------------------------------------- #
def _insert_between(loop: list[int], p: int, q: int, new: int) -> list[int]:
    n = len(loop)
    ip = loop.index(p)
    if loop[(ip + 1) % n] == q:
        return loop[:ip + 1] + [new] + loop[ip + 1:]
    if loop[(ip - 1) % n] == q:
        return loop[:ip] + [new] + loop[ip:]
    raise ValueError("p and q are not adjacent in loop")


def _cell_neighbors(T: Tissue, ci: int) -> set[int]:
    out = set()
    loop = T.cells[ci]
    n = len(loop)
    for k in range(n):
        e = (loop[k], loop[(k + 1) % n])
        key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
        for cc in T.edge_cells.get(key, []):
            if cc != ci:
                out.add(cc)
    return out


def _cell_centroid(T: Tissue, ci: int) -> np.ndarray:
    return T.cell_polygon(ci).mean(axis=0)
