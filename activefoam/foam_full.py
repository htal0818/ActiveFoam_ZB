"""Faithful vertex-network active-foam model (curved edges + extracellular spaces).

Direct port of the reference MATLAB structure (Kim et al., Nat. Phys. 2021):

  * physical vertices  -- triple junctions, each with 3 incident edges / 3 faces
  * intermediate vertices -- extra points along every contact, giving it curvature
    (paper Fig 1g, blue dots); an edge is a *polyline* ``e_mid`` of these points
  * faces               -- cells (index >= 0, with osmotic pressure) and the
    extracellular space, encoded as face id ``SPACE = -1`` (the code's face 0)

This module ports, as literally as practical, the geometry helpers
(``ts_crdLocal``, ``ts_edgeMidVrtxAverage``, ``ts_faceArea`` ...), the force
iteration (``ts_iteration`` / ``ts_edgeNormalForce`` / ``ts_edgeFixedTension``),
the space-creation transition ``ts_t2ReverseTransition`` and the T1 transition.

Everything is 0-indexed; ``-1`` denotes "no edge/face" or the extracellular space.
"""

from __future__ import annotations

import numpy as np

from .topology import min_image

SPACE = -1


# --------------------------------------------------------------------------- #
# periodic geometry helpers (ports of ts_crdLocal / ts_edgeLen)
# --------------------------------------------------------------------------- #
def crd_local(vr, bs, sstn=0.0):
    """Unwrap a polyline so consecutive points are minimum-image neighbours.

    Faithful port of ts_crdLocal.m (Lees-Edwards shear sstn along x wrt y).
    """
    vr = np.array(vr, float)
    out = vr.copy()
    for j in range(1, len(vr)):
        if out[j, 1] - out[j - 1, 1] > bs / 2:
            out[j, 1] -= bs
            out[j, 0] -= bs * sstn
        if out[j, 1] - out[j - 1, 1] < -bs / 2:
            out[j, 1] += bs
            out[j, 0] += bs * sstn
        if out[j, 0] - out[j - 1, 0] > bs / 2:
            out[j, 0] -= bs
        if out[j, 0] - out[j - 1, 0] < -bs / 2:
            out[j, 0] += bs
    return out


def edge_len(ecrd):
    d = np.diff(ecrd, axis=0)
    return np.hypot(d[:, 0], d[:, 1])


def edge_mid_vrtx_average(emd, edpc):
    """Re-sample a contact polyline to ~uniform spacing edpc between its ends.

    Faithful port of ts_edgeMidVrtxAverage.m: keeps the two endpoints, places
    ``rfn = floor(total_len/edpc)`` interior points evenly along arclength.
    Returns (new_polyline, rfn).
    """
    emd = np.asarray(emd, float)
    seg = np.hypot(np.diff(emd[:, 0]), np.diff(emd[:, 1]))
    total = seg.sum()
    rfn = int(np.floor(total / edpc))
    if rfn < 1:
        return np.array([emd[0], emd[-1]]), 0  # no interior points (1 segment)
    frac = seg / total
    cum = np.concatenate([[0.0], np.cumsum(frac)])       # 0..1 arclength at nodes
    targets = (1.0 / (rfn + 1)) * np.arange(1, rfn + 1)  # interior fractions
    new = np.empty((rfn, 2))
    for j, t in enumerate(targets):
        k = np.searchsorted(cum, t, side="right") - 1
        k = min(max(k, 0), len(seg) - 1)
        local = (t - cum[k]) / (cum[k + 1] - cum[k] + 1e-30)
        new[j] = emd[k] + local * (emd[k + 1] - emd[k])
    return np.vstack([emd[0], new, emd[-1]]), rfn


def poly_area_centroid(pts):
    """Shoelace area (signed) and centroid of an (unwrapped) polygon."""
    x, y = pts[:, 0], pts[:, 1]
    a = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return a, pts.mean(axis=0)


def edge_vector(evs, evf, fcn, bs, sstn=0.0):
    """Tangent + outward normal of a contact segment (port of ts_edgeVector)."""
    tv = evf - evs
    md = 0.5 * (evf + evs)
    eln = np.hypot(tv[0], tv[1])
    tv = tv / eln
    nv = np.array([-tv[1], tv[0]])
    fcmp = crd_local(np.array([fcn, md]), bs, sstn)
    fv = fcmp[1] - fcmp[0]
    if np.dot(nv, fv) < 0:
        nv = -nv
    return tv, nv, eln


class FoamTissue:
    """Vertex-network foam: curved contacts + extracellular spaces (face id -1)."""

    def __init__(self, base_tissue, w=0.0, rho=1.0, P0=10.0, tauT=10.0,
                 dt=0.005, edpc=0.15, beta=0.0, seed=0):
        self.bs = base_tissue.L
        self.sstn = 0.0
        self.rho = rho
        self.psi = np.sqrt(rho)
        self.lsc = self.psi                 # length scale L0 (= psi*sqrt(A0), A0=1)
        self.A0 = self.lsc ** 2             # preferred area
        self.P0 = P0
        self.w = w
        self.beta = beta
        self.gam = w                        # W/T0
        self.tauT = tauT
        self.dt = dt
        self.edpc = edpc
        self.rng = np.random.default_rng(seed)
        self._build_from(base_tissue)

    # ------------------------------------------------------------------ #
    def _build_from(self, T):
        self.vpos = T.pos.copy()
        Nv = len(self.vpos)
        cells = [list(c) for c in T.cells]
        self.nFa = len(cells)

        # unique undirected edges -> endpoints + the (up to 2) cells
        edge_map: dict[tuple[int, int], int] = {}
        e_v, e_f = [], []
        for ci, loop in enumerate(cells):
            n = len(loop)
            for k in range(n):
                a, b = loop[k], loop[(k + 1) % n]
                key = (a, b) if a < b else (b, a)
                if key not in edge_map:
                    edge_map[key] = len(e_v)
                    e_v.append([key[0], key[1]])
                    e_f.append([ci, SPACE])
                else:
                    e_f[edge_map[key]][1] = ci
        self.e_v = np.array(e_v, np.int64)
        Ne = len(e_v)
        # sort faces so SPACE(-1) sits in column 0, cell in column 1
        e_f = np.array(e_f, np.int64)
        for i in range(Ne):
            if e_f[i, 0] > e_f[i, 1] and e_f[i, 1] != SPACE:
                e_f[i] = e_f[i, ::-1]
            if e_f[i, 0] == SPACE:
                pass
            elif e_f[i, 1] == SPACE:
                e_f[i] = e_f[i, ::-1]
            elif e_f[i, 0] > e_f[i, 1]:
                e_f[i] = e_f[i, ::-1]
        self.e_f = e_f

        # incident edges/faces per vertex (trivalent)
        self.v_e = np.full((Nv, 3), -1, np.int64)
        self.v_f = np.full((Nv, 3), -1, np.int64)
        vc = np.zeros(Nv, int)
        for ei in range(Ne):
            for a in self.e_v[ei]:
                self.v_e[a, vc[a]] = ei
                vc[a] += 1
        for ci, loop in enumerate(cells):
            for v in loop:
                row = self.v_f[v]
                if ci not in row:
                    row[np.argmax(row == -1)] = ci

        # curved polyline per edge (initially straight, refined to edpc)
        self.e_mid = [None] * Ne
        self.e_rfn = np.zeros(Ne, np.int64)
        for ei in range(Ne):
            p = crd_local(self.vpos[self.e_v[ei]], self.bs, self.sstn)
            mid, rfn = edge_mid_vrtx_average(p, self.edpc)
            self.e_mid[ei] = mid
            self.e_rfn[ei] = rfn

        # signed edge loops per cell face
        self.f_e = []
        for ci, loop in enumerate(cells):
            n = len(loop)
            fe = []
            for k in range(n):
                a, b = loop[k], loop[(k + 1) % n]
                key = (a, b) if a < b else (b, a)
                ei = edge_map[key]
                fe.append(ei if (self.e_v[ei, 0] == a) else -(ei + 1))
            self.f_e.append(fe)
        # store cell vertex loops too
        self.f_v = [list(c) for c in cells]

        # tension fixed points
        self.e_t = np.array([self.fixed_tension(ei) for ei in range(Ne)])
        self._update_faces()

    # ------------------------------------------------------------------ #
    def _signed_edge_pts(self, se):
        """Vertices of a signed edge id (>=0 forward; encoded -(e+1) reverse)."""
        if se >= 0:
            return self.e_mid[se]
        e = -se - 1
        return self.e_mid[e][::-1]

    def _cell_polyline(self, ci):
        pieces = []
        for se in self.f_e[ci]:
            pts = self._signed_edge_pts(se)
            pieces.append(pts[:-1])            # drop last (shared with next)
        pts = np.vstack(pieces)
        return crd_local(pts, self.bs, self.sstn)

    def cell_area_center(self, ci):
        pts = self._cell_polyline(ci)
        a, c = poly_area_centroid(pts)
        return a, np.mod(c, self.bs)

    def _update_faces(self):
        self.f_area = np.zeros(self.nFa)
        self.f_center = np.zeros((self.nFa, 2))
        for ci in range(self.nFa):
            a, c = self.cell_area_center(ci)
            self.f_area[ci] = a
            self.f_center[ci] = c

    # ------------------------------------------------------------------ #
    def fixed_tension(self, ei):
        """Port of ts_edgeFixedTension: interior 2-W, free boundary 1 (beta=0)."""
        etn = 0.0
        f = self.e_f[ei]
        both = True
        for ff in f:
            if ff != SPACE:
                etn += 1.0 + self.beta * 0.0      # perimeter term (beta=0)
            else:
                both = False
        if f[0] != SPACE and f[1] != SPACE:
            etn -= self.gam
        return etn

    def normal_force(self, ei, eln):
        f = self.e_f[ei]
        eNf = 0.0
        for ii in (0, 1):
            if f[ii] != SPACE:
                A = self.f_area[f[ii]]
                sign = -1.0 if ii == 0 else 1.0
                eNf += self.P0 * (1.0 / (A / self.lsc ** 2) - 1.0) * eln / 2.0 / self.lsc * sign
        return eNf

    # ------------------------------------------------------------------ #
    def _wrap_vertices(self):
        bs, s = self.bs, self.sstn
        p = self.vpos
        over = p[:, 1] >= bs
        under = p[:, 1] < 0
        p[over, 1] -= bs; p[over, 0] -= bs * s
        p[under, 1] += bs; p[under, 0] += bs * s
        p[:, 0] = np.mod(p[:, 0] - s * p[:, 1], bs) + s * p[:, 1]

    def step(self, mu=0.0):
        """One Euler-Maruyama step (port of ts_iteration.m)."""
        Ne = len(self.e_mid)
        Nv = len(self.vpos)
        Fv = np.zeros((Nv, 2))
        new_mid = [None] * Ne
        for i in range(Ne):
            rfn = self.e_rfn[i]
            eMd = self.e_mid[i]
            fCn = self.f_center[self.e_f[i, 1]]
            eTf = max(0.0, self.e_t[i])
            vTn = np.zeros((rfn + 2, 2))
            for jj in range(rfn + 1):
                tv, nv, eln = edge_vector(eMd[jj], eMd[jj + 1], fCn, self.bs, self.sstn)
                eNf = self.normal_force(i, eln)
                vTn[jj] += eTf * tv
                vTn[jj + 1] -= eTf * tv
                vTn[jj] += eNf * nv
                vTn[jj + 1] += eNf * nv
            Fv[self.e_v[i, 0]] += vTn[0]
            Fv[self.e_v[i, 1]] += vTn[-1]
            new_mid[i] = eMd + self.psi * self.dt * vTn

        self.vpos = self.vpos + self.psi * self.dt * Fv
        # fix endpoints of each polyline to the (updated) physical vertices
        for i in range(Ne):
            eMd = new_mid[i]
            eMd[0] = self.vpos[self.e_v[i, 0]]
            eMd[-1] = self.vpos[self.e_v[i, 1]]
            self.e_mid[i] = crd_local(eMd, self.bs, self.sstn)

        # Ornstein-Uhlenbeck tension update
        etn = np.array([self.fixed_tension(i) for i in range(Ne)])
        etp = np.sum(self.e_f != SPACE, axis=1).astype(float)
        xi = self.rng.standard_normal(Ne)
        self.e_t = (self.e_t
                    - self.dt / self.tauT * (self.e_t - etn)
                    + mu / self.tauT * np.sqrt(self.dt) / 2.0 * etp * xi)

        # re-refine edges whose sampling drifted (port of ts_iteration tail)
        for i in range(Ne):
            emd = self.e_mid[i]
            eln = edge_len(emd)
            tot = eln.sum()
            if (tot < self.edpc * (self.e_rfn[i] - 1 / 3) or
                    tot > self.edpc * (self.e_rfn[i] + 4 / 3) or
                    (eln.max() / max(eln.min(), 1e-12) > 2)):
                self.e_mid[i], self.e_rfn[i] = edge_mid_vrtx_average(emd, self.edpc)

        self._wrap_vertices()
        self._update_faces()

    # ------------------------------------------------------------------ #
    def volume_fraction(self):
        return np.sum(np.clip(self.f_area, 0, None)) / (self.bs ** 2)

    def shape_factors(self):
        s = []
        for ci in range(self.nFa):
            A = self.f_area[ci]
            if A > 1e-9:
                s.append(self._perimeter(ci) / np.sqrt(A))
        return np.array(s)

    def _perimeter(self, ci):
        tot = 0.0
        for se in self.f_e[ci]:
            e = se if se >= 0 else -se - 1
            tot += edge_len(self.e_mid[e]).sum()
        return tot

    def neighbor_number(self):
        z = np.zeros(self.nFa)
        for ei in range(len(self.e_v)):
            f = self.e_f[ei]
            if f[0] != SPACE and f[1] != SPACE:
                z[f[0]] += 1
                z[f[1]] += 1
        return z

    # ================================================================== #
    #  topological transitions
    # ================================================================== #
    def _add_vertex(self, pos):
        self.vpos = np.vstack([self.vpos, pos])
        self.v_e = np.vstack([self.v_e, [-1, -1, -1]])
        self.v_f = np.vstack([self.v_f, [-1, -1, -1]])
        return len(self.vpos) - 1

    def _add_edge(self, v0, v1, f0, f1, mid, rfn, t):
        self.e_v = np.vstack([self.e_v, [v0, v1]])
        self.e_f = np.vstack([self.e_f, [f0, f1]])
        self.e_t = np.append(self.e_t, t)
        self.e_rfn = np.append(self.e_rfn, rfn)
        self.e_mid.append(mid)
        return len(self.e_v) - 1

    def _sortf(self, arr):
        return sorted(arr)          # SPACE=-1 sorts first, like the code's 0

    def _vrtx_sort_ccw(self, vlist):
        pts = crd_local(self.vpos[vlist], self.bs, self.sstn)
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        return [vlist[i] for i in np.argsort(ang)]

    def _face_vrtx_insert(self, vr1, vr2, vrin, fvrs):
        fvrs = list(fvrs)
        p = sorted([fvrs.index(vr1), fvrs.index(vr2)])
        if p[0] == 0 and p[1] == len(fvrs) - 1:
            return fvrs + [vrin]
        return fvrs[:p[0] + 1] + [vrin] + fvrs[p[1]:]

    def _face_edge_id(self, ci):
        """Signed edge loop for cell ci from its vertex loop (port ts_faceEdgeId)."""
        fvr = self.f_v[ci]
        n = len(fvr)
        fed = []
        for k in range(n):
            a, b = fvr[k], fvr[(k + 1) % n]
            cand = [e for e in range(len(self.e_v))
                    if {int(self.e_v[e, 0]), int(self.e_v[e, 1])} == {a, b}
                    and ci in (int(self.e_f[e, 0]), int(self.e_f[e, 1]))]
            e = cand[0]
            if int(self.e_v[e, 0]) == a and int(self.e_v[e, 1]) == b:
                fed.append(e)
            else:
                fed.append(-(e + 1))
        return fed

    def _recompute_cell(self, ci):
        self.f_e[ci] = self._face_edge_id(ci)
        a, c = self.cell_area_center(ci)
        if a < 0:                                   # keep CCW / positive area
            self.f_v[ci] = self.f_v[ci][::-1]
            self.f_e[ci] = self._face_edge_id(ci)
            a, c = self.cell_area_center(ci)
        self.f_area[ci] = a
        self.f_center[ci] = np.mod(c, self.bs)

    def t2_reverse(self, t2Id):
        """Open a small triangular extracellular space at vertex t2Id.

        Faithful port of ts_t2ReverseTransition.m (0-indexed; SPACE=-1).
        """
        # three neighbours of t2Id (CCW)
        inc = [e for e in self.v_e[t2Id] if e != -1]
        nbrs = []
        for e in inc:
            other = int(self.e_v[e, 0]) if int(self.e_v[e, 1]) == t2Id else int(self.e_v[e, 1])
            nbrs.append(other)
        vl = self._vrtx_sort_ccw(nbrs)                       # vl[0:3] neighbours
        vnew1 = self._add_vertex(self.vpos[t2Id].copy())
        vnew2 = self._add_vertex(self.vpos[t2Id].copy())
        vlId = vl + [vnew1, vnew2]                           # [n0,n1,n2,vnew1,vnew2]

        # map each incident edge to the slot of its neighbour
        elId = [-1, -1, -1, None, None, None]
        for e in inc:
            other = int(self.e_v[e, 0]) if int(self.e_v[e, 1]) == t2Id else int(self.e_v[e, 1])
            elId[vlId.index(other)] = e
        # three new triangle edges (placeholders, filled below)
        el3 = self._add_edge(t2Id, vnew1, SPACE, SPACE,
                             np.array([self.vpos[t2Id], self.vpos[vnew1]]), 0, 0.0)
        el4 = self._add_edge(vnew1, vnew2, SPACE, SPACE,
                             np.array([self.vpos[vnew1], self.vpos[vnew2]]), 0, 0.0)
        el5 = self._add_edge(t2Id, vnew2, SPACE, SPACE,
                             np.array([self.vpos[t2Id], self.vpos[vnew2]]), 0, 0.0)
        elId[3], elId[4], elId[5] = el3, el4, el5

        def faces_of(e):
            return {int(self.e_f[e, 0]), int(self.e_f[e, 1])}
        fl0 = ((faces_of(elId[0]) & faces_of(elId[1])) - {SPACE}).pop()
        fl1 = ((faces_of(elId[1]) & faces_of(elId[2])) - {SPACE}).pop()
        fl2 = ((faces_of(elId[2]) & faces_of(elId[0])) - {SPACE}).pop()
        flId = [fl0, fl1, fl2, SPACE]

        # vertex edge/face lists
        self.v_e[t2Id] = self._sortf([elId[0], elId[3], elId[5]])
        self.v_e[vnew1] = self._sortf([elId[1], elId[3], elId[4]])
        self.v_e[vnew2] = self._sortf([elId[2], elId[4], elId[5]])
        self.v_f[t2Id] = self._sortf([flId[0], flId[2], flId[3]])
        self.v_f[vnew1] = self._sortf([flId[0], flId[1], flId[3]])
        self.v_f[vnew2] = self._sortf([flId[1], flId[2], flId[3]])

        # positions: tiny triangle, 1% toward each neighbour
        vloc = crd_local(np.vstack([self.vpos[t2Id], self.vpos[vl[0]],
                                    self.vpos[vl[1]], self.vpos[vl[2]]]),
                         self.bs, self.sstn)
        self.vpos[t2Id] = vloc[0] + 0.01 * (vloc[1] - vloc[0])
        self.vpos[vnew1] = vloc[0] + 0.01 * (vloc[2] - vloc[0])
        self.vpos[vnew2] = vloc[0] + 0.01 * (vloc[3] - vloc[0])

        # edge endpoints
        self.e_v[elId[1]] = self._sortf([vlId[1], vnew1])
        self.e_v[elId[2]] = self._sortf([vlId[2], vnew2])
        self.e_v[elId[3]] = self._sortf([t2Id, vnew1])
        self.e_v[elId[4]] = self._sortf([vnew1, vnew2])
        self.e_v[elId[5]] = self._sortf([t2Id, vnew2])
        self.e_f[elId[3]] = self._sortf([flId[0], SPACE])
        self.e_f[elId[4]] = self._sortf([flId[1], SPACE])
        self.e_f[elId[5]] = self._sortf([flId[2], SPACE])

        # rebuild polylines: original edges keep neighbour end, move t2-end to new vertex
        vvId = [t2Id, vnew1, vnew2]
        for ii in range(3):
            e = elId[ii]
            endpts = crd_local(self.vpos[self.e_v[e]], self.bs, self.sstn)
            mid, rfn = edge_mid_vrtx_average(endpts, self.edpc)
            self.e_mid[e] = mid
            self.e_rfn[e] = rfn
        for ii in range(3, 6):
            e = elId[ii]
            endpts = crd_local(self.vpos[self.e_v[e]], self.bs, self.sstn)
            mid, rfn = edge_mid_vrtx_average(endpts, self.edpc)
            self.e_mid[e] = mid
            self.e_rfn[e] = rfn

        # face vertex loops
        self.f_v[flId[0]] = self._face_vrtx_insert(vlId[1], t2Id, vnew1, self.f_v[flId[0]])
        self.f_v[flId[2]] = self._face_vrtx_insert(vlId[2], t2Id, vnew2, self.f_v[flId[2]])
        fv = self._face_vrtx_insert(vlId[1], t2Id, vnew1, self.f_v[flId[1]])
        fv = self._face_vrtx_insert(vlId[2], t2Id, vnew2, fv)
        fv = [v for v in fv if v != t2Id]
        self.f_v[flId[1]] = fv

        for ci in (flId[0], flId[1], flId[2]):
            self._recompute_cell(ci)

        # tensions of triangle edges (free boundaries -> fixed point 1)
        for e in (elId[3], elId[4], elId[5]):
            self.e_t[e] = self.fixed_tension(e)

    # ------------------------------------------------------------------ #
    #  T1 neighbour exchange (curved edges + spaces)
    # ------------------------------------------------------------------ #
    def _cell_dir(self, v1, v2):
        """Cell whose loop contains v1 immediately followed by v2 (or None)."""
        for ci in range(self.nFa):
            loop = self.f_v[ci]
            n = len(loop)
            for k in range(n):
                if loop[k] == v1 and loop[(k + 1) % n] == v2:
                    return ci, k
        return None, None

    def _edge_between(self, a, b):
        for e in self.v_e[a]:
            if e == -1:
                continue
            if {int(self.e_v[e, 0]), int(self.e_v[e, 1])} == {a, b}:
                return e
        return None

    def _third_face(self, v, exclude):
        for f in self.v_f[v]:
            if f not in exclude:
                return int(f)
        return SPACE

    def _snapshot(self):
        import copy
        return (self.vpos.copy(), self.v_e.copy(), self.v_f.copy(),
                self.e_v.copy(), self.e_f.copy(), self.e_t.copy(),
                self.e_rfn.copy(), [m.copy() for m in self.e_mid],
                [list(x) for x in self.f_v], [list(x) for x in self.f_e],
                self.f_area.copy(), self.f_center.copy())

    def _restore(self, s):
        (self.vpos, self.v_e, self.v_f, self.e_v, self.e_f, self.e_t,
         self.e_rfn, self.e_mid, self.f_v, self.f_e, self.f_area,
         self.f_center) = (s[0], s[1], s[2], s[3], s[4], s[5], s[6],
                           s[7], s[8], s[9], s[10], s[11])

    def t1_foam(self, e):
        """Neighbour exchange on cell-cell edge e (handles adjacent spaces)."""
        v1, v2 = int(self.e_v[e, 0]), int(self.e_v[e, 1])
        fa, fb = int(self.e_f[e, 0]), int(self.e_f[e, 1])
        if fa == SPACE or fb == SPACE:
            return False
        resL = self._cell_dir(v1, v2)
        resR = self._cell_dir(v2, v1)
        cL, kL = resL
        cR, kR = resR
        if cL is None or cR is None or cL == cR:
            return False
        loopL, loopR = self.f_v[cL], self.f_v[cR]
        nL, nR = len(loopL), len(loopR)
        if nL <= 3 or nR <= 3:
            return False
        a = loopL[(kL - 1) % nL]
        b = loopL[(kL + 2) % nL]
        c = loopR[(kR - 1) % nR]
        d = loopR[(kR + 2) % nR]
        c1 = self._third_face(v1, {cL, cR})
        c2 = self._third_face(v2, {cL, cR})
        if c1 == c2 and c1 != SPACE:
            return False
        # both cells and already neighbours -> would double an edge
        if c1 != SPACE and c2 != SPACE:
            nbr = set()
            for ee in range(len(self.e_v)):
                f0, f1 = int(self.e_f[ee, 0]), int(self.e_f[ee, 1])
                if c1 in (f0, f1):
                    nbr.add(f0); nbr.add(f1)
            if c2 in nbr:
                return False
        e_a = self._edge_between(v1, a)
        e_c = self._edge_between(v2, c)
        if e_a is None or e_c is None or e_a == e_c:
            return False

        snap = self._snapshot()
        try:
            # geometry: rotate edge 90 deg about midpoint
            p1 = self.vpos[v1]
            p2 = p1 + min_image(self.vpos[v2] - p1, self.bs)
            mid = 0.5 * (p1 + p2)
            ev = p2 - p1
            perp = np.array([-ev[1], ev[0]])
            npp = np.hypot(*perp)
            perp = perp / npp if npp > 1e-12 else np.array([1.0, 0.0])
            half = 0.5 * max(self.edpc, np.hypot(*ev)) * 1.2
            self.vpos[v1] = np.mod(mid + half * perp, self.bs)
            self.vpos[v2] = np.mod(mid - half * perp, self.bs)

            # combinatorics
            self.e_f[e] = self._sortf([c1, c2])
            self.e_v[e_a] = self._sortf([a, v2])
            self.e_v[e_c] = self._sortf([c, v1])
            e_d = self._edge_between(v1, d)   # (recompute after? still v1-d) -> unchanged
            e_b = self._edge_between(v2, b)
            self.v_e[v1] = self._sortf([e, e_c, e_d])
            self.v_e[v2] = self._sortf([e, e_a, e_b])
            self.v_f[v1] = self._sortf([cR, c1, c2])
            self.v_f[v2] = self._sortf([cL, c1, c2])

            self.f_v[cL] = [v for v in loopL if v != v1]
            self.f_v[cR] = [v for v in loopR if v != v2]
            if c1 != SPACE:
                self.f_v[c1] = self._face_vrtx_insert(v1, a, v2, self.f_v[c1])
            if c2 != SPACE:
                self.f_v[c2] = self._face_vrtx_insert(v2, c, v1, self.f_v[c2])

            # rebuild affected polylines (straight; relaxation re-curves them)
            for ee in (e, e_a, e_c):
                pts = crd_local(self.vpos[self.e_v[ee]], self.bs, self.sstn)
                self.e_mid[ee], self.e_rfn[ee] = edge_mid_vrtx_average(pts, self.edpc)
            self.e_t[e] = self.fixed_tension(e)
            for ci in (cL, cR, c1, c2):
                if ci != SPACE:
                    self._recompute_cell(ci)
            # validate
            for ci in (cL, cR, c1, c2):
                if ci != SPACE and (len(self.f_v[ci]) < 3 or self.f_area[ci] <= 1e-4):
                    raise ValueError("degenerate cell")
            return True
        except Exception:
            self._restore(snap)
            return False

    def do_transitions(self, mu=0.0, create_spaces=False):
        """Apply T1 (short cell-cell edges) + T2 annihilation (+ optional creation)."""
        n1 = 0
        # T1 on short cell-cell edges
        lens = np.array([edge_len(self.e_mid[i]).sum() for i in range(len(self.e_mid))])
        thr = 0.01 * 2 * np.sqrt(np.pi)
        order = np.argsort(lens)
        for i in order:
            if lens[i] >= thr:
                break
            if int(self.e_f[i, 0]) != SPACE and int(self.e_f[i, 1]) != SPACE:
                if self.t1_foam(int(i)):
                    n1 += 1
        # after T1s, any edge left between two spaces is merged away (T3-reverse)
        self.resolve_space_space_edges()
        # T2: annihilate collapsed spaces
        na = self.t2_annihilate_spaces()
        # optional ongoing space creation at all-cell vertices (topTransitionType2)
        n2 = 0
        if create_spaces:
            for v in range(len(self.vpos)):
                if np.all(self.v_f[v] != SPACE):
                    try:
                        self.t2_reverse(v); n2 += 1
                    except Exception:
                        pass
        self._update_faces()
        return n1, n2, na

    # ------------------------------------------------------------------ #
    #  T2: annihilate a collapsed triangular extracellular space
    # ------------------------------------------------------------------ #
    def _remove_ve(self, vdel, edel):
        """Delete vertices/edges and remap all ids (port of ts_newIdAssign)."""
        vdel, edel = set(vdel), set(edel)
        vkeep = [v for v in range(len(self.vpos)) if v not in vdel]
        ekeep = [e for e in range(len(self.e_v)) if e not in edel]
        vmap = {old: new for new, old in enumerate(vkeep)}
        emap = {old: new for new, old in enumerate(ekeep)}

        def rm_e(e):
            return emap[e] if e in emap else -1
        self.vpos = self.vpos[vkeep]
        self.v_e = np.array([[rm_e(e) if e != -1 else -1 for e in self.v_e[v]]
                             for v in vkeep], np.int64)
        self.v_f = self.v_f[vkeep]
        self.e_v = np.array([[vmap[int(self.e_v[e, 0])], vmap[int(self.e_v[e, 1])]]
                             for e in ekeep], np.int64)
        self.e_f = self.e_f[ekeep]
        self.e_t = self.e_t[ekeep]
        self.e_rfn = self.e_rfn[ekeep]
        self.e_mid = [self.e_mid[e] for e in ekeep]
        self.f_v = [[vmap[v] for v in loop] for loop in self.f_v]
        for ci in range(self.nFa):
            self.f_e[ci] = self._face_edge_id(ci)

    def _find_collapsible_triangle(self, area_thr):
        """Return (a,b,c,eab,eac,ebc) of one tiny space triangle, or None."""
        space_edges = [e for e in range(len(self.e_v))
                       if int(self.e_f[e, 0]) == SPACE or int(self.e_f[e, 1]) == SPACE]
        adj = {}
        for e in space_edges:
            a, b = int(self.e_v[e, 0]), int(self.e_v[e, 1])
            adj.setdefault(a, {})[b] = e
            adj.setdefault(b, {})[a] = e
        for a in adj:
            nbrs = list(adj[a])
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    b, c = nbrs[i], nbrs[j]
                    if b in adj.get(c, {}):
                        tri = crd_local(self.vpos[[a, b, c]], self.bs, self.sstn)
                        if abs(poly_area_centroid(tri)[0]) < area_thr:
                            return (a, b, c, adj[a][b], adj[a][c], adj[c][b])
        return None

    def t2_annihilate_spaces(self, area_thr=1e-5, max_events=200):
        """Collapse fully-collapsed triangular spaces to triple junctions.

        The threshold is well below the 1%-triangle area at which spaces are
        seeded, so only genuinely closed spaces are removed (growing spaces
        survive to set the equilibrium volume fraction).
        """
        removed = 0
        for _ in range(max_events):
            tri = self._find_collapsible_triangle(area_thr)
            if tri is None:
                break
            if not self._collapse_triangle(*tri):
                break
            removed += 1
        if removed:
            self._update_faces()
        return removed

    def _collapse_triangle(self, a, b, c, eab, eac, ebc):
        """Merge b,c into a; drop the 3 triangle edges (reverse ts_t2Reverse)."""
        snap = self._snapshot()
        try:
            # outer edge of each corner (the one not on the triangle)
            def outer(v, tri_edges):
                for e in self.v_e[v]:
                    if e != -1 and e not in tri_edges:
                        return int(e)
                return None
            oa = outer(a, {eab, eac})
            ob = outer(b, {eab, ebc})
            oc = outer(c, {eac, ebc})
            if None in (oa, ob, oc):
                raise ValueError
            # reconnect b's and c's outer edges to a
            for (v, ov) in ((b, ob), (c, oc)):
                if int(self.e_v[ov, 0]) == v:
                    self.e_v[ov, 0] = a
                else:
                    self.e_v[ov, 1] = a
            # a's incident edges: its outer + b's outer + c's outer
            self.v_e[a] = self._sortf([oa, ob, oc])
            # a's faces: union of the 3 corners' non-space faces
            faces = set()
            for v in (a, b, c):
                for f in self.v_f[v]:
                    if f != SPACE:
                        faces.add(int(f))
            faces = sorted(faces)[:3]
            while len(faces) < 3:
                faces.append(SPACE)
            self.v_f[a] = self._sortf(faces)
            # position -> triangle centroid
            tri = crd_local(self.vpos[[a, b, c]], self.bs, self.sstn)
            self.vpos[a] = np.mod(tri.mean(axis=0), self.bs)
            # cells: replace b,c by a, drop duplicates
            for ci in range(self.nFa):
                loop = self.f_v[ci]
                if b in loop or c in loop:
                    new = [a if v in (b, c) else v for v in loop]
                    dedup = [new[0]]
                    for v in new[1:]:
                        if v != dedup[-1]:
                            dedup.append(v)
                    if len(dedup) > 1 and dedup[0] == dedup[-1]:
                        dedup.pop()
                    self.f_v[ci] = dedup
            # rebuild outer-edge polylines to end at a
            for ov in (oa, ob, oc):
                pts = crd_local(self.vpos[self.e_v[ov]], self.bs, self.sstn)
                self.e_mid[ov], self.e_rfn[ov] = edge_mid_vrtx_average(pts, self.edpc)
            self._remove_ve({b, c}, {eab, eac, ebc})
            # recompute the touched cells
            for ci in range(self.nFa):
                self._recompute_cell(ci)
            for ci in range(self.nFa):
                if len(self.f_v[ci]) < 3 or self.f_area[ci] <= 1e-4:
                    raise ValueError("degenerate cell after collapse")
            return True
        except Exception:
            self._restore(snap)
            return False

    # ------------------------------------------------------------------ #
    #  T3-reverse: remove a space-space edge, merging the two spaces
    # ------------------------------------------------------------------ #
    def t3_reverse(self, deId):
        """Remove space-space edge deId (both faces SPACE), merging the spaces.

        Port of ts_t3ReverseTransition.m.  Removes the edge and its two vertices,
        merging the two adjacent edges at each endpoint.
        """
        if not (int(self.e_f[deId, 0]) == SPACE and int(self.e_f[deId, 1]) == SPACE):
            return False
        cv = [int(self.e_v[deId, 0]), int(self.e_v[deId, 1])]
        fId = []
        for k in range(2):
            cells = [int(f) for f in self.v_f[cv[k]] if f != SPACE]
            if len(cells) != 1:
                return False                      # only the simple T1-created case
            fId.append(cells[0])
        # two non-deId edges at each endpoint
        e01 = [int(x) for x in self.v_e[cv[0]] if x != deId and x != -1]
        e23 = [int(x) for x in self.v_e[cv[1]] if x != deId and x != -1]
        if len(e01) != 2 or len(e23) != 2:
            return False
        eId = e01 + e23                            # [eA, eB, eC, eD]

        def far(e, notv):
            a, b = int(self.e_v[e, 0]), int(self.e_v[e, 1])
            return a if b == notv else b
        if eId[0] == eId[1] or eId[2] == eId[3]:
            return False                           # 2-gon degenerate; skip
        vId = [far(eId[0], cv[0]), far(eId[1], cv[0]),
               far(eId[2], cv[1]), far(eId[3], cv[1])]
        if len({vId[0], vId[1], cv[0]}) < 3 or len({vId[2], vId[3], cv[1]}) < 3:
            return False

        snap = self._snapshot()
        try:
            # reconnect far vertices: eId[1]->eId[0] at vId[1]; eId[3]->eId[2] at vId[3]
            self.v_e[vId[1]] = self._sortf(
                [eId[0] if x == eId[1] else int(x) for x in self.v_e[vId[1]]])
            self.v_e[vId[3]] = self._sortf(
                [eId[2] if x == eId[3] else int(x) for x in self.v_e[vId[3]]])
            # merged edge endpoints
            self.e_v[eId[0]] = self._sortf([vId[0], vId[1]])
            self.e_v[eId[2]] = self._sortf([vId[2], vId[3]])
            self.e_t[eId[0]] = 0.5 * (self.e_t[eId[0]] + self.e_t[eId[1]])
            self.e_t[eId[2]] = 0.5 * (self.e_t[eId[2]] + self.e_t[eId[3]])
            # merge polylines
            for (ea, eb, shared, keep) in [(eId[0], eId[1], cv[0], eId[0]),
                                           (eId[2], eId[3], cv[1], eId[2])]:
                emd1 = self.e_mid[ea]
                if int(snap[3][ea, 0]) == shared:
                    emd1 = emd1[::-1]
                emd2 = self.e_mid[eb]
                if int(snap[3][eb, 1]) == shared:
                    emd2 = emd2[::-1]
                vr = crd_local(np.vstack([emd1, emd2[1:]]), self.bs, self.sstn)
                self.e_mid[keep], self.e_rfn[keep] = edge_mid_vrtx_average(vr, self.edpc)
            # cells lose the removed vertices
            self.f_v[fId[0]] = [x for x in self.f_v[fId[0]] if x != cv[0]]
            self.f_v[fId[1]] = [x for x in self.f_v[fId[1]] if x != cv[1]]
            # remove vertices cv, edges deId,eId[1],eId[3]
            self._remove_ve({cv[0], cv[1]}, {deId, eId[1], eId[3]})
            for ci in set(fId):
                self._recompute_cell(ci)
            for ci in set(fId):
                if len(self.f_v[ci]) < 3 or self.f_area[ci] <= 1e-4:
                    raise ValueError("degenerate cell after t3-reverse")
            return True
        except Exception:
            self._restore(snap)
            return False

    def resolve_space_space_edges(self, max_events=400):
        """Remove all space-space edges via T3-reverse (re-detect after each)."""
        removed = 0
        for _ in range(max_events):
            ss = [e for e in range(len(self.e_v))
                  if int(self.e_f[e, 0]) == SPACE and int(self.e_f[e, 1]) == SPACE]
            if not ss:
                break
            if not self.t3_reverse(ss[0]):
                # can't resolve this one; drop it to avoid an infinite loop
                ss = ss[1:]
                progressed = False
                for e in ss:
                    if self.t3_reverse(e):
                        removed += 1; progressed = True; break
                if not progressed:
                    break
            else:
                removed += 1
        return removed
