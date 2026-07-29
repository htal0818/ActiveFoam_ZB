"""Deformable-particle model with adhesion -> extracellular spaces (Fig 2).

The confluent vertex model cannot open spaces between cells.  The paper's
non-confluent structural results (Fig 2a-d: adhesion-dependent extracellular
spaces, volume fraction phi, contact number z) come from cells that can round up
and physically separate.  This is the deformable-particle model with adhesion
(Boromand et al., Phys. Rev. Lett. 121, 248003 (2018) / Soft Matter 2019 --
the paper's ref. 28, which it "extends ... to arbitrary adhesion levels").

Each cell is a closed ring of ``m`` boundary vertices ("beads").  Energy:

    * area (osmotic):    E_A = 1/2 K_A (A_i/A0 - 1)^2         -> keeps area ~ A0
    * cortical tension:  E_T = T0 * perimeter_i               -> rounds the cell
    * inter-cell beads (different cells only):
          repulsion for gap < 0          (no overlap)
          adhesion  for 0 < gap < d_adh  (strength proportional to W)

Density is set by the box size:  rho = N A0 / A_T, so A_T = N A0 / rho and
L = sqrt(A_T).  For rho < 1 cells prefer to be smaller than the available area
and, at low adhesion, round up leaving extracellular spaces; increasing W flattens
contacts and drives the system confluent (phi -> 1).

Equilibrium configurations (paper Fig 2, Delta T = 0) are obtained by FIRE energy
minimisation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class DeformableTissue:
    def __init__(self, n_cells=36, m=24, rho=1.0, w=0.0, A0=1.0,
                 P0=10.0, T0=1.0, seed=0,
                 sigma=0.16, d_adh=0.25, k_rep=10.0, adh_scale=5.0):
        self.N = n_cells
        self.m = m
        self.A0 = A0
        self.rho = rho
        self.w = w
        self.P0 = P0
        self.T0 = T0
        self.sigma = sigma
        self.d_adh = d_adh
        self.k_rep = k_rep
        self.adh_scale = adh_scale
        rng = np.random.default_rng(seed)

        self.L = np.sqrt(n_cells * A0 / rho)
        # initial cell centres on a jittered grid
        s = int(np.ceil(np.sqrt(n_cells)))
        gx, gy = np.meshgrid(np.arange(s), np.arange(s))
        centres = np.stack([gx.ravel(), gy.ravel()], 1)[:n_cells].astype(float)
        centres = (centres + 0.5) * (self.L / s)
        centres += rng.normal(0, 0.05 * self.L / s, centres.shape)
        # circular initial cells of radius r0
        r0 = 0.9 * np.sqrt(A0 / np.pi)
        ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
        ring = np.stack([np.cos(ang), np.sin(ang)], 1) * r0
        self.pos = (centres[:, None, :] + ring[None, :, :])  # (N, m, 2)
        self.cell_id = np.repeat(np.arange(n_cells), m)

    # ------------------------------------------------------------------ #
    def areas(self):
        p = self.pos
        x, y = p[:, :, 0], p[:, :, 1]
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, 1))

    def perimeters(self):
        d = np.roll(self.pos, -1, 1) - self.pos
        return np.sum(np.hypot(d[:, :, 0], d[:, :, 1]), 1)

    def _intra_forces(self):
        p = self.pos
        N, m = self.N, self.m
        F = np.zeros_like(p)
        # signed area + gradient
        x, y = p[:, :, 0], p[:, :, 1]
        area_signed = 0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, 1)
        sign = np.sign(area_signed)
        A = np.abs(area_signed)
        # osmotic pressure (paper form): P_i = P0 (A0/A - 1), force = P_i * dA/dr
        coeff = (self.P0 * (self.A0 / A - 1.0)) * sign                    # (N,)
        xn, xp = np.roll(x, -1, 1), np.roll(x, 1, 1)
        yn, yp = np.roll(y, -1, 1), np.roll(y, 1, 1)
        dAx = 0.5 * (yn - yp)
        dAy = 0.5 * (xp - xn)
        F[:, :, 0] += coeff[:, None] * dAx
        F[:, :, 1] += coeff[:, None] * dAy
        # cortical tension: pull each vertex toward its two neighbours
        nxt = np.roll(p, -1, 1) - p
        prv = np.roll(p, 1, 1) - p
        ln_n = np.hypot(nxt[:, :, 0], nxt[:, :, 1])[:, :, None]
        ln_p = np.hypot(prv[:, :, 0], prv[:, :, 1])[:, :, None]
        F += self.T0 * (nxt / (ln_n + 1e-12) + prv / (ln_p + 1e-12))
        return F

    def _inter_forces(self):
        beads = (self.pos.reshape(-1, 2)) % self.L
        cutoff = self.sigma + self.d_adh
        tree = cKDTree(beads, boxsize=self.L)
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        F = np.zeros((self.N * self.m, 2))
        if len(pairs) == 0:
            return F.reshape(self.N, self.m, 2)
        i, j = pairs[:, 0], pairs[:, 1]
        same = self.cell_id[i] == self.cell_id[j]
        i, j = i[~same], j[~same]
        if len(i) == 0:
            return F.reshape(self.N, self.m, 2)
        dr = beads[i] - beads[j]
        dr -= self.L * np.round(dr / self.L)
        r = np.hypot(dr[:, 0], dr[:, 1])
        r = np.clip(r, 1e-9, None)
        rhat = dr / r[:, None]
        gap = r - self.sigma
        fmag = np.zeros_like(r)
        # repulsion (gap < 0): push apart
        rep = gap < 0
        fmag[rep] = -self.k_rep * gap[rep]                 # >0 => along +rhat (apart)
        # adhesion (0 <= gap < d_adh): pull together, strength ~ W
        adh = (gap >= 0) & (gap < self.d_adh)
        eW = self.adh_scale * self.w
        fmag[adh] = -eW * (1.0 - gap[adh] / self.d_adh)    # <0 => along -rhat (together)
        f = fmag[:, None] * rhat
        np.add.at(F, i, f)
        np.add.at(F, j, -f)
        return F.reshape(self.N, self.m, 2)

    def forces(self):
        return self._intra_forces() + self._inter_forces()

    # ------------------------------------------------------------------ #
    def minimize(self, n_steps=1500, dt=0.02):
        vel = np.zeros_like(self.pos)
        alpha = 0.1
        for step in range(n_steps):
            F = self.forces()
            power = np.sum(F * vel)
            fnorm = np.sqrt(np.sum(F * F))
            if fnorm < 1e-8:
                break
            vnorm = np.sqrt(np.sum(vel * vel))
            vel = (1 - alpha) * vel + alpha * (F / (fnorm + 1e-30)) * vnorm
            if power > 0:
                dt = min(dt * 1.1, 0.05)
                alpha *= 0.99
            else:
                vel[:] = 0
                dt *= 0.5
                alpha = 0.1
            vel = vel + dt * F
            self.pos = self.pos + dt * vel
        return self

    # ------------------------------------------------------------------ #
    def volume_fraction(self):
        return np.sum(self.areas()) / (self.L * self.L)

    def contact_number(self, contact_gap=None, min_beads=2):
        """Mean number of neighbouring cells sharing a real contact (z).

        Two cells count as neighbours only if at least ``min_beads`` of their
        boundary vertices lie within ``contact_gap`` -- i.e. they share a contact
        of finite length, not just a single touching corner.
        """
        if contact_gap is None:
            contact_gap = 1.1 * self.sigma
        beads = self.pos.reshape(-1, 2) % self.L
        tree = cKDTree(beads, boxsize=self.L)
        pairs = tree.query_pairs(contact_gap, output_type="ndarray")
        from collections import Counter
        cnt: Counter = Counter()
        for a, b in pairs:
            ca, cb = self.cell_id[a], self.cell_id[b]
            if ca != cb:
                cnt[(min(ca, cb), max(ca, cb))] += 1
        neigh = [set() for _ in range(self.N)]
        for (ca, cb), c in cnt.items():
            if c >= min_beads:
                neigh[ca].add(cb)
                neigh[cb].add(ca)
        return np.mean([len(s) for s in neigh])

    def polygons(self):
        return [self.pos[c] for c in range(self.N)]
