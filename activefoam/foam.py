"""Foam / emulsion jamming limit (Fig 2f).

In the limit of vanishing cell adhesion (W = 0) the active-foam model reduces to
a packing of soft, repulsive deformable particles -- i.e. the classic jamming
problem of foams/emulsions (O'Hern et al., Phys. Rev. E 68, 011306 (2003); the
paper's ref. 36).  The isostatic condition z_c = 4 in 2D sets the critical volume
fraction phi_c ~ 0.83-0.84, with

        z - z_c = z0 (phi - phi_c)^{1/2} + z1 (phi - phi_c).

We reproduce this with bidisperse harmonic soft disks compressed through the
jamming point, measuring the mean contact number (excluding rattlers) vs phi.
"""

from __future__ import annotations

import numpy as np


def _pair_forces(pos, radii, L):
    """Harmonic repulsion. Returns (energy, forces, n_contacts_matrix rows)."""
    n = len(pos)
    F = np.zeros_like(pos)
    contacts = [set() for _ in range(n)]
    energy = 0.0
    # all pairs (n small: ~100-400); minimum image
    for i in range(n - 1):
        dr = pos[i + 1:] - pos[i]
        dr -= L * np.round(dr / L)
        dist = np.hypot(dr[:, 0], dr[:, 1])
        sigma = radii[i] + radii[i + 1:]
        overlap = sigma - dist
        touching = np.where(overlap > 0)[0]
        for t in touching:
            j = i + 1 + t
            d = dist[t]
            if d < 1e-12:
                continue
            fmag = overlap[t] / sigma[t]        # harmonic: dU/dr scaled
            fij = (fmag / d) * dr[t]
            F[i] -= fij
            F[j] += fij
            energy += 0.5 * (overlap[t] / sigma[t]) ** 2
            contacts[i].add(j)
            contacts[j].add(i)
    return energy, F, contacts


def minimize(pos, radii, L, n_steps=4000, lr=0.05, ftol=1e-12):
    """FIRE-lite: damped gradient descent to the nearest energy minimum."""
    pos = pos.copy()
    vel = np.zeros_like(pos)
    alpha = 0.1
    dt = lr
    for step in range(n_steps):
        e, F, _ = _pair_forces(pos, radii, L)
        fnorm = np.sqrt(np.sum(F * F))
        if fnorm < ftol:
            break
        # FIRE update
        power = np.sum(F * vel)
        vel = (1 - alpha) * vel + alpha * (F / (fnorm + 1e-30)) * np.sqrt(np.sum(vel * vel))
        if power > 0:
            dt = min(dt * 1.1, 0.1)
            alpha *= 0.99
        else:
            vel[:] = 0
            dt *= 0.5
            alpha = 0.1
        vel = vel + dt * F
        pos = pos + dt * vel
        pos = np.mod(pos, L)
    return pos


def contact_number_at_phi(phi, n=256, ratio=1.4, seed=0, n_steps=3000):
    """Compress a bidisperse disk packing to volume fraction phi; return z (no rattlers)."""
    rng = np.random.default_rng(seed)
    # bidisperse 50:50
    r = np.empty(n)
    r[: n // 2] = 1.0
    r[n // 2:] = ratio
    r = rng.permutation(r)
    area_particles = np.sum(np.pi * r ** 2)
    L = np.sqrt(area_particles / phi)
    pos = rng.random((n, 2)) * L
    pos = minimize(pos, r, L, n_steps=n_steps)
    _, _, contacts = _pair_forces(pos, r, L)

    # iteratively remove rattlers (particles with < 3 contacts)
    deg = np.array([len(c) for c in contacts])
    alive = np.ones(n, bool)
    changed = True
    contacts = [set(c) for c in contacts]
    while changed:
        changed = False
        for i in range(n):
            if alive[i] and sum(1 for j in contacts[i] if alive[j]) < 3:
                alive[i] = False
                changed = True
    if alive.sum() == 0:
        return 0.0
    z = np.mean([sum(1 for j in contacts[i] if alive[j]) for i in range(n) if alive[i]])
    return z


def sweep_phi(phis, n=256, seeds=(0, 1, 2, 3), n_steps=3000):
    zmean = []
    zstd = []
    for phi in phis:
        zs = [contact_number_at_phi(phi, n=n, seed=s, n_steps=n_steps) for s in seeds]
        zmean.append(np.mean(zs))
        zstd.append(np.std(zs))
    return np.array(zmean), np.array(zstd)
