"""Measurement protocols: MSD, neighbour-exchange rate, stress relaxation."""

from __future__ import annotations

import numpy as np

from .model import ActiveFoam


def _burn(af: ActiveFoam, n_tauT: float, t1_every: int = 3):
    steps = int(round(n_tauT * af.tauT / af.dt))
    for it in range(steps):
        af.step()
        if it % t1_every == 0:
            af.do_t1_transitions()


def run_msd(dT, w=1.0, rho=1.0, n_side=6, t_max_tauT=100.0, burn_tauT=20.0,
            sample_every=8, seed=0, l_t1=0.035449, t1_every=3):
    """Track unwrapped cell centroids; return (t/tauT lags, MSD/L0^2, NE_rate).

    NE_rate is the mean number of T1 events per cell per tau_R over the run.
    """
    af = ActiveFoam(n_side=n_side, w=w, dT=dT, rho=rho, seed=seed, l_t1=l_t1)
    _burn(af, burn_tauT, t1_every)

    steps = int(round(t_max_tauT * af.tauT / af.dt))
    L = af.tissue.L
    prev = af.cell_centroids()
    unwrapped = prev.copy()
    traj, times = [unwrapped.copy()], [af.time]
    n_t1 = 0
    for it in range(steps):
        af.step()
        if it % t1_every == 0:
            n_t1 += af.do_t1_transitions()
        if it % sample_every == 0:
            cur = af.cell_centroids()
            d = cur - prev
            d -= L * np.round(d / L)
            unwrapped = unwrapped + d
            prev = cur
            traj.append(unwrapped.copy())
            times.append(af.time)
    traj = np.asarray(traj)
    times = np.asarray(times)
    dt_s = times[1] - times[0]
    # measure displacements in the tissue centre-of-mass frame
    traj = traj - traj.mean(axis=1, keepdims=True)

    nS = len(times)
    lags = np.unique(np.round(np.logspace(0, np.log10(nS - 1), 30)).astype(int))
    msd = np.empty(len(lags))
    for i, lag in enumerate(lags):
        disp = traj[lag:] - traj[:-lag]
        msd[i] = np.mean(np.sum(disp ** 2, axis=2))
    tlag = dt_s * lags / af.tauT
    ne_rate = n_t1 / af.n_cells / (steps * af.dt)   # per cell per tau_R
    return tlag, msd, ne_rate


def msd_exponent(tlag, msd, t_lo=1.0, t_hi=100.0):
    """Fit MSD ~ t^alpha in the long-time window [t_lo, t_hi] (units of tau_T)."""
    m = (tlag >= t_lo) & (tlag <= t_hi) & (msd > 0)
    if m.sum() < 2:
        return np.nan
    p = np.polyfit(np.log(tlag[m]), np.log(msd[m]), 1)
    return p[0]


def run_stress_relaxation(dT, w=1.0, rho=1.0, n_side=6, strain=0.5,
                          pre_tauT=20.0, relax_tauR=400.0, seed=0,
                          l_t1=0.035449, t1_every=3, sample_every=4):
    """Apply an affine shear step then record sigma_xy(t) during relaxation.

    Returns (t/tauR, sigma_xy(t)/sigma0) where sigma0 = sqrt(rho)*T0/L0 (=sqrt(rho)).
    Also returns the active-stress level sigma_A (std of sigma_xy pre-step).
    """
    af = ActiveFoam(n_side=n_side, w=w, dT=dT, rho=rho, seed=seed, l_t1=l_t1)
    _burn(af, pre_tauT, t1_every)

    # measure active shear-stress level (std over 10 tauR, no imposed strain)
    pre = []
    for it in range(int(round(10 * af.tauT / af.dt))):
        af.step()
        if it % t1_every == 0:
            af.do_t1_transitions()
        if it % sample_every == 0:
            pre.append(af.shear_stress())
    sigma_A = np.std(pre)

    # apply affine shear step: x -> x + strain*y   (Lees-Edwards offset too)
    af.shear = strain
    pos = af.tissue.pos.copy()
    pos[:, 0] += strain * pos[:, 1]
    af.tissue.pos = af._wrap_shear(pos)

    sigma0 = np.sqrt(rho)
    steps = int(round(relax_tauR / af.dt))
    ts, sig = [], []
    for it in range(steps):
        af.step()
        if it % t1_every == 0:
            af.do_t1_transitions()
        if it % sample_every == 0:
            ts.append(af.time)
            sig.append(af.shear_stress())
    ts = np.asarray(ts)
    ts = ts - ts[0]
    sig = np.asarray(sig) / sigma0
    return ts, sig, sigma_A / sigma0
