"""Fig 2j (paper Methods): yield stress sigma_Y vs W/T0 (equilibrium, dT=0).

Uses the paper's stress tensor (Eq. 4/5) implemented in the foam model plus the
affine-shear-step + relaxation protocol from the Methods. sigma_Y is the residual
shear stress at long times after the step; it is maximal at the structural
transition (~W/T0=0.5) and vanishes at W/T0=0 (jamming) and W/T0=2 (vanishing
tension) -- the fluid limits.  Saves data/fig2j.npz.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.topology import build_periodic_voronoi
from activefoam.foam_full import FoamTissue

os.makedirs("data", exist_ok=True)

WS = np.round(np.arange(0.0, 2.01, 0.2), 3)
SEEDS = [1, 2, 3]
STRAIN = 0.5


def yield_stress(w, rho=1.0, seed=1):
    T = build_periodic_voronoi(6, np.random.default_rng(100 + seed))
    ft = FoamTissue(T, w=w, rho=rho, seed=seed)
    for _ in range(150):
        ft.step(mu=0.0)
    for v in range(len(ft.vpos)):
        ft.t2_reverse(v)
    ft._update_faces()
    for it in range(600):                       # equilibrate
        ft.step(mu=0.0)
        if it % 15 == 0:
            ft.do_transitions()
    if ft.volume_fraction() > 1.08 or np.any(ft.f_area > 3 * ft.A0):
        return np.nan
    s_pre = ft.shear_stress()
    ft.apply_affine_shear(STRAIN)               # affine shear step
    tail = []
    for it in range(500):                       # relax; average the tail
        ft.step(mu=0.0)
        if it % 15 == 0:
            ft.do_transitions()
        if it >= 350 and it % 10 == 0:
            tail.append(ft.shear_stress())
    if ft.volume_fraction() > 1.08:
        return np.nan
    return float(np.mean(tail) - s_pre)


def main():
    t0 = time.time()
    sig = np.zeros(len(WS))
    err = np.zeros(len(WS))
    for j, w in enumerate(WS):
        vals = [yield_stress(w, seed=s) for s in SEEDS]
        vals = [v for v in vals if np.isfinite(v)]
        sig[j] = np.mean(vals) if vals else np.nan
        err[j] = np.std(vals) if len(vals) > 1 else 0.0
        print(f"W/T0={w:.1f}: sigma_Y={sig[j]:.3f} +/- {err[j]:.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    np.savez("data/fig2j.npz", WS=WS, sigmaY=sig, err=err, strain=STRAIN)
    print("saved data/fig2j.npz  total", round(time.time()-t0), "s")


if __name__ == "__main__":
    main()
