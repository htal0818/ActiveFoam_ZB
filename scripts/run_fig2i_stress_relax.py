"""Fig 2i (paper Methods): shear-stress relaxation sigma_xy(t)/sigma0 vs t/tau_R
for varying adhesion W/T0 (equilibrium, dT=0).

Foam model + paper Eq. 4/5 stress + affine-shear step.  The initial stress jump
is largest at W=0 and vanishes as effective tensions vanish (W->2); the stress
then relaxes with a characteristic time ~tau_R towards the yield stress.
Saves data/fig2i.npz.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.topology import build_periodic_voronoi
from activefoam.foam_full import FoamTissue

os.makedirs("data", exist_ok=True)

WS = [0.0, 0.5, 1.0, 1.5, 2.0]
SEEDS = [1, 2]
STRAIN = 1.0
RELAX_TAUR = 30.0
SAMPLE = 25


def relax_curve(w, rho=1.0, seed=1):
    T = build_periodic_voronoi(6, np.random.default_rng(100 + seed))
    ft = FoamTissue(T, w=w, rho=rho, seed=seed)
    for _ in range(150):
        ft.step(mu=0.0)
    for v in range(len(ft.vpos)):
        ft.t2_reverse(v)
    ft._update_faces()
    for it in range(600):
        ft.step(mu=0.0)
        if it % 15 == 0:
            ft.do_transitions()
    if ft.volume_fraction() > 1.08:
        return None, None
    s0 = ft.shear_stress()
    ft.apply_affine_shear(STRAIN)
    steps = int(round(RELAX_TAUR / ft.dt))
    ts, sig = [], []
    for it in range(steps):
        ft.step(mu=0.0)
        if it % 15 == 0:
            ft.do_transitions()
        if it % SAMPLE == 0:
            ts.append(it * ft.dt)
            sig.append(ft.shear_stress() - s0)
    return np.array(ts), np.array(sig)


def main():
    t0 = time.time()
    curves = {}
    for w in WS:
        stack, ts = [], None
        for s in SEEDS:
            t, sig = relax_curve(w, seed=s)
            if t is None:
                continue
            ts = t
            stack.append(sig)
        curves[w] = (ts, np.mean(stack, axis=0), np.std(stack, axis=0))
        print(f"W/T0={w}: jump={curves[w][1][0]:.3f} "
              f"end={curves[w][1][-1]:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    np.savez("data/fig2i.npz",
             WS=np.array(WS),
             ts=curves[WS[0]][0],
             sig=np.array([curves[w][1] for w in WS]),
             err=np.array([curves[w][2] for w in WS]))
    print("saved data/fig2i.npz  total", round(time.time()-t0), "s")


if __name__ == "__main__":
    main()
