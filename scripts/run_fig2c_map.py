"""Fig 2c as drawn in the paper: contact number z over the (W/T0, rho) plane.

Same equilibrium protocol as run_fig2b_foam_full.py (confluent Voronoi ->
T2-reverse at every vertex -> anneal -> quench), now with the complete
transition set.  The non-adjacent T4 (ts_t4Transition) is the reference code's
contact-FORMATION step: two cells' free edges that overlap are joined by a new
cell-cell edge.  Without it contacts could only break, which is what drove the
earlier W->0 over-fragmentation.

Saves data/fig2c_map.npz (phi and z on the grid).
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.topology import build_periodic_voronoi
from activefoam.foam_full import FoamTissue

os.makedirs("data", exist_ok=True)

WS = np.round(np.arange(0.0, 0.71, 0.1), 3)
RHOS = np.round(np.arange(0.80, 1.151, 0.05), 3)
SEEDS = [1]
TRANS_EVERY = 10


def equilibrate(w, rho, seed):
    T = build_periodic_voronoi(6, np.random.default_rng(100 + seed))
    ft = FoamTissue(T, w=w, rho=rho, seed=seed)
    ncell = ft.nFa
    for _ in range(150):
        ft.step(mu=0.0)
    for v in range(len(ft.vpos)):
        ft.t2_reverse(v)
    ft._update_faces()
    for it in range(350):
        ft.step(mu=0.5)                       # anneal
        if it % TRANS_EVERY == 0:
            ft.do_transitions()
    for it in range(700):
        ft.step(mu=0.0)                       # quench
        if it % TRANS_EVERY == 0:
            ft.do_transitions()
    phi = ft.volume_fraction()
    areas = ft.f_area[:ncell]
    if phi > 1.08 or np.any(areas < -1e-6) or np.any(areas > 3 * ft.A0):
        return np.nan, np.nan
    z = ft.neighbor_number()[:ncell]
    return phi, float(z.mean())


def _task(a):
    i, j, rho, w, s = a
    try:
        p, z = equilibrate(w, rho, s)
    except Exception as ex:                       # one bad run must not kill the sweep
        print(f"  !! rho={rho} W={w} seed={s} failed: {ex!r}", flush=True)
        p = z = np.nan
    return i, j, p, z


def main():
    from multiprocessing import Pool
    t0 = time.time()
    tasks = [(i, j, float(rho), float(w), s) for i, rho in enumerate(RHOS)
             for j, w in enumerate(WS) for s in SEEDS]
    pacc = {(i, j): [] for i in range(len(RHOS)) for j in range(len(WS))}
    zacc = {(i, j): [] for i in range(len(RHOS)) for j in range(len(WS))}
    with Pool(processes=4) as pool:
        for n, (i, j, p, z) in enumerate(pool.imap_unordered(_task, tasks), 1):
            pacc[(i, j)].append(p); zacc[(i, j)].append(z)
            print(f"[{n}/{len(tasks)}] rho={RHOS[i]:.2f} W={WS[j]:.1f}: "
                  f"phi={p:.3f} z={z:.2f}  [{time.time()-t0:.0f}s]", flush=True)

    def mean(acc):
        out = np.full((len(RHOS), len(WS)), np.nan)
        for (i, j), v in acc.items():
            v = [x for x in v if np.isfinite(x)]
            if v:
                out[i, j] = np.mean(v)
        return out
    np.savez("data/fig2c_map.npz", WS=WS, RHOS=RHOS, phi=mean(pacc), z=mean(zacc))
    print("saved data/fig2c_map.npz  total", round(time.time()-t0), "s")


if __name__ == "__main__":
    main()
