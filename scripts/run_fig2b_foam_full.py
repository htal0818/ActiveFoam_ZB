"""Fig 2b/c from the FAITHFUL curved-edge foam model with extracellular spaces.

Protocol (paper Methods): confluent Voronoi -> relax -> introduce a triangular
extracellular space at every vertex (T2-reverse) -> anneal with small tension
fluctuations (Delta T = 0.5) -> quench (Delta T = 0).  Measures volume fraction
phi and cell-cell contact number z vs adhesion W/T0 and density rho.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.topology import build_periodic_voronoi
from activefoam.foam_full import FoamTissue

os.makedirs("data", exist_ok=True)

WS = np.round(np.arange(0.0, 1.51, 0.25), 3)
RHOS = [1.0, 0.9, 0.8]
SEEDS = [1, 2]


def equilibrate(w, rho, seed):
    T = build_periodic_voronoi(6, np.random.default_rng(100 + seed))
    ft = FoamTissue(T, w=w, rho=rho, seed=seed)
    for _ in range(150):
        ft.step(mu=0.0)
    nv0 = len(ft.vpos)
    for v in range(nv0):
        ft.t2_reverse(v)
    ft._update_faces()
    for it in range(350):
        ft.step(mu=0.5)               # anneal
        if it % 20 == 0:
            ft.do_transitions()       # T1 + T2 annihilation
    for it in range(700):
        ft.step(mu=0.0)               # quench
        if it % 20 == 0:
            ft.do_transitions()
    z = ft.neighbor_number()
    return ft.volume_fraction(), z[z > 0].mean() if np.any(z > 0) else 0.0


def main():
    t0 = time.time()
    phi = np.zeros((len(RHOS), len(WS)))
    zz = np.zeros((len(RHOS), len(WS)))
    for i, rho in enumerate(RHOS):
        for j, w in enumerate(WS):
            pv, zv = [], []
            for s in SEEDS:
                p, z = equilibrate(w, rho, s)
                pv.append(p); zv.append(z)
            phi[i, j] = np.mean(pv)
            zz[i, j] = np.mean(zv)
        print(f"rho={rho}: phi={np.round(phi[i],3)}  [{time.time()-t0:.0f}s]", flush=True)
        print(f"          z={np.round(zz[i],2)}", flush=True)
    np.savez("data/fig2b_full.npz", WS=WS, RHOS=np.array(RHOS), phi=phi, z=zz)
    print("saved data/fig2b_full.npz  total", round(time.time()-t0), "s")


if __name__ == "__main__":
    main()
