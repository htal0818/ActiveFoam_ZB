"""Fig 4d (tau_SR map) and Fig 4e (fluid/solid phase diagram) vs (dT/T0, W/T0).

Paper methods: the long-timescale stress relaxation is driven by actively
induced neighbour-exchange (NE) events -- T1 transitions in confluent regions
and contact loss/formation in non-confluent regions (Fig 3d, Fig 4b).  The
stress-relaxation time tau_SR is set by the inverse cellular NE rate,
tau_SR ~ 1/k_NE (Extended Data Fig 1: MSD(10^2 tau_T) ~ k_NE^{3/4}).

We measure the steady-state cellular NE rate k_NE (topological events per cell
per tau_R) directly from the full foam model at each (dT/T0, W/T0), with an
*adaptive* window: quiet (near-solid) states are simulated for up to MAX_TAUR
so that a near-zero NE rate -> very large tau_SR is properly resolved, while
active/fragmenting states terminate early.  phi locates the confluent region.

  * Fig 4d : colour = log10(tau_SR/tau_R),  tau_SR = 1/k_NE.
  * Fig 4e : fluid if tau_SR/tau_T < 10^2  (tau_T = 10 tau_R -> tau_SR < 10^3 tau_R),
             else solid -- the paper's Fig 4e criterion.

Saves data/fig4de.npz.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.topology import build_periodic_voronoi
from activefoam.foam_full import FoamTissue

os.makedirs("data", exist_ok=True)

DTS = [0.5, 0.75, 1.0, 1.25, 1.5]
WS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
SEEDS = [1, 2]
EQUIL = 250          # equilibration steps
MIN_TAUR = 8.0       # measure at least this long
MAX_TAUR = 60.0      # ... and at most this long (resolves near-zero rates)
TARGET_EV = 6        # stop early once this many NE events are counted
TRANS_EVERY = 15
VCAP = 320           # vertex cap: fragmentation past this = fully fluid
KNE_FLOOR = 1e-6     # tau_SR cap 1e6 tau_R (matches paper colourbar top)
KNE_CEIL = 0.5       # tau_SR floor 2 tau_R (fully fluid)


def measure(dt_act, w, seed=1):
    """Return (k_NE per cell per tau_R, volume_fraction phi)."""
    T = build_periodic_voronoi(6, np.random.default_rng(100 + seed))
    ft = FoamTissue(T, w=w, rho=1.0, seed=seed)
    ncell = ft.nFa
    for _ in range(150):
        ft.step(mu=dt_act)
    for v in range(len(ft.vpos)):               # open spaces once at weak regions
        ft.t2_reverse(v)
    ft._update_faces()
    for it in range(EQUIL):                      # settle (spaces close where adhesion wins)
        ft.step(mu=dt_act)
        if it % TRANS_EVERY == 0:
            ft.do_transitions(mu=dt_act, create_spaces=False)
        if len(ft.vpos) > VCAP:
            return KNE_CEIL, ft.volume_fraction()
    # adaptive NE counting
    min_steps = int(round(MIN_TAUR / ft.dt))
    max_steps = int(round(MAX_TAUR / ft.dt))
    n_ev = 0
    phis = []
    it = 0
    while it < max_steps:
        ft.step(mu=dt_act)
        if it % TRANS_EVERY == 0:
            n1, n2, na = ft.do_transitions(mu=dt_act, create_spaces=False)
            n_ev += n1 + n2 + na
        if len(ft.vpos) > VCAP:
            return KNE_CEIL, ft.volume_fraction()
        if it % 200 == 0:
            phis.append(ft.volume_fraction())
        it += 1
        if n_ev >= TARGET_EV and it >= min_steps:
            break
    elapsed = it * ft.dt
    kne = n_ev / ncell / elapsed
    kne = min(max(kne, 0.0), KNE_CEIL)
    return kne, float(np.median(phis))


def _task(arg):
    i, j, d, w, s = arg
    k, p = measure(d, w, seed=s)
    return i, j, k, p


def main():
    from multiprocessing import Pool
    t0 = time.time()
    nD, nW = len(DTS), len(WS)
    tasks = [(i, j, d, w, s)
             for i, d in enumerate(DTS)
             for j, w in enumerate(WS)
             for s in SEEDS]
    kacc = {(i, j): [] for i in range(nD) for j in range(nW)}
    pacc = {(i, j): [] for i in range(nD) for j in range(nW)}
    with Pool(processes=4) as pool:
        for n, (i, j, k, p) in enumerate(pool.imap_unordered(_task, tasks), 1):
            kacc[(i, j)].append(k); pacc[(i, j)].append(p)
            print(f"[{n}/{len(tasks)}] dT={DTS[i]:.2f} W={WS[j]:.1f}: "
                  f"kNE={k:.4f} phi={p:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    kne = np.array([[np.mean(kacc[(i, j)]) for j in range(nW)] for i in range(nD)])
    phi = np.array([[np.mean(pacc[(i, j)]) for j in range(nW)] for i in range(nD)])
    tauSR = 1.0 / np.clip(kne, KNE_FLOOR, None)
    np.savez("data/fig4de.npz", DTS=np.array(DTS), WS=np.array(WS),
             kne=kne, phi=phi, tauSR=tauSR)
    print("saved data/fig4de.npz  total", round(time.time()-t0), "s")


if __name__ == "__main__":
    main()
