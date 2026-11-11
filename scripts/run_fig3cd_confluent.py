"""Long-time extension of Fig 3c/3d on the confluent branch.

The foam-model sweep (run_fig3cd.py) covers the whole adhesion range but only
over a 20 tau_R window, so it cannot resolve NE rates below ~1e-3 nor the
paper's MSD lag of 10^2 tau_T.  Where the tissue is confluent (phi = 1,
W/T0 >= 0.75 for all activities studied) the fast vectorised confluent model
describes the same dynamics and runs ~100x longer:

  * MSD at the paper's lag t = 10^2 tau_T (cell centroids, centre-of-mass frame)
  * NE rate = net cell-cell contacts lost between snapshots one tau_T apart,
    per cell per tau_R.  (Net losses, not raw T1 flips: an edge that flips and
    flips straight back is not a neighbour exchange.)

Saves data/fig3cd_confluent.npz.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.model import ActiveFoam

os.makedirs("data", exist_ok=True)

WS = [0.6, 0.75, 0.9, 1.05, 1.2]
DTS = [0.25, 0.5, 1.0, 1.5]
SEEDS = [1]
BURN_TAUT = 20
RUN_TAUT = 200
LAG_TAUT = 100


def contact_pairs(af):
    """Set of neighbouring cell pairs (cells sharing an edge)."""
    owner = {}
    pairs = set()
    for ci, loop in enumerate(af.tissue.cells):
        n = len(loop)
        for k in range(n):
            a, b = loop[k], loop[(k + 1) % n]
            key = (a, b) if a < b else (b, a)
            if key in owner:
                cj = owner[key]
                if cj != ci:
                    pairs.add((cj, ci) if cj < ci else (ci, cj))
            else:
                owner[key] = ci
    return pairs


def measure(dT, w, seed):
    af = ActiveFoam(n_side=6, w=w, dT=dT, seed=seed)
    spt = int(round(af.tauT / af.dt))            # steps per tau_T
    for it in range(BURN_TAUT * spt):
        af.step()
        if it % 3 == 0:
            af.do_t1_transitions()
    L = af.tissue.L
    prev = af.cell_centroids()
    unwrapped = prev.copy()
    traj = [unwrapped.copy()]
    pairs = contact_pairs(af)
    lost = 0
    for it in range(1, RUN_TAUT * spt + 1):
        af.step()
        if it % 3 == 0:
            af.do_t1_transitions()
        if it % 200 == 0:                        # keep the unwrapping well resolved
            cur = af.cell_centroids()
            d = cur - prev
            d -= L * np.round(d / L)
            unwrapped = unwrapped + d
            prev = cur
        if it % spt == 0:
            traj.append(unwrapped.copy())
            new = contact_pairs(af)
            lost += len(pairs - new)
            pairs = new
    traj = np.asarray(traj)
    traj = traj - traj.mean(axis=1, keepdims=True)
    disp = traj[LAG_TAUT:] - traj[:-LAG_TAUT]
    msd = float(np.mean(np.sum(disp ** 2, axis=2)))          # L0 = 1 at rho = 1
    kne = lost / af.n_cells / (RUN_TAUT * af.tauT)
    return kne, msd


def _task(a):
    i, j, d, w, s = a
    k, m = measure(d, w, s)
    return i, j, k, m


def main():
    from multiprocessing import Pool
    t0 = time.time()
    nD, nW = len(DTS), len(WS)
    tasks = [(i, j, d, w, s) for i, d in enumerate(DTS)
             for j, w in enumerate(WS) for s in SEEDS]
    kacc = {(i, j): [] for i in range(nD) for j in range(nW)}
    macc = {(i, j): [] for i in range(nD) for j in range(nW)}
    with Pool(processes=4) as pool:
        for n, (i, j, k, m) in enumerate(pool.imap_unordered(_task, tasks), 1):
            kacc[(i, j)].append(k); macc[(i, j)].append(m)
            print(f"[{n}/{len(tasks)}] dT={DTS[i]:.2f} W={WS[j]:.2f}: "
                  f"kNE={k:.2e} MSD(100 tauT)={m:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    kne = np.array([[np.mean(kacc[(i, j)]) for j in range(nW)] for i in range(nD)])
    msd = np.array([[np.mean(macc[(i, j)]) for j in range(nW)] for i in range(nD)])
    np.savez("data/fig3cd_confluent.npz", DTS=np.array(DTS), WS=np.array(WS),
             kne=kne, msd=msd, lag_taut=LAG_TAUT,
             kne_floor=1.0 / (36 * RUN_TAUT * 10.0 * len(SEEDS)))
    print("saved data/fig3cd_confluent.npz  total", round(time.time()-t0), "s")


if __name__ == "__main__":
    main()
