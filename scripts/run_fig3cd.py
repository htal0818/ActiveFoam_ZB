"""Fig 3c (MSD at long times) and Fig 3d (cellular NE rate) vs adhesion W/T0.

Both quantities are non-monotonic in W/T0 with a minimum at the structural
(non-confluent <-> confluent) transition: the tissue is maximally solid-like
there.  Low W/T0 is mobile because cell contacts continually break and reform
(non-confluent); high W/T0 is mobile because tension fluctuations drive T1
neighbour exchanges (confluent).

Measured directly from the curved-edge foam model, which has BOTH branches:
  * cellular NE rate k_NE = T1 events on cell-cell edges per cell per tau_R       (Fig 3d, log)
  * MSD of the 36 persistent cell centroids over the measurement window,
    in the tissue centre-of-mass frame, in units of L0^2                     (Fig 3c)

Saves data/fig3cd.npz.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.topology import build_periodic_voronoi
from activefoam.foam_full import FoamTissue
from activefoam.budget import call_with_budget, BudgetExceeded

os.makedirs("data", exist_ok=True)

WS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
DTS = [0.25, 0.5, 1.0, 1.5]
SEEDS = [1, 2]
EQUIL = 250
WIN_TAUR = 20.0
TRANS_EVERY = 15
SAMPLE = 40          # MSD trajectory sampling interval (steps)
VCAP = 420
MAX_SECONDS = 1200     # per run; a run that crawls (tangled) is recorded as invalid
PHI_MAX = 1.05         # phi above this means self-overlapping cells: invalid


def _centroids(ft, ncell):
    c = np.zeros((ncell, 2))
    for ci in range(ncell):
        _, cen = ft.cell_area_center(ci)
        c[ci] = cen
    return c


def measure(dt_act, w, seed=1):
    T = build_periodic_voronoi(6, np.random.default_rng(100 + seed))
    ft = FoamTissue(T, w=w, rho=1.0, seed=seed)
    ncell = ft.nFa
    L = ft.bs
    for _ in range(150):
        ft.step(mu=dt_act)
    for v in range(len(ft.vpos)):                 # open spaces at weak regions
        ft.t2_reverse(v)
    ft._update_faces()
    frag = False
    for it in range(EQUIL):
        ft.step(mu=dt_act)
        if it % TRANS_EVERY == 0:
            ft.do_transitions(mu=dt_act, create_spaces=False)
        if len(ft.vpos) > VCAP:
            frag = True; break

    steps = int(round(WIN_TAUR / ft.dt))
    prev = _centroids(ft, ncell)
    unwrapped = prev.copy()
    traj, n_ev, phis = [unwrapped.copy()], 0, []
    for it in range(steps):
        ft.step(mu=dt_act)
        if it % TRANS_EVERY == 0:
            n1, n2 = ft.do_transitions(mu=dt_act, create_spaces=False)
            n_ev += n1 + n2
        if len(ft.vpos) > VCAP:
            frag = True
        if it % SAMPLE == 0:
            cur = _centroids(ft, ncell)
            d = cur - prev
            d -= L * np.round(d / L)              # min-image displacement
            unwrapped = unwrapped + d
            prev = cur
            traj.append(unwrapped.copy())
            phis.append(ft.volume_fraction())
    traj = np.asarray(traj)                        # (nS, ncell, 2)
    traj = traj - traj.mean(axis=1, keepdims=True)  # centre-of-mass frame
    # time-averaged MSD at the largest available lag
    nS = len(traj)
    lag = nS - 1
    disp = traj[lag:] - traj[:-lag]
    msd = float(np.mean(np.sum(disp ** 2, axis=2))) / (ft.lsc ** 2)
    kne = n_ev / ncell / (steps * ft.dt)
    return kne, msd, float(np.median(phis))


def _task(a):
    i, j, d, w, s = a
    try:
        k, m, p = call_with_budget(MAX_SECONDS, measure, d, w, seed=s)
        if not (np.isfinite(k) and np.isfinite(m)):
            raise FloatingPointError("non-finite result")
        if not p <= PHI_MAX:
            raise FloatingPointError(f"tangled, phi={p:.2f}")
    except (Exception, BudgetExceeded) as ex:                       # one bad run must not kill the sweep
        print(f"  !! dT={d} W={w} seed={s} failed: {ex!r}", flush=True)
        k = m = p = np.nan
    return i, j, k, m, p


def main():
    from multiprocessing import Pool
    t0 = time.time()
    nD, nW = len(DTS), len(WS)
    tasks = [(i, j, d, w, s) for i, d in enumerate(DTS)
             for j, w in enumerate(WS) for s in SEEDS]
    acc = {q: {(i, j): [] for i in range(nD) for j in range(nW)} for q in "kmp"}
    with Pool(processes=4) as pool:
        for n, (i, j, k, m, p) in enumerate(pool.imap_unordered(_task, tasks), 1):
            acc["k"][(i, j)].append(k); acc["m"][(i, j)].append(m); acc["p"][(i, j)].append(p)
            print(f"[{n}/{len(tasks)}] dT={DTS[i]:.2f} W={WS[j]:.2f}: "
                  f"kNE={k:.4f} MSD={m:.4f} phi={p:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    def mean(q):
        out = np.full((nD, nW), np.nan)
        for (i, j), v in acc[q].items():
            v = [x for x in v if np.isfinite(x)]
            if v:
                out[i, j] = np.mean(v)
        return out
    kne, msd, phi = mean("k"), mean("m"), mean("p")
    np.savez("data/fig3cd.npz", DTS=np.array(DTS), WS=np.array(WS),
             kne=kne, msd=msd, phi=phi, win_taur=WIN_TAUR,
             kne_floor=1.0 / (36 * WIN_TAUR * len(SEEDS)))   # one event in the window
    print("saved data/fig3cd.npz  total", round(time.time()-t0), "s")


if __name__ == "__main__":
    main()
