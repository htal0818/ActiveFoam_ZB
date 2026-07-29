"""Fig 4g / Fig 2g: time-averaged cell shape factor.

 * Fig 4g: dynamic s-bar vs W/T0 for several dT (rho=1).
 * Fig 2g: equilibrium s-bar vs W/T0 up to the vanishing-tension point W/T0=2 (dT=0).

Saves data/shapefactor.npz.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.model import ActiveFoam

os.makedirs("data", exist_ok=True)


def mean_shape(w, dT, n_side=6, burn_tauT=25, avg_tauT=25, seed=0, sample_every=200):
    af = ActiveFoam(n_side=n_side, w=w, dT=dT, seed=seed)
    spt = int(round(af.tauT / af.dt))
    for it in range(int(burn_tauT * spt)):
        af.step()
        if it % 3 == 0:
            af.do_t1_transitions()
    vals = []
    for it in range(int(avg_tauT * spt)):
        af.step()
        if it % 3 == 0:
            af.do_t1_transitions()
        if it % sample_every == 0:
            vals.append(np.mean(af.shape_factors()))
    return np.mean(vals), np.std(vals)


def main():
    t0 = time.time()
    seeds = [1, 2, 3]

    # ---- Fig 4g: dynamic shape factor vs W/T0 ----
    W4 = np.round(np.arange(0.0, 1.21, 0.15), 3)
    DT4 = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    s4 = np.zeros((len(DT4), len(W4)))
    for i, dT in enumerate(DT4):
        for j, w in enumerate(W4):
            sv = [mean_shape(w, dT, seed=s)[0] for s in seeds]
            s4[i, j] = np.mean(sv)
        print(f"[4g] dT={dT}: {np.round(s4[i],3)}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- Fig 2g: equilibrium shape factor vs W/T0 (up to 2) ----
    W2 = np.round(np.arange(1.5, 2.001, 0.1), 3)
    s2 = np.zeros(len(W2))
    s2sd = np.zeros(len(W2))
    for j, w in enumerate(W2):
        sv = [mean_shape(w, 0.0, seed=s, burn_tauT=40, avg_tauT=15)[0] for s in seeds]
        s2[j] = np.mean(sv); s2sd[j] = np.std(sv)
    print(f"[2g] W={W2}\n     s={np.round(s2,3)}  [{time.time()-t0:.0f}s]", flush=True)

    np.savez("data/shapefactor.npz",
             W4=W4, DT4=np.array(DT4), s4=s4,
             W2=W2, s2=s2, s2sd=s2sd)
    print("saved data/shapefactor.npz  total", round(time.time()-t0), "s")


if __name__ == "__main__":
    main()
