"""Fig 4a: shear-stress relaxation after an affine strain step, for varying dT.
Fits a stretched exponential and reports the relaxation time.  Saves data/fig4a.npz.
"""
import os, sys, time
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.measure import run_stress_relaxation

os.makedirs("data", exist_ok=True)

DT_LIST = [0.0, 0.5, 1.0, 1.5]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
W = 0.5
STRAIN = 1.5           # large strain step, matching the paper (eps_xy = 1.5)
RELAX = 250.0          # tau_R
N_SIDE = 8             # 64 cells -> cleaner long-time stress statistics

def stretched(t, sinf, s0, tau, beta):
    return sinf + (s0 - sinf) * np.exp(-(np.clip(t, 0, None) / tau) ** beta)

def main():
    t0 = time.time()
    out = {}
    for dT in DT_LIST:
        sig_stack = []
        sA_stack = []
        ts = None
        for s in SEEDS:
            t, sig, sA = run_stress_relaxation(dT, w=W, strain=STRAIN,
                                               relax_tauR=RELAX, seed=s,
                                               n_side=N_SIDE)
            ts = t
            sig_stack.append(sig)
            sA_stack.append(sA)
        sig_mean = np.mean(sig_stack, axis=0)
        sig_sd = np.std(sig_stack, axis=0)
        sA = np.mean(sA_stack)
        out[dT] = (ts, sig_mean, sig_sd, sA)
        # fit stretched exponential (skip the very first fast jump)
        tt = ts / 10.0  # in tau_T for reporting; keep tau_R for fit
        try:
            p0 = [max(sA, sig_mean[-1]), sig_mean[3], 30.0, 0.5]
            popt, _ = curve_fit(stretched, ts, sig_mean, p0=p0, maxfev=40000,
                                bounds=([0, 0, 0.1, 0.1], [2, 3, 1e6, 1.5]))
        except Exception as e:
            popt = [np.nan] * 4
            print("fit failed", dT, e)
        print(f"dT={dT:.2f}  sigma_A={sA:.3f}  sigma(end)={sig_mean[-1]:.3f}  "
              f"tauSR={popt[2]:.1f} beta={popt[3]:.2f}  [{time.time()-t0:.0f}s]",
              flush=True)
        out[dT] = (ts, sig_mean, sig_sd, sA, np.array(popt))

    np.savez("data/fig4a.npz",
             dT_list=np.array(DT_LIST),
             ts=out[DT_LIST[0]][0],
             sig=np.array([out[d][1] for d in DT_LIST]),
             sig_sd=np.array([out[d][2] for d in DT_LIST]),
             sigma_A=np.array([out[d][3] for d in DT_LIST]),
             fit=np.array([out[d][4] for d in DT_LIST]))
    print("saved data/fig4a.npz  total", round(time.time()-t0), "s")

if __name__ == "__main__":
    main()
