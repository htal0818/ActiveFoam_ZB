"""Fig 3a: MSD(t) for varying tension fluctuations (rho=1, W/T0=1), + alpha inset.
Also collects Fig 3c/3d style NE-rate vs W/T0.  Saves data/fig3.npz.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.measure import run_msd, msd_exponent

os.makedirs("data", exist_ok=True)

DT_LIST = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
SEEDS = list(range(1, 9))        # 8 independent simulations
N_SIDE = 6
T_MAX = 100.0

def main():
    t0 = time.time()
    curves = {}
    alphas = {}
    ne_rates = {}
    for dT in DT_LIST:
        msd_stack = []
        ne_stack = []
        tlag = None
        for s in SEEDS:
            tl, msd, ne = run_msd(dT, w=1.0, n_side=N_SIDE, t_max_tauT=T_MAX,
                                  burn_tauT=20.0, seed=s)
            tlag = tl
            msd_stack.append(msd)
            ne_stack.append(ne)
        msd_mean = np.mean(msd_stack, axis=0)
        curves[dT] = (tlag, msd_mean, np.std(msd_stack, axis=0))
        # asymptotic exponent for t >> tau_T
        alphas[dT] = msd_exponent(tlag, msd_mean, t_lo=10.0, t_hi=100.0)
        ne_rates[dT] = np.mean(ne_stack)
        print(f"dT={dT:.2f}  alpha={alphas[dT]:.2f}  "
              f"MSD(t=100)={msd_mean[-1]:.3e}  NE={ne_rates[dT]:.2e}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    np.savez("data/fig3.npz",
             dT_list=np.array(DT_LIST),
             tlag=np.array([curves[d][0] for d in DT_LIST]),
             msd=np.array([curves[d][1] for d in DT_LIST]),
             msd_std=np.array([curves[d][2] for d in DT_LIST]),
             alpha=np.array([alphas[d] for d in DT_LIST]),
             ne_rate=np.array([ne_rates[d] for d in DT_LIST]))
    print("saved data/fig3.npz  total", round(time.time()-t0), "s")

if __name__ == "__main__":
    main()
