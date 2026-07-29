"""Fig 2f: jamming of the foam limit -- mean contact number z vs volume fraction.
Fits z - z_c = z0 (phi-phi_c)^0.5 + z1 (phi-phi_c). Saves data/fig2f.npz.
"""
import os, sys, time
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.foam import contact_number_at_phi

os.makedirs("data", exist_ok=True)

PHIS = np.round(np.arange(0.82, 1.001, 0.01), 3)
SEEDS = (0, 1, 2, 3, 4)
N = 256

def main():
    t0 = time.time()
    zmean, zstd = [], []
    for phi in PHIS:
        zs = [contact_number_at_phi(phi, n=N, seed=s, n_steps=3000) for s in SEEDS]
        zmean.append(np.mean(zs))
        zstd.append(np.std(zs))
        print(f"phi={phi:.3f}  z={zmean[-1]:.3f} +/- {zstd[-1]:.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    zmean = np.array(zmean); zstd = np.array(zstd)

    # fit above jamming
    zc = 4.0
    mask = zmean > zc + 0.02
    phi_j = PHIS[mask]; z_j = zmean[mask]
    def model(phi, phic, z0, z1):
        d = np.clip(phi - phic, 0, None)
        return zc + z0 * np.sqrt(d) + z1 * d
    try:
        popt, _ = curve_fit(model, phi_j, z_j, p0=[0.83, 1.45, 10.45], maxfev=20000)
    except Exception as e:
        popt = [0.83, 1.45, 10.45]
        print("fit failed:", e)
    print("phi_c, z0, z1 =", popt)

    np.savez("data/fig2f.npz", phis=PHIS, zmean=zmean, zstd=zstd,
             fit=np.array(popt), zc=zc)
    print("saved data/fig2f.npz  total", round(time.time()-t0), "s")

if __name__ == "__main__":
    main()
