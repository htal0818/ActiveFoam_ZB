"""Score a model's (W/T0, rho) sweep against the published Fig 2b/2c source data.

Usage:  python scripts/compare_fig2bc_to_source.py [model.npz]
  model.npz needs WS, RHOS, phi[rho, W], z[rho, W] (default data/fig2c_map.npz).
Run scripts/fetch_paper_source_data.py first.  The published grid is
interpolated onto the model's points; NaN model points are skipped.
Writes figures/compare_fig2bc_source.png and prints error summaries.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    mpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "fig2c_map.npz")
    src = np.load(os.path.join(ROOT, "data", "paper_fig2bc_source.npz"))
    m = np.load(mpath)
    R, W = src["rhos"], src["Ws"]
    WW, RR = np.meshgrid(m["WS"], m["RHOS"])
    inside = (WW <= W[-1]) & (RR >= R[0]) & (RR <= R[-1])
    pts = np.c_[np.clip(RR.ravel(), R[0], R[-1]), np.clip(WW.ravel(), W[0], W[-1])]
    ref = {q: RegularGridInterpolator((R, W), src[q])(pts).reshape(WW.shape) for q in ("phi", "z")}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    spec = {"phi": (r"$\phi$", 0.65, 1.0, "viridis", [0.999]),
            "z": (r"$z$", 1.5, 6.0, "RdYlBu_r", [4, 5, 5.99])}
    for row, q in enumerate(("phi", "z")):
        lab, lo, hi, cmap, levels = spec[q]
        ax = axes[row, 0]
        im = ax.pcolormesh(W, R, src[q], cmap=cmap, vmin=lo, vmax=hi, shading="nearest")
        cs = ax.contour(W, R, src[q], levels=levels, colors="k", linewidths=1)
        ax.clabel(cs, fmt="%.3g", fontsize=8)
        ax.set_title(f"Published source data: {lab}  (16 x 41 grid)", fontsize=10)
        fig.colorbar(im, ax=ax)
        ax = axes[row, 1]
        mod = np.ma.masked_invalid(m[q])
        ax.set_facecolor("0.8")
        im = ax.pcolormesh(m["WS"], m["RHOS"], mod, cmap=cmap, vmin=lo, vmax=hi, shading="nearest")
        ax.set_xlim(W[0] - 0.01, W[-1] + 0.01); ax.set_ylim(R[0] - 0.02, R[-1] + 0.02)
        ax.set_title(f"Model: {lab}  ({os.path.basename(mpath)}; grey = no valid run)", fontsize=10)
        fig.colorbar(im, ax=ax)
        ax = axes[row, 2]
        diff = np.where(inside, m[q] - ref[q], np.nan)
        lim = 0.15 if q == "phi" else 1.5
        ax.set_facecolor("0.8")
        im = ax.pcolormesh(m["WS"], m["RHOS"], np.ma.masked_invalid(diff), cmap="RdBu_r",
                           vmin=-lim, vmax=lim, shading="nearest")
        ax.set_xlim(W[0] - 0.01, W[-1] + 0.01); ax.set_ylim(R[0] - 0.02, R[-1] + 0.02)
        ok = np.isfinite(diff)
        ax.set_title(f"Model - published: mean {np.nanmean(diff):+.3f}, "
                     f"RMS {np.sqrt(np.nanmean(diff**2)):.3f}, n={ok.sum()}", fontsize=10)
        fig.colorbar(im, ax=ax)
        print(f"{q}: n={ok.sum()} mean diff={np.nanmean(diff):+.4f} "
              f"RMS={np.sqrt(np.nanmean(diff**2)):.4f} max|diff|={np.nanmax(abs(diff)):.3f}")
        for a in axes[row]:
            a.set_xlabel(r"$W/T_0$"); a.set_ylabel(r"$\rho$")
    fig.suptitle("Fig. 2b (volume fraction) and 2c (mean number of contacts): "
                 "published source data vs model", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(ROOT, "figures", "compare_fig2bc_source.png")
    fig.savefig(out, dpi=110); print("wrote", out)


if __name__ == "__main__":
    main()
