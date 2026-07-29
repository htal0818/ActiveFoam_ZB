"""Fig 4h: dynamic cell-shape snapshots at fixed adhesion (W/T0=1) for
increasing tension-fluctuation magnitude Delta T/T0.

Confluent dynamic vertex model; cells coloured by shape factor s = P/sqrt(A)
(paper colourbar 3.6..>4.6).  Higher activity -> larger, more heterogeneous
shape factors and a fluid state (red outline); low activity stays low and
uniform (solid, blue outline), reproducing the paper's s_bar ~ 3.83 / 3.89 / 4.08.
Writes figures/compare_fig4h_snapshots.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.model import ActiveFoam

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS = os.path.join(ROOT, "figures", "paper_panels")

N_SIDE = 8              # 64 cells
W = 1.0
DTS = [1.5, 1.0, 0.5]  # top -> bottom, as in the paper
SEED = 5
BURN_TAUT = 12
CMAP = plt.get_cmap("RdYlBu_r")
NORM = mcolors.Normalize(vmin=3.6, vmax=4.6)
RIGID = 3.85           # fluid (red frame) above, solid (blue frame) below


def config(dT):
    af = ActiveFoam(n_side=N_SIDE, w=W, dT=dT, seed=SEED)
    spt = int(round(af.tauT / af.dt))
    for it in range(BURN_TAUT * spt):
        af.step()
        if it % 3 == 0:
            af.do_t1_transitions()
    return af


def draw(af, ax):
    T = af.tissue
    L = T.L
    A, P = T.all_areas_perimeters()
    sf = P / np.sqrt(A)
    pats, cols = [], []
    for ci in range(T.n_cells):
        poly = T.cell_polygon(ci)
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                pp = poly + np.array([sx * L, sy * L])
                if (pp[:, 0].max() < -0.03 * L or pp[:, 0].min() > 1.03 * L or
                        pp[:, 1].max() < -0.03 * L or pp[:, 1].min() > 1.03 * L):
                    continue
                pats.append(Polygon(pp, closed=True)); cols.append(CMAP(NORM(sf[ci])))
    ax.add_collection(PatchCollection(pats, facecolor=cols, edgecolor="k", lw=0.5))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    sbar = float(np.mean(sf))
    fluid = sbar > RIGID
    for sp in ax.spines.values():
        sp.set_edgecolor("#e3271f" if fluid else "#2b3fd6"); sp.set_linewidth(3)
    return sbar, fluid


def main():
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig = plt.figure(figsize=(9.5, 7.0))
    axp = fig.add_axes([0.02, 0.02, 0.30, 0.86])
    cp = os.path.join(PANELS, "crop_fig4h.png")
    if os.path.exists(cp):
        axp.imshow(mpimg.imread(cp))
    axp.set_title("Paper (Kim et al. 2021)", fontsize=10, pad=2); axp.axis("off")

    for i, dT in enumerate(DTS):
        af = config(dT)
        ax = fig.add_axes([0.40, 0.60 - 0.29 * i, 0.33, 0.28])
        sbar, fluid = draw(af, ax)
        ax.text(0.03, 0.93, fr"$\Delta T/T_0={dT}$", transform=ax.transAxes,
                fontsize=10, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.text(0.97, 0.06, fr"$\bar s={sbar:.2f}$", transform=ax.transAxes,
                fontsize=10, ha="right",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.text(-0.04, 0.5, "fluid" if fluid else "solid", transform=ax.transAxes,
                va="center", ha="right", rotation=90, fontsize=9,
                color="#e3271f" if fluid else "#2b3fd6")
        print(f"dT={dT}: s_bar={sbar:.3f} ({'fluid' if fluid else 'solid'})", flush=True)

    cax = fig.add_axes([0.80, 0.15, 0.025, 0.62])
    fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, label=r"$s$")
    fig.text(0.5, 0.955, r"Fig. 4h  |  Cell-shape snapshots vs activity "
             r"($W/T_0=1$) — shapes grow and fluidise with $\Delta T/T_0$",
             ha="center", fontsize=12)
    out = os.path.join(ROOT, "figures", "compare_fig4h_snapshots.png")
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    main()
