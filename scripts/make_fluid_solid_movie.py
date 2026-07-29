"""Fig 5j-style movie: fluid MPZ vs solid PSM cell-shape dynamics.

Two active tissues side by side at the same adhesion (W/T0=1) but different
tension-fluctuation magnitude: a high-activity, fluid tissue (fast cell-shape
dynamics, many T1 neighbour exchanges -- the MPZ) and a low-activity, solid
tissue (largely static boundaries -- the PSM).  Cells are coloured by shape
factor s = P/sqrt(A); a few cells are tracked to expose the very different
mobilities.  Writes figures/fluid_solid_movie.gif.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.model import ActiveFoam

N_SIDE = 8
W = 1.0
DT_FLUID, DT_SOLID = 1.5, 0.3     # MPZ (active/fluid) vs PSM (quiet/solid)
SEED = 7
BURN_TAUT = 6
FRAMES = 150
STEPS_PER_FRAME = 90
N_TRACK = 5
CMAP = plt.get_cmap("YlGnBu_r")
CLIM = (3.74, 4.20)


def make(dT):
    af = ActiveFoam(n_side=N_SIDE, w=W, dT=dT, seed=SEED)
    spt = int(round(af.tauT / af.dt))
    for it in range(BURN_TAUT * spt):
        af.step()
        if it % 3 == 0:
            af.do_t1_transitions()
    return af


def patches(af):
    T = af.tissue; L = T.L
    A, P = T.all_areas_perimeters()
    sf = P / np.sqrt(A)
    pats, cv = [], []
    for ci in range(T.n_cells):
        poly = T.cell_polygon(ci)
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                pp = poly + np.array([sx * L, sy * L])
                if (pp[:, 0].max() < -0.03 * L or pp[:, 0].min() > 1.03 * L or
                        pp[:, 1].max() < -0.03 * L or pp[:, 1].min() > 1.03 * L):
                    continue
                pats.append(Polygon(pp, closed=True)); cv.append(sf[ci])
    return pats, np.array(cv)


def main():
    os.makedirs("figures", exist_ok=True)
    afs = {"fluid": make(DT_FLUID), "solid": make(DT_SOLID)}
    Ls = {k: v.tissue.L for k, v in afs.items()}
    tcol = plt.cm.tab10(np.linspace(0, 1, N_TRACK))
    track, prev, unwr, trails = {}, {}, {}, {}
    for k, af in afs.items():
        cen = af.cell_centroids()
        track[k] = list(np.argsort(cen[:, 0] + 0.7 * cen[:, 1])[
            np.linspace(0, af.n_cells - 1, N_TRACK).astype(int)])
        prev[k] = af.cell_centroids(); unwr[k] = prev[k].copy()
        trails[k] = {c: [unwr[k][c].copy()] for c in track[k]}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    n_t1 = {"fluid": 0, "solid": 0}
    imgs = []
    titles = {"fluid": fr"MPZ (fluid) · $\Delta T/T_0={DT_FLUID}$",
              "solid": fr"PSM (solid) · $\Delta T/T_0={DT_SOLID}$"}
    for f in range(FRAMES):
        for k, af in afs.items():
            for it in range(STEPS_PER_FRAME):
                af.step()
                if it % 3 == 0:
                    n_t1[k] += af.do_t1_transitions()
            L = Ls[k]
            cur = af.cell_centroids(); d = cur - prev[k]; d -= L * np.round(d / L)
            unwr[k] = unwr[k] + d; prev[k] = cur
            for c in track[k]:
                trails[k][c].append(unwr[k][c].copy())
        for ax, k in zip(axes, ("fluid", "solid")):
            af = afs[k]; L = Ls[k]
            ax.clear()
            pats, cv = patches(af)
            pc = PatchCollection(pats, cmap=CMAP, edgecolor="#20303a", linewidths=0.7)
            pc.set_array(cv); pc.set_clim(*CLIM); ax.add_collection(pc)
            for i, c in enumerate(track[k]):
                trw = np.mod(np.array(trails[k][c]), L)
                seg = [trw[0]]
                for a, b in zip(trw[:-1], trw[1:]):
                    if np.hypot(*(b - a)) > 0.5 * L:
                        ax.plot(*np.array(seg).T, color=tcol[i], lw=1.5, alpha=.9); seg = [b]
                    else:
                        seg.append(b)
                ax.plot(*np.array(seg).T, color=tcol[i], lw=1.5, alpha=.9)
                ax.plot(*trw[-1], "o", color=tcol[i], ms=5, mec="white", mew=0.8)
            ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{titles[k]}   T1: {n_t1[k]}", fontsize=10)
        fig.suptitle(fr"Active fluidisation ($W/T_0=1$)   "
                     fr"$t={f*STEPS_PER_FRAME*afs['fluid'].dt/afs['fluid'].tauT:4.1f}\,\tau_T$",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw()
        im = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                              fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
        imgs.append(im.resize((760, 420), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=128))
        if f % 25 == 0:
            print(f"frame {f}/{FRAMES}  T1 fluid={n_t1['fluid']} solid={n_t1['solid']}", flush=True)

    out = "figures/fluid_solid_movie.gif"
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=60, loop=0, optimize=True)
    print("wrote", out, f"({os.path.getsize(out)//1024} KB, {len(imgs)} frames)")


if __name__ == "__main__":
    main()
