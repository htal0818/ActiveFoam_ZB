"""Render a movie of the active-foam dynamics (confluent vertex model).

Cells fluctuate and rearrange (T1 neighbour exchanges) under active tension
fluctuations; a few cells are tracked to show their trajectories (cf. Fig 3b).
Writes an animated GIF to figures/active_foam_movie.gif.
"""
import os, sys, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activefoam.model import ActiveFoam
from activefoam.topology import min_image

# ---- parameters -----------------------------------------------------------
N_SIDE = 8            # 64 cells
W = 1.0              # W/T0
DT = 1.0            # Delta T / T0  (active, uncaging regime)
SEED = 7
BURN_TAUT = 8
FRAMES = 190
STEPS_PER_FRAME = 130  # ~0.65 tau_R per frame -> ~24 tau_T total
N_TRACK = 6

CMAP = "YlGnBu_r"


def tissue_patches(af, shift_track):
    """Build unwrapped cell polygons + shape-factor colours for the primary box."""
    T = af.tissue
    L = T.L
    A, P = T.all_areas_perimeters()
    sf = P / np.sqrt(A)
    patches, cvals = [], []
    for ci in range(T.n_cells):
        poly = T.cell_polygon(ci)
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                pp = poly + np.array([sx * L, sy * L])
                if (pp[:, 0].max() < -0.03 * L or pp[:, 0].min() > 1.03 * L or
                        pp[:, 1].max() < -0.03 * L or pp[:, 1].min() > 1.03 * L):
                    continue
                patches.append(Polygon(pp, closed=True))
                cvals.append(sf[ci])
    return patches, np.array(cvals)


def main():
    af = ActiveFoam(n_side=N_SIDE, w=W, dT=DT, seed=SEED)
    L = af.tissue.L
    spt = int(round(af.tauT / af.dt))
    for it in range(BURN_TAUT * spt):
        af.step()
        if it % 3 == 0:
            af.do_t1_transitions()

    # pick tracked cells spread across the box
    cen0 = af.cell_centroids()
    track = list(np.argsort(cen0[:, 0] + 0.7 * cen0[:, 1])[
        np.linspace(0, af.n_cells - 1, N_TRACK).astype(int)])
    tcol = plt.cm.tab10(np.linspace(0, 1, N_TRACK))
    prev = af.cell_centroids()
    unwrapped = prev.copy()
    trails = {c: [unwrapped[c].copy()] for c in track}

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    fig.subplots_adjust(0, 0, 1, 0.94)
    imgs = []
    n_t1_total = 0
    for f in range(FRAMES):
        for k in range(STEPS_PER_FRAME):
            af.step()
            it = f * STEPS_PER_FRAME + k
            if it % 3 == 0:
                n_t1_total += af.do_t1_transitions()
        cur = af.cell_centroids()
        d = cur - prev
        d -= L * np.round(d / L)
        unwrapped = unwrapped + d
        prev = cur
        for c in track:
            trails[c].append(unwrapped[c].copy())

        ax.clear()
        patches, cvals = tissue_patches(af, None)
        pc = PatchCollection(patches, cmap=CMAP, edgecolor="#20303a", linewidths=0.7)
        pc.set_array(cvals); pc.set_clim(3.74, 4.15)
        ax.add_collection(pc)
        # trajectories, wrapped into the box, drawn as fading trails
        for i, c in enumerate(track):
            tr = np.array(trails[c])
            trw = np.mod(tr, L)
            # split where it wraps to avoid long horizontal jumps
            seg = [trw[0]]
            for p_prev, p_cur in zip(trw[:-1], trw[1:]):
                if np.hypot(*(p_cur - p_prev)) > 0.5 * L:
                    ax.plot(*np.array(seg).T, color=tcol[i], lw=1.6, alpha=0.9)
                    seg = [p_cur]
                else:
                    seg.append(p_cur)
            ax.plot(*np.array(seg).T, color=tcol[i], lw=1.6, alpha=0.9)
            ax.plot(*trw[-1], "o", color=tcol[i], ms=6, mec="white", mew=0.8)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Active foam  ·  ρ=1, W/T₀={W}, ΔT/T₀={DT}   "
                     f"t={f*STEPS_PER_FRAME*af.dt/af.tauT:4.1f} τ_T   "
                     f"T1 events: {n_t1_total}", fontsize=10)
        fig.canvas.draw()
        im = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                              fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
        im = im.resize((540, 540), Image.LANCZOS)
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
        if f % 40 == 0:
            print(f"frame {f}/{FRAMES}  T1s={n_t1_total}", flush=True)

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures", "active_foam_movie.gif")
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=55, loop=0, optimize=True)
    print("wrote", out, f"({os.path.getsize(out)//1024} KB, {len(imgs)} frames)")


if __name__ == "__main__":
    main()
